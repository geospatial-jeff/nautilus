#!/usr/bin/env python3
"""Distributed shuffle benchmark: run a shuffle-heavy pipeline across the daemon cluster under each
placement — intra-node (all daemons on one host, shuffle over loopback) and cross-node (one daemon per
host, shuffle over the pod network) — and print the throughput and wire-cost difference. That difference
is the cross-node shuffle penalty, the number a Rust port's networking most has to answer to.

It drives ``kubectl`` only (no Python cluster client): for each overlay it applies the kustomization,
waits for the daemons to be Ready, then for each benchmark runs a coordinator Job that dials the daemons
and prints its telemetry report, which this parses back for the headline metrics.

    python bench.py --image <acct>.dkr.ecr.<region>.amazonaws.com/nautilus:<tag>

Prereqs: the image built + pushed, ``kubectl`` pointed at the cluster, the cluster secret created (see
README), and — for cross-node — at least ``--replicas`` schedulable nodes; for intra-node, one node
labeled ``nautilus.dev/bench-node=intra-node`` big enough to fit them all.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
COORDINATOR = "nautilus-coordinator"
BEGIN, END = "NAUTILUS_REPORT_BEGIN", "NAUTILUS_REPORT_END"


def kubectl(*args: str, stdin: str | None = None, capture: bool = True) -> str:
    """Run kubectl, returning stdout. Raises on a non-zero exit so a broken step stops the sweep loudly."""
    out = subprocess.run(
        ["kubectl", *args],
        input=stdin,
        capture_output=capture,
        text=True,
    )
    if out.returncode != 0:
        sys.exit(f"kubectl {' '.join(args)} failed:\n{out.stderr or out.stdout}")
    return out.stdout


def deploy(overlay: str, image: str, replicas: int, ns: str) -> None:
    """Apply an overlay (which sets the daemon placement) with the chosen image + replica count, then wait
    for every daemon to be Ready. Switching overlays re-schedules the daemons onto their new placement.
    """
    rendered = kubectl("kustomize", str(HERE / "overlays" / overlay))
    # Patch the image and replica count into the rendered manifests before applying, so the same overlay
    # serves any registry/size without editing files.
    docs = []
    for doc in rendered.split("\n---\n"):
        doc = doc.replace("image: nautilus:latest", f"image: {image}")
        if "kind: StatefulSet" in doc:
            doc = doc.replace("replicas: 4", f"replicas: {replicas}")
        docs.append(doc)
    kubectl("apply", "-f", "-", stdin="\n---\n".join(docs))
    print(f"  waiting for {replicas} daemons ({overlay})…", flush=True)
    kubectl(
        "-n",
        ns,
        "rollout",
        "status",
        "statefulset/nautilus-worker",
        "--timeout=300s",
        capture=False,
    )


def coordinator_job(
    bench: str, parallelism: int, replicas: int, scale: dict[str, str], image: str, ns: str
) -> dict:
    """The coordinator Job as a dict: dial the N daemons by stable DNS, run ``bench`` at ``parallelism``
    (so the keyed shuffle is a real N×N all-to-all), and bracket the JSON report with sentinels so this
    harness can lift it back out of the pod log."""
    daemons = ",".join(f"nautilus-worker-{i}.nautilus-workers:9000" for i in range(replicas))
    cmd = (
        f"nautilus run {bench} --parallelism {parallelism} --daemons {daemons} "
        f"--telemetry full --save /tmp/r.json --show none "
        f"&& echo {BEGIN} && cat /tmp/r.json && echo {END}"
    )
    env = [
        {
            "name": "NAUTILUS_CLUSTER_SECRET",
            "valueFrom": {"secretKeyRef": {"name": "nautilus-cluster-secret", "key": "secret"}},
        }
    ]
    env += [{"name": k, "value": v} for k, v in scale.items()]
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": COORDINATOR, "namespace": ns},
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 600,
            "template": {
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "coordinator",
                            "image": image,
                            "command": ["sh", "-c", cmd],
                            "env": env,
                            "resources": {"requests": {"cpu": "1", "memory": "1Gi"}},
                        }
                    ],
                }
            },
        },
    }


def run_bench(
    bench: str, parallelism: int, replicas: int, scale: dict[str, str], image: str, ns: str
) -> dict:
    """Run one benchmark on the current placement and return its parsed telemetry report."""
    kubectl("-n", ns, "delete", "job", COORDINATOR, "--ignore-not-found", "--wait=true")
    kubectl(
        "apply",
        "-f",
        "-",
        stdin=json.dumps(coordinator_job(bench, parallelism, replicas, scale, image, ns)),
    )
    # Poll for terminal state — kubectl wait can only block on one condition, and we want either.
    for _ in range(120):
        status = kubectl("-n", ns, "get", "job", COORDINATOR, "-o", "json")
        st = json.loads(status).get("status", {})
        if st.get("succeeded"):
            break
        if st.get("failed"):
            log = kubectl("-n", ns, "logs", f"job/{COORDINATOR}", "--tail=40")
            sys.exit(f"coordinator failed on {bench}:\n{log}")
        time.sleep(5)
    else:
        sys.exit(f"coordinator timed out on {bench}")
    log = kubectl("-n", ns, "logs", f"job/{COORDINATOR}")
    blob = log.split(BEGIN, 1)[-1].split(END, 1)[0].strip()
    return json.loads(blob)


def metrics(report: dict) -> dict[str, float]:
    """The headline numbers from a report. transport_mb should be ~equal across placements (same data
    shuffled); the interesting deltas are rows_per_s (network-bound cross-node) and credit_wait (flow
    control stalling on the socket)."""
    wall_s = report["meta"]["wall_micros"] / 1e6
    rows_out = report["summary"]["total_rows_out"]

    def total(counter: str) -> float:
        return sum(
            c["value"]
            for op in report.get("operators", [])
            for c in op.get("counters", [])
            if c["name"] == counter
        )

    return {
        "rows_per_s": rows_out / wall_s if wall_s else 0.0,
        "transport_mb": total("transport.bytes_sent") / 1e6,
        "credit_wait_ms": total("edge.credit_wait_micros") / 1e3,
        "wall_s": wall_s,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--image", required=True, help="the nautilus image the daemons + coordinator run"
    )
    ap.add_argument("--namespace", default="nautilus-bench")
    ap.add_argument(
        "--replicas", type=int, default=4, help="daemon count (>= this many nodes for cross-node)"
    )
    ap.add_argument(
        "--parallelism", type=int, default=0, help="operator parallelism (default: = replicas)"
    )
    ap.add_argument(
        "--benches", default="bench-keyed,bench-chain", help="comma-separated shuffle benchmarks"
    )
    ap.add_argument("--overlays", default="intra-node,cross-node", help="placements to compare")
    ap.add_argument("--rows", default="20000000")
    ap.add_argument("--batch", default="4096")
    ap.add_argument("--keys", default="10000")
    args = ap.parse_args()

    parallelism = args.parallelism or args.replicas
    scale = {
        "NAUTILUS_BENCH_ROWS": args.rows,
        "NAUTILUS_BENCH_BATCH": args.batch,
        "NAUTILUS_BENCH_KEYS": args.keys,
    }
    benches = args.benches.split(",")
    overlays = args.overlays.split(",")

    results: dict[tuple[str, str], dict[str, float]] = {}
    for overlay in overlays:
        print(f"\n=== placement: {overlay} ===", flush=True)
        deploy(overlay, args.image, args.replicas, args.namespace)
        for bench in benches:
            print(f"  running {bench}…", flush=True)
            report = run_bench(bench, parallelism, args.replicas, scale, args.image, args.namespace)
            results[(overlay, bench)] = metrics(report)

    # Comparison: for each benchmark, intra vs cross throughput and the cross-node penalty.
    print(f"\n=== results ({args.replicas} daemons, parallelism {parallelism}) ===")
    print(
        f"{'benchmark':16} {'placement':12} {'rows/s':>14} {'transport MB':>13} {'credit-wait ms':>15}"
    )
    for bench in benches:
        for overlay in overlays:
            m = results.get((overlay, bench))
            if m:
                print(
                    f"{bench:16} {overlay:12} {m['rows_per_s']:14,.0f} {m['transport_mb']:13,.1f} {m['credit_wait_ms']:15,.1f}"
                )
        intra = results.get(("intra-node", bench))
        cross = results.get(("cross-node", bench))
        if intra and cross and intra["rows_per_s"]:
            penalty = (intra["rows_per_s"] - cross["rows_per_s"]) / intra["rows_per_s"] * 100
            print(f"{bench:16} {'':12} {'cross-node penalty:':>28} {penalty:+.1f}% throughput")


if __name__ == "__main__":
    main()
