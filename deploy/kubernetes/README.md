# Distributed shuffle benchmark on Kubernetes

Measure the keyed shuffle's throughput two ways and diff them: **intra-node** (all worker daemons on one
host, so the shuffle crosses containers but stays on loopback) versus **cross-node** (one daemon per host,
so the shuffle crosses the real pod-to-pod network). That difference is the **cross-node network penalty** —
the cost a single-box `nautilus dashboard --workers` run cannot show, and the one a Rust port's transport
has to answer to.

## Why the same manifests test both

`placement.py` maps each operator instance to a *worker daemon*, round-robin, and knows nothing about
physical nodes. So whether two daemons share a host is decided entirely by the Kubernetes scheduler — which
means intra vs cross is a one-line change in the overlay, not a code change:

- **`overlays/cross-node`** — `podAntiAffinity` puts each daemon on its own node.
- **`overlays/intra-node`** — a `nodeSelector` pins them all to one labeled node.

The daemons themselves are identical to what `docker-compose.yml` runs (the integration-tested multi-node
path): a headless Service gives each StatefulSet pod a stable DNS name the coordinator dials on the control
port, while the shuffle's data edges use an **ephemeral pod-to-pod port** (each daemon advertises its pod
IP), so only the control port needs a Service.

## Prerequisites

1. **Build and push the image** (repo-root `Dockerfile` — one image runs both the daemon and coordinator):

   ```
   docker build -t <acct>.dkr.ecr.<region>.amazonaws.com/nautilus:<tag> .
   docker push  <acct>.dkr.ecr.<region>.amazonaws.com/nautilus:<tag>
   ```

2. **`kubectl`** pointed at the cluster.

3. **The cluster secret** (every node authenticates with it):

   ```
   kubectl create namespace nautilus-bench
   kubectl -n nautilus-bench create secret generic nautilus-cluster-secret \
     --from-literal=secret=$(openssl rand -hex 32)
   ```

4. **Nodes.** Cross-node needs at least `--replicas` schedulable nodes (default 4). Intra-node needs one
   node big enough for all daemons — label it, and use the same instance type as the cross-node nodes so the
   only variable between the two runs is the shuffle's network path:

   ```
   kubectl label node <node-name> nautilus.dev/bench-node=intra-node
   ```

## Run the comparison

```
python deploy/kubernetes/bench.py --image <acct>.dkr.ecr.<region>.amazonaws.com/nautilus:<tag>
```

It deploys each placement, runs each benchmark across the daemons, and prints rows/s, transport MB, and
credit-wait (flow-control stall) per `(placement, benchmark)`, plus the **cross-node penalty** in
throughput. `transport MB` should be roughly equal across placements — the same data is shuffled either
way; the throughput drop and the extra credit-wait cross-node are the network cost.

Watch a run live with the dashboard (the per-instance flow view shows the all-to-all shuffle mesh and
per-worker load):

```
kubectl -n nautilus-bench port-forward statefulset/nautilus-worker 8787 &   # or run the dashboard as a Job
```

## Manual smoke test (before the sweep)

```
kubectl apply -k deploy/kubernetes/overlays/cross-node
kubectl -n nautilus-bench rollout status statefulset/nautilus-worker
kubectl apply -f deploy/kubernetes/coordinator-job.yaml   # set its image first
kubectl -n nautilus-bench logs -f job/nautilus-coordinator
```

## Tuning

| knob | flag | note |
|---|---|---|
| daemon / node count | `--replicas` | cross-node needs this many nodes |
| operator parallelism | `--parallelism` | defaults to `--replicas` (one instance per daemon) |
| workload size | `--rows` / `--batch` / `--keys` | bigger `--rows` = longer, steadier shuffle |
| which benchmarks | `--benches` | `bench-keyed` (keyed shuffle) · `bench-chain` (256-byte payload → real wire bytes) |

The daemon pods request **3 cpu / 4Gi as guaranteed QoS** (`requests == limits`) so throttling or eviction
never skews a number — tune these in `base/workers-statefulset.yaml` to your instance type (the intra-node
node must fit `replicas ×` that).

## Security

Every node authenticates with `NAUTILUS_CLUSTER_SECRET`; for an untrusted network add
`NAUTILUS_CLUSTER_TLS_CERT` / `_KEY` / `_CA` env to encrypt the wire (mutual TLS) — see the `nautilus
worker` daemon docstring.
