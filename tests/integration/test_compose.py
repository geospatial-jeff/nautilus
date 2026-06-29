"""Stage 4.3: the genuine multi-node run, across separate containers via docker-compose.

This is the proof that nautilus runs across machines, not just processes: two worker daemons in separate
containers (distinct network namespaces, addressed by service DNS) and a coordinator that *dials* them.
At parallelism 2 the keyed shuffle crosses the container boundary over a real socket, and the result must
match a single-process run — checked here by conserved row count and by the keyed operator having run on
*both* container hosts (its host-tagged telemetry nodes are distinct), which is only possible if the
shuffle genuinely crossed.

Marked ``docker`` and skipped by default (``addopts = -m 'not docker'`` in pyproject), so the unit suite
stays hermetic; run it with ``pytest -m docker``. It also self-skips when ``docker compose`` is absent.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from nautilus.bench import run_once
from nautilus.telemetry import Tier

pytestmark = pytest.mark.docker

_ROOT = Path(__file__).resolve().parents[2]
_OUT = _ROOT / ".compose-out"


def _docker_compose_available() -> bool:
    try:
        return (
            subprocess.run(
                ["docker", "compose", "version"], capture_output=True, timeout=20
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.skipif(not _docker_compose_available(), reason="docker compose not available")
def test_multinode_wordcount_across_containers() -> None:
    # The single-process baseline: same pipeline, one process. Its conserved row count is what the
    # distributed run must reproduce.
    serial = run_once("wordcount", parallelism=1, workers=1, capacity=16, tier=Tier.COUNTERS)
    expected_rows_out = serial.telemetry.summary.total_rows_out

    _OUT.mkdir(exist_ok=True)
    for stale in _OUT.glob("*"):
        stale.unlink()
    try:
        up = subprocess.run(
            [
                "docker",
                "compose",
                "up",
                "--build",
                "--abort-on-container-exit",
                "--exit-code-from",
                "coordinator",
            ],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert up.returncode == 0, f"compose run failed:\n{up.stdout[-3000:]}\n{up.stderr[-3000:]}"

        report = json.loads((_OUT / "report.json").read_text())
        # Rows are conserved across the container boundary — the distributed run matches the serial one.
        assert report["summary"]["total_rows_out"] == expected_rows_out
        assert report["summary"]["total_errors"] == 0
        # The keyed operator ran on BOTH containers: its node labels carry each daemon's advertised host,
        # so two distinct hosts prove the shuffle genuinely crossed the container boundary.
        op1_nodes = {o["node"] for o in report["operators"] if o["operator_id"] == "op1"}
        assert op1_nodes == {"worker-0@worker-0", "worker-1@worker-1"}, op1_nodes
    finally:
        subprocess.run(
            ["docker", "compose", "down", "-v"],
            cwd=_ROOT,
            capture_output=True,
            timeout=120,
        )
