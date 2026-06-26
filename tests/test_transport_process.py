"""A pipeline running across two real processes over a TCP loopback connection."""

from __future__ import annotations

from nautilus import run
from nautilus.core.records import EOS_FRAME
from nautilus.operators import InMemorySource, KeyedCount
from nautilus.pipelines import wordcount
from nautilus.telemetry import Tier
from nautilus.testing import data
from nautilus.transport import run_two_process


def _counts(rows: list[dict]) -> dict[str, int]:
    return {r["word"]: r["count"] for r in rows}


def test_two_process_wordcount_matches_single_process() -> None:
    expected = _counts(run(*wordcount()).to_pylist())

    source, transforms = wordcount()  # fresh operator instances
    result = run_two_process(source, transforms, capacity=4, tier=Tier.COUNTERS)

    assert _counts(result.to_pylist()) == expected
    assert result.telemetry is not None


def test_two_process_large_workload_high_capacity_conserves_rows() -> None:
    # The shutdown race that dropped data showed up at higher capacity with many frames in flight.
    # Run a larger keyed workload at capacity 16 and assert every input row is accounted for.
    per_batch, n_batches = 20, 30
    total = per_batch * n_batches
    frames = [
        data(k=[f"w{(j * per_batch + i) % 7}" for i in range(per_batch)]) for j in range(n_batches)
    ]
    frames.append(EOS_FRAME)

    result = run_two_process(InMemorySource(frames), [KeyedCount("k")], capacity=16)

    assert sum(row["count"] for row in result.to_pylist()) == total
