"""A parallel keyed word-count via the keyed shuffle, in a single process.

The same word-count topology as ``examples/wordcount.py`` (on its own input), but ``KeyedCount`` runs as
several instances. The keyed
shuffle (a ``KeyGroupPartitioner`` — the identity table at G == N, so byte-identical to a direct hash)
routes every occurrence of a word to the one instance that owns it, so each instance counts a disjoint
key range and the union is the full result — identical, as a multiset, to the single-instance run.

Run with:  python examples/parallel_word_count.py

(``nautilus run <pipeline> --parallelism N`` — add ``--workers W`` to spread across processes — now
drives this from the command line; this example shows the same run from Python.)
"""

from __future__ import annotations

import asyncio

from nautilus.core.records import EOS_FRAME
from nautilus.operators import InMemorySource, KeyedCount, Tokenize
from nautilus.runtime.local import run_local_chain
from nautilus.runtime.parallel import Stage, run_parallel_chain
from nautilus.testing import data


def _source() -> InMemorySource:
    return InMemorySource(
        [
            data(line=["the quick brown fox", "the lazy dog"]),
            data(line=["the fox jumped", "the lazy fox ran"]),
            EOS_FRAME,
        ]
    )


async def main() -> None:
    parallelism = 3
    # source -> Tokenize (1 instance) -> [hash shuffle on "word"] -> KeyedCount (N instances) -> sink
    result = await run_parallel_chain(
        _source(),
        [
            Stage(lambda: Tokenize("line", "word")),
            Stage(lambda: KeyedCount("word"), parallelism, ["word"]),
        ],
    )
    counts = {row["word"]: row["count"] for row in result.to_pylist()}
    for word, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{count:3d}  {word}")

    # The parallel result matches the single-instance baseline.
    serial = await run_local_chain(_source(), [Tokenize("line", "word"), KeyedCount("word")])
    serial_counts = {row["word"]: row["count"] for row in serial.to_pylist()}
    print(f"\nmatches single-instance result: {counts == serial_counts}")

    # Telemetry: KeyedCount (op1) ships one OperatorStats row per instance, summing to the totals.
    rep = result.telemetry
    instances = sorted({o.subtask_index for o in rep.operators if o.operator_id == "op1"})
    print(f"telemetry: {rep.summary.total_rows_in} rows in, {rep.summary.total_rows_out} rows out")
    print(f"  KeyedCount ran as {len(instances)} instances: subtasks {instances}")


if __name__ == "__main__":
    asyncio.run(main())
