"""Stage 2d demo: run a keyed word-count across two worker processes.

``deploy`` compiles the graph, places its instances across the spawned workers (same-index subtasks
co-located), and runs it over a mix of in-process and cross-worker socket edges. With ``KeyedCount`` at
parallelism 2 the shuffle genuinely crosses the two workers, yet the result matches the single-process
run. ``deploy`` is synchronous and spawns processes, so this script must guard ``main()`` behind
``if __name__ == "__main__"`` — without it, each spawned worker would re-run the module and spawn again.

Run with:  python examples/two_worker.py
"""

from __future__ import annotations

from nautilus.core.records import EOS_FRAME
from nautilus.dsl import source
from nautilus.operators import InMemorySource
from nautilus.testing import data


def _source() -> InMemorySource:
    return InMemorySource(
        [
            data(line=["the quick brown fox the lazy dog"]),
            data(line=["the fox jumped the lazy fox ran a dog and a cat"]),
            EOS_FRAME,
        ]
    )


def main() -> None:
    # .run(workers=2) spawns the workers, runs, and aggregates one report — the same graph the
    # single-process run below builds, just deployed across processes.
    result = (
        source(_source()).tokenize("line", "word").count_by("word", parallelism=2).run(workers=2)
    )

    counts = {row["word"]: row["count"] for row in result.to_pylist()}
    print("word counts:")
    for word, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {count:3d}  {word}")

    # The keyed operator genuinely ran on both workers — the shuffle crossed a socket.
    nodes = sorted({o.node for o in result.telemetry.operators if o.operator_id == "op1"})
    print(f"\ncount_by ran on: {nodes}")

    serial = source(_source()).tokenize("line", "word").count_by("word").run()
    serial_counts = {row["word"]: row["count"] for row in serial.to_pylist()}
    print(f"matches the single-process result: {counts == serial_counts}")


if __name__ == "__main__":
    main()
