"""Stage 2a demo: build a logical graph, compile it to a physical plan, run the plan in one process.

This is the same parallel keyed word-count as ``examples/parallel_word_count.py``, but it goes through
the Stage 2 path instead of the legacy mesh runner. You describe the job as a ``LogicalGraph`` —
operators, parallelism, key columns, and nothing physical — ``compile_graph`` lowers it to a
serializable ``PhysicalPlan``, and ``run_plan`` runs that plan single-process. The plan is exactly what
a coordinator will cloudpickle to a worker in a cluster; here we run it locally and show it survives a
serialization round-trip unchanged.

Run with:  python examples/compiled_wordcount.py
"""

from __future__ import annotations

import asyncio

import cloudpickle

from nautilus.api import LogicalVertex, linear_graph
from nautilus.compile import compile_graph
from nautilus.core.records import EOS_FRAME
from nautilus.operators import InMemorySource, KeyedCount, Tokenize
from nautilus.runtime.run import run_compiled
from nautilus.testing import data


def _source() -> InMemorySource:
    return InMemorySource(
        [
            data(line=["the quick brown fox", "the lazy dog"]),
            data(line=["the fox jumped", "the lazy fox ran"]),
            EOS_FRAME,
        ]
    )


def build_graph(parallelism: int):
    """source -> Tokenize (1 instance) -> [keyed shuffle on 'word'] -> KeyedCount (N instances) -> sink.

    The sink is not in the graph — the compiler synthesizes the collecting sink. Tokenize is keyless, so
    its edge from the source is a forward; KeyedCount declares ``key_columns=('word',)``, so the compiler
    selects a keyed shuffle (a KeyGroupPartitioner, identity table at G == N) into its N instances.
    """
    return linear_graph(
        _source,
        [
            LogicalVertex("tokenize", lambda: Tokenize("line", "word"), "one_input"),
            LogicalVertex("count", lambda: KeyedCount("word"), "one_input", parallelism, ("word",)),
        ],
    )


async def main() -> None:
    graph = build_graph(parallelism=3)

    # Compile once: the LogicalGraph becomes a PhysicalPlan of named operators and partitioner specs.
    plan = compile_graph(graph)
    print("compiled plan:")
    for op in plan.operators:
        print(
            f"  {op.operator_id:8s} {op.op_class:14s} kind={op.kind:9s} parallelism={op.parallelism}"
        )
    for edge in plan.edges:
        print(
            f"  edge {edge.src_operator_id} -> {edge.dst_operator_id} via {edge.spec.partitioner_name}"
        )

    result = await run_compiled(plan)  # run the plan we just compiled and printed (not a recompile)
    counts = {row["word"]: row["count"] for row in result.to_pylist()}
    print("\nword counts:")
    for word, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {count:3d}  {word}")

    # The plan is the unit of distribution: cloudpickle it (as a coordinator would, to a worker) and the
    # reloaded plan runs to the identical result and structural digest.
    reloaded = cloudpickle.loads(cloudpickle.dumps(plan))
    again = await run_compiled(reloaded)
    same = {row["word"]: row["count"] for row in again.to_pylist()} == counts
    digest_same = result.telemetry.structural_digest() == again.telemetry.structural_digest()
    print(f"\nround-trips through cloudpickle identically: {same and digest_same}")


if __name__ == "__main__":
    asyncio.run(main())
