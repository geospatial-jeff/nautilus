"""Run a pipeline across worker processes with ``deploy``.

The same word-count topology as ``examples/wordcount.py`` (on its own input), but run across two spawned
workers: ``KeyedCount`` at
parallelism 2 forces a keyed shuffle that genuinely crosses a worker boundary over a socket, while
co-located edges stay in-process. The distributed result matches the single-process one.

Run with:  python examples/multiprocess.py

The ``if __name__ == "__main__"`` guard below is required: ``deploy`` uses ``multiprocessing`` with the
``spawn`` start method, which re-imports this file in each worker.
"""

from __future__ import annotations

from nautilus import run
from nautilus.dsl import source
from nautilus.pipelines import wordcount


def _counts(rows: list[dict]) -> dict[str, int]:
    return {row["word"]: row["count"] for row in rows}


def main() -> None:
    # Baseline: the whole pipeline in one process.
    single = _counts(run(*wordcount()).to_pylist())

    # Same word-count across two workers: tokenize feeds a parallelism-2 count_by through a keyed
    # shuffle, so each word's count is computed on exactly one worker. .run(workers=2) is the only
    # change from the single-process run — the same graph, deployed.
    src, _ = wordcount()
    result = (
        source(src)
        .tokenize("line", "word")
        .count_by("word", parallelism=2)
        .run(workers=2, capacity=4)
    )
    across = _counts(result.to_pylist())

    for word, count in sorted(across.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{count:3d}  {word}")
    print(f"\nmatches single-process result: {across == single}")

    # Telemetry is aggregated at the coordinator; KeyedCount ran across both workers.
    summary = result.telemetry.summary
    nodes = sorted({o.node for o in result.telemetry.operators if o.operator_id == "op1"})
    print(f"telemetry: {summary.total_rows_in} rows in, {summary.total_rows_out} rows out")
    print(f"  KeyedCount ran on workers: {nodes}")


if __name__ == "__main__":
    main()
