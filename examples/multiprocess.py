"""Run a pipeline across two processes (Stage 1).

The same word-count as ``examples/wordcount.py``, but the source runs in this process and the
transforms and sink run in a spawned child process, joined by one TCP edge on the loopback interface
with credit-based flow control (see ``nautilus.transport``). The two-process result matches the
single-process one.

Run with:  python examples/multiprocess.py

The ``if __name__ == "__main__"`` guard below is required: ``run_two_process`` uses ``multiprocessing``
with the ``spawn`` start method, which re-imports this file in the child process.
"""

from __future__ import annotations

from nautilus import run
from nautilus.pipelines import wordcount
from nautilus.transport import run_two_process


def _counts(rows: list[dict]) -> dict[str, int]:
    return {row["word"]: row["count"] for row in rows}


def main() -> None:
    # Baseline: the whole pipeline in one process.
    single = _counts(run(*wordcount()).to_pylist())

    # Same pipeline across two processes: source here, transforms + sink in a spawned child,
    # with a credit window of 4 data frames on the cross-process edge.
    result = run_two_process(*wordcount(), capacity=4)
    across = _counts(result.to_pylist())

    for word, count in sorted(across.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{count:3d}  {word}")
    print(f"\nmatches single-process result: {across == single}")

    # Telemetry from the child process (the operator side of the edge).
    summary = result.telemetry.summary
    print(f"telemetry (child): {summary.total_rows_in} rows in, {summary.total_rows_out} rows out")
    for op in summary.per_operator:
        print(f"  {op.operator_id:7s} rows_out={op.rows_out_total:3d} errors={op.error_count}")


if __name__ == "__main__":
    main()
