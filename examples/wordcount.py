"""Stage 0 demo: a bounded streaming word-count in a single process.

Run with:  python examples/wordcount.py
"""

from __future__ import annotations

from nautilus.operators import KeyedCount, Tokenize
from nautilus.runtime.local import run
from nautilus.testing import data, from_batches


def main() -> None:
    source = from_batches(
        data(line=["the quick brown fox", "the lazy dog"]),
        data(line=["the fox jumped", "the dog slept"]),
    )
    result = run(source, [Tokenize("line", "word"), KeyedCount("word")])

    counts = {row["word"]: row["count"] for row in result.to_pylist()}
    for word, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{count:3d}  {word}")

    # Telemetry comes with every run — the raw facts behind the output above.
    summary = result.telemetry.summary
    print(f"\ntelemetry: {summary.total_rows_in} rows in, {summary.total_rows_out} rows out")
    for op in summary.per_operator:
        print(f"  {op.operator_id:7s} rows_out={op.rows_out_total:3d} errors={op.error_count}")


if __name__ == "__main__":
    main()
