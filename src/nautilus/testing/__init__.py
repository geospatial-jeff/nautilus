"""Test helpers: deterministic batch/frame builders and a tiny pipeline driver.

Kept in the package (not just under ``tests/``) so examples and downstream users can write
deterministic tests against operators in isolation.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pyarrow as pa

from nautilus.core.operator import OneInputOperator
from nautilus.core.records import (
    ACTIVE_FRAME,
    EOS_FRAME,
    IDLE_FRAME,
    Batch,
    Frame,
    Watermark,
)
from nautilus.core.time import TestClock
from nautilus.operators import InMemorySource, from_batches
from nautilus.runtime.local import run_local_chain
from nautilus.runtime.result import RunResult
from nautilus.telemetry.report import RunReport
from nautilus.tensors import tensor_array

__all__ = [
    "TestClock",
    "ACTIVE_FRAME",
    "EOS_FRAME",
    "IDLE_FRAME",
    "batch",
    "data",
    "wm",
    "from_batches",
    "run_ops",
    "op_counter",
    "multiset",
]


def batch(**columns: Any) -> pa.RecordBatch:
    """Build a RecordBatch from ``name=values`` keyword columns.

    Each value may be a list of scalars (built with ``pa.array``), a pre-built ``pa.Array`` or
    ``pa.ChunkedArray`` (used as is), or a numpy ndarray with ``ndim >= 2`` / a sequence of
    equal-shape ndarrays (built as a fixed-shape tensor column via
    :func:`nautilus.tensors.tensor_array`).
    """
    return pa.record_batch({name: _column(value) for name, value in columns.items()})


def data(**columns: Any) -> Batch:
    """Build a data :class:`Batch` frame from keyword columns."""
    return Batch(batch(**columns))


def wm(t: int) -> Watermark:
    return Watermark(t)


async def run_ops(frames: list[Frame], *transforms: OneInputOperator) -> RunResult:
    """Drive ``transforms`` over a fixed ``frames`` sequence and return the result (batches + telemetry)."""
    return await run_local_chain(InMemorySource(frames), list(transforms))


def op_counter(report: RunReport, operator_id: str, name: str) -> int:
    """Sum a named counter across *all* per-subtask ``OperatorStats`` sharing ``operator_id`` — unlike
    :meth:`RunReport.operator`, which returns only the first match, so this is what a parallel run needs.
    """
    return sum(
        p.value
        for o in report.operators
        if o.operator_id == operator_id
        for p in o.counters
        if p.name == name
    )


def multiset(result: RunResult) -> Counter[tuple[tuple[str, Any], ...]]:
    """A schema-agnostic multiset of a result's rows (each a sorted tuple of items), so any pipeline's
    output compares by content regardless of row or batch order."""
    return Counter(tuple(sorted(row.items())) for row in result.to_pylist())


def _column(value: Any) -> Any:
    if hasattr(value, "ndim"):  # a numpy ndarray
        return tensor_array(value) if value.ndim >= 2 else pa.array(value)
    if isinstance(value, (list, tuple)) and value and all(hasattr(v, "ndim") for v in value):
        return tensor_array(value)
    if hasattr(value, "to_pylist") and hasattr(value, "type"):  # pre-built pa.Array / ChunkedArray
        return value
    return pa.array(value)
