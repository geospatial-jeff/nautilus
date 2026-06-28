"""The value a run returns: the emitted batches, plus the telemetry report.

``RunResult`` wraps the list of emitted :class:`pyarrow.RecordBatch` es — it iterates, indexes, and
exposes ``.batches`` and ``.telemetry``. It is deliberately NOT a ``Sequence`` subclass and rejects
slicing, because a slice would return a bare list that silently drops ``.telemetry``; read the batches
via ``.batches`` or ``.to_table()`` instead. ``to_table``/``to_pylist``/``to_pydict`` are the
Arrow-first readers. ``RunReport`` is imported only under ``TYPE_CHECKING``, so this data-path module
never pulls in the report layer at runtime.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, cast

import pyarrow as pa

if TYPE_CHECKING:
    from nautilus.telemetry.report import RunReport


class RunResult:
    """The emitted batches (iterable / indexable) plus the run's ``.telemetry`` report."""

    def __init__(self, batches: list[pa.RecordBatch], telemetry: RunReport) -> None:
        self.batches = batches
        self.telemetry = telemetry

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        return iter(self.batches)

    def __getitem__(self, index: int) -> pa.RecordBatch:
        if isinstance(index, slice):
            raise TypeError(
                "RunResult does not support slicing (it would drop .telemetry); "
                "slice .batches or use .to_table()"
            )
        return self.batches[index]

    def __len__(self) -> int:
        return len(self.batches)

    def to_table(self) -> pa.Table:
        """All emitted batches as one :class:`pyarrow.Table` (an empty schemaless table when the run
        emitted no batches)."""
        if self.batches:
            return pa.Table.from_batches(self.batches)
        return pa.table({})

    def to_pylist(self) -> list[dict[str, Any]]:
        """All emitted rows as a list of ``{column: value}`` dicts."""
        return cast("list[dict[str, Any]]", self.to_table().to_pylist())

    def to_pydict(self) -> dict[str, list[Any]]:
        """All emitted rows as a column-oriented ``{column: [values]}`` dict."""
        return cast("dict[str, list[Any]]", self.to_table().to_pydict())
