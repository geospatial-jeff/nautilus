"""The value a run returns: the emitted batches, plus the telemetry report.

``RunResult`` is a thin, explicit wrapper over the list of emitted :class:`pyarrow.RecordBatch` es: it
iterates and indexes (``for b in result``, ``result[0]``, ``len(result)``) and exposes ``.batches`` and
``.telemetry``. It is deliberately NOT a ``Sequence`` subclass — slicing one would return a bare list
that silently drops ``.telemetry``, and the inherited value-equality mixins are misleading on
RecordBatches. The Arrow-first readers (``to_table``/``to_pylist``/``to_pydict``) collapse the per-batch
zip every consumer would otherwise repeat. The ``RunReport`` type is referenced only for annotations
(under ``TYPE_CHECKING``), so this data-path module never imports the boundary report layer at runtime.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, cast

import pyarrow as pa

if TYPE_CHECKING:
    from nautilus.telemetry.report import RunReport


class RunResult:
    """The emitted batches (iterable / indexable) plus the run's ``.telemetry`` report."""

    def __init__(
        self,
        batches: list[pa.RecordBatch],
        telemetry: RunReport,
        *,
        schema: pa.Schema | None = None,
    ) -> None:
        self.batches = batches
        self.telemetry = telemetry
        # Output schema, so to_table() can return an empty-but-typed table when a run emits zero
        # batches (e.g. everything was filtered out). Falls back to the first batch's schema, or None.
        self._schema = schema or (batches[0].schema if batches else None)

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        return iter(self.batches)

    def __getitem__(self, index: int) -> pa.RecordBatch:
        return self.batches[index]

    def __len__(self) -> int:
        return len(self.batches)

    def to_table(self) -> pa.Table:
        """All emitted batches as one :class:`pyarrow.Table` (empty-but-typed when there are zero
        batches and the schema is known; an empty schemaless table otherwise)."""
        if self.batches:
            return pa.Table.from_batches(self.batches)
        if self._schema is not None:
            return self._schema.empty_table()
        return pa.table({})

    def to_pylist(self) -> list[dict[str, Any]]:
        """All emitted rows as a list of ``{column: value}`` dicts."""
        return cast("list[dict[str, Any]]", self.to_table().to_pylist())

    def to_pydict(self) -> dict[str, list[Any]]:
        """All emitted rows as a column-oriented ``{column: [values]}`` dict."""
        return cast("dict[str, list[Any]]", self.to_table().to_pydict())
