"""Partitioners: pure, local routing of a data batch to downstream instances.

A partitioner decides, on the *sender*, which downstream instance(s) each row goes to. It is a pure
function of the batch and the downstream fan-out — no central entity is consulted. This is exactly
where "no central scheduler" lives on the data path.

Stage 0 ships :class:`Forward` (1:1) and :class:`Broadcast`. Key-group hashing (the keyed shuffle)
and round-robin rebalancing arrive with multi-instance parallelism in Stage 2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pyarrow as pa


class Partitioner(ABC):
    """Splits a batch into ``(downstream_index, sub_batch)`` pairs."""

    @abstractmethod
    def route(
        self, batch: pa.RecordBatch, num_downstream: int
    ) -> list[tuple[int, pa.RecordBatch]]: ...


class Forward(Partitioner):
    """1:1 forwarding. Requires a single downstream instance (upstream P == downstream Q == 1 in
    Stage 0); the whole batch goes to instance 0."""

    def route(self, batch: pa.RecordBatch, num_downstream: int) -> list[tuple[int, pa.RecordBatch]]:
        if num_downstream != 1:
            raise ValueError(f"Forward requires a single downstream instance, got {num_downstream}")
        return [(0, batch)]


class Broadcast(Partitioner):
    """Sends a full copy of every batch to every downstream instance."""

    def route(self, batch: pa.RecordBatch, num_downstream: int) -> list[tuple[int, pa.RecordBatch]]:
        return [(i, batch) for i in range(num_downstream)]
