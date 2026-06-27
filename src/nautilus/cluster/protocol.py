"""Control-plane messages between the coordinator and its workers.

These cross a :class:`multiprocessing.Queue`, which pickles them, so they are plain frozen dataclasses.
Two payloads do *not* ride as ordinary pickled objects, by necessity:

* the **plan** ships separately as cloudpickled bytes (a spawn argument) — stdlib pickle, which the
  queue uses, cannot carry the plan's lambda operator factories;
* **sink batches** cross as Arrow IPC bytes (:func:`encode_batches`), not pickled ``RecordBatch`` es,
  so canonical extension types (e.g. ``fixed_shape_tensor``) survive the boundary.

Recorder snapshots are ordinary pickled objects (plain numbers and tuples), carried inline in
:class:`Done`. The coordinator aggregates them into the one report at the job boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from nautilus.telemetry.model import InstanceSnapshot


@dataclass(frozen=True)
class Register:
    """worker → coordinator: I bound my listener at ``(host, port)`` and am ready to be wired."""

    worker_id: int
    host: str
    port: int


@dataclass(frozen=True)
class Done:
    """worker → coordinator: my slice finished. ``snapshots`` are my recorders' readings; ``sink_batches``
    is the Arrow IPC of the sink's output (empty unless this worker hosts the sink)."""

    worker_id: int
    snapshots: list[InstanceSnapshot]
    sink_batches: bytes


@dataclass(frozen=True)
class Failed:
    """worker → coordinator: my slice raised. ``traceback`` is the child's, so ``deploy`` can re-raise it."""

    worker_id: int
    traceback: str


def encode_batches(batches: list[pa.RecordBatch]) -> bytes:
    """Serialize the sink's collected batches as one Arrow IPC stream (empty bytes for no batches)."""
    if not batches:
        return b""
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batches[0].schema) as writer:
        for batch in batches:
            writer.write_batch(batch)
    return bytes(sink.getvalue().to_pybytes())


def decode_batches(data: bytes) -> list[pa.RecordBatch]:
    """Read back the batches written by :func:`encode_batches`."""
    if not data:
        return []
    with pa.ipc.open_stream(pa.py_buffer(data)) as reader:
        return list(reader)
