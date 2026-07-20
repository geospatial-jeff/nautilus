"""Tier 4 property-based conformance tests for transport-framing (:mod:`nautilus.transport.framing`).

Where ``test_transport_framing.py`` pins named boundary cases, this file pins the invariants that must
hold over the *whole* input space: the wire header layout, encode determinism, and the ``encode ->
split -> decode`` round trip. A Rust port that diverges on any of these silently corrupts the stream, so
each property is asserted over many random inputs (fixed seed, no ``hypothesis``) rather than a handful of
literals.
"""

from __future__ import annotations

import random

import msgpack
import pyarrow as pa

from nautilus.core.records import EOS_FRAME, Barrier, Batch
from nautilus.testing import batch
from nautilus.transport.framing import (
    Kind,
    decode,
    encode_credit,
    encode_frame,
    split,
)

_SEED = 1234
# Payload-length boundaries the spec names: each side of a 1-, 2-, and 4-byte length rollover.
_LENGTH_BOUNDARIES = [0, 1, 127, 128, 255, 256, 65535, 65536]
# int64 credit boundaries: each side of the signed 32-bit rollover, plus the signed 64-bit extremes.
_CREDIT_BOUNDARIES = [0, 1, -1, 2**31 - 1, 2**31, 2**63 - 1, -(2**63)]


def _random_batch(rng: random.Random) -> pa.RecordBatch:
    """A RecordBatch of random width/height with non-alphabetic column names and mixed nullable types."""
    n_rows = rng.randrange(0, 6)
    n_cols = rng.randrange(1, 4)
    columns: dict[str, list[object]] = {}
    for c in range(n_cols):
        # Non-alphabetic, non-sorted names so a port that reorders columns by name is caught.
        name = f"{rng.randrange(9000, 9999)}_{c}"
        kind = rng.randrange(0, 3)
        if kind == 0:
            values: list[object] = [rng.randrange(-(2**40), 2**40) for _ in range(n_rows)]
        elif kind == 1:
            values = [rng.random() for _ in range(n_rows)]
        else:
            values = [f"s{rng.randrange(0, 1000)}" for _ in range(n_rows)]
        # Punch nulls so the round trip exercises null bitmaps too.
        for i in range(n_rows):
            if rng.random() < 0.25:
                values[i] = None
        columns[name] = values
    return batch(**columns)


def test_length_encoding_big_endian_4byte() -> None:
    """For every frame, bytes[1:5] is the big-endian payload length: from_bytes(msg[1:5]) == len(msg)-5."""
    rng = random.Random(_SEED)
    # Named payload sizes: a single binary column of `size` bytes drives the IPC payload past each
    # 1-, 2-, and 4-byte length rollover; the header field must equal the real payload length.
    for size in _LENGTH_BOUNDARIES:
        rb = pa.record_batch({"7c": pa.array([b"x" * size])})
        msg = encode_frame(Batch(rb))
        assert int.from_bytes(msg[1:5], "big") == len(msg) - 5

    for _ in range(200):
        rb = _random_batch(rng)
        for msg in (
            encode_frame(Batch(rb)),
            encode_frame(Barrier(rng.randrange(0, 2**63))),
            encode_frame(EOS_FRAME),
            encode_credit(rng.randrange(-(2**63), 2**63)),
        ):
            declared = int.from_bytes(msg[1:5], "big")
            assert declared == len(msg) - 5
            assert msg[1:5] == declared.to_bytes(4, "big")  # exactly 4 bytes, big-endian


def test_encoding_determinism() -> None:
    """Encoding the same frame twice yields byte-for-byte identical wire messages, for every frame kind."""
    rng = random.Random(_SEED)
    for _ in range(200):
        rb = _random_batch(rng)
        cid = rng.randrange(0, 2**63)
        credit = rng.randrange(-(2**63), 2**63)
        assert encode_frame(Batch(rb)) == encode_frame(Batch(rb))
        assert encode_frame(Barrier(cid)) == encode_frame(Barrier(cid))
        assert encode_frame(EOS_FRAME) == encode_frame(EOS_FRAME)
        assert encode_credit(credit) == encode_credit(credit)


def test_split_decode_inverse() -> None:
    """For every frame, decode(*split(encode_frame(frame))) reproduces it, and split's parts re-pack it."""
    rng = random.Random(_SEED)
    for _ in range(200):
        rb = _random_batch(rng)
        cid = rng.randrange(0, 2**63)
        frames = [Batch(rb), Barrier(cid), EOS_FRAME]
        for frame in frames:
            msg = encode_frame(frame)
            kind, payload = split(msg)
            out = decode(kind, payload)
            if isinstance(frame, Batch):
                assert isinstance(out, Batch)
                assert out.data.equals(frame.data)
            else:
                assert out == frame
            # Re-packing split()'s output reconstructs the original message byte-for-byte.
            assert bytes([int(kind)]) + len(payload).to_bytes(4, "big") + payload == msg


def test_row_and_column_order_preservation() -> None:
    """A Batch round-trips (encode->split->decode) with identical row order and column order preserved."""
    rng = random.Random(_SEED)
    for _ in range(200):
        rb = _random_batch(rng)
        kind, payload = split(encode_frame(Batch(rb)))
        out = decode(kind, payload)
        assert isinstance(out, Batch)
        # Column order: schema field names come back in the exact order they were written.
        assert out.data.schema.names == rb.schema.names
        # Row order: every column's values come back in the same order (no reordering/sorting).
        for name in rb.schema.names:
            assert out.data.column(name).to_pylist() == rb.column(name).to_pylist()
        assert out.data.equals(rb)


def test_credit_count_range_preservation() -> None:
    """encode_credit(c) round-trips via split+decode back to c across the signed 64-bit range."""
    rng = random.Random(_SEED)
    values = list(_CREDIT_BOUNDARIES)
    values += [rng.randrange(-(2**63), 2**63) for _ in range(200)]
    for c in values:
        kind, payload = split(encode_credit(c))
        assert kind == Kind.CREDIT
        out = decode(kind, payload)
        assert out == c
        assert isinstance(out, int)


def test_msgpack_field_order_determinism() -> None:
    """A control frame (Barrier/EOS) encodes to identical bytes on repeat: deterministic msgpack key order."""
    rng = random.Random(_SEED)
    for _ in range(200):
        cid = rng.randrange(0, 2**63)
        barrier_bytes = encode_frame(Barrier(cid))
        assert barrier_bytes == encode_frame(Barrier(cid))
        # The payload is a msgpack map with a fixed insertion order; unpacking and re-packing the same
        # map must reproduce the exact payload bytes a port would emit.
        _, payload = split(barrier_bytes)
        assert msgpack.packb(msgpack.unpackb(payload, raw=False)) == payload

    eos_bytes = encode_frame(EOS_FRAME)
    assert eos_bytes == encode_frame(EOS_FRAME)
    _, eos_payload = split(eos_bytes)
    assert msgpack.packb(msgpack.unpackb(eos_payload, raw=False)) == eos_payload
