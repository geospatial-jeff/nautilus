"""Characterization tests pinning tensor-column dtype/layout behavior for the Rust port.

Tier 1: these lock the observable contract of :mod:`nautilus.tensors` — exact Arrow value types,
value order across chunk boundaries, and the precise error a bad input raises — so a future rewrite
that passes this suite reproduces the same round-trips and rejections. Every golden here (value
types, error classes, error strings) was read from the running code, not assumed.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from nautilus.tensors import embedding_array, is_tensor, tensor_array, to_numpy


# (a) round-trip int16/int32/int64/float64 to the IDENTICAL numpy dtype, pinning Arrow value_type.
@pytest.mark.parametrize(
    ("np_dtype", "arrow_value_type"),
    [
        (np.int16, pa.int16()),
        (np.int32, pa.int32()),
        (np.int64, pa.int64()),
        (np.float64, pa.float64()),
    ],
)
def test_tensor_array_roundtrips_dtype_exactly(np_dtype, arrow_value_type) -> None:
    src = np.arange(2 * 3 * 3, dtype=np_dtype).reshape(2, 3, 3)
    col = tensor_array(src)
    assert col.type.value_type == arrow_value_type
    out = to_numpy(col)
    assert out.dtype == np.dtype(np_dtype)
    np.testing.assert_array_equal(out, src)


# (b) A bool element dtype builds a column but is rejected on read-back: pyarrow's tensor conversion
# does not accept bool, so the round-trip raises ArrowInvalid at to_numpy (not at tensor_array).
def test_bool_element_dtype_rejected_on_readback() -> None:
    col = tensor_array(np.zeros((2, 3, 3), dtype=bool))
    assert col.type.value_type == pa.bool_()  # build succeeds; the wrapper accepts bool
    with pytest.raises(pa.lib.ArrowInvalid, match="bool is not valid data type for a tensor"):
        to_numpy(col)


# (c) A Fortran-ordered (non-C-contiguous) input is copied once and round-trips to the same VALUES.
def test_fortran_ordered_input_roundtrips_values() -> None:
    src = np.asfortranarray(np.arange(2 * 3 * 4, dtype=np.int32).reshape(2, 3, 4))
    assert not src.flags["C_CONTIGUOUS"]
    out = to_numpy(tensor_array(src))
    np.testing.assert_array_equal(out, src)


# (d) A byte-swapped/big-endian dtype is rejected at build time by pyarrow.
def test_big_endian_dtype_rejected() -> None:
    src = np.arange(2 * 3 * 3, dtype=">i4").reshape(2, 3, 3)
    with pytest.raises(pa.lib.ArrowNotImplementedError, match="Byte-swapped arrays not supported"):
        tensor_array(src)


# (e) to_numpy over a multi-chunk ChunkedArray equals concatenate([A, B]) — value order is preserved
# across the chunk boundary.
def test_to_numpy_multi_chunk_preserves_order() -> None:
    a = np.arange(2 * 3 * 3, dtype=np.int32).reshape(2, 3, 3)
    b = np.arange(100, 100 + 3 * 3 * 3, dtype=np.int32).reshape(3, 3, 3)
    chunked = pa.chunked_array([tensor_array(a), tensor_array(b)])
    assert chunked.num_chunks == 2
    out = to_numpy(chunked)
    np.testing.assert_array_equal(out, np.concatenate([a, b]))


# (f) N=0 empty input: pyarrow's from_numpy_ndarray rejects an empty leading axis.
def test_empty_batch_axis_rejected() -> None:
    with pytest.raises(ValueError, match="Expected a non-empty ndarray"):
        tensor_array(np.zeros((0, 3, 3), dtype=np.int32))


# (g) to_numpy on a non-tensor pa.array raises TypeError naming the type; is_tensor(None) is False.
def test_to_numpy_on_non_tensor_names_type() -> None:
    with pytest.raises(TypeError, match="expects a fixed_shape_tensor column, got int64"):
        to_numpy(pa.array([1, 2, 3]))


def test_is_tensor_none_is_false() -> None:
    assert is_tensor(None) is False


# (h) A sequence sharing shape but not dtype, and an empty sequence, raise distinct ValueErrors.
def test_sequence_dtype_mismatch_rejected() -> None:
    a = np.zeros((3, 3), np.int32)
    b = np.zeros((3, 3), np.int64)
    with pytest.raises(ValueError, match="arrays must share one dtype"):
        tensor_array([a, b])


def test_empty_sequence_rejected() -> None:
    with pytest.raises(ValueError, match="empty sequence of arrays"):
        tensor_array([])


# (i) N=1 keeps the batch axis: a (1, H, W, C) input stays one row and reads back with the axis intact.
def test_single_row_keeps_batch_axis() -> None:
    src = np.arange(1 * 4 * 4 * 3, dtype=np.uint8).reshape(1, 4, 4, 3)
    col = tensor_array(src)
    assert len(col) == 1
    out = to_numpy(col)
    assert out.shape == (1, 4, 4, 3)
    np.testing.assert_array_equal(out, src)


# (j) embedding_array passes a float32 input through bit-equal (no re-cast); dtype="float16" yields a
# float16 (halffloat) value type.
def test_embedding_float32_passes_through_bit_equal() -> None:
    src = np.random.default_rng(1).random((5, 8)).astype(np.float32)
    out = to_numpy(embedding_array(src))
    assert out.dtype == np.float32
    assert np.array_equal(out.view(np.uint8), src.view(np.uint8))  # bit-for-bit, no re-cast


def test_embedding_float16_dtype() -> None:
    src = np.random.default_rng(1).random((5, 8)).astype(np.float32)
    col = embedding_array(src, dtype="float16")
    assert col.type.value_type == pa.float16()
    assert col.storage.type.value_type == pa.float16()
