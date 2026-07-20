"""Tier 4 property-based conformance tests for tensors.

Where :mod:`tests.port.test_tensors` pins a handful of golden round-trips, these tests quantify the
same contract of :mod:`nautilus.tensors` over its whole input space: they drive many seeded-random
shapes, batch sizes, and dtypes and assert each invariant on every one. A Rust port that diverges on
any input the goldens miss — a stray shape, a dtype whose bits it truncates, a chunk order it flips —
fails here. Randomness is fully seeded so the sampled inputs are identical on every run.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from nautilus.tensors import embedding_array, tensor_array, to_numpy

# The value dtypes tensor_array carries losslessly. bool is excluded: it builds but pyarrow rejects it
# on read-back (pinned in test_tensors.py), so it has no round-trip to quantify here.
_ROUNDTRIP_DTYPES = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
]


def _random_tensor(rng: np.random.Generator, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    """A random ``shape`` array of ``dtype`` spanning that dtype's full value range.

    Integers cover the whole [min, max] interval (so int64 extremes and unsigned wrap-around appear);
    floats mix uniform draws with nan/+-inf/+-0.0 so bit-level round-trips exercise the payloads a
    naive port might normalize away.
    """
    np_dtype = np.dtype(dtype)
    if np.issubdtype(np_dtype, np.integer):
        # Draw raw bit patterns of the right width and reinterpret, so the full [min, max] range is
        # reachable for both signed and unsigned types without overflowing an intermediate integer.
        unsigned = np.dtype(f"uint{np_dtype.itemsize * 8}")
        high = 2 ** (np_dtype.itemsize * 8)
        raw = rng.integers(0, high, size=shape, dtype=unsigned)
        return raw.view(np_dtype)
    values = rng.standard_normal(size=shape).astype(dtype)
    specials = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0], dtype=dtype)
    flat = values.reshape(-1)
    if flat.size:
        idx = rng.integers(0, flat.size, size=min(flat.size, specials.size))
        flat[idx] = specials[: idx.size]
    return values


def _bits_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Bit-for-bit equality, so nan payloads and -0.0 count as different from 0.0."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return np.array_equal(a.view(np.uint8), b.view(np.uint8))


def test_shape_and_values_roundtrip() -> None:
    """For every (N, *shape) array with N>=1, to_numpy(tensor_array(src)) matches src in shape and values."""
    rng = np.random.default_rng(1234)
    for _ in range(200):
        n = int(rng.integers(1, 6))
        rank = int(rng.integers(1, 4))
        shape = (n, *(int(rng.integers(1, 5)) for _ in range(rank)))
        src = _random_tensor(rng, "int32", shape)
        out = to_numpy(tensor_array(src))
        assert out.shape == src.shape
        np.testing.assert_array_equal(out, src)


def test_row_count_equals_column_length() -> None:
    """len(tensor_array(src)) and len(embedding_array(v)) equal the leading batch dimension N."""
    rng = np.random.default_rng(1234)
    for _ in range(200):
        n = int(rng.integers(1, 20))
        rank = int(rng.integers(1, 4))
        shape = (n, *(int(rng.integers(1, 5)) for _ in range(rank)))
        assert len(tensor_array(_random_tensor(rng, "float32", shape))) == n
        dim = int(rng.integers(1, 16))
        assert len(embedding_array(_random_tensor(rng, "float64", (n, dim)))) == n


def test_dtypes_preserved_bitwise() -> None:
    """For each int/uint/float width, tensor_array round-trips the dtype and every value bit-for-bit."""
    rng = np.random.default_rng(1234)
    for dtype in _ROUNDTRIP_DTYPES:
        for _ in range(20):
            n = int(rng.integers(1, 6))
            rank = int(rng.integers(1, 3))
            shape = (n, *(int(rng.integers(1, 5)) for _ in range(rank)))
            src = _random_tensor(rng, dtype, shape)
            out = to_numpy(tensor_array(src))
            assert out.dtype == np.dtype(dtype)
            assert _bits_equal(out, src)


def test_embedding_array_cast_deterministic() -> None:
    """embedding_array(float64 input) casts to float32 bit-identically to src.astype(float32)."""
    rng = np.random.default_rng(1234)
    for _ in range(200):
        n = int(rng.integers(1, 8))
        dim = int(rng.integers(1, 32))
        src = _random_tensor(rng, "float64", (n, dim))
        out = to_numpy(embedding_array(src))
        assert out.dtype == np.float32
        assert _bits_equal(out, src.astype(np.float32))


def test_chunked_array_concatenation_order() -> None:
    """to_numpy on a multi-chunk ChunkedArray equals concatenate of its per-chunk arrays, in order."""
    rng = np.random.default_rng(1234)
    for _ in range(100):
        rank = int(rng.integers(1, 3))
        shape = tuple(int(rng.integers(1, 5)) for _ in range(rank))
        chunks = [
            _random_tensor(rng, "int64", (int(rng.integers(1, 5)), *shape))
            for _ in range(int(rng.integers(2, 5)))
        ]
        chunked = pa.chunked_array([tensor_array(c) for c in chunks])
        assert chunked.num_chunks == len(chunks)
        np.testing.assert_array_equal(to_numpy(chunked), np.concatenate(chunks))
