"""Fixed-shape tensor columns: imagery and embeddings."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from nautilus import FilterRows, MapBatch, TelemetryConfig, Tier, from_batches, run
from nautilus.pipelines import image_embed
from nautilus.tensors import embedding_array, is_tensor, tensor_array, tensor_type, to_numpy
from nautilus.testing import batch, data


def test_image_roundtrip() -> None:
    imgs = np.arange(4 * 8 * 8 * 3, dtype=np.uint8).reshape(4, 8, 8, 3)
    col = tensor_array(imgs)
    assert is_tensor(col.type)
    assert tuple(col.type.shape) == (8, 8, 3)
    out = to_numpy(col)
    assert out.shape == (4, 8, 8, 3)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, imgs)


def test_embedding_roundtrip_and_storage() -> None:
    vecs = np.random.default_rng(0).random((5, 16))  # float64 on the way in
    col = embedding_array(vecs)  # cast to float32
    assert tuple(col.type.shape) == (16,)
    # vector-search tooling keys on the fixed_size_list storage, not the extension wrapper
    assert pa.types.is_fixed_size_list(col.storage.type)
    assert col.storage.type.value_type == pa.float32()
    assert col.storage.type.list_size == 16
    out = to_numpy(col)
    assert out.shape == (5, 16)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, vecs.astype(np.float32))


def test_tensor_array_from_sequence_and_dim_names() -> None:
    arrs = [np.zeros((4, 4, 3), np.uint8), np.ones((4, 4, 3), np.uint8)]
    col = tensor_array(arrs, dim_names=["H", "W", "C"])
    assert len(col) == 2
    assert tuple(col.type.shape) == (4, 4, 3)
    assert list(col.type.dim_names) == ["H", "W", "C"]


def test_tensor_type() -> None:
    ty = tensor_type(pa.uint8(), (8, 8, 3), dim_names=["H", "W", "C"])
    assert is_tensor(ty)
    assert tuple(ty.shape) == (8, 8, 3)


def test_batch_helper_builds_tensor_column() -> None:
    imgs = np.zeros((3, 4, 4, 3), np.uint8)
    rb = batch(idx=[0, 1, 2], image=imgs)
    assert rb.num_rows == 3
    assert is_tensor(rb.column("image").type)
    # a 1-D numpy column stays a scalar column, not a tensor
    rb2 = batch(x=np.array([1, 2, 3]))
    assert not is_tensor(rb2.column("x").type)


def test_passes_through_map() -> None:
    imgs = np.arange(3 * 2 * 2, dtype=np.uint8).reshape(3, 2, 2)
    result = run(from_batches(data(idx=[0, 1, 2], image=imgs)), [MapBatch(lambda rb: rb)])
    out = to_numpy(result.to_table().column("image"))
    assert out.shape == (3, 2, 2)
    np.testing.assert_array_equal(out, imgs)


def test_passes_through_filter() -> None:
    imgs = np.arange(3 * 2 * 2, dtype=np.uint8).reshape(3, 2, 2)
    src = from_batches(data(idx=[0, 1, 2], image=imgs))
    result = run(src, [FilterRows(lambda rb: pc.greater(rb.column("idx"), 0))])
    out = to_numpy(result.to_table().column("image"))
    assert out.shape == (2, 2, 2)
    np.testing.assert_array_equal(out, imgs[1:])


def test_to_numpy_on_sliced_column() -> None:
    imgs = np.arange(5 * 2 * 2, dtype=np.uint8).reshape(5, 2, 2)
    col = tensor_array(imgs)
    out = to_numpy(col[1:4])
    assert out.shape == (3, 2, 2)
    np.testing.assert_array_equal(out, imgs[1:4])


def test_full_telemetry_counts_tensor_bytes() -> None:
    imgs = np.zeros((4, 16, 16, 3), np.uint8)
    src = from_batches(data(idx=[0, 1, 2, 3], image=imgs))
    result = run(src, [MapBatch(lambda rb: rb)], telemetry=TelemetryConfig(tier=Tier.FULL))
    # the Arrow API telemetry uses for byte accounting must see the tensor buffers
    assert result.to_table().get_total_buffer_size() >= 4 * 16 * 16 * 3


def test_embedding_array_rejects_1d() -> None:
    with pytest.raises(ValueError):
        embedding_array(np.zeros(16, np.float32))


def test_tensor_array_rejects_unequal_shapes() -> None:
    with pytest.raises(ValueError):
        tensor_array([np.zeros((2, 2)), np.zeros((3, 3))])


def test_image_embed_example_runs() -> None:
    source, transforms = image_embed()
    table = run(source, transforms).to_table()
    assert table.num_rows == 7  # 4 + 3 tiles
    emb = to_numpy(table.column("embedding"))
    assert emb.shape == (7, 3)  # per-tile mean over H, W -> one value per channel
    assert emb.dtype == np.float32
