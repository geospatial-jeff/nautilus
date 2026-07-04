"""numpy-backed tensor columns for imagery and embeddings.

Imagery (multidimensional arrays) and embeddings (1-D vectors) are carried as Arrow ``fixed_shape_tensor``
extension columns (:class:`pyarrow.FixedShapeTensorArray`): one tensor per row, stored row-major as a
``fixed_size_list``, with the shape held in the column type. The column length is the batch
dimension, so an ``(N, *shape)`` numpy array maps to an N-row column.

:func:`tensor_array` and :func:`embedding_array` build a column from numpy; :func:`to_numpy` reads it
back. Conversion is zero-copy when the numpy input is C-contiguous.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

_TENSOR_EXTENSION_NAME = "arrow.fixed_shape_tensor"


def tensor_type(
    value_type: pa.DataType, shape: Sequence[int], *, dim_names: Sequence[str] | None = None
) -> pa.DataType:
    """Arrow ``fixed_shape_tensor`` type for tensors of ``shape`` with ``value_type`` elements."""
    return pa.fixed_shape_tensor(
        value_type,
        list(shape),
        dim_names=list(dim_names) if dim_names is not None else None,
    )


def tensor_array(
    arrays: NDArray[Any] | Sequence[NDArray[Any]], *, dim_names: Sequence[str] | None = None
) -> pa.FixedShapeTensorArray:
    """Build a fixed-shape tensor column from numpy.

    ``arrays`` is either an ``(N, *shape)`` ndarray (the first axis is the row axis) or a non-empty
    sequence of ndarrays that share one shape and dtype. The conversion is zero-copy when the input
    is C-contiguous; a non-contiguous input is copied once.
    """
    stacked = np.ascontiguousarray(_stack(arrays))
    if stacked.ndim < 2:
        raise ValueError(
            f"tensor_array expects an (N, *shape) array with ndim >= 2, got shape {stacked.shape}"
        )
    array = pa.FixedShapeTensorArray.from_numpy_ndarray(stacked)
    if dim_names is None:
        return array
    named = tensor_type(array.type.value_type, list(array.type.shape), dim_names=dim_names)
    return pa.ExtensionArray.from_storage(named, array.storage)


def embedding_array(
    vectors: NDArray[Any] | Sequence[NDArray[Any]], *, dtype: Any = "float32"
) -> pa.FixedShapeTensorArray:
    """Build an embedding column: an ``(N, dim)`` array becomes ``fixed_shape_tensor((dim,))``.

    Each row is one ``dim``-length vector. The result's ``.storage`` is ``fixed_size_list<dtype, dim>``,
    the layout vector indexes (LanceDB, DuckDB) operate on. Values are cast to ``dtype`` (float32 by
    default).
    """
    stacked = _stack(vectors)
    if stacked.ndim != 2:
        raise ValueError(
            f"embedding_array expects a 2-D (N, dim) array, got shape {stacked.shape}; "
            "pass one row per embedding"
        )
    if np.dtype(dtype) != stacked.dtype:
        stacked = stacked.astype(dtype)
    return tensor_array(stacked)


def to_numpy(column: Any) -> NDArray[Any]:
    """Read a fixed-shape tensor column back to an ``(N, *shape)`` numpy array.

    Accepts a :class:`pyarrow.FixedShapeTensorArray` or a :class:`pyarrow.ChunkedArray` of one. A
    multi-chunk column is combined first; a sliced column is handled by pyarrow.
    """
    if not is_tensor(getattr(column, "type", None)):
        got = getattr(column, "type", type(column))
        raise TypeError(f"to_numpy expects a fixed_shape_tensor column, got {got}")
    array = column if hasattr(column, "to_numpy_ndarray") else column.combine_chunks()
    return np.asarray(array.to_numpy_ndarray())


def is_tensor(type_: Any) -> bool:
    """True if ``type_`` is the ``fixed_shape_tensor`` extension type."""
    return getattr(type_, "extension_name", None) == _TENSOR_EXTENSION_NAME


def _stack(arrays: NDArray[Any] | Sequence[NDArray[Any]]) -> NDArray[Any]:
    if isinstance(arrays, np.ndarray):  # already an (N, *shape) ndarray
        return arrays
    items = list(arrays)
    if not items:
        raise ValueError("empty sequence of arrays")  # caller-agnostic: _stack serves both builders
    shapes = {item.shape for item in items}
    if len(shapes) != 1:
        raise ValueError(f"arrays must share one shape, got {sorted(shapes)}")
    dtypes = {str(item.dtype) for item in items}
    if len(dtypes) != 1:
        raise ValueError(f"arrays must share one dtype, got {sorted(dtypes)}")
    return np.asarray(np.stack(items))
