"""Imagery + embeddings example.

A stream of ``(H, W, C)`` uint8 image tiles, each reduced to a per-tile embedding vector. Tiles and
embeddings are carried as Arrow ``fixed_shape_tensor`` columns (see ``nautilus.tensors``). Run
directly with ``python examples/imagery.py`` or via the CLI with ``nautilus run image-embed``.
"""

from __future__ import annotations

from nautilus import run
from nautilus.pipelines import image_embed
from nautilus.tensors import to_numpy


def main() -> None:
    source, transforms = image_embed()
    result = run(source, transforms)
    for record_batch in result:
        ids = record_batch.column("tile_id").to_pylist()
        embeddings = to_numpy(record_batch.column("embedding"))
        print(f"tiles {ids} -> embeddings {embeddings.shape} {embeddings.dtype}")
    print(result.telemetry.summary)


if __name__ == "__main__":
    main()
