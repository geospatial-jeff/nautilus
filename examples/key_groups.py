"""Stage 2b demo: the same keyed run at G == Q and at G > Q produces the same result.

A keyed shuffle routes through key groups: each key is hashed to one of G groups, and a static
group→instance table maps groups to the Q instances. Raising G above Q (the ``key_groups`` argument) is
the rescale seam — a key's group is fixed, so a later change to the instance count is a new table over
the same groups, never a re-hash of state.

This demo runs one keyed word-count at several G values. The **output multiset is identical for every
G >= Q** (co-partitioning holds). The **structural digest** equals the direct-hash run exactly when *Q
divides G* — because an instance is ``(hash(key) mod G) mod Q``, which equals ``hash(key) mod Q`` only
then — so G in {3, 6, 12} match the digest while G = 7 keeps the result but routes keys to different
instances, so its digest differs.

Run with:  python examples/key_groups.py
"""

from __future__ import annotations

import asyncio
from collections import Counter

from nautilus.core.records import EOS_FRAME
from nautilus.operators import InMemorySource, KeyedCount, Tokenize
from nautilus.driver.parallel import Stage, graph_from_stages
from nautilus.driver.run import run_plan
from nautilus.testing import data

_Q = 3  # the KeyedCount stage runs as 3 instances


def _source() -> InMemorySource:
    return InMemorySource(
        [
            data(line=["the quick brown fox the lazy dog"]),
            data(line=["the fox jumped the lazy fox ran a dog and a cat"]),
            EOS_FRAME,
        ]
    )


def _counts(result) -> dict[str, int]:
    return {row["word"]: row["count"] for row in result.to_pylist()}


async def main() -> None:
    def graph():
        return graph_from_stages(
            _source(),
            [
                Stage(lambda: Tokenize("line", "word")),
                Stage(lambda: KeyedCount("word"), _Q, ["word"]),
            ],
        )

    baseline_counts: Counter | None = None
    baseline_digest: str | None = None
    print(f"KeyedCount runs as Q={_Q} instances; routing each run through G key groups:\n")
    for g in (_Q, _Q, 6, 7, 12):  # G == Q (identity), then G > Q (multiples and non-multiples)
        result = await run_plan(graph(), key_groups=g)
        counts = Counter(_counts(result))
        digest = result.telemetry.structural_digest()
        if baseline_counts is None:
            baseline_counts, baseline_digest = counts, digest
        same_result = counts == baseline_counts
        same_digest = digest == baseline_digest
        tag = "Q divides G: same digest" if g % _Q == 0 else "Q does not divide G: keys re-spread"
        print(
            f"  G={g:2d}  same result: {same_result!s:5}  same digest: {same_digest!s:5}  ({tag})"
        )

    print(f"\nword counts (identical for every G >= Q={_Q}):")
    for word, count in sorted(baseline_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {count:3d}  {word}")


if __name__ == "__main__":
    asyncio.run(main())
