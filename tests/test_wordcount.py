"""Stage 0 demo: a bounded word-count returns a deterministic result in one process."""

from nautilus.core.operator import ListCollector
from nautilus.core.records import EOS_FRAME
from nautilus.driver.local import run, run_local_chain
from nautilus.operators import InMemorySource, KeyedCount, Tokenize, from_batches
from nautilus.testing import batch, data

_EXPECTED = {"the": 4, "cat": 3, "sat": 1, "dog": 1, "ran": 1}


def test_tokenize_drops_empty_tokens() -> None:
    # Whitespace runs, leading/trailing whitespace, and empty/null rows must yield no empty 'word' cells
    # — the str.split() property that a naive columnar split would break by emitting empty-string tokens
    # (which would then become real keys in the shuffle/count).
    coll = ListCollector()
    Tokenize("line", "word").process(batch(line=["  the   cat  ", "", "dog\t\nfox", None]), coll)
    words = [w for rb in coll.drain() for w in rb.column("word").to_pylist()]
    assert words == ["the", "cat", "dog", "fox"]


async def test_bounded_wordcount():
    frames = [
        data(line=["the cat sat", "the dog ran"]),
        data(line=["the cat the cat"]),
        EOS_FRAME,
    ]
    results = await run_local_chain(
        InMemorySource(frames), [Tokenize("line", "word"), KeyedCount("word")]
    )

    counts: dict[str, int] = {}
    for rb in results:
        for word, count in zip(
            rb.column("word").to_pylist(), rb.column("count").to_pylist(), strict=True
        ):
            counts[word] = count

    assert counts == _EXPECTED


def test_sync_run_with_from_batches():
    # The sync run() one-liner + from_batches factory (appends EOS) + Arrow-first reader — the
    # minimal client surface, no asyncio.run/async def or manual EOS_FRAME needed.
    result = run(
        from_batches(data(line=["the cat sat", "the dog ran"]), data(line=["the cat the cat"])),
        [Tokenize("line", "word"), KeyedCount("word")],
    )
    assert {r["word"]: r["count"] for r in result.to_pylist()} == _EXPECTED
