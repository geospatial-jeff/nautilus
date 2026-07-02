"""S3: the agent-facing surface — markdown digest (numbers ⊆ JSON), query helpers that project but
never diagnose, and per-operator metrics (like the end-of-stream flush count) attributed to the operator
that recorded them."""

import re

from nautilus.core.records import EOS_FRAME
from nautilus.core.time import TestClock
from nautilus.driver.local import run_local_chain
from nautilus.operators import InMemorySource, KeyedCount, Tokenize
from nautilus.testing import data

WORDS = [data(line=["the cat sat", "the dog ran"]), data(line=["the cat the cat"]), EOS_FRAME]


async def _wordcount_report():
    result = await run_local_chain(
        InMemorySource(list(WORDS)),
        [Tokenize("line", "word"), KeyedCount("word")],
        clock=TestClock(),
    )
    return result.telemetry


async def test_markdown_numbers_are_a_subset_of_json():
    rep = await _wordcount_report()
    md = rep.to_markdown(token_budget=2000)
    js = rep.to_json()
    for token in set(re.findall(r"\d+", md)):
        assert token in js, f"number {token!r} in markdown is not present in the JSON report"


async def test_markdown_has_summary_and_explicit_rankings():
    md = (await _wordcount_report()).to_markdown()
    assert "## summary" in md
    assert "by self-time" in md
    assert "by send-wait" in md
    assert "to_json()" in md  # points the agent at the full data


async def test_query_helpers_project_not_diagnose():
    rep = await _wordcount_report()
    assert rep.operator("op0") is not None
    assert rep.operator("does-not-exist") is None
    assert rep.edge("source", "op0") is not None
    assert rep.edge("op0", "source") is None

    ids = {s.operator_id for s in rep.by_self_time()}
    assert ids == {"source", "op0", "op1", "sink"}
    busy = [s.busy_micros_total for s in rep.by_self_time()]
    assert busy == sorted(busy, reverse=True)  # ranked, descending, by the stated axis
    sw = [s.send_wait_micros_total for s in rep.by_send_wait()]
    assert sw == sorted(sw, reverse=True)


async def test_on_eos_flush_of_global_aggregation_counted_once():
    op1 = (await _wordcount_report()).operator("op1")
    assert op1 is not None
    calls = sum(p.value for p in op1.counters if p.name == "operator.on_eos_calls")
    assert (
        calls == 1
    )  # KeyedCount, a global aggregation, flushes exactly once when its inputs close
