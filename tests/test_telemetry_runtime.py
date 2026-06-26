"""S2: every run ships a telemetry report end-to-end, deterministically, with no behavior change."""

from nautilus.core.records import EOS_FRAME
from nautilus.core.time import TestClock
from nautilus.operators import InMemorySource, KeyedCount, Tokenize
from nautilus.runtime.local import run_local_chain
from nautilus.runtime.result import RunResult
from nautilus.telemetry.catalog import Tier
from nautilus.telemetry.recorder import TelemetryConfig
from nautilus.telemetry.report import BufferSink
from nautilus.testing import data

WORDS = [data(line=["the cat sat", "the dog ran"]), data(line=["the cat the cat"]), EOS_FRAME]


def _chain():
    return InMemorySource(list(WORDS)), [Tokenize("line", "word"), KeyedCount("word")]


def _counter(report, op_id, name):
    op = next(o for o in report.operators if o.operator_id == op_id)
    return sum(p.value for p in op.counters if p.name == name)


async def test_run_returns_result_with_telemetry_and_batches():
    src, ops = _chain()
    result = await run_local_chain(src, ops, clock=TestClock())
    assert isinstance(result, RunResult)
    # iterates and indexes over the emitted batches
    assert len(result) >= 1
    counts = {}
    for rb in result:  # iteration delegates to batches
        for w, c in zip(rb.column("word").to_pylist(), rb.column("count").to_pylist(), strict=True):
            counts[w] = c
    assert counts == {"the": 4, "cat": 3, "sat": 1, "dog": 1, "ran": 1}
    assert result[0] is result.batches[0]
    # Arrow-first readers collapse the per-batch zip
    assert {r["word"]: r["count"] for r in result.to_pylist()} == counts
    assert set(result.to_pydict()) == {"word", "count"}
    assert result.telemetry.schema_version == 2


async def test_rows_are_conserved_across_edges():
    src, ops = _chain()
    rep = (await run_local_chain(src, ops, clock=TestClock())).telemetry
    # source emits 3 lines; Tokenize explodes them to words; KeyedCount folds to distinct words.
    assert _counter(rep, "source", "operator.rows_out") == 3
    # words crossing op0->op1 are conserved
    assert _counter(rep, "op0", "operator.rows_out") == _counter(rep, "op1", "operator.rows_in")
    # KeyedCount emits 5 distinct words; the sink receives exactly those
    assert _counter(rep, "op1", "operator.rows_out") == 5
    assert _counter(rep, "sink", "operator.rows_in") == 5
    # eos accounting: every operator saw EOS on its single input
    assert _counter(rep, "op0", "eos.received") == 1


async def test_topology_edges_resolve_to_nodes():
    src, ops = _chain()
    rep = (await run_local_chain(src, ops, clock=TestClock())).telemetry
    assert rep.topology is not None
    node_ids = {n.operator_id for n in rep.topology.nodes}
    assert node_ids == {"source", "op0", "op1", "sink"}
    for e in rep.topology.edges:
        assert e.src_operator_id in node_ids and e.dst_operator_id in node_ids


async def test_structural_digest_is_stable_across_runs():
    digests = set()
    for _ in range(50):
        src, ops = _chain()
        rep = (await run_local_chain(src, ops, clock=TestClock())).telemetry
        digests.add(rep.structural_digest())
    assert len(digests) == 1, f"structural digest is not deterministic: {digests}"


async def test_sink_receives_report():
    src, ops = _chain()
    sink = BufferSink()
    await run_local_chain(src, ops, clock=TestClock(), sink=sink)
    assert len(sink.reports) == 1


async def test_off_tier_does_zero_catalog_lookups(monkeypatch):
    import nautilus.telemetry.recorder as rec_mod

    calls: list[str] = []
    orig = rec_mod.metric_spec
    monkeypatch.setattr(rec_mod, "metric_spec", lambda name: calls.append(name) or orig(name))

    src, ops = _chain()
    result = await run_local_chain(src, ops, telemetry=TelemetryConfig(tier=Tier.OFF))
    assert calls == [], f"OFF tier performed catalog lookups: {calls}"
    assert result.telemetry.summary.total_rows_in == 0  # nothing recorded


async def test_bytes_metrics_only_at_full_tier():
    src, ops = _chain()
    counters_run = (
        await run_local_chain(
            src, ops, telemetry=TelemetryConfig(tier=Tier.COUNTERS, clock=TestClock())
        )
    ).telemetry
    assert _counter(counters_run, "op0", "operator.bytes_in") == 0  # absent at COUNTERS

    src, ops = _chain()
    full_run = (
        await run_local_chain(
            src, ops, telemetry=TelemetryConfig(tier=Tier.FULL, clock=TestClock())
        )
    ).telemetry
    assert _counter(full_run, "op0", "operator.bytes_in") > 0  # present at FULL
