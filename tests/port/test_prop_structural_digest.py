"""Tier 4 property-based conformance tests for structural-digest.

``structural_digest()`` is the machine-independent correctness anchor a benchmark diff and a future Rust
port compare against: the same logical run must yield the same 64-hex digest regardless of scheduling,
placement, machine, or timing. Tier 0 (``test_conformance_vectors.py``) pins one golden input/output
pair; this file pins the *universally-quantified* laws the digest must obey over the whole input space,
because a divergence the golden vector happens not to exercise still silently breaks a port.

Each test drives many random reports from a fixed seed and asserts one law for every one. Randomness is
seeded (``random.Random(_SEED)``) so a failure is reproducible and the suite is deterministic; no
hypothesis, no new dependencies.
"""

from __future__ import annotations

import random

import nautilus
from nautilus.core.time import TestClock
from nautilus.telemetry.recorder import InstanceRecorder, TelemetryConfig
from nautilus.telemetry.report import RunMeta, build_report
from nautilus.telemetry.report.report import Edge, OperatorNode, Topology

_SEED = 1234
_CASES = 200

#: The counters the digest hashes (see ``STRUCTURAL_METRICS``). Their label keyword differs: the row
#: counters are keyed by ``subtask_index``, ``eos.received`` by ``input_index`` — a recorder rejects the
#: wrong label keyword, so each structural counter must be written with its own.
_ROW_METRICS = ("operator.rows_in", "operator.rows_out")

#: int64 extremes plus small values — a port hashes the decimal text of each counter total, so the digest
#: must be stable at the range boundaries, not only for the "hundreds of rows" the golden vector uses.
_INT_CHOICES = (0, 1, 2, 7, 100, (1 << 31) - 1, 1 << 32, (1 << 63) - 1)

_KINDS = ("source", "one_input", "two_input")


def _meta(**overrides: object) -> RunMeta:
    """A RunMeta whose timing/identity fields the digest must ignore; overrides let a test vary exactly
    the metadata field it is proving irrelevant."""
    base: dict[str, object] = {
        "run_id": "prop-run",
        "started_at_micros": 0,
        "ended_at_micros": 1000,
        "wall_micros": 1000,
        "clock_kind": "TestClock",
        "nautilus_version": nautilus.__version__,
        "python_version": "3.12",
        "config_digest": "deadbeef",
        "capacity": 16,
    }
    base.update(overrides)
    return RunMeta(**base)  # type: ignore[arg-type]


def _snapshot(
    operator_id: str,
    op_class: str,
    kind: str,
    *,
    rows_in: int | None = None,
    rows_out: int | None = None,
    eos: int | None = None,
    subtask_index: int = 0,
    step_micros: int | None = None,
):
    """One InstanceSnapshot carrying the requested structural counters, plus optional ``step_micros`` —
    a timing counter the digest must exclude, recorded so a snapshot looks realistic."""
    r = InstanceRecorder(
        operator_id=operator_id,
        op_class=op_class,
        kind=kind,
        subtask_index=subtask_index,
        config=TelemetryConfig(clock=TestClock(0)),
    )
    if rows_in is not None:
        r.incr("operator.rows_in", rows_in, operator_id=operator_id, subtask_index=subtask_index)
    if rows_out is not None:
        r.incr("operator.rows_out", rows_out, operator_id=operator_id, subtask_index=subtask_index)
    if eos is not None:
        r.incr("eos.received", eos, operator_id=operator_id, input_index=0)
    if step_micros is not None:
        r.incr(
            "runtime.step_micros", step_micros, operator_id=operator_id, subtask_index=subtask_index
        )
    return r.snapshot()


def _random_operators(rng: random.Random) -> list[dict[str, object]]:
    """A random set of operator specs (id + class + kind + structural counter values). Returns the specs,
    not snapshots, so a test can realize the same logical run in different orders or with different
    timing."""
    specs: list[dict[str, object]] = []
    for i in range(rng.randint(0, 5)):
        specs.append(
            {
                "operator_id": f"op{i}",
                "op_class": rng.choice(("Numbers", "Filter", "Count", "Join")),
                "kind": rng.choice(_KINDS),
                "rows_in": rng.choice(_INT_CHOICES),
                "rows_out": rng.choice(_INT_CHOICES),
                "eos": rng.randint(0, 2),
            }
        )
    return specs


def _snapshots_from_specs(specs: list[dict[str, object]], rng: random.Random | None = None) -> list:
    """Realize operator specs into snapshots. When ``rng`` is given, each spec also gets a random
    ``step_micros`` — timing the digest must ignore — so equal-structure/different-timing runs can be
    compared."""
    snaps = []
    for s in specs:
        step = rng.randint(0, 10_000_000) if rng is not None else None
        snaps.append(
            _snapshot(
                str(s["operator_id"]),
                str(s["op_class"]),
                str(s["kind"]),
                rows_in=int(s["rows_in"]),  # type: ignore[arg-type]
                rows_out=int(s["rows_out"]),  # type: ignore[arg-type]
                eos=int(s["eos"]),  # type: ignore[arg-type]
                step_micros=step,
            )
        )
    return snaps


def _random_topology(specs: list[dict[str, object]], rng: random.Random) -> Topology | None:
    """A topology over the given operators: one node each, plus random forward edges between them. Returns
    None sometimes so the topology-absent path is exercised too."""
    if not specs or rng.random() < 0.2:
        return None
    nodes = tuple(
        OperatorNode(str(s["operator_id"]), str(s["op_class"]), str(s["kind"])) for s in specs
    )
    edges = []
    ids = [str(s["operator_id"]) for s in specs]
    for _ in range(rng.randint(0, len(ids))):
        src = rng.choice(ids)
        dst = rng.choice(ids)
        edges.append(
            Edge(
                src_operator_id=src,
                dst_operator_id=dst,
                channel_index=rng.randint(0, 2),
                partitioner=rng.choice(("Forward", "Hash", "Broadcast")),
                capacity=rng.choice((1, 16, 256)),
            )
        )
    return Topology(nodes=nodes, edges=tuple(edges))


def test_digest_invariant_to_snapshot_order() -> None:
    """For every run, structural_digest() is unchanged by the order snapshots are passed to build_report."""
    rng = random.Random(_SEED)
    for _ in range(_CASES):
        specs = _random_operators(rng)
        topology = _random_topology(specs, rng)
        snaps = _snapshots_from_specs(specs)
        base = build_report(snaps, meta=_meta(), topology=topology).structural_digest()
        shuffled = list(snaps)
        rng.shuffle(shuffled)
        got = build_report(shuffled, meta=_meta(), topology=topology).structural_digest()
        assert got == base


def test_digest_byte_determinism_by_content() -> None:
    """For every run, identical structural content yields the same digest regardless of timing metadata."""
    rng = random.Random(_SEED)
    for _ in range(_CASES):
        specs = _random_operators(rng)
        topology = _random_topology(specs, rng)
        # Same structural specs, but each side gets independently-random step_micros and a fully
        # different RunMeta (run_id, wall, version, clock). Only the structural facts are shared.
        left = build_report(
            _snapshots_from_specs(specs, rng), meta=_meta(), topology=topology
        ).structural_digest()
        right = build_report(
            _snapshots_from_specs(specs, rng),
            meta=_meta(
                run_id="other-run",
                started_at_micros=42,
                ended_at_micros=999_999,
                wall_micros=999_999,
                clock_kind="OtherClock",
                nautilus_version="0.0.0",
                python_version="9.9",
                config_digest="ffffffff",
                capacity=1,
            ),
            topology=topology,
        ).structural_digest()
        assert left == right


def test_digest_row_conservation_and_aggregation() -> None:
    """For every operator, splitting a counter across two same-key snapshots that sum to it yields the
    digest of one snapshot carrying the sum."""
    rng = random.Random(_SEED)
    for _ in range(_CASES):
        operator_id = f"op{rng.randint(0, 3)}"
        op_class = rng.choice(("Numbers", "Filter", "Count"))
        kind = rng.choice(_KINDS)
        total_out = rng.choice(_INT_CHOICES)
        total_in = rng.choice(_INT_CHOICES)
        split_out = rng.randint(0, total_out)
        split_in = rng.randint(0, total_in)

        # Two snapshots for the same (operator_id, subtask, node), each holding a partial count; the
        # report aggregation must sum them into the one-snapshot digest.
        parts = [
            _snapshot(
                operator_id,
                op_class,
                kind,
                rows_in=split_in,
                rows_out=split_out,
                eos=1,
            ),
            _snapshot(
                operator_id,
                op_class,
                kind,
                rows_in=total_in - split_in,
                rows_out=total_out - split_out,
                eos=0,
            ),
        ]
        summed = [
            _snapshot(
                operator_id,
                op_class,
                kind,
                rows_in=total_in,
                rows_out=total_out,
                eos=1,
            )
        ]
        got = build_report(parts, meta=_meta()).structural_digest()
        want = build_report(summed, meta=_meta()).structural_digest()
        assert got == want


def test_digest_sensitive_to_structural_change() -> None:
    """For every run, mutating any structural fact (a rows_out count, an operator, or an edge) changes the
    digest, while mutating only timing/metadata does not."""
    rng = random.Random(_SEED)
    for _ in range(_CASES):
        specs = _random_operators(rng)
        if not specs:
            continue  # an empty run has no structural fact to mutate; other cases cover it
        topology = _random_topology(specs, rng)
        snaps = _snapshots_from_specs(specs)
        base = build_report(snaps, meta=_meta(), topology=topology).structural_digest()

        # (a) timing/metadata-only change must NOT move the digest.
        timing_only = build_report(
            _snapshots_from_specs(specs, rng),
            meta=_meta(run_id="x", wall_micros=1, ended_at_micros=1, nautilus_version="9.9.9"),
            topology=topology,
        ).structural_digest()
        assert timing_only == base

        # (b) bump one operator's rows_out by 1 — a structural fact — the digest MUST move.
        mutated = list(specs)
        idx = rng.randrange(len(mutated))
        mutated[idx] = {**mutated[idx], "rows_out": int(mutated[idx]["rows_out"]) + 1}  # type: ignore[arg-type]
        rows_changed = build_report(
            _snapshots_from_specs(mutated), meta=_meta(), topology=topology
        ).structural_digest()
        assert rows_changed != base

        # (c) add an operator — a structural fact — the digest MUST move.
        extra = specs + [
            {
                "operator_id": "op_extra",
                "op_class": "Extra",
                "kind": "one_input",
                "rows_in": 1,
                "rows_out": 1,
                "eos": 1,
            }
        ]
        op_added = build_report(
            _snapshots_from_specs(extra), meta=_meta(), topology=topology
        ).structural_digest()
        assert op_added != base

        # (d) with a topology present, adding an edge — a structural fact — the digest MUST move.
        if topology is not None:
            ids = [str(s["operator_id"]) for s in specs]
            new_edge = Edge(ids[0], ids[-1], 99, "Forward", 16)
            if new_edge not in topology.edges:
                edge_added = build_report(
                    snaps,
                    meta=_meta(),
                    topology=Topology(nodes=topology.nodes, edges=topology.edges + (new_edge,)),
                ).structural_digest()
                assert edge_added != base


def test_digest_topology_edge_order() -> None:
    """For every graph, permuting the edge and node tuples of the topology yields the same digest."""
    rng = random.Random(_SEED)
    for _ in range(_CASES):
        specs = _random_operators(rng)
        topology = _random_topology(specs, rng)
        if topology is None:
            continue  # nothing to permute; the topology-absent path is covered elsewhere
        snaps = _snapshots_from_specs(specs)
        base = build_report(snaps, meta=_meta(), topology=topology).structural_digest()

        nodes = list(topology.nodes)
        edges = list(topology.edges)
        rng.shuffle(nodes)
        rng.shuffle(edges)
        permuted = Topology(nodes=tuple(nodes), edges=tuple(edges))
        got = build_report(snaps, meta=_meta(), topology=permuted).structural_digest()
        assert got == base
