"""Tier 0 cross-language conformance vectors.

``conformance/vectors.json`` is a language-neutral golden file: each vector stores an input and the
output the *current* Python implementation produces for it. These tests load that file and assert the
code still reproduces every vector, so the JSON is a contract a future Rust port can load and check
itself against. When you change any pinned behavior here, you must also regenerate the vector (the file
is not hand-maintained) — a divergence is a real behavior change, not a test to loosen.

Coverage:
- ``partition.stable_bucket`` — the process-stable key hash the keyed shuffle routes on. Pins both the
  msgpack canonicalization and the blake2b digest, since co-location depends on both being identical in
  any process and any language.
- ``handshake._proof`` — the HMAC challenge-response proof; a Rust peer must produce byte-identical
  proofs or it cannot join the cluster.
- ``transport.framing`` — the kind byte and the exact wire bytes for control/credit messages, plus a
  decoder round-trip.
- ``report.structural_digest`` — the machine-independent correctness anchor a benchmark diff compares;
  a port that computes the same results must emit the same digest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import msgpack

import nautilus
from nautilus.core.records import EOS_FRAME, Barrier, Batch
from nautilus.core.time import TestClock
from nautilus.runtime.partition import stable_bucket
from nautilus.security.handshake import _proof
from nautilus.telemetry.recorder import InstanceRecorder, TelemetryConfig
from nautilus.telemetry.report import RunMeta, build_report
from nautilus.telemetry.report.report import Edge, OperatorNode, Topology
from nautilus.testing import batch
from nautilus.transport.framing import Kind, decode, encode_credit, encode_frame, split

_VECTORS_PATH = Path(__file__).resolve().parent.parent / "conformance" / "vectors.json"


def _vectors() -> dict:
    return json.loads(_VECTORS_PATH.read_text())


def _decode_key_elem(elem: list) -> object:
    """Reverse the language-neutral [type, value] tagging vectors.json uses for a key scalar. Kept in
    lockstep with the encoder in the generation script (see the file's ``_doc`` fields)."""
    tag, value = elem
    if tag == "null":
        return None
    if tag == "bool":
        return bool(value)
    if tag == "int":
        return int(value)
    if tag == "str":
        return str(value)
    if tag == "bytes":
        return bytes.fromhex(value)
    raise AssertionError(f"unknown key-elem tag {tag!r}")


# --- partition hash ----------------------------------------------------------------------------


def test_partition_vectors_reproduce_bucket_and_hash() -> None:
    section = _vectors()["partition_stable_bucket"]
    num_downstream = section["num_downstream"]
    for vec in section["vectors"]:
        key = tuple(_decode_key_elem(e) for e in vec["key"])

        # The msgpack canonicalization and the blake2b digest are both part of the contract: a port
        # must reproduce the exact bytes and the exact 64-bit int, not just the final bucket.
        raw = msgpack.packb(list(key), use_bin_type=True)
        assert raw.hex() == vec["packb_hex"], f"packb changed for {key!r}"
        digest_u64 = int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")
        assert digest_u64 == vec["digest_u64"], f"blake2b digest changed for {key!r}"

        assert digest_u64 % num_downstream == vec["bucket_mod_4"]
        assert (
            stable_bucket(key, num_downstream) == vec["bucket_mod_4"]
        ), f"bucket changed for {key!r}"


def test_partition_bool_and_int_do_not_collide() -> None:
    # bool ⊆ int in Python, but msgpack type-tags them apart so True and 1 route independently; the
    # vectors pin both, and this guards the property directly. Keyed by the tagged encoding, not the
    # decoded tuple, because (True,) and (1,) collapse to one Python dict key (hash(True) == hash(1)).
    section = _vectors()["partition_stable_bucket"]
    by_tag = {tuple(tuple(e) for e in v["key"]): v for v in section["vectors"]}
    true_key = (("bool", True),)
    int_key = (("int", 1),)
    assert true_key in by_tag and int_key in by_tag
    assert by_tag[true_key]["packb_hex"] != by_tag[int_key]["packb_hex"]
    # every vector packs to distinct bytes — no two keys collide on the wire
    assert len({v["packb_hex"] for v in section["vectors"]}) == len(section["vectors"])


# --- auth proof --------------------------------------------------------------------------------


def test_auth_proof_vector() -> None:
    v = _vectors()["auth_proof"]
    proof = _proof(
        bytes.fromhex(v["secret_hex"]),
        bytes.fromhex(v["server_nonce_hex"]),
        bytes.fromhex(v["client_nonce_hex"]),
        bytes.fromhex(v["tag_hex"]),
    )
    assert proof.hex() == v["proof_hex"]


# --- transport frame ---------------------------------------------------------------------------


def test_frame_kind_bytes() -> None:
    kinds = _vectors()["transport_frame"]["kind_bytes"]
    assert int(Kind.DATA) == kinds["DATA"]
    assert int(Kind.CONTROL) == kinds["CONTROL"]
    assert int(Kind.CREDIT) == kinds["CREDIT"]


def test_frame_eos_encodes_and_roundtrips() -> None:
    v = _vectors()["transport_frame"]["eos"]
    encoded = encode_frame(EOS_FRAME)
    assert encoded.hex() == v["frame_hex"]
    kind, payload = split(encoded)
    assert kind == Kind.CONTROL
    assert payload.hex() == v["control_payload_hex"]
    # control_shape is the msgpack map the payload decodes to (compared as a dict — msgpack key order is
    # fixed by the encoder, but JSON sorted the vector's shape, so decode rather than re-pack the shape).
    assert msgpack.unpackb(payload, raw=False) == v["control_shape"]
    assert decode(kind, payload) == EOS_FRAME


def test_frame_barrier_encodes_and_roundtrips() -> None:
    v = _vectors()["transport_frame"]["barrier"]
    barrier = Barrier(v["checkpoint_id"])
    encoded = encode_frame(barrier)
    assert encoded.hex() == v["frame_hex"]
    kind, payload = split(encoded)
    assert kind == Kind.CONTROL
    assert payload.hex() == v["control_payload_hex"]
    assert msgpack.unpackb(payload, raw=False) == v["control_shape"]
    assert decode(kind, payload) == barrier


def test_frame_credit_encodes_and_roundtrips() -> None:
    v = _vectors()["transport_frame"]["credit"]
    assert msgpack.packb(v["count"]).hex() == v["control_payload_hex"]
    encoded = encode_credit(v["count"])
    assert encoded.hex() == v["frame_hex"]
    kind, payload = split(encoded)
    assert kind == Kind.CREDIT
    assert decode(kind, payload) == v["count"]


def test_frame_data_kind_byte() -> None:
    # A DATA frame's Arrow IPC payload is not byte-stable to pin as a literal, but the kind tag is; keep
    # the DATA path in the round-trip contract by checking the leading byte the port must emit.
    kind, _payload = split(encode_frame(Batch(batch(x=[1]))))
    assert kind == Kind.DATA
    assert int(kind) == _vectors()["transport_frame"]["kind_bytes"]["DATA"]


# --- structural digest -------------------------------------------------------------------------


def _digest_meta() -> RunMeta:
    # Timing/run_id fields are deliberately arbitrary: structural_digest must ignore them (asserted by
    # test_structural_digest_ignores_run_metadata), so their values do not affect the golden.
    return RunMeta(
        run_id="conformance-report",
        started_at_micros=0,
        ended_at_micros=1000,
        wall_micros=1000,
        clock_kind="TestClock",
        nautilus_version=nautilus.__version__,
        python_version="3.12",
        config_digest="deadbeef",
        capacity=16,
    )


def _digest_snapshots(step_src: int = 555, step_snk: int = 999) -> list:
    """Two operators — a source emitting 100 rows and a sink counting them to 7 — carrying exactly the
    STRUCTURAL_METRICS the digest hashes (rows_in, rows_out, eos.received). runtime.step_micros is
    recorded too so the report is realistic, but it is timing and the digest must exclude it."""
    src = InstanceRecorder(
        operator_id="src",
        op_class="Numbers",
        kind="source",
        config=TelemetryConfig(clock=TestClock(0)),
    )
    src.incr("operator.rows_out", 100, operator_id="src", subtask_index=0)
    src.incr("runtime.step_micros", step_src, operator_id="src", subtask_index=0)
    src.incr("eos.received", 1, operator_id="src", subtask_index=0)

    snk = InstanceRecorder(
        operator_id="snk",
        op_class="Count",
        kind="one_input",
        config=TelemetryConfig(clock=TestClock(0)),
    )
    snk.incr("operator.rows_in", 100, operator_id="snk", subtask_index=0)
    snk.incr("operator.rows_out", 7, operator_id="snk", subtask_index=0)
    snk.incr("runtime.step_micros", step_snk, operator_id="snk", subtask_index=0)
    snk.incr("eos.received", 1, operator_id="snk", subtask_index=0)
    return [src.snapshot(), snk.snapshot()]


def _digest_topology() -> Topology:
    return Topology(
        nodes=(
            OperatorNode(operator_id="src", op_class="Numbers", kind="source"),
            OperatorNode(operator_id="snk", op_class="Count", kind="one_input"),
        ),
        edges=(
            Edge(
                src_operator_id="src",
                dst_operator_id="snk",
                channel_index=0,
                partitioner="Forward",
                capacity=16,
            ),
        ),
    )


def _build_digest_report(step_src: int = 555, step_snk: int = 999, meta: RunMeta | None = None):
    return build_report(
        _digest_snapshots(step_src, step_snk),
        meta=meta or _digest_meta(),
        topology=_digest_topology(),
    )


def test_structural_digest_vector() -> None:
    digest = _build_digest_report().structural_digest()
    assert len(digest) == 64
    assert digest == _vectors()["structural_digest"]["digest_hex"]


def test_structural_digest_ignores_run_metadata() -> None:
    # The golden must be reproducible from the structural facts alone: different wall/run_id/step timing
    # yields the same digest. This is the property that makes the vector portable across machines and
    # languages, so a Rust port need only reproduce the row/EOS counts and the topology.
    other_meta = RunMeta(
        run_id="different-run",
        started_at_micros=5,
        ended_at_micros=99999,
        wall_micros=99999,
        clock_kind="OtherClock",
        nautilus_version="0.0.0",
        python_version="9.9",
        config_digest="ffffffff",
        capacity=1,
    )
    other = _build_digest_report(step_src=1, step_snk=88888, meta=other_meta)
    assert other.structural_digest() == _vectors()["structural_digest"]["digest_hex"]
