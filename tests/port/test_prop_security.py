"""Tier 4 property-based conformance tests for security.

These pin invariants of the shared-secret handshake MAC and the cluster-secret length floor that must
hold over the WHOLE input space, not just the hand-picked vectors the Tier-1 characterization tests
(``test_security.py``) fix. A future Python->Rust port re-implements :func:`_proof` and
:func:`cluster_secret`; a divergence that shows only on some inputs — a reordered MAC input, a dropped
version tag, an off-by-one length check — is one a port could ship silently, so each test drives many
random inputs from a FIXED seed and asserts the universally-quantified property on every one.

Inputs are drawn to hit the edges the invariants name: empty and long secrets, all-equal and swapped
nonces, both direction tags, and the 15/16-byte boundary of the secret floor.
"""

from __future__ import annotations

import hmac
import random

import pytest

from nautilus.security.handshake import _VERSION, _proof
from nautilus.security.secret import SecretError, cluster_secret

_SECRET_ENV = "NAUTILUS_CLUSTER_SECRET"


def _rand_bytes(rng: random.Random, n: int) -> bytes:
    return bytes(rng.getrandbits(8) for _ in range(n))


def _rand_inputs(rng: random.Random) -> tuple[bytes, bytes, bytes, bytes]:
    """A (secret, server_nonce, client_nonce, tag) draw covering the edges the invariants name: a
    variable-length secret including the empty and single-byte cases, 32-byte nonces (the real width)
    that are sometimes equal, and one of the two real direction tags."""
    secret = _rand_bytes(rng, rng.randint(0, 40))
    server_nonce = _rand_bytes(rng, 32)
    # Sometimes reuse the server nonce so the swap/order invariants are exercised on equal nonces too.
    client_nonce = server_nonce if rng.random() < 0.2 else _rand_bytes(rng, 32)
    tag = rng.choice([b"C", b"S"])
    return secret, server_nonce, client_nonce, tag


def test_proof_deterministic_and_32_bytes() -> None:
    """For all inputs, _proof returns the same 32-byte value on every call."""
    rng = random.Random(1234)
    for _ in range(500):
        secret, server_nonce, client_nonce, tag = _rand_inputs(rng)
        first = _proof(secret, server_nonce, client_nonce, tag)
        assert len(first) == 32
        assert first == _proof(secret, server_nonce, client_nonce, tag)


def test_proof_input_order_matters() -> None:
    """For all distinct nonces, swapping server_nonce and client_nonce changes the proof."""
    rng = random.Random(1234)
    for _ in range(500):
        secret, server_nonce, client_nonce, tag = _rand_inputs(rng)
        if server_nonce == client_nonce:
            continue  # a swap of equal nonces is a no-op, so it cannot change the proof
        assert _proof(secret, server_nonce, client_nonce, tag) != _proof(
            secret, client_nonce, server_nonce, tag
        )


def test_direction_tag_domain_separation() -> None:
    """For all secrets/nonces, the b'C' proof and the b'S' proof differ."""
    rng = random.Random(1234)
    for _ in range(500):
        secret, server_nonce, client_nonce, _ = _rand_inputs(rng)
        assert _proof(secret, server_nonce, client_nonce, b"C") != _proof(
            secret, server_nonce, client_nonce, b"S"
        )


def test_version_tag_in_mac_input() -> None:
    """_VERSION is b'nautilus-auth-v1' and the first MAC component: hmac_sha256(secret, _VERSION+server+client+tag) == _proof."""
    assert _VERSION == b"nautilus-auth-v1"
    rng = random.Random(1234)
    for _ in range(500):
        secret, server_nonce, client_nonce, tag = _rand_inputs(rng)
        recomputed = hmac.new(
            secret, _VERSION + server_nonce + client_nonce + tag, "sha256"
        ).digest()
        assert recomputed == _proof(secret, server_nonce, client_nonce, tag)


def test_proof_changes_with_any_input() -> None:
    """For all inputs, flipping the secret, either nonce, or the tag changes the proof."""
    rng = random.Random(1234)
    for _ in range(500):
        secret, server_nonce, client_nonce, tag = _rand_inputs(rng)
        base = _proof(secret, server_nonce, client_nonce, tag)

        # Perturb by flipping an existing byte (or, for the empty secret, appending a non-zero one). A
        # trailing *zero* byte would NOT change the proof: HMAC right-zero-pads a sub-block-size key to
        # the 64-byte block, so `K` and `K + b"\x00"` are the same key — a real HMAC property, not the
        # kind of change this invariant is about.
        if secret:
            other_secret = bytes([secret[0] ^ 0x01]) + secret[1:]
        else:
            other_secret = b"\x01"
        assert _proof(other_secret, server_nonce, client_nonce, tag) != base

        flipped_server = bytes([server_nonce[0] ^ 0x01]) + server_nonce[1:]
        assert _proof(secret, flipped_server, client_nonce, tag) != base

        flipped_client = bytes([client_nonce[0] ^ 0x01]) + client_nonce[1:]
        assert _proof(secret, server_nonce, flipped_client, tag) != base

        other_tag = b"S" if tag == b"C" else b"C"
        assert _proof(secret, server_nonce, client_nonce, other_tag) != base


def test_secret_length_minimum_16(monkeypatch: pytest.MonkeyPatch) -> None:
    """For all lengths, cluster_secret rejects a secret < 16 bytes (SecretError) and accepts one >= 16."""
    rng = random.Random(1234)
    # The env value is decoded to bytes, so a printable ASCII char is one byte; drive lengths across the
    # 16-byte floor, always including the 15/16 boundary and the empty (0-byte) case.
    for length in [0, 15, 16, 17, *(rng.randint(1, 64) for _ in range(500))]:
        monkeypatch.setenv(_SECRET_ENV, "x" * length)
        if length < 16:
            with pytest.raises(SecretError, match="at least 16"):
                cluster_secret()
        else:
            assert cluster_secret() == b"x" * length
