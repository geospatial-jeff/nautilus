"""Tier 1 characterization tests pinning the security layer's observable contracts.

These pin the auth/secret/TLS boundary so a future Python->Rust rewrite is provably faithful. They are
hermetic: env vars are driven with monkeypatch, DNS resolution is stubbed with monkeypatched
``socket.getaddrinfo`` (never the real network), and TLS contexts are built from a self-signed cert made
in a temp dir. Goldens (the HMAC proof bytes, the secret error strings, the hex token) were derived by
running the real code and hardcoding its actual output.
"""

from __future__ import annotations

import shutil
import socket
import ssl
import subprocess

import pytest

from nautilus.security.handshake import _VERSION, _proof
from nautilus.security.secret import (
    SecretError,
    cluster_secret,
    dashboard_token,
    is_loopback,
)
from nautilus.security.tls import client_tls_context, server_tls_context

_SECRET_ENV = "NAUTILUS_CLUSTER_SECRET"
_TOKEN_ENV = "NAUTILUS_DASHBOARD_TOKEN"

# A 42-byte secret used across the dashboard-token and proof tests; comfortably over the 16-byte floor.
SECRET = b"a-sufficiently-long-test-secret-0123456789"
SECRET_HEX = "612d73756666696369656e746c792d6c6f6e672d746573742d7365637265742d30313233343536373839"


def _clear_secret_env(monkeypatch):
    monkeypatch.delenv(_SECRET_ENV, raising=False)
    monkeypatch.delenv(_TOKEN_ENV, raising=False)


# --- (a) dashboard_token derivation --------------------------------------------------------------


def test_dashboard_token_none_when_nothing_set(monkeypatch):
    # No explicit token and no cluster secret -> the dashboard runs open (None means "no auth").
    _clear_secret_env(monkeypatch)
    assert dashboard_token() is None


def test_dashboard_token_explicit_wins(monkeypatch):
    # An explicit token is returned verbatim even when a cluster secret is also configured.
    _clear_secret_env(monkeypatch)
    monkeypatch.setenv(_TOKEN_ENV, "explicit-tok")
    monkeypatch.setenv(_SECRET_ENV, SECRET.decode())
    assert dashboard_token() == "explicit-tok"


def test_dashboard_token_falls_back_to_secret_hex(monkeypatch):
    # No explicit token, but a cluster secret is set -> the token is that secret as lowercase hex.
    _clear_secret_env(monkeypatch)
    monkeypatch.setenv(_SECRET_ENV, SECRET.decode())
    token = dashboard_token()
    assert token == SECRET_HEX
    assert token == SECRET.hex()
    assert token == token.lower()  # .hex() is lowercase


def test_dashboard_token_empty_explicit_falls_through(monkeypatch):
    # An empty explicit token is falsy, so it falls through to the secret-hex derivation rather than
    # returning "" (empty string is not a usable bearer token).
    _clear_secret_env(monkeypatch)
    monkeypatch.setenv(_TOKEN_ENV, "")
    monkeypatch.setenv(_SECRET_ENV, SECRET.decode())
    assert dashboard_token() == SECRET_HEX


def test_dashboard_token_empty_explicit_and_no_secret_is_none(monkeypatch):
    # Empty explicit token falls through, and with no secret to fall back to the result is None.
    _clear_secret_env(monkeypatch)
    monkeypatch.setenv(_TOKEN_ENV, "")
    assert dashboard_token() is None


# --- (b) cluster_secret boundaries ---------------------------------------------------------------


def test_cluster_secret_unset_is_none(monkeypatch):
    # Unset is a distinct outcome from empty: None (no secret configured), not an error.
    _clear_secret_env(monkeypatch)
    assert cluster_secret() is None


def test_cluster_secret_16_bytes_accepted(monkeypatch):
    # Exactly the 16-byte floor is accepted, returned as raw bytes.
    _clear_secret_env(monkeypatch)
    monkeypatch.setenv(_SECRET_ENV, "x" * 16)
    assert cluster_secret() == b"x" * 16


def test_cluster_secret_15_bytes_rejected(monkeypatch):
    # One byte under the floor is refused loudly.
    _clear_secret_env(monkeypatch)
    monkeypatch.setenv(_SECRET_ENV, "x" * 15)
    with pytest.raises(SecretError, match="at least 16"):
        cluster_secret()


def test_cluster_secret_empty_string_rejected(monkeypatch):
    # An empty *set* value is a 0-byte secret -> SecretError, distinct from unset -> None above.
    _clear_secret_env(monkeypatch)
    monkeypatch.setenv(_SECRET_ENV, "")
    with pytest.raises(SecretError, match="at least 16"):
        cluster_secret()


def test_cluster_secret_at_missing_file_raises(monkeypatch, tmp_path):
    # An "@path" pointing at a file that does not exist is an operator error, surfaced as SecretError
    # (never a silently-absent secret).
    missing = tmp_path / "does-not-exist"
    monkeypatch.setenv(_SECRET_ENV, f"@{missing}")
    with pytest.raises(SecretError, match="could not be read"):
        cluster_secret()


def test_cluster_secret_at_file_read_and_stripped(monkeypatch, tmp_path):
    # "@path" reads the file's contents; a trailing newline is stripped (a common editor artifact).
    f = tmp_path / "secret"
    f.write_bytes(SECRET + b"\n")
    monkeypatch.setenv(_SECRET_ENV, f"@{f}")
    assert cluster_secret() == SECRET


# --- (c) is_loopback -----------------------------------------------------------------------------


def test_is_loopback_ipv6_literal():
    # "::1" is the IPv6 loopback literal; recognized without any resolution.
    assert is_loopback("::1") is True


def test_is_loopback_ipv4_literal():
    # "127.0.0.1" is the IPv4 loopback literal; recognized without any resolution.
    assert is_loopback("127.0.0.1") is True


def test_is_loopback_wildcard_binds_are_not_loopback():
    # A wildcard bind exposes every interface, so it is emphatically not loopback.
    assert is_loopback("0.0.0.0") is False
    assert is_loopback("::") is False
    assert is_loopback("") is False


def test_is_loopback_unresolvable_host_is_false(monkeypatch):
    # A name that fails to resolve (getaddrinfo raises) is treated as not-loopback — fail closed, and no
    # real DNS is hit because getaddrinfo is stubbed.
    def _boom(*_a, **_k):
        raise socket.gaierror("stubbed resolution failure")

    monkeypatch.setattr(socket, "getaddrinfo", _boom)
    assert is_loopback("nope.invalid.host") is False


def test_is_loopback_name_resolving_to_loopback_only_is_true(monkeypatch):
    # A name that resolves to loopback addresses only is loopback. getaddrinfo is stubbed to that.
    def _loop(*_a, **_k):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", _loop)
    assert is_loopback("my-local-name") is True


def test_is_loopback_name_resolving_to_public_addr_is_false(monkeypatch):
    # A name that resolves to a non-loopback address is not loopback (needs a secret to bind).
    def _public(*_a, **_k):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", _public)
    assert is_loopback("my-remote-name") is False


# --- (d) TLS context contracts -------------------------------------------------------------------


def _self_signed(tmp_path):
    """A self-signed cert usable as both the presented cert and the verifying CA, so one cert drives mTLS
    hermetically. Returns (certfile, keyfile) as strings."""
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-nodes",
            "-subj",
            "/CN=nautilus-test",
            "-addext",
            "basicConstraints=CA:TRUE",
        ],
        check=True,
        capture_output=True,
    )
    return str(cert), str(key)


@pytest.fixture
def _cert(tmp_path):
    if shutil.which("openssl") is None:
        pytest.skip("openssl not available to make a self-signed cert hermetically")
    return _self_signed(tmp_path)


def test_server_tls_context_requires_client_cert(_cert):
    cert, key = _cert
    ctx = server_tls_context(cert, key, cert)
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.minimum_version == ssl.TLSVersion.TLSv1_2


def test_client_tls_context_verifies_and_skips_hostname(_cert):
    cert, key = _cert
    ctx = client_tls_context(cert, key, cert)
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.minimum_version == ssl.TLSVersion.TLSv1_2
    # Hostname checking is off: peers are dialed by service DNS / bare IP, not a cert hostname.
    assert ctx.check_hostname is False


# --- HMAC proof determinism ----------------------------------------------------------------------


def test_proof_is_deterministic_and_pinned():
    # Same inputs -> same 32-byte HMAC-SHA256 output, every call. The exact bytes are the golden; a
    # rewrite that changes the MAC construction (version tag, ordering, algorithm) would diverge here.
    server_nonce = b"\x01" * 32
    client_nonce = b"\x02" * 32
    got = _proof(SECRET, server_nonce, client_nonce, b"C")
    assert got == _proof(SECRET, server_nonce, client_nonce, b"C")  # determinism
    assert len(got) == 32
    assert got.hex() == "cb9c70427e5d4c21d05494fb4d0461c88f4ecff1379f636b7f22db1dd156d83a"


def test_proof_direction_tag_separates_server_from_client():
    # The direction tag domain-separates the two proofs, so a client proof can never be replayed as a
    # server proof (the core of the mutual check).
    server_nonce = b"\x01" * 32
    client_nonce = b"\x02" * 32
    client_proof = _proof(SECRET, server_nonce, client_nonce, b"C")
    server_proof = _proof(SECRET, server_nonce, client_nonce, b"S")
    assert client_proof != server_proof


def test_version_tag_is_pinned():
    # The version tag domain-separates this MAC from any other use of the same secret; pin its value.
    assert _VERSION == b"nautilus-auth-v1"
