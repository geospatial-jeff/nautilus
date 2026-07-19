"""The cluster security primitives: the shared-secret handshake, the fail-closed bind rule, secret
loading, and the TLS context builder. These pin the boundary that makes the cloudpickle control wire safe
— a peer without the secret must never complete a connection — so each failure mode is asserted, not just
the happy path."""

from __future__ import annotations

import asyncio
import socket
import ssl

import pytest

from nautilus.security import (
    AuthError,
    authenticate_client,
    authenticate_client_sync,
    authenticate_server,
    cluster_secret,
    is_loopback,
    require_secret_for_bind,
)
from nautilus.security.secret import SecretError
from nautilus.security.tls import tls_from_env

SECRET = b"a-sufficiently-long-test-secret-0123456789"


async def _serve_once(secret, ready):
    """Accept one connection, run the server handshake, report success/failure through ``ready``."""
    result: dict = {}

    async def handle(reader, writer):
        try:
            await authenticate_server(reader, writer, secret)
            result["ok"] = True
        except AuthError as e:
            result["err"] = str(e)
        finally:
            writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    ready.set_result((server.sockets[0].getsockname(), result, server))
    async with server:
        await asyncio.sleep(0.5)


async def _run_pair(server_secret, client_secret, *, sync_client=False):
    """Bring up a one-shot server and drive a client at it; return (server_result, client_error|None)."""
    ready: asyncio.Future = asyncio.get_running_loop().create_future()
    server_task = asyncio.create_task(_serve_once(server_secret, ready))
    (host, port), result, server = await ready
    client_err = None
    try:
        if sync_client:
            sock = socket.create_connection((host, port))
            try:
                await asyncio.get_running_loop().run_in_executor(
                    None, authenticate_client_sync, sock, client_secret
                )
            finally:
                sock.close()
        else:
            reader, writer = await asyncio.open_connection(host, port)
            try:
                await authenticate_client(reader, writer, client_secret)
            finally:
                writer.close()
    except AuthError as e:
        client_err = str(e)
    await asyncio.sleep(0.05)
    server.close()
    server_task.cancel()
    return result, client_err


async def test_matching_secret_authenticates_both_ways():
    result, client_err = await _run_pair(SECRET, SECRET)
    assert result.get("ok") is True and client_err is None


async def test_wrong_client_secret_is_rejected_by_server():
    result, client_err = await _run_pair(SECRET, b"the-wrong-secret-but-long-enough-xx")
    assert "err" in result  # server refused
    assert client_err is not None  # client also sees the failure (no server proof)


async def test_mismatched_secret_client_also_refuses():
    # Server holds a different secret: it rejects the client's proof and closes before proving itself, so
    # the client sees the failure too (an unauthenticated server can never satisfy it).
    _result, client_err = await _run_pair(b"impostor-secret-also-long-enough-xx", SECRET)
    assert client_err is not None


async def test_client_catches_bad_server_proof():
    # A fake server that answers with a garbage server-proof (an impostor that doesn't hold the secret):
    # the client must verify the server's proof and refuse, not trust a completed handshake.
    async def bad_server(reader, writer):
        writer.write(b"NAUS" + b"\x00" * 32)  # magic + a server nonce
        await writer.drain()
        await reader.readexactly(64)  # consume the client's reply
        writer.write(b"\xff" * 32)  # a wrong server proof
        await writer.drain()
        await asyncio.sleep(0.1)
        writer.close()

    server = await asyncio.start_server(bad_server, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()
    async with server:
        reader, writer = await asyncio.open_connection(host, port)
        try:
            with pytest.raises(AuthError, match="impostor"):
                await authenticate_client(reader, writer, SECRET)
        finally:
            writer.close()


async def test_no_secret_skips_handshake():
    # Both sides secretless (the loopback default): the connection proceeds with no handshake at all.
    result, client_err = await _run_pair(None, None)
    assert result.get("ok") is True and client_err is None


async def test_sync_client_matches_async_server():
    result, client_err = await _run_pair(SECRET, SECRET, sync_client=True)
    assert result.get("ok") is True and client_err is None


async def test_sync_client_wrong_secret_rejected():
    result, client_err = await _run_pair(
        SECRET, b"wrong-secret-long-enough-000000000", sync_client=True
    )
    assert client_err is not None


# --- fail-closed bind rule ----------------------------------------------------------------------


def test_loopback_recognized():
    assert is_loopback("127.0.0.1") and is_loopback("localhost")
    assert not is_loopback("0.0.0.0") and not is_loopback("") and not is_loopback("::")


def test_require_secret_refuses_wildcard_bind_without_secret():
    with pytest.raises(SecretError, match="without a cluster secret"):
        require_secret_for_bind("0.0.0.0", None)


def test_require_secret_allows_loopback_without_secret():
    require_secret_for_bind("127.0.0.1", None)  # local dev: no secret needed, no raise


def test_require_secret_allows_wildcard_with_secret():
    require_secret_for_bind("0.0.0.0", SECRET)  # secret present: the bind is allowed


# --- secret loading -----------------------------------------------------------------------------


def test_cluster_secret_from_env(monkeypatch):
    monkeypatch.setenv("NAUTILUS_CLUSTER_SECRET", SECRET.decode())
    assert cluster_secret() == SECRET


def test_cluster_secret_unset_is_none(monkeypatch):
    monkeypatch.delenv("NAUTILUS_CLUSTER_SECRET", raising=False)
    assert cluster_secret() is None


def test_cluster_secret_too_short_rejected(monkeypatch):
    monkeypatch.setenv("NAUTILUS_CLUSTER_SECRET", "short")
    with pytest.raises(SecretError, match="at least"):
        cluster_secret()


def test_cluster_secret_from_file(monkeypatch, tmp_path):
    f = tmp_path / "secret"
    f.write_bytes(SECRET + b"\n")  # trailing newline is stripped
    monkeypatch.setenv("NAUTILUS_CLUSTER_SECRET", f"@{f}")
    assert cluster_secret() == SECRET


def test_tls_disabled_when_env_unset(monkeypatch):
    for k in ("NAUTILUS_CLUSTER_TLS_CERT", "NAUTILUS_CLUSTER_TLS_KEY", "NAUTILUS_CLUSTER_TLS_CA"):
        monkeypatch.delenv(k, raising=False)
    assert tls_from_env() is None


def _self_signed(tmp_path):
    """A self-signed cert usable as both the presented cert and the verifying CA (so one cert drives mTLS
    in a hermetic test). Returns (certfile, keyfile)."""
    import subprocess

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


async def test_tls_wraps_the_authenticated_connection(monkeypatch, tmp_path):
    # With TLS configured, the connection is encrypted AND mutually cert-verified, and the shared-secret
    # handshake then runs over it. A plaintext or wrong-CA peer never reaches the handshake.
    cert, key = _self_signed(tmp_path)
    monkeypatch.setenv("NAUTILUS_CLUSTER_TLS_CERT", cert)
    monkeypatch.setenv("NAUTILUS_CLUSTER_TLS_KEY", key)
    monkeypatch.setenv("NAUTILUS_CLUSTER_TLS_CA", cert)
    ctxs = tls_from_env()
    assert ctxs is not None
    server_ctx, client_ctx = ctxs

    ok: dict = {}

    async def handle(reader, writer):
        try:
            await authenticate_server(reader, writer, SECRET)
            ok["server"] = True
        except AuthError:
            ok["server"] = False
        finally:
            writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0, ssl=server_ctx)
    host, port = server.sockets[0].getsockname()
    async with server:
        reader, writer = await asyncio.open_connection(
            host, port, ssl=client_ctx, server_hostname=host
        )
        try:
            assert writer.get_extra_info("ssl_object") is not None  # the socket is really TLS
            await authenticate_client(reader, writer, SECRET)
            ok["client"] = True
        finally:
            writer.close()
        await asyncio.sleep(0.05)
    assert ok.get("server") is True and ok.get("client") is True

    # A plaintext client cannot even open the connection to the TLS-required server.
    server2 = await asyncio.start_server(
        lambda r, w: w.close(), "127.0.0.1", 0, ssl=server_ctx, ssl_handshake_timeout=2
    )
    host2, port2 = server2.sockets[0].getsockname()
    async with server2:
        with pytest.raises((ssl.SSLError, OSError, asyncio.IncompleteReadError)):
            r, w = await asyncio.open_connection(host2, port2)  # no ssl → rejected at the TLS layer
            try:
                await r.readexactly(1)
            finally:
                w.close()


def test_tls_half_config_rejected(monkeypatch):
    monkeypatch.setenv("NAUTILUS_CLUSTER_TLS_CERT", "/tmp/cert.pem")
    monkeypatch.delenv("NAUTILUS_CLUSTER_TLS_KEY", raising=False)
    monkeypatch.delenv("NAUTILUS_CLUSTER_TLS_CA", raising=False)
    with pytest.raises(ValueError, match="half-configured"):
        tls_from_env()


# --- dashboard auth (5d) ------------------------------------------------------------------------


def test_dashboard_requires_token_when_configured(monkeypatch):
    import urllib.error
    import urllib.request

    from nautilus.telemetry.live import LiveServer, StaticSnapshotSource

    monkeypatch.setenv("NAUTILUS_DASHBOARD_TOKEN", "sekrit-token")
    server = LiveServer(
        StaticSnapshotSource('{"ok":1}'), b"<html></html>", host="127.0.0.1", port=0
    )
    server.start()
    try:
        base = f"http://127.0.0.1:{server.port}"
        assert urllib.request.urlopen(base + "/healthz").status == 200  # health probe stays open
        with pytest.raises(urllib.error.HTTPError) as ei:  # the report needs the token
            urllib.request.urlopen(base + "/api/telemetry.json")
        assert ei.value.code == 401
        req = urllib.request.Request(
            base + "/api/telemetry.json", headers={"Authorization": "Bearer sekrit-token"}
        )
        assert urllib.request.urlopen(req).status == 200
    finally:
        server.stop()


def test_dashboard_refuses_nonloopback_bind_without_token(monkeypatch):
    from nautilus.telemetry.live import LiveServer, StaticSnapshotSource

    monkeypatch.delenv("NAUTILUS_DASHBOARD_TOKEN", raising=False)
    monkeypatch.delenv("NAUTILUS_CLUSTER_SECRET", raising=False)
    with pytest.raises(SecretError, match="without a token"):
        LiveServer(StaticSnapshotSource("{}"), b"", host="0.0.0.0", port=0)
