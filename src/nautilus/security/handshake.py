"""A mutual HMAC challenge-response run once per connection, before any payload crosses.

The threat it closes: both the control wire (cloudpickle) and the data edges deserialize peer bytes, so a
peer that reaches the port and is *not* the holder of the shared secret must never get that far. The
server speaks first with a random nonce; each side proves it holds the secret by returning an HMAC over
both nonces and a fixed direction tag, and verifies the other's with a constant-time compare. Neither the
secret nor a reusable token ever crosses the wire, and the fresh per-connection nonces make a captured
transcript worthless to replay.

The messages are fixed-size, so there is nothing to length-parse and nothing to allocate before the peer
is authenticated — the flood a length-prefixed frame invites cannot happen here. Three entry points share
one core because the two ends run differently: the daemon and the edge listener are asyncio servers
(:func:`authenticate_server`), the edge connector is an asyncio client (:func:`authenticate_client`), and
the coordinator dials daemons on a blocking socket (:func:`authenticate_client_sync`).

When ``secret`` is ``None`` the handshake is skipped — the local/loopback default, where the
process-isolation boundary (or nothing crossing a network at all) already stands in for it. A non-loopback
bind may not run without a secret; :func:`nautilus.security.secret.require_secret_for_bind` enforces that,
so a skipped handshake only ever means loopback.
"""

from __future__ import annotations

import asyncio
import hmac
import os
import socket

_VERSION = b"nautilus-auth-v1"  # domain-separates this MAC from any other use of the same secret
_NONCE = 32  # random challenge per side; 256 bits, fresh per connection (anti-replay)
_MAC = 32  # HMAC-SHA256 output width
_SERVER_HELLO = 4 + _NONCE  # magic + server nonce
_CLIENT_REPLY = _NONCE + _MAC  # client nonce + client proof
_MAGIC = b"NAUS"  # nautilus auth server-hello; a mismatch is a foreign/garbage connection
_TIMEOUT = 10.0  # a peer that connects then stalls mid-handshake is dropped, not held open


class AuthError(RuntimeError):
    """The peer failed the shared-secret handshake — wrong secret, a stalled/short exchange, or (on the
    server hello) not a nautilus connection at all."""


def _proof(secret: bytes, server_nonce: bytes, client_nonce: bytes, tag: bytes) -> bytes:
    return hmac.new(secret, _VERSION + server_nonce + client_nonce + tag, "sha256").digest()


async def authenticate_server(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, secret: bytes | None
) -> None:
    """Server side (daemon, edge listener): challenge the peer and verify its proof before any payload is
    read. No-op when ``secret`` is ``None``. Raises :class:`AuthError` if the peer cannot prove the secret;
    the caller closes the connection."""
    if secret is None:
        return
    server_nonce = os.urandom(_NONCE)
    try:
        writer.write(_MAGIC + server_nonce)
        await writer.drain()
        reply = await asyncio.wait_for(reader.readexactly(_CLIENT_REPLY), _TIMEOUT)
    except (TimeoutError, asyncio.IncompleteReadError, OSError) as exc:
        raise AuthError(f"peer did not complete the handshake: {exc}") from exc
    client_nonce, client_proof = reply[:_NONCE], reply[_NONCE:]
    if not hmac.compare_digest(client_proof, _proof(secret, server_nonce, client_nonce, b"C")):
        raise AuthError("peer failed authentication (wrong cluster secret)")
    writer.write(_proof(secret, server_nonce, client_nonce, b"S"))
    await writer.drain()


async def authenticate_client(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, secret: bytes | None
) -> None:
    """Client side (edge connector): answer the server's challenge and verify the server's proof in return,
    so a hijacker that cannot prove the secret is rejected *and* an impostor server is caught. No-op when
    ``secret`` is ``None``. Raises :class:`AuthError` on failure."""
    if secret is None:
        return
    try:
        hello = await asyncio.wait_for(reader.readexactly(_SERVER_HELLO), _TIMEOUT)
        if hello[:4] != _MAGIC:
            raise AuthError(f"not a nautilus authenticated endpoint (magic {hello[:4]!r})")
        server_nonce = hello[4:]
        client_nonce = os.urandom(_NONCE)
        writer.write(client_nonce + _proof(secret, server_nonce, client_nonce, b"C"))
        await writer.drain()
        server_proof = await asyncio.wait_for(reader.readexactly(_MAC), _TIMEOUT)
    except (TimeoutError, asyncio.IncompleteReadError, OSError) as exc:
        raise AuthError(f"handshake did not complete: {exc}") from exc
    if not hmac.compare_digest(server_proof, _proof(secret, server_nonce, client_nonce, b"S")):
        raise AuthError("server failed authentication (wrong cluster secret or impostor)")


def authenticate_client_sync(sock: socket.socket, secret: bytes | None) -> None:
    """Client side on a blocking socket (the coordinator dialing a daemon's control port). The synchronous
    twin of :func:`authenticate_client`, same protocol and same mutual check. No-op when ``secret`` is
    ``None``. Raises :class:`AuthError` on failure; the caller closes the socket."""
    if secret is None:
        return
    prev = sock.gettimeout()
    sock.settimeout(_TIMEOUT)
    try:
        hello = _recv_exact(sock, _SERVER_HELLO)
        if hello[:4] != _MAGIC:
            raise AuthError(f"not a nautilus authenticated endpoint (magic {hello[:4]!r})")
        server_nonce = hello[4:]
        client_nonce = os.urandom(_NONCE)
        sock.sendall(client_nonce + _proof(secret, server_nonce, client_nonce, b"C"))
        server_proof = _recv_exact(sock, _MAC)
    except (TimeoutError, OSError) as exc:
        raise AuthError(f"handshake did not complete: {exc}") from exc
    finally:
        sock.settimeout(prev)
    if not hmac.compare_digest(server_proof, _proof(secret, server_nonce, client_nonce, b"S")):
        raise AuthError("server failed authentication (wrong cluster secret or impostor)")


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks: list[bytes] = []
    remaining = n
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise AuthError("peer closed during the handshake")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


__all__ = [
    "AuthError",
    "authenticate_server",
    "authenticate_client",
    "authenticate_client_sync",
]
