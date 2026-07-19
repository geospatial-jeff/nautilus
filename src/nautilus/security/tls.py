"""Optional mutual TLS for the control and data connections.

The :mod:`.handshake` HMAC already authenticates the peer and, with fresh per-connection nonces, is not
replayable — enough for the documented threat model (a private network where a port may get accidentally
published, so the risk is an *off-path* attacker who merely reaches the port). TLS is the additional layer
for a genuinely *untrusted* network, where an *on-path* attacker can read the cleartext plan and results
or rewrite frames: it adds confidentiality and per-record integrity that the connection-level HMAC does
not.

It is opt-in and cert-based. When ``NAUTILUS_CLUSTER_TLS_CERT`` / ``_KEY`` / ``_CA`` are set, both ends
present a certificate and require the peer's to chain to the shared CA (mutual TLS) — so TLS also
authenticates, redundantly with the HMAC. When they are unset, :func:`tls_from_env` returns ``None`` and
the connections run in cleartext, relying on the HMAC handshake alone.
"""

from __future__ import annotations

import os
import ssl

_CERT = "NAUTILUS_CLUSTER_TLS_CERT"
_KEY = "NAUTILUS_CLUSTER_TLS_KEY"
_CA = "NAUTILUS_CLUSTER_TLS_CA"


def server_tls_context(certfile: str, keyfile: str, cafile: str) -> ssl.SSLContext:
    """A TLS context for the accepting side (daemon, edge listener) that presents ``certfile`` and
    *requires* a client certificate chaining to ``cafile`` — mutual TLS, so an unauthenticated client is
    rejected at the TLS layer before any nautilus byte is read."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(certfile, keyfile)
    ctx.load_verify_locations(cafile)
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx


def client_tls_context(certfile: str, keyfile: str, cafile: str) -> ssl.SSLContext:
    """A TLS context for the dialing side (coordinator, edge connector) that presents ``certfile`` and
    verifies the server's certificate chains to ``cafile``. Hostname checking is off: peers are dialed by
    service DNS or bare IP and identity is established by the shared CA plus the HMAC handshake, not by a
    hostname in the certificate."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.check_hostname = False
    ctx.load_cert_chain(certfile, keyfile)
    ctx.load_verify_locations(cafile)
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx


def tls_from_env() -> tuple[ssl.SSLContext, ssl.SSLContext] | None:
    """Build ``(server_context, client_context)`` from the ``NAUTILUS_CLUSTER_TLS_*`` env vars, or ``None``
    if TLS is not configured. Raises :class:`ValueError` if only some of the three are set (a half-config
    is a mistake that would silently fall back to cleartext)."""
    cert, key, ca = os.environ.get(_CERT), os.environ.get(_KEY), os.environ.get(_CA)
    if not any((cert, key, ca)):
        return None
    if not all((cert, key, ca)):
        raise ValueError(
            f"TLS is half-configured: set all of {_CERT}, {_KEY}, {_CA} (or none). "
            f"got cert={bool(cert)} key={bool(key)} ca={bool(ca)}"
        )
    assert cert and key and ca
    return server_tls_context(cert, key, ca), client_tls_context(cert, key, ca)


__all__ = ["server_tls_context", "client_tls_context", "tls_from_env"]
