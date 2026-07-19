"""Cluster security primitives — shared by the control plane (:mod:`nautilus.cluster`) and the data plane
(:mod:`nautilus.transport`), so it depends on neither and imports only the standard library. This keeps
the import-linter layer boundaries intact (the data path must not import the control plane): a secret or
TLS context is *injected* into transport by the code that runs a worker, never imported across the layer.

Two mechanisms, both keyed by one operator-provided shared secret:

- **Authentication** (:mod:`.handshake`) — a mutual HMAC challenge-response run once per connection, on
  both planes, *before* any payload is read. It is what makes the cloudpickle control wire safe: an
  unauthenticated peer never gets its bytes deserialized, so the arbitrary-code-execution-on-receipt
  surface is closed to anyone without the secret.
- **Encryption** (:mod:`.tls`) — an optional TLS layer for confidentiality and integrity on a genuinely
  untrusted network (the authentication handshake alone covers an off-path attacker who merely reaches a
  published port; TLS additionally defeats an on-path attacker who can read or rewrite the cleartext).

:mod:`.secret` loads the shared secret and enforces the fail-closed rule that binding a non-loopback
interface without one is refused — so the insecure configuration cannot be reached by default.
"""

from __future__ import annotations

from nautilus.security.handshake import (
    AuthError,
    authenticate_client,
    authenticate_client_sync,
    authenticate_server,
)
from nautilus.security.secret import (
    cluster_secret,
    dashboard_token,
    is_loopback,
    require_secret_for_bind,
)
from nautilus.security.tls import client_tls_context, server_tls_context, tls_from_env

__all__ = [
    "AuthError",
    "authenticate_client",
    "authenticate_client_sync",
    "authenticate_server",
    "cluster_secret",
    "dashboard_token",
    "is_loopback",
    "require_secret_for_bind",
    "client_tls_context",
    "server_tls_context",
    "tls_from_env",
]
