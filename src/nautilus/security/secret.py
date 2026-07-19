"""The shared cluster secret, and the fail-closed rule that a non-loopback bind requires one.

The secret is provided out of band by the operator — the same value on every coordinator and daemon of a
deployment — as ``NAUTILUS_CLUSTER_SECRET`` (its value, or ``@/path/to/file`` to read a file so the secret
never sits in the process environment or ``ps`` output). It is never sent on the wire; it only keys the
:mod:`.handshake` HMAC.

The fail-closed rule is the backstop for the container ``0.0.0.0`` bind: a worker in a container must bind
all interfaces (it cannot know its routable address to bind), so the *bind address* cannot be the security
boundary — the secret is. :func:`require_secret_for_bind` therefore refuses to bind anything but loopback
without a secret configured, so the exposed-and-unauthenticated configuration is unreachable by default
rather than merely discouraged by docs.
"""

from __future__ import annotations

import ipaddress
import os
import socket

_ENV = "NAUTILUS_CLUSTER_SECRET"
_MIN_LEN = 16  # a too-short secret is almost certainly a mistake; refuse it loudly


class SecretError(RuntimeError):
    """The cluster secret is missing where it is required, or malformed."""


def cluster_secret() -> bytes | None:
    """The shared secret from ``NAUTILUS_CLUSTER_SECRET`` — its literal value, or the contents of the file
    named by an ``@path`` value — or ``None`` if unset. Raises :class:`SecretError` on an empty or
    implausibly short secret (a silent weak secret is worse than a loud failure)."""
    raw = os.environ.get(_ENV)
    if raw is None:
        return None
    if raw.startswith("@"):
        path = raw[1:]
        try:
            value = open(path, "rb").read().strip()  # noqa: SIM115 — one-shot read at startup
        except OSError as exc:
            raise SecretError(f"{_ENV}=@{path} could not be read: {exc}") from exc
    else:
        value = raw.encode()
    if len(value) < _MIN_LEN:
        raise SecretError(
            f"cluster secret is {len(value)} bytes; use at least {_MIN_LEN} (e.g. `openssl rand -hex 32`)"
        )
    return value


def is_loopback(host: str) -> bool:
    """Whether ``host`` names only the local machine — a literal loopback IP, or a name that resolves to
    loopback addresses only. A wildcard bind (``0.0.0.0`` / ``::`` / ``""``) is emphatically *not*
    loopback: it exposes every interface, so it needs a secret."""
    if host in ("", "0.0.0.0", "::"):
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    return bool(infos) and all(ipaddress.ip_address(info[4][0]).is_loopback for info in infos)


def dashboard_token() -> str | None:
    """The bearer token the live-dashboard HTTP API requires, or ``None`` for no auth. An explicit
    ``NAUTILUS_DASHBOARD_TOKEN``, else the cluster secret (hex) if one is configured — so a secured cluster
    secures its dashboard with the same value, no extra knob to forget."""
    explicit = os.environ.get("NAUTILUS_DASHBOARD_TOKEN")
    if explicit:
        return explicit
    secret = cluster_secret()
    return secret.hex() if secret is not None else None


def require_secret_for_bind(host: str, secret: bytes | None) -> None:
    """Fail-closed: refuse to bind ``host`` on a non-loopback interface without a secret. Loopback (local
    dev) is allowed secretless; anything reachable from another machine is not. Raises
    :class:`SecretError` with a fix hint."""
    if secret is None and not is_loopback(host):
        raise SecretError(
            f"refusing to bind {host!r} (a non-loopback interface) without a cluster secret: set "
            f"{_ENV} (the same value on every node) — e.g. `export {_ENV}=$(openssl rand -hex 32)` — or "
            f"bind 127.0.0.1 for a local-only run"
        )


__all__ = [
    "SecretError",
    "cluster_secret",
    "dashboard_token",
    "is_loopback",
    "require_secret_for_bind",
]
