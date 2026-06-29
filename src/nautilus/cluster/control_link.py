"""The framed TCP control wire between a coordinator and a worker daemon — the remote replacement for the
``multiprocessing`` queues.

Locally the control messages cross ``mp.Queue``s, which pickle them implicitly. Across machines they cross
one TCP connection per worker, so they need explicit framing. Each message is
``[4-byte magic][4-byte big-endian length][payload]``, mirroring the data-plane handshake
(:mod:`nautilus.transport.handshake`) and frame format (:mod:`nautilus.transport.framing`); the length is
bounded *before* allocating so a garbage or truncated stream is rejected rather than sized into a huge
buffer. The bound is large because a :class:`~nautilus.cluster.protocol.Done` carries the sink-hosting
worker's entire Arrow-IPC result inline.

The payload is **cloudpickle** — the same codec the plan already requires for its lambda operator
factories, and a superset of the stdlib pickle the queues use, so one codec covers ``Launch``/``Abort``
and the reused ``Register``/``Done``/``Failed``. The framer is payload-agnostic: it never imports a
message type (an address book rides as a cloudpickled :class:`~nautilus.cluster.membership.AddressBook`
object that the daemon reconstructs at load time, so this module stays free of any
:mod:`nautilus.runtime`-reaching import). **Cloudpickle over a socket is arbitrary-code-execution on
receipt — the Stage-5 surface**; the length-prefixed framing leaves room to swap the payload codec for a
schema'd one without touching the framer.

Two readers, because the two ends run differently: the coordinator drives a synchronous ``selectors``
loop and feeds bytes to :func:`take_message`; the daemon runs asyncio and awaits :func:`read_message`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import cloudpickle

from nautilus.telemetry import TelemetryConfig

_MAGIC = b"NCTL"  # nautilus control link; a mismatch is a foreign/corrupt stream
_HEADER = 8  # 4 magic + 4 length
_MAX_PAYLOAD = (
    1024 * 1024 * 1024
)  # 1 GiB: sized for a Done carrying the sink's full Arrow-IPC result


class ControlLinkError(RuntimeError):
    """A control-wire frame was malformed: bad magic or an oversized declared length."""


@dataclass(frozen=True)
class Launch:
    """coordinator → daemon: run this slice. Carries what the local path passes as spawn arguments — the
    cloudpickled plan, the placement, the channel capacity, and the telemetry config — keyed by the
    ``worker_id`` the coordinator assigned (the daemon's roster index). The daemon supplies its own bind
    and advertise hosts, so they are *not* here."""

    worker_id: int
    plan_bytes: bytes
    placement: dict[tuple[str, int], int]
    capacity: int
    config: TelemetryConfig


@dataclass(frozen=True)
class Abort:
    """coordinator → daemon: stop the current job. Sent at reap; a control-connection drop means the same
    thing, so this is the explicit, graceful form."""


def encode(message: Any) -> bytes:
    """Frame one control message as ``[magic][length][cloudpickle payload]``."""
    payload: bytes = cloudpickle.dumps(message)
    if len(payload) > _MAX_PAYLOAD:
        raise ControlLinkError(
            f"control message of {len(payload)} bytes exceeds max {_MAX_PAYLOAD}"
        )
    return _MAGIC + len(payload).to_bytes(4, "big") + payload


def take_message(buffer: bytearray) -> Any | None:
    """Pop one complete message off the front of ``buffer`` and return it, or ``None`` if a whole frame
    has not arrived yet. Consumes the framed bytes in place; leaves any trailing partial/next frame. The
    synchronous counterpart to :func:`read_message`, for the coordinator's selectors loop where one
    readable event may carry a partial frame or several frames."""
    if len(buffer) < _HEADER:
        return None
    if bytes(buffer[:4]) != _MAGIC:
        raise ControlLinkError(f"not a control frame (magic {bytes(buffer[:4])!r})")
    length = int.from_bytes(buffer[4:_HEADER], "big")
    if length > _MAX_PAYLOAD:
        raise ControlLinkError(f"control frame length {length} exceeds max {_MAX_PAYLOAD}")
    if len(buffer) < _HEADER + length:
        return None
    payload = bytes(buffer[_HEADER : _HEADER + length])
    del buffer[: _HEADER + length]
    return cloudpickle.loads(payload)


async def read_message(reader: asyncio.StreamReader) -> Any:
    """Read exactly one message off a stream. Raises :class:`asyncio.IncompleteReadError` at EOF (the peer
    closed) and :class:`ControlLinkError` on a bad magic or oversized length."""
    header = await reader.readexactly(_HEADER)
    if header[:4] != _MAGIC:
        raise ControlLinkError(f"not a control frame (magic {header[:4]!r})")
    length = int.from_bytes(header[4:_HEADER], "big")
    if length > _MAX_PAYLOAD:
        raise ControlLinkError(f"control frame length {length} exceeds max {_MAX_PAYLOAD}")
    payload = await reader.readexactly(length)
    return cloudpickle.loads(payload)


__all__ = ["Launch", "Abort", "ControlLinkError", "encode", "take_message", "read_message"]
