"""Cross-process transport: framed frames over a socket with credit-based flow control.

A :class:`~nautilus.transport.socket_channel.SocketChannel` is a drop-in
:class:`~nautilus.runtime.channel.Channel` whose two ends live in different processes, connected by a
TCP socket (loopback between two local processes, a node-to-node connection in a cluster). Data frames
are credit-limited so a fast producer cannot outpace a slow consumer; control frames (end-of-stream)
are sent without credit so they are never delayed behind a full data window.

Stage 2c added the *addressed* edge: a producer dials a worker's :class:`EdgeListener` and announces the
edge with a one-shot handshake; the listener routes the socket to the consumer awaiting that edge. The
:class:`SocketConnector` implements the runtime :class:`~nautilus.runtime.connector.Connector` over this
seam, so the executor wires a cross-worker edge with the same code as an in-process one. The Stage 2
coordinator (:func:`nautilus.cluster.deploy`) drives workers over it.
"""

from __future__ import annotations

from nautilus.transport.connector import SocketConnector
from nautilus.transport.handshake import HandshakeError
from nautilus.transport.listener import EdgeListener
from nautilus.transport.socket_channel import SocketChannel, TransportClosed

__all__ = [
    "SocketChannel",
    "TransportClosed",
    "EdgeListener",
    "SocketConnector",
    "HandshakeError",
]
