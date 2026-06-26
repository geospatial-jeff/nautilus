"""Cross-process transport: framed frames over a socket with credit-based flow control.

A :class:`~nautilus.transport.socket_channel.SocketChannel` is a drop-in
:class:`~nautilus.runtime.channel.Channel` whose two ends live in different processes, connected by a
TCP socket (loopback between two local processes, a node-to-node connection in a cluster). Data frames
are credit-limited so a fast producer cannot outpace a slow consumer; control frames (watermark,
end-of-stream) are sent without credit so they are never delayed behind a full data window.
"""

from __future__ import annotations

from nautilus.transport.process import run_two_process
from nautilus.transport.socket_channel import SocketChannel, TransportClosed

__all__ = ["SocketChannel", "TransportClosed", "run_two_process"]
