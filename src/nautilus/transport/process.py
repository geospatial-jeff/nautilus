"""Run a pipeline across two processes connected by a TCP socket on the loopback interface.

Stage 1 minimal split: the source runs in the parent process; the transforms and sink run in a
spawned child. The single edge between them is a :class:`SocketChannel` over TCP, so credit-based
flow control is exercised across a real process boundary using the same transport that connects nodes
in a cluster. Placement and a general launcher are Stage 2 (``nautilus.cluster``); this function is
the local two-process harness, not the cluster control plane.

Failure handling is explicit: the child always reports an outcome (``ok`` with batches+telemetry, or
``error`` with a traceback) back to the parent, so a failing operator raises in the caller instead of
hanging; and the parent always reaps the child.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import queue
import time
import traceback
from collections.abc import AsyncIterator
from typing import Any

from nautilus.core.operator import OneInputOperator, OperatorContext, SourceOperator
from nautilus.core.records import EOS, Frame
from nautilus.runtime.actor import Output, run_source
from nautilus.runtime.channel import DEFAULT_CAPACITY
from nautilus.runtime.local import run_local_chain
from nautilus.runtime.partition import Forward
from nautilus.runtime.result import RunResult
from nautilus.telemetry import Tier
from nautilus.transport.socket_channel import SocketChannel

_LOOPBACK = "127.0.0.1"


class _SocketSource(SourceOperator):
    """Relays the frames arriving on a :class:`SocketChannel` — the child's input edge as a source."""

    def __init__(self, channel: SocketChannel) -> None:
        self._channel = channel

    async def frames(self) -> AsyncIterator[Frame]:
        while True:
            frame = await self._channel.recv()
            yield frame
            if isinstance(frame, EOS):
                return


def run_two_process(
    source: SourceOperator,
    transforms: list[OneInputOperator],
    *,
    capacity: int = DEFAULT_CAPACITY,
    tier: Tier = Tier.COUNTERS,
) -> RunResult:
    """Run ``source`` in this process and ``transforms`` + sink in a spawned child, joined by one TCP
    edge on the loopback interface. Returns the child's :class:`RunResult` (its output batches and
    telemetry). Raises if the child's pipeline fails, surfacing the child's traceback."""
    ctx = mp.get_context("spawn")
    result_q: Any = ctx.Queue()
    address_q: Any = ctx.Queue()  # child reports the (host, port) it bound
    proc = ctx.Process(
        target=_worker_main, args=(transforms, int(tier), capacity, result_q, address_q)
    )
    proc.start()
    try:
        host, port = _await_address(proc, address_q)
        asyncio.run(_drive_source(source, host, port, capacity))
        tag, payload = _get_result(proc, result_q)
        if tag == "error":
            raise RuntimeError(f"transport worker failed:\n{payload}")
        batches, report = payload
        return RunResult(batches, report)
    finally:
        _reap(proc)


def _await_address(proc: Any, address_q: Any, timeout: float = 30.0) -> tuple[str, int]:
    start = time.monotonic()
    while True:
        try:
            return address_q.get(timeout=0.2)  # type: ignore[no-any-return]
        except queue.Empty:
            if not proc.is_alive():
                raise RuntimeError(
                    f"transport worker exited during startup (code {proc.exitcode})"
                ) from None
            if time.monotonic() - start > timeout:
                raise RuntimeError(
                    f"transport worker did not start within {timeout:.0f}s"
                ) from None


def _get_result(proc: Any, result_q: Any, timeout: float = 60.0) -> tuple[str, Any]:
    start = time.monotonic()
    while True:
        try:
            return result_q.get(timeout=0.2)  # type: ignore[no-any-return]
        except queue.Empty:
            if not proc.is_alive():
                raise RuntimeError(
                    f"transport worker exited without a result (code {proc.exitcode})"
                ) from None
            if time.monotonic() - start > timeout:
                raise RuntimeError(
                    f"transport worker did not return within {timeout:.0f}s"
                ) from None


def _reap(proc: Any) -> None:
    if proc.is_alive():
        proc.terminate()
    proc.join(timeout=10)
    if proc.is_alive():  # last resort, then reap so it never becomes a zombie
        proc.kill()
        proc.join(timeout=5)


async def _drive_source(source: SourceOperator, host: str, port: int, capacity: int) -> None:
    reader, writer = await asyncio.open_connection(host, port)
    channel = SocketChannel(reader, writer, capacity=capacity)
    try:
        await run_source(source, OperatorContext("source"), [Output([channel], Forward())])
        await channel.finish()  # drain returning credits to the consumer's end before tearing down
    finally:
        await channel.close()  # closes even if the source raised, so the child sees the disconnect


def _worker_main(
    transforms: list[OneInputOperator],
    tier_value: int,
    capacity: int,
    result_q: Any,
    address_q: Any,
) -> None:
    asyncio.run(_worker(transforms, tier_value, capacity, result_q, address_q))


async def _worker(
    transforms: list[OneInputOperator],
    tier_value: int,
    capacity: int,
    result_q: Any,
    address_q: Any,
) -> None:
    from nautilus.telemetry import TelemetryConfig

    outcome: dict[str, tuple[str, Any]] = {}
    done = asyncio.Event()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            channel = SocketChannel(reader, writer, capacity=capacity)
            try:
                result = await run_local_chain(
                    _SocketSource(channel),
                    transforms,
                    capacity=capacity,
                    telemetry=TelemetryConfig(tier=Tier(tier_value)),
                )
                outcome["result"] = ("ok", (list(result.batches), result.telemetry))
            finally:
                await channel.close()  # consumer FIN: releases the producer's finish() drain
        except Exception:
            outcome["result"] = ("error", traceback.format_exc())
        finally:
            done.set()

    server = await asyncio.start_server(handle, _LOOPBACK, 0)
    port = server.sockets[0].getsockname()[1]
    address_q.put((_LOOPBACK, port))
    try:
        async with server:
            await done.wait()
    finally:
        result_q.put(outcome.get("result", ("error", "transport worker produced no result")))
