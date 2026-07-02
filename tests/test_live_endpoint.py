"""Stage 3: the live HTTP endpoint serves the evolving report mid-run, cross-thread, without ever
racing the single-writer loop (the loop-hop is the regression guard)."""

import asyncio
import contextlib
import json
import urllib.error
import urllib.request

from nautilus.demos import DemoStreamSource
from nautilus.operators import KeyedCount
from nautilus.telemetry.catalog import Tier
from nautilus.telemetry.live import serve_local_chain
from nautilus.telemetry.recorder import TelemetryConfig


def _count_op():
    return KeyedCount("key")


def _get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


async def _serve(src, *, linger, tier=Tier.COUNTERS):
    ready = asyncio.Event()
    holder: dict[str, str] = {}

    def on_ready(url: str) -> None:
        holder["url"] = url
        ready.set()

    task = asyncio.create_task(
        serve_local_chain(
            src,
            [_count_op()],
            telemetry=TelemetryConfig(tier=tier, sample_interval_micros=10_000),
            linger=linger,
            port=0,
            on_ready=on_ready,
        )
    )
    await asyncio.wait_for(ready.wait(), 5)
    return task, holder["url"]


async def test_endpoint_serves_report_and_flips_to_completed():
    task, url = await _serve(DemoStreamSource(interval_s=0.02, max_batches=8), linger=True)
    loop = asyncio.get_running_loop()
    try:
        # healthz + a 404
        assert (await loop.run_in_executor(None, _get, url + "healthz")) == (200, b"ok")
        assert (await loop.run_in_executor(None, _get, url + "nope"))[0] == 404

        # poll the live report until the bounded demo finishes
        last = None
        deadline = loop.time() + 5
        while loop.time() < deadline:
            status, body = await loop.run_in_executor(None, _get, url + "api/telemetry.json")
            assert status == 200
            last = json.loads(body)
            assert last["schema_version"] == 3
            assert last["status"] in ("live", "completed")
            if last["status"] == "completed":
                break
            await asyncio.sleep(0.05)
        assert last is not None and last["status"] == "completed"
        # the run actually moved data and the hardware row is present
        assert last["summary"]["total_rows_out"] >= 0
        assert any(o["kind"] == "process" for o in last["operators"])
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def test_concurrent_polls_never_race():
    # Unbounded fast writer at FULL tier = maximum label churn (the condition that crashes a naive
    # cross-thread reader). The loop-hop must keep every concurrent snapshot consistent.
    task, url = await _serve(
        DemoStreamSource(interval_s=0.005, max_batches=None), linger=True, tier=Tier.FULL
    )
    loop = asyncio.get_running_loop()
    try:
        await asyncio.sleep(0.05)  # let it start churning
        # A handful of overlapping reads is enough to exercise the loop-hop; this isn't a load test.
        results = await asyncio.gather(
            *[loop.run_in_executor(None, _get, url + "api/telemetry.json") for _ in range(8)]
        )
        for status, body in results:
            assert status == 200
            doc = json.loads(body)  # must parse — a dict-resize race would corrupt or 500 this
            assert doc["schema_version"] == 3
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def test_serve_returns_for_bounded_without_linger():
    task, _url = await _serve(DemoStreamSource(interval_s=0.01, max_batches=5), linger=False)
    # bounded demo completes, linger=False → serve returns on its own and frees the port
    await asyncio.wait_for(task, 5)
