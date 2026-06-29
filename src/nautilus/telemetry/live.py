"""A live telemetry endpoint: this process serves its own telemetry over HTTP.

A stdlib HTTP server on a daemon thread exposes the live
:class:`~nautilus.telemetry.report.RunReport` as JSON, plus the dashboard HTML.

The HTTP thread must not read a recorder directly — that races the single-writer asyncio loop, which
adds label keys mid-run ("dict changed size during iteration"). Each request is instead scheduled onto
the loop with :func:`asyncio.run_coroutine_threadsafe`, so ``snapshot_all`` runs between actor steps.
Each response is a point-in-time snapshot.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from collections.abc import Callable
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Protocol

from nautilus.compile import compile_graph
from nautilus.core.operator import OneInputOperator, SourceOperator
from nautilus.core.time import SystemClock
from nautilus.driver.meta import make_run_meta
from nautilus.driver.pipeline import graph_from_pipeline
from nautilus.driver.run import plan_to_topology, run_compiled
from nautilus.runtime.channel import DEFAULT_CAPACITY
from nautilus.telemetry import RecorderRegistry, TelemetryConfig
from nautilus.telemetry.report import RunMeta, Topology, build_report

_FALLBACK_HTML = b"""<!doctype html><html><head><meta charset=utf-8><title>nautilus</title></head>
<body><h1>nautilus telemetry</h1><p>Live report at <a href="/api/telemetry.json">/api/telemetry.json</a>.
The full dashboard HTML is not packaged in this build.</p></body></html>"""


def load_dashboard_html() -> bytes:
    path = Path(__file__).parent / "dashboard.html"
    return path.read_bytes() if path.exists() else _FALLBACK_HTML


class Snapshotter(Protocol):
    """Anything the server can ask for a JSON report body."""

    def render_json(self, *, timeout: float = ...) -> str: ...


class StaticSnapshotSource:
    """Serves a fixed report (for ``nautilus serve --report``): no loop hop, no history."""

    def __init__(self, json_text: str) -> None:
        self._text = json_text

    def render_json(self, *, timeout: float = 2.0) -> str:
        return self._text


class SnapshotSource:
    """Builds the live report JSON on the asyncio loop thread (via a thread-safe hop), so reading the
    recorders never races the writer."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        registry: RecorderRegistry,
        topology: Topology,
        meta_fn: Callable[[], RunMeta],
        status_fn: Callable[[], str],
    ) -> None:
        self._loop = loop
        self._registry = registry
        self._topology = topology
        self._meta_fn = meta_fn
        self._status_fn = status_fn

    async def _snapshot(self) -> str:
        # Runs on the loop thread, between actor steps → single-writer recorders are read safely.
        report = build_report(
            self._registry.snapshot_all(), meta=self._meta_fn(), topology=self._topology
        )
        doc = report.to_dict()
        doc["status"] = self._status_fn()
        doc["sampled_at_micros"] = SystemClock().now_micros()
        return json.dumps(doc, sort_keys=True)

    def render_json(self, *, timeout: float = 2.0) -> str:
        future = asyncio.run_coroutine_threadsafe(self._snapshot(), self._loop)
        return future.result(timeout=timeout)


class _Server(ThreadingHTTPServer):
    allow_reuse_address = True  # so relaunch on the same port works after an abrupt exit
    daemon_threads = True
    request_queue_size = (
        128  # absorb a burst of concurrent pollers (many tabs) instead of RST-ing them
    )


def _handler_class(source: Snapshotter, html: bytes) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass  # quiet

        def _send(self, code: int, ctype: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (stdlib handler API)
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", html)
            elif path == "/api/telemetry.json":
                try:
                    self._send(200, "application/json", source.render_json().encode())
                except TimeoutError:
                    self._send(503, "application/json", b'{"error":"snapshot timed out"}')
                except Exception as e:  # never crash the server thread
                    self._send(500, "application/json", json.dumps({"error": str(e)}).encode())
            elif path == "/healthz":
                self._send(200, "text/plain", b"ok")
            else:
                self._send(404, "text/plain", b"not found")

    return Handler


class LiveServer:
    """A daemon-thread HTTP server. Pure reader — it only ever calls ``source.render_json()``."""

    def __init__(
        self, source: Snapshotter, html: bytes, *, host: str = "127.0.0.1", port: int = 8787
    ) -> None:
        self._httpd = _Server((host, port), _handler_class(source, html))
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._stopped = False
        self.host = host
        self.port = int(self._httpd.server_address[1])

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2.0)


async def serve_local_chain(
    source: SourceOperator,
    transforms: list[OneInputOperator],
    *,
    capacity: int = DEFAULT_CAPACITY,
    telemetry: TelemetryConfig | None = None,
    host: str = "127.0.0.1",
    port: int = 8787,
    linger: bool = True,
    max_seconds: float | None = None,
    open_browser: bool = False,
    on_ready: Callable[[str], None] | None = None,
) -> None:
    """Serve a live dashboard while running a pipeline. Bounded pipelines complete then (with
    ``linger``) keep serving the frozen final snapshot until cancelled; ``max_seconds`` caps unbounded
    ones. Cancellation (Ctrl-C from the CLI) unwinds cleanly and frees the port."""
    clk = SystemClock()
    config = telemetry or TelemetryConfig(clock=clk)
    if config.run_id is None:
        config = replace(config, run_id=f"run-{clk.now_micros()}")
    run_id = config.run_id
    assert run_id is not None  # set just above; bind to a non-optional local for make_run_meta
    registry = RecorderRegistry()
    # Compile the chain to a plan and serve/run that — the same engine a non-live run uses — so the live
    # topology is exactly what executes (one topology builder, plan_to_topology).
    plan = compile_graph(graph_from_pipeline(source, transforms, 1))
    topology = plan_to_topology(plan, capacity)
    loop = asyncio.get_running_loop()
    started_at = clk.now_micros()
    started_ns = perf_counter_ns()
    # frozen_meta is captured once the run ends, so wall_micros/ended_at (and every wall-derived ratio)
    # stop advancing while the dashboard lingers on the final snapshot — "frozen" as the docstring says.
    state: dict[str, Any] = {"status": "live", "frozen_meta": None}

    def meta_now() -> RunMeta:
        return make_run_meta(
            run_id=run_id,
            started_at=started_at,
            ended_at=clk.now_micros(),
            wall_micros=(perf_counter_ns() - started_ns) // 1000,
            clk=clk,
            topology=topology,
            config=config,
            capacity=capacity,
        )

    def meta_fn() -> RunMeta:
        frozen = state["frozen_meta"]
        return frozen if frozen is not None else meta_now()

    snap = SnapshotSource(loop, registry, topology, meta_fn, lambda: state["status"])
    server = LiveServer(snap, load_dashboard_html(), host=host, port=port)
    server.start()
    if on_ready is not None:
        on_ready(server.url)
    if open_browser:
        import webbrowser

        webbrowser.open(server.url)

    run_task = asyncio.create_task(
        run_compiled(plan, capacity=capacity, clock=clk, telemetry=config, registry=registry)
    )

    async def drive() -> None:
        # A bounded run completing flips to "completed" and — if lingering — holds the frozen snapshot
        # until cancelled. A run *error* re-raises out of drive() (the finally still freezes/flips) and
        # propagates to serve_local_chain's shutdown, so an errored run does NOT linger. An external
        # cancel (Ctrl-C) likewise propagates rather than being swallowed into the linger wait, so the
        # shutdown gather never hangs.
        try:
            await run_task  # bounded completes; unbounded blocks until cancelled
        finally:
            state["frozen_meta"] = meta_now()  # freeze wall/ended_at at the instant the run ended
            state["status"] = "completed"
        if linger:
            await asyncio.Event().wait()  # serve the frozen final snapshot until cancelled

    drive_task = asyncio.create_task(drive())
    try:
        if max_seconds is not None:
            # max_seconds bounds the WHOLE session (run + linger) so unbounded demos / CI exit.
            with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(drive_task, timeout=max_seconds)
        else:
            await drive_task
    finally:
        state["status"] = "completed"
        for task in (drive_task, run_task):
            if not task.done():
                task.cancel()
        with contextlib.suppress(BaseException):
            await asyncio.gather(drive_task, run_task, return_exceptions=True)
        server.stop()


async def serve_report(
    report_json: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    on_ready: Callable[[str], None] | None = None,
) -> None:
    """Serve a saved report JSON statically in the dashboard until cancelled (``nautilus serve``)."""
    doc = json.loads(report_json)
    doc["status"] = "completed"
    server = LiveServer(
        StaticSnapshotSource(json.dumps(doc, sort_keys=True)),
        load_dashboard_html(),
        host=host,
        port=port,
    )
    server.start()
    if on_ready is not None:
        on_ready(server.url)
    try:
        await asyncio.Event().wait()
    finally:
        server.stop()
