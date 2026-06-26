"""Stage 4: the packaged dashboard HTML is served and stays facts-only; the static report viewer and
the `dashboard` CLI command both work."""

import asyncio
import contextlib
import json
import re
import urllib.request

from typer.testing import CliRunner

from nautilus.cli import app
from nautilus.telemetry.catalog import BANNED_ANALYSIS_WORDS
from nautilus.telemetry.live import load_dashboard_html, serve_report

runner = CliRunner()
_WORD = re.compile(r"[a-z_]+")


def test_dashboard_html_is_packaged_and_facts_only():
    html = load_dashboard_html().decode()
    assert "<title>nautilus telemetry</title>" in html
    assert "/api/telemetry.json" in html
    tokens = set(_WORD.findall(html.lower()))
    leaked = tokens & BANNED_ANALYSIS_WORDS
    assert not leaked, f"the dashboard shows verdicts, not just facts: {sorted(leaked)}"


async def test_static_report_viewer_serves_saved_report():
    saved = json.dumps(
        {
            "schema_version": 1,
            "run_id": "run-x",
            "meta": {"wall_micros": 1, "clock_kind": "TestClock"},
            "summary": {"total_rows_in": 3, "total_rows_out": 3, "total_errors": 0},
            "operators": [],
            "edges": [],
            "events": [],
            "errors": [],
            "topology": {"nodes": [], "edges": []},
        }
    )
    ready = asyncio.Event()
    holder: dict[str, str] = {}

    def on_ready(url: str) -> None:
        holder["url"] = url
        ready.set()

    task = asyncio.create_task(serve_report(saved, port=0, on_ready=on_ready))
    try:
        await asyncio.wait_for(ready.wait(), 5)
        loop = asyncio.get_running_loop()
        body = await loop.run_in_executor(
            None,
            lambda: urllib.request.urlopen(holder["url"] + "api/telemetry.json", timeout=5).read(),
        )
        doc = json.loads(body)
        assert doc["status"] == "completed"
        assert doc["run_id"] == "run-x"
        # GET / serves the dashboard HTML
        index = await loop.run_in_executor(
            None, lambda: urllib.request.urlopen(holder["url"], timeout=5).read()
        )
        assert b"nautilus telemetry" in index
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def test_cli_dashboard_runs_bounded_and_exits():
    result = runner.invoke(
        app,
        ["dashboard", "demo-stream", "--max-seconds", "0.4", "--port", "0", "--no-linger"],
    )
    assert result.exit_code == 0, result.output
    assert "live dashboard at" in result.output
