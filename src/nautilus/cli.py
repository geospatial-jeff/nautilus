"""The ``nautilus`` command-line interface.

Run pipelines and read their telemetry. ``nautilus task`` prints a prompt for an AI coding agent: the
task, a run's telemetry, what each metric means, and which files to read.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

import nautilus
from nautilus.bench import (
    DEFAULT_BASELINE,
    DEFAULT_THRESHOLD,
    DEFAULT_TRIALS,
    DEFAULT_WARMUP,
    BenchResult,
    Comparison,
    compare,
    is_failure,
    load_baseline,
    measure,
    measure_like,
    save_baseline,
)
from nautilus.benchmarks import DEFAULT_BATCH, DEFAULT_KEYS, DEFAULT_ROWS, DEFAULT_WM_EVERY
from nautilus.cluster import deploy
from nautilus.core.time import SystemClock
from nautilus.pipelines import EXAMPLES, load_pipeline
from nautilus.runtime.local import run_local_chain
from nautilus.runtime.parallel import graph_from_pipeline
from nautilus.runtime.result import RunResult
from nautilus.runtime.run import run_plan
from nautilus.telemetry import METRIC_SPECS, TelemetryConfig, Tier
from nautilus.telemetry.report.reference import render_reference, write_reference
from nautilus.telemetry.report.report import RunReport

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Nautilus — a decentralized, streaming compute framework with built-in telemetry.",
)
console = Console()

_TIERS = {
    "off": Tier.OFF,
    "counters": Tier.COUNTERS,
    "events": Tier.COUNTERS_PLUS_EVENTS,
    "full": Tier.FULL,
}


def _tier(name: str) -> Tier:
    try:
        return _TIERS[name.lower()]
    except KeyError:
        raise typer.BadParameter(f"telemetry must be one of: {', '.join(_TIERS)}") from None


def _run(
    pipeline: str, tier: Tier, capacity: int, workers: int = 1, parallelism: int = 1
) -> RunResult:
    try:
        source, transforms = load_pipeline(pipeline)
    except (KeyError, ImportError, AttributeError) as e:
        console.print(f"[red]could not load pipeline[/red] {pipeline!r}: {e}")
        raise typer.Exit(code=2) from None
    config = TelemetryConfig(tier=tier, clock=SystemClock())
    if workers == 1 and parallelism == 1:
        return asyncio.run(run_local_chain(source, transforms, capacity=capacity, telemetry=config))
    # Parallel / distributed: lower the pipeline to a graph (each transform at `parallelism`, keyed by
    # its declared key columns). One worker runs it in-process; more workers spawn and deploy it.
    graph = graph_from_pipeline(source, transforms, parallelism)
    if workers == 1:
        return asyncio.run(run_plan(graph, capacity=capacity, telemetry=config))
    return deploy(graph, num_workers=workers, capacity=capacity, telemetry=config)


def _summary_table(report: RunReport) -> Table:
    op_class = {o.operator_id: o.op_class for o in report.operators}
    table = Table(
        title="operators (most work first)",
        caption=(
            "rows_out = rows produced · busy µs = time computing · "
            "wait µs = time blocked because the next step was full · errors = exceptions"
        ),
        caption_style="dim",
        header_style="bold",
        expand=False,
    )
    table.add_column("operator")
    table.add_column("class")
    table.add_column("rows_out", justify="right")
    table.add_column("busy µs", justify="right")
    table.add_column("wait µs", justify="right")
    table.add_column("errors", justify="right")
    for s in report.by_self_time():
        table.add_row(
            s.operator_id,
            op_class.get(s.operator_id, ""),
            str(s.rows_out_total),
            str(s.busy_micros_total),
            str(s.send_wait_micros_total),
            str(s.error_count) if s.error_count == 0 else f"[red]{s.error_count}[/red]",
        )
    return table


def _hardware_line(report: RunReport) -> str | None:
    """A one-line process resource summary from the 'process' row, or None if unsampled."""
    proc = report.operator("process")
    if proc is None:
        return None
    gauges = {g.name: g.last for g in proc.gauges}
    parts = []
    if "process.cpu_percent" in gauges:
        parts.append(f"CPU {gauges['process.cpu_percent']:.0f}%")
    if "process.rss_bytes" in gauges:
        parts.append(f"RSS {gauges['process.rss_bytes'] / 1_000_000:.0f} MB")
    if "process.num_fds" in gauges:
        parts.append(f"fds {gauges['process.num_fds']:.0f}")
    lag = next((h for h in proc.histograms if h.name == "runtime.loop_lag_micros"), None)
    if lag is not None and lag.max is not None:
        parts.append(f"loop-lag max {lag.max} µs")
    return "  ·  ".join(parts) if parts else None


def _preview_table(batch: pa.RecordBatch, head: int) -> Table:
    table = Table(
        title=f"output (first {min(head, batch.num_rows)} of {batch.num_rows} rows)",
        header_style="bold cyan",
    )
    for name in batch.schema.names:
        table.add_column(name)
    columns = [batch.column(i).to_pylist() for i in range(batch.num_columns)]
    for r in range(min(head, batch.num_rows)):
        table.add_row(*(str(col[r]) for col in columns))
    return table


@app.command()
def run(
    pipeline: str = typer.Argument(..., help="A built-in example name, or 'module:function'."),
    telemetry: str = typer.Option("counters", help="off | counters | events | full"),
    show: str = typer.Option("summary", help="summary | markdown | json | none"),
    save: Path | None = typer.Option(None, help="Write the full JSON report to this path."),
    capacity: int = typer.Option(16, help="Channel capacity (backpressure bound)."),
    head: int = typer.Option(5, help="Rows of pipeline output to preview."),
    workers: int = typer.Option(
        1, help="Worker processes to deploy across (>1 spawns and distributes)."
    ),
    parallelism: int = typer.Option(1, help="Instances per operator (keyed ops shuffle by key)."),
) -> None:
    """Run a PIPELINE and show its output and telemetry."""
    result = _run(pipeline, _tier(telemetry), capacity, workers, parallelism)
    report = result.telemetry

    if head > 0 and len(result) > 0:
        console.print(_preview_table(result[0], head))

    if show == "summary":
        s = report.summary
        deepest = (
            f"  deepest queue: {report.summary.deepest_queue[0]} "
            f"(depth {report.summary.deepest_queue[1]})"
            if report.summary.deepest_queue
            else ""
        )
        hardware = _hardware_line(report)
        hw_line = f"\nhardware: {hardware}" if hardware else ""
        console.print(
            Panel.fit(
                f"run [bold]{report.run_id}[/bold] · {s.total_rows_in} rows in → "
                f"{s.total_rows_out} rows out · {s.total_errors} errors\n"
                f"wall {report.meta.wall_micros} µs · telemetry tier '{telemetry}'{deepest}{hw_line}",
                title="nautilus run",
            )
        )
        console.print(_summary_table(report))
    elif show == "markdown":
        # Render the markdown so it's readable in a terminal (use --show summary for the default
        # tables, or `nautilus task` / --save for the raw text to hand to an agent).
        console.print(Markdown(report.to_markdown()))
    elif show == "json":
        console.print_json(report.to_json())
    elif show != "none":
        raise typer.BadParameter("show must be one of: summary, markdown, json, none")

    if save is not None:
        save.write_text(report.to_json(indent=2))
        console.print(f"[green]wrote[/green] {save}")


@app.command()
def examples() -> None:
    """List the built-in example pipelines."""
    table = Table(title="built-in pipelines", header_style="bold")
    table.add_column("name")
    table.add_column("description")
    for name, builder in EXAMPLES.items():
        table.add_row(name, (builder.__doc__ or "").strip().splitlines()[0])
    console.print(table)
    console.print("\nrun one with:  [bold]nautilus run <name>[/bold]")


@app.command()
def catalog(
    md: bool = typer.Option(False, "--md", help="Print the full markdown reference instead."),
) -> None:
    """Show every metric nautilus records, with its unit, tier, and meaning."""
    if md:
        console.print(render_reference())
        return
    table = Table(title="telemetry metrics", header_style="bold", show_lines=False)
    table.add_column("metric")
    table.add_column("unit")
    table.add_column("tier")
    table.add_column("means")
    for name in sorted(METRIC_SPECS):
        spec = METRIC_SPECS[name]
        table.add_row(name, spec.unit, Tier(spec.min_tier).name.lower(), spec.meaning)
    console.print(table)
    console.print("\nfull detail (labels, relations):  [bold]nautilus catalog --md[/bold]")


def _present_metric_defs(report: RunReport) -> list[tuple[str, str, str]]:
    present: set[str] = set()
    for o in report.operators:
        present.update(p.name for p in o.counters)
        present.update(p.name for p in o.gauges)
        present.update(p.name for p in o.histograms)
    out = []
    for name in sorted(present):
        spec = METRIC_SPECS.get(name)
        if spec is not None:
            out.append((name, spec.unit, spec.meaning))
    return out


def _agent_prompt(description: str, report: RunReport | None) -> str:
    lines = [
        "# Task for an agent working on nautilus",
        "",
        description.strip(),
    ]
    if report is not None:
        lines += [
            "",
            "## Telemetry from the latest run (these are facts; draw your own conclusions)",
            "",
            "```",
            report.to_markdown(token_budget=3000),
            "```",
            "",
            "## What those numbers mean",
            "",
        ]
        lines += [
            f"- `{name}` ({unit}): {meaning}"
            for name, unit, meaning in _present_metric_defs(report)
        ]
    lines += [
        "",
        "## Where to look",
        "",
        "- `DESIGN.md` — architecture and the telemetry design",
        "- `IMPLEMENTATION_PLAN.md` — staged plan and current status",
        "- `src/nautilus/` — the framework source",
        "- `docs/telemetry-reference.md` — every metric nautilus records, defined",
        "",
        "## How to work",
        "",
        "Read the telemetry, identify performance issues / bugs / optimizations, make the change,",
        "then re-run the same pipeline and compare the numbers.",
    ]
    return "\n".join(lines)


@app.command()
def task(
    description: str = typer.Argument(..., help="What you want the agent to do, in your words."),
    on: str | None = typer.Option(
        None, "--on", help="Run this pipeline first and include its telemetry."
    ),
    telemetry: str = typer.Option("counters", help="off | counters | events | full"),
    capacity: int = typer.Option(16, help="Channel capacity (backpressure bound)."),
    save: Path | None = typer.Option(None, help="Write the prompt to this file instead of stdout."),
) -> None:
    """Print a ready-to-paste prompt for an AI coding agent — your TASK plus the run's telemetry."""
    report = _run(on, _tier(telemetry), capacity).telemetry if on else None
    prompt = _agent_prompt(description, report)
    if save is not None:
        save.write_text(prompt)
        console.print(f"[green]wrote agent prompt to[/green] {save}")
    else:
        # Plain print (not Rich) so it is clean to copy/pipe into an agent.
        print(prompt)


_STATUS_STYLE = {
    "IMPROVED": "green",
    "REGRESSED": "red",
    "OUTPUT-CHANGED": "red bold",
    "unchanged": "dim",
    "machine-differs": "yellow",
}


def _fmt(n: float) -> str:
    return f"{n:,.0f}"


def _bench_panel(r: BenchResult) -> Panel:
    t, e = r.throughput_rows_per_sec, r.environment
    determinism = "deterministic" if r.deterministic else "[red]NONDETERMINISTIC[/red]"
    lines = [
        f"pipeline [bold]{r.pipeline}[/bold] · {r.trials} trials · tier {Tier(r.scale['tier']).name.lower()}",
        f"scale rows={_fmt(r.scale['rows'])} batch={r.scale['batch']} keys={r.scale['keys']} "
        f"parallelism={r.scale['parallelism']} workers={r.scale['workers']}",
        "",
        f"throughput [bold]{_fmt(t.median)}[/bold] rows/s (median)  ·  IQR {_fmt(t.iqr)}  ·  "
        f"noise {t.rel_spread:.1%}  ·  range {_fmt(t.min)}–{_fmt(t.max)}",
        f"digest {r.structural_digest[:12]} ({determinism})",
        f"on {e.platform} · py {e.python_version} · nautilus {e.nautilus_version} · commit {e.commit or '—'}",
    ]
    return Panel.fit("\n".join(lines), title="nautilus bench")


def _comparison_line(c: Comparison) -> str:
    style = _STATUS_STYLE.get(c.status, "")
    return (
        f"vs baseline: [{style}]{c.status}[/{style}] {c.delta:+.1%} "
        f"(median {_fmt(c.base_median)} → {_fmt(c.new_median)} rows/s; needs ±{c.threshold:.1%} to count)"
    )


@app.command()
def bench(
    pipeline: str = typer.Argument(..., help="A built-in example name, or 'module:function'."),
    trials: int = typer.Option(DEFAULT_TRIALS, help="Measured runs reduced to a median + IQR."),
    warmup: int = typer.Option(DEFAULT_WARMUP, help="Warmup runs, discarded (cold-cache)."),
    rows: int = typer.Option(DEFAULT_ROWS, help="bench-* total rows (ignored by fixed pipelines)."),
    batch: int = typer.Option(DEFAULT_BATCH, help="bench-* rows per batch."),
    keys: int = typer.Option(DEFAULT_KEYS, help="bench-* distinct keys."),
    wm_every: int = typer.Option(DEFAULT_WM_EVERY, "--wm-every", help="bench-* watermark cadence."),
    parallelism: int = typer.Option(1, help="Instances per operator (keyed ops shuffle by key)."),
    workers: int = typer.Option(1, help="Worker processes (>1 deploys across them)."),
    capacity: int = typer.Option(16, help="Channel capacity (backpressure bound)."),
    telemetry: str = typer.Option("counters", help="Tier to measure at (>= counters; the digest needs it)."),
    json_out: bool = typer.Option(False, "--json", help="Emit the result as JSON."),
    baseline: Path = typer.Option(DEFAULT_BASELINE, help="Baseline file to compare against / update."),
    update: bool = typer.Option(False, "--update", help="Write this result into the baseline file."),
    label: str = typer.Option(
        "", help="Baseline entry name (default: the pipeline). Use to keep a --workers variant alongside the single-process one."
    ),
) -> None:
    """Measure a pipeline's throughput over repeated trials (median + IQR, not best-of-N), compare to the
    baseline if one exists, and optionally update it. This is how to produce the before/after numbers a
    PERFORMANCE_CHANGELOG.md entry records."""
    key = label or pipeline
    tier = _tier(telemetry)
    if tier <= Tier.OFF:
        raise typer.BadParameter("bench needs telemetry >= counters (the structural digest needs it)")
    try:
        with console.status(f"measuring {pipeline} · {warmup}+{trials} runs…"):
            result = measure(
                pipeline, rows=rows, batch=batch, keys=keys, wm_every=wm_every,
                parallelism=parallelism, workers=workers, capacity=capacity, tier=tier,
                trials=trials, warmup=warmup,
                recorded_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
    except (KeyError, ImportError, AttributeError) as e:
        console.print(f"[red]could not load pipeline[/red] {pipeline!r}: {e}")
        raise typer.Exit(code=2) from None

    if json_out:
        console.print_json(json.dumps(result.to_dict()))
    else:
        console.print(_bench_panel(result))
        base = load_baseline(baseline) if baseline.exists() else {}
        if key in base:
            console.print(_comparison_line(compare(base[key], result)))

    if update:
        base = load_baseline(baseline) if baseline.exists() else {}
        base[key] = result
        save_baseline(baseline, base)
        console.print(f"[green]updated baseline[/green] {baseline} · [bold]{key}[/bold]")


@app.command(name="bench-check")
def bench_check(
    baseline: Path = typer.Option(DEFAULT_BASELINE, help="Baseline file to check against."),
    threshold: float = typer.Option(DEFAULT_THRESHOLD, help="Floor (fraction) a change must clear to count."),
    update: bool = typer.Option(False, "--update", help="Rewrite the baseline from this run instead of checking."),
) -> None:
    """Re-run every pipeline in the baseline at its recorded scale and fail (exit 1) on any regression or
    output change — the regression ratchet for CI. A change counts only when it clears both the threshold
    and twice the measured noise; an output change (digest mismatch) always fails, on any machine."""
    if not baseline.exists():
        console.print(
            f"[red]no baseline at[/red] {baseline}  "
            "(create one with `nautilus bench <pipeline> --update`)"
        )
        raise typer.Exit(code=2)
    base = load_baseline(baseline)
    if not base:
        console.print(f"[yellow]baseline is empty[/yellow] {baseline}")
        return
    now = datetime.now(UTC).isoformat(timespec="seconds")
    table = Table(title=f"bench-check vs {baseline}", header_style="bold")
    table.add_column("pipeline")
    for col in ("baseline rows/s", "now rows/s", "Δ", "noise"):
        table.add_column(col, justify="right")
    table.add_column("status")

    failures: list[str] = []
    drifted: list[str] = []
    updated: dict[str, BenchResult] = {}
    for name, b in base.items():
        with console.status(f"re-running {name} · {b.trials} trials…"):
            cur = measure_like(b, recorded_at=now)
        cmp = compare(b, cur, min_threshold=threshold)
        updated[name] = cur
        style = _STATUS_STYLE.get(cmp.status, "")
        table.add_row(
            name, _fmt(cmp.base_median), _fmt(cmp.new_median), f"{cmp.delta:+.1%}",
            f"{cur.throughput_rows_per_sec.rel_spread:.1%}", f"[{style}]{cmp.status}[/{style}]",
        )
        if is_failure(cmp.status):
            failures.append(name)
        if cmp.status == "machine-differs":
            drifted.append(name)
    console.print(table)

    if update:
        save_baseline(baseline, updated)
        console.print(f"[green]rewrote baseline[/green] {baseline}")
        return
    if drifted:
        console.print(
            f"[yellow]note:[/yellow] {', '.join(drifted)} ran on different hardware than the baseline — "
            "throughput is not comparable there (digest still checked). Re-baseline on this machine."
        )
    if failures:
        console.print(f"[red bold]{len(failures)} failure(s):[/red bold] {', '.join(failures)}")
        raise typer.Exit(code=1)
    console.print("[green]no regressions[/green]")


@app.command()
def reference(
    write: bool = typer.Option(False, "--write", help="(Re)generate docs/telemetry-reference.md."),
) -> None:
    """Print (or regenerate) the telemetry reference document."""
    if write:
        path = write_reference()
        console.print(f"[green]wrote[/green] {path}")
    else:
        console.print(render_reference())


@app.command()
def dashboard(
    pipeline: str = typer.Argument(..., help="A built-in example name, or 'module:function'."),
    port: int = typer.Option(8787, help="Port to serve on (0 = OS-assigned)."),
    host: str = typer.Option("127.0.0.1", help="Bind host (use 0.0.0.0 to expose; needs auth)."),
    telemetry: str = typer.Option("counters", help="off | counters | events | full"),
    capacity: int = typer.Option(16, help="Channel capacity (backpressure bound)."),
    linger: bool = typer.Option(
        True, "--linger/--no-linger", help="Keep serving after a bounded run."
    ),
    max_seconds: float | None = typer.Option(
        None, help="Stop after N seconds (caps unbounded runs)."
    ),
    open_browser: bool = typer.Option(False, "--open", help="Open the dashboard in a browser."),
) -> None:
    """Run a PIPELINE and serve a live telemetry dashboard in the browser."""
    from nautilus.telemetry.live import serve_local_chain

    try:
        source, transforms = load_pipeline(pipeline)
    except (KeyError, ImportError, AttributeError) as e:
        console.print(f"[red]could not load pipeline[/red] {pipeline!r}: {e}")
        raise typer.Exit(code=2) from None

    def on_ready(url: str) -> None:
        console.print(
            Panel.fit(
                f"live dashboard at [bold]{url}[/bold]\nrunning '{pipeline}' · press Ctrl-C to stop",
                title="nautilus dashboard",
            )
        )

    config = TelemetryConfig(tier=_tier(telemetry), clock=SystemClock())
    try:
        asyncio.run(
            serve_local_chain(
                source,
                transforms,
                capacity=capacity,
                telemetry=config,
                host=host,
                port=port,
                linger=linger,
                max_seconds=max_seconds,
                open_browser=open_browser,
                on_ready=on_ready,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[dim]stopped[/dim]")
    except OSError as e:
        console.print(
            f"[red]could not bind {host}:{port}[/red]: {e}  (try --port 0 or another --port)"
        )
        raise typer.Exit(code=1) from None


@app.command()
def serve(
    report: Path = typer.Option(..., "--report", help="A saved JSON report (from `run --save`)."),
    port: int = typer.Option(8787, help="Port to serve on (0 = OS-assigned)."),
    host: str = typer.Option("127.0.0.1", help="Bind host."),
) -> None:
    """View a saved JSON report statically in the dashboard."""
    from nautilus.telemetry.live import serve_report as _serve_report

    def on_ready(url: str) -> None:
        console.print(
            Panel.fit(
                f"viewing {report} at [bold]{url}[/bold]\npress Ctrl-C to stop",
                title="nautilus serve",
            )
        )

    try:
        asyncio.run(_serve_report(report.read_text(), host=host, port=port, on_ready=on_ready))
    except KeyboardInterrupt:
        console.print("\n[dim]stopped[/dim]")
    except OSError as e:
        console.print(
            f"[red]could not bind {host}:{port}[/red]: {e}  (try --port 0 or another --port)"
        )
        raise typer.Exit(code=1) from None


@app.command()
def version() -> None:
    """Print the nautilus version."""
    console.print(f"nautilus {nautilus.__version__}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
