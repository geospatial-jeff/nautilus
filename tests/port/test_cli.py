"""Characterization tests for the ``nautilus`` CLI's command wiring and argument parsing.

Pins the observable contract of the commands and helpers that had no CLI coverage — the exact exit codes,
the stdout/stderr split (``--json`` keeps stdout pure JSON; status and diagnostics go to stderr), the
``BadParameter`` messages, and how host:port / daemon rosters / telemetry tiers parse. A future Rust port
that reproduces these is faithful.

Every golden here (exit codes, error strings, JSON shape) was read from the running code, not invented.
Wide ``COLUMNS`` stops Rich from wrapping the asserted phrases across lines.
"""

from __future__ import annotations

import json
import re

import pytest
import typer
from typer.testing import CliRunner

import nautilus.cli as cli
from nautilus.bench import BenchResult, Comparison, Environment, Stats, save_baseline
from nautilus.cli import _parse_daemons, _split_host_port, _tier, app
from nautilus.telemetry import Tier

runner = CliRunner()

# A wide terminal so the Rich error panel keeps each asserted message on one line.
WIDE = {"COLUMNS": "200"}

# A tiny, fast scale for the real bench path: two trials, no warmup, a hundred rows.
FAST = ["--trials", "2", "--warmup", "0", "--rows", "100", "--batch", "50", "--keys", "4"]

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_BORDER = re.compile(r"[│┃╭╮╰╯─━┏┓┗┛┌┐└┘|]")


def _norm(text: str) -> str:
    """Rich renders a ``BadParameter`` error in a width-dependent bordered panel; strip ANSI + box borders
    and collapse whitespace so an asserted message matches regardless of terminal width or line wrapping
    (``COLUMNS`` is not always honored under CliRunner, e.g. on CI)."""
    return re.sub(r"\s+", " ", _BORDER.sub(" ", _ANSI.sub("", text)))


def _result(pipeline: str = "bench-linear", *, deterministic: bool = True) -> BenchResult:
    """A hand-built BenchResult, so tests that monkeypatch ``measure``/``measure_like`` never run a real
    pipeline. Values are placeholders — only the structural shape and the ``deterministic`` flag matter.
    """
    return BenchResult(
        pipeline=pipeline,
        scale={"rows": 100, "batch": 50, "keys": 4, "parallelism": 1, "workers": 1, "tier": 1},
        trials=2,
        throughput_rows_per_sec=Stats((1.0, 2.0), 1.5, 0.5, 0.33, 1.0, 2.0),
        structural_digest="deadbeef",
        deterministic=deterministic,
        environment=Environment("0", "3", "plat", "proc", None),
        recorded_at="2026-01-01",
    )


# --- (a) bench --json: pure-JSON stdout of the to_dict() shape -----------------------------------


def test_bench_json_stdout_is_pure_json_of_to_dict_shape():
    result = runner.invoke(app, ["bench", "bench-linear", "--json", *FAST])
    assert result.exit_code == 0, result.output
    doc = json.loads(result.stdout)  # stdout parses as JSON on its own
    assert set(doc) == {
        "pipeline",
        "scale",
        "trials",
        "throughput_rows_per_sec",
        "structural_digest",
        "deterministic",
        "environment",
        "recorded_at",
    }
    assert set(doc["scale"]) == {"rows", "batch", "keys", "parallelism", "workers", "tier"}
    assert set(doc["throughput_rows_per_sec"]) == {
        "samples",
        "median",
        "iqr",
        "rel_spread",
        "min",
        "max",
    }
    assert set(doc["environment"]) == {
        "nautilus_version",
        "python_version",
        "platform",
        "processor",
        "commit",
    }
    assert doc["pipeline"] == "bench-linear"
    assert doc["scale"] == {
        "rows": 100,
        "batch": 50,
        "keys": 4,
        "parallelism": 1,
        "workers": 1,
        "tier": int(Tier.COUNTERS),
    }
    assert doc["trials"] == 2


def test_bench_json_keeps_status_off_stdout_even_with_baseline(tmp_path):
    # Seed a baseline entry, then re-run with --json: the comparison/status line is suppressed under
    # --json (only the JSON is printed), so stdout stays pure JSON and no "vs baseline" leaks anywhere.
    baseline = tmp_path / "baseline.json"
    seed = runner.invoke(
        app, ["bench", "bench-linear", "--json", "--update", "--baseline", str(baseline), *FAST]
    )
    assert seed.exit_code == 0, seed.output
    result = runner.invoke(
        app, ["bench", "bench-linear", "--json", "--baseline", str(baseline), *FAST]
    )
    assert result.exit_code == 0, result.output
    json.loads(result.stdout)  # still pure JSON
    assert "vs baseline" not in result.stdout
    # The one status line the --update seed emitted ("updated baseline …") went to stderr, not stdout.
    assert "updated baseline" in seed.stderr
    assert "updated baseline" not in seed.stdout


# --- (b) bench --update refuses a nondeterministic result ----------------------------------------


def test_bench_update_refuses_nondeterministic_and_leaves_baseline_untouched(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.json"
    baseline.write_text('{"version": 1, "results": {}}\n')
    before = baseline.read_text()
    monkeypatch.setattr(cli, "measure", lambda *a, **k: _result(deterministic=False))
    result = runner.invoke(
        app,
        ["bench", "bench-linear", "--update", "--baseline", str(baseline), *FAST],
        env=WIDE,
    )
    assert result.exit_code == 1
    assert "not updating baseline" in result.stderr
    assert "nondeterministic" in result.stderr
    assert baseline.read_text() == before  # the write was refused


# --- (c) bench-check exit-code matrix ------------------------------------------------------------


def test_bench_check_missing_baseline_exits_2(tmp_path):
    missing = tmp_path / "nope.json"
    result = runner.invoke(app, ["bench-check", "--baseline", str(missing)], env=WIDE)
    assert result.exit_code == 2
    assert "no baseline at" in result.output


def test_bench_check_empty_baseline_exits_0(tmp_path):
    baseline = tmp_path / "empty.json"
    baseline.write_text('{"version": 1, "results": {}}\n')
    result = runner.invoke(app, ["bench-check", "--baseline", str(baseline)], env=WIDE)
    assert result.exit_code == 0
    assert "baseline is empty" in result.output


def test_bench_check_regression_exits_1_and_names_pipeline(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.json"
    save_baseline(baseline, {"foo": _result("foo")})
    monkeypatch.setattr(cli, "measure_like", lambda b, **k: b)
    monkeypatch.setattr(
        cli,
        "compare",
        lambda b, c, min_threshold=0.07: Comparison("foo", "REGRESSED", -0.5, 0.07, 1.5, 0.7, 0.1),
    )
    result = runner.invoke(app, ["bench-check", "--baseline", str(baseline)], env=WIDE)
    assert result.exit_code == 1
    assert "failure" in result.output
    assert "foo" in result.output


def test_bench_check_all_pass_exits_0(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.json"
    save_baseline(baseline, {"foo": _result("foo")})
    monkeypatch.setattr(cli, "measure_like", lambda b, **k: b)
    monkeypatch.setattr(
        cli,
        "compare",
        lambda b, c, min_threshold=0.07: Comparison("foo", "unchanged", 0.0, 0.07, 1.5, 1.5, 0.1),
    )
    result = runner.invoke(app, ["bench-check", "--baseline", str(baseline)], env=WIDE)
    assert result.exit_code == 0
    assert "no regressions" in result.output


# --- (d) _split_host_port and _parse_daemons -----------------------------------------------------


def test_split_host_port_keeps_everything_before_the_last_colon():
    # rpartition on the last colon: "a:b:9000" is host "a:b", port 9000.
    assert _split_host_port("a:b:9000") == ("a:b", 9000)
    assert _split_host_port("host:9000") == ("host", 9000)


@pytest.mark.parametrize("value", ["nocolon", ":9000"])
def test_split_host_port_rejects_missing_host_or_colon(value):
    with pytest.raises(typer.BadParameter, match=r"expected HOST:PORT, got"):
        _split_host_port(value)


def test_split_host_port_rejects_non_integer_port():
    with pytest.raises(typer.BadParameter, match=r"invalid port in 'host:notaport'"):
        _split_host_port("host:notaport")


def test_parse_daemons_splits_strips_and_skips_blanks():
    assert _parse_daemons("a:1, b:2 ,, c:3") == [("a", 1), ("b", 2), ("c", 3)]


def test_parse_daemons_returns_none_when_arg_none_and_env_unset(monkeypatch):
    monkeypatch.delenv("NAUTILUS_DAEMONS", raising=False)
    assert _parse_daemons(None) is None
    assert _parse_daemons("") is None  # empty string is falsy, same as unset


def test_parse_daemons_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("NAUTILUS_DAEMONS", "x:10,y:20")
    assert _parse_daemons(None) == [("x", 10), ("y", 20)]


def test_parse_daemons_whitespace_only_env_yields_empty_list(monkeypatch):
    # A whitespace-only env var is truthy, so it passes the emptiness guard; every item then strips to
    # blank and is skipped, leaving [] (not None). Pinned as the actual behavior — see notes.
    monkeypatch.setenv("NAUTILUS_DAEMONS", "   ")
    assert _parse_daemons(None) == []


# --- (e) run --key-groups < parallelism ----------------------------------------------------------


def test_run_key_groups_below_parallelism_is_rejected():
    result = runner.invoke(
        app, ["run", "wordcount", "--key-groups", "2", "--parallelism", "3"], env=WIDE
    )
    assert result.exit_code == 2
    assert "--key-groups (2) must be >= --parallelism (3)" in _norm(result.output)


# --- (f) _tier case-insensitivity / unknown tier / bench telemetry floor -------------------------


def test_tier_is_case_insensitive():
    assert _tier("FULL") is Tier.FULL
    assert _tier("full") is Tier.FULL
    assert _tier("CoUnTeRs") is Tier.COUNTERS


def test_tier_unknown_lists_the_valid_tiers():
    with pytest.raises(
        typer.BadParameter, match=r"telemetry must be one of: off, counters, events, full"
    ):
        _tier("bogus")


def test_run_unknown_tier_exits_2_and_lists_tiers():
    result = runner.invoke(app, ["run", "wordcount", "--telemetry", "bogus"], env=WIDE)
    assert result.exit_code == 2
    assert "telemetry must be one of: off, counters, events, full" in _norm(result.output)


def test_bench_telemetry_off_is_rejected():
    result = runner.invoke(
        app,
        ["bench", "bench-linear", "--telemetry", "off", "--trials", "1", "--warmup", "0"],
        env=WIDE,
    )
    assert result.exit_code == 2
    assert "bench needs telemetry >= counters (the structural digest needs it)" in _norm(
        result.output
    )
