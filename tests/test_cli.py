"""The CLI: humans run pipelines and read telemetry; `task` prints an agent-ready prompt."""

import json

import pytest
from typer.testing import CliRunner

import nautilus
from nautilus.cli import app

runner = CliRunner()


def test_examples_lists_pipelines():
    result = runner.invoke(app, ["examples"])
    assert result.exit_code == 0
    assert "wordcount" in result.stdout


def test_run_shows_summary_and_output():
    result = runner.invoke(app, ["run", "wordcount"])
    assert result.exit_code == 0
    assert "rows in" in result.stdout
    assert "operator" in result.stdout


def test_run_save_writes_valid_json_report(tmp_path):
    out = tmp_path / "report.json"
    result = runner.invoke(app, ["run", "wordcount", "--show", "none", "--save", str(out)])
    assert result.exit_code == 0
    doc = json.loads(out.read_text())
    assert doc["schema_version"] == 3
    assert doc["summary"]["total_rows_out"] > 0


def test_run_unknown_pipeline_exits_nonzero():
    result = runner.invoke(app, ["run", "does-not-exist"])
    assert result.exit_code == 2


def test_task_prompt_includes_task_telemetry_and_definitions():
    result = runner.invoke(app, ["task", "make tokenize faster", "--on", "wordcount"])
    assert result.exit_code == 0
    assert "make tokenize faster" in result.stdout
    assert "Telemetry from the latest run" in result.stdout
    # the prompt (printed plain, not via Rich) carries full, untruncated metric definitions
    assert "operator.process_micros" in result.stdout
    assert "Where to look" in result.stdout


def test_task_without_pipeline_omits_telemetry():
    result = runner.invoke(app, ["task", "add a feature"])
    assert result.exit_code == 0
    assert "add a feature" in result.stdout
    assert "Telemetry from the latest run" not in result.stdout


def test_task_can_write_prompt_to_file(tmp_path):
    out = tmp_path / "prompt.md"
    result = runner.invoke(app, ["task", "optimize", "--on", "wordcount", "--save", str(out)])
    assert result.exit_code == 0
    assert "optimize" in out.read_text()


def test_catalog_markdown_lists_every_metric():
    result = runner.invoke(app, ["catalog", "--md"])
    assert result.exit_code == 0
    assert "operator.process_micros" in result.stdout
    assert "Nautilus telemetry reference" in result.stdout


def test_dashboard_single_process_serves_and_exits():
    # --no-linger + a bounded pipeline: serve, run to completion, exit on its own. COLUMNS keeps the Rich
    # panel from wrapping the phrases asserted below.
    result = runner.invoke(
        app,
        ["dashboard", "wordcount", "--no-linger", "--no-open", "--port", "0"],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0, result.output
    assert "live dashboard" in result.stdout
    assert "single process" in result.stdout


def test_dashboard_distributed_serves_across_workers():
    # --workers 2 --parallelism 2 routes through serve_cluster: two *real* workers (parallelism must match,
    # or deploy caps the count), serving the aggregated report, then exiting on its own with --no-linger.
    result = runner.invoke(
        app,
        [
            "dashboard",
            "wordcount",
            "--workers",
            "2",
            "--parallelism",
            "2",
            "--no-linger",
            "--no-open",
            "--port",
            "0",
        ],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0, result.output
    assert "across 2 workers" in result.stdout


@pytest.mark.filterwarnings("ignore:requested")  # the cap warning is asserted in test_cluster_scale
def test_dashboard_reports_capped_worker_count_honestly():
    # The confusing case, made honest: asking for more workers than the pipeline's parallelism must not
    # claim workers that don't exist. wordcount at the default parallelism 1 fills one worker, so three
    # requested reads as "1 of 3", not "3".
    result = runner.invoke(
        app,
        ["dashboard", "wordcount", "--workers", "3", "--no-linger", "--no-open", "--port", "0"],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0, result.output
    assert "across 1 of 3 workers" in result.stdout


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert nautilus.__version__ in result.stdout
