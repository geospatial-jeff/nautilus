"""Generate ``docs/telemetry-reference.md`` from the static CATALOG.

The reference is the offline, self-describing surface an agent reads to learn what every nautilus
metric and event measures — derived entirely from :mod:`nautilus.telemetry.catalog`, so it can never
drift from the code. A no-drift test asserts the committed file equals :func:`render_reference`.

Regenerate with::

    python -m nautilus.telemetry.report.reference
"""

from __future__ import annotations

from pathlib import Path

from nautilus.telemetry.catalog import EVENT_SPECS, METRIC_SPECS, Tier
from nautilus.telemetry.report.report import REPORT_SCHEMA_VERSION
from nautilus.telemetry.report.serialize import CATALOG_VERSION


def render_reference() -> str:
    """Render the full telemetry reference markdown (deterministic; sorted by name)."""
    lines: list[str] = [
        "<!-- Generated from nautilus.telemetry.catalog — do not edit by hand; regenerate with "
        "`python -m nautilus.telemetry.report.reference`. -->",
        "# Nautilus telemetry reference",
        "",
        "Every metric Nautilus records, generated from the metric catalog "
        f"(report schema v{REPORT_SCHEMA_VERSION}, catalog v{CATALOG_VERSION}).",
        "",
        "Each entry gives what the metric measures, its unit and tier, and the metrics it relates to.",
        "",
        "## Metrics",
        "",
        "| name | kind | unit | tier | reduction | labels | meaning | relates_to | derivation |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in sorted(METRIC_SPECS):
        s = METRIC_SPECS[name]
        lines.append(
            f"| `{s.name}` | {s.kind} | {s.unit} | {Tier(s.min_tier).name} | {s.reduction} | "
            f"{', '.join(s.labels)} | {s.meaning} | {', '.join(s.relates_to)} | {s.derivation or ''} |"
        )
    lines += ["", "## Events", "", "| name | tier | fields | meaning |", "|---|---|---|---|"]
    for name in sorted(EVENT_SPECS):
        e = EVENT_SPECS[name]
        lines.append(
            f"| `{e.name}` | {Tier(e.min_tier).name} | {', '.join(e.fields)} | {e.meaning} |"
        )
    lines.append("")
    return "\n".join(lines)


def reference_path() -> Path:
    """Canonical location of the committed reference: ``<repo>/docs/telemetry-reference.md``."""
    return Path(__file__).resolve().parents[4] / "docs" / "telemetry-reference.md"


def write_reference() -> Path:
    path = reference_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_reference())
    return path


if __name__ == "__main__":
    written = write_reference()
    print(f"wrote {written}")
