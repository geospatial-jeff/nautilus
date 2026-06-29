"""S4: the generated telemetry reference is committed, complete, and never stale."""

from nautilus.telemetry.catalog import EVENT_SPECS, METRIC_SPECS
from nautilus.telemetry.report.reference import reference_path, render_reference


def test_reference_is_committed_and_not_stale():
    path = reference_path()
    assert path.exists(), (
        "docs/telemetry-reference.md is missing — "
        "regenerate with `python -m nautilus.telemetry.report.reference`"
    )
    assert path.read_text() == render_reference(), (
        "docs/telemetry-reference.md is stale — "
        "regenerate with `python -m nautilus.telemetry.report.reference`"
    )


def test_reference_documents_every_catalog_entry():
    text = render_reference()
    for name in METRIC_SPECS:
        assert f"`{name}`" in text, f"metric {name} missing from the reference"
    for name in EVENT_SPECS:
        assert f"`{name}`" in text, f"event {name} missing from the reference"
