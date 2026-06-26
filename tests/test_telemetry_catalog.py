"""The catalog is the self-describing surface; these tests enforce that: facts only, never verdicts;
units that match names; and integrity of cross-references."""

import re

import pytest

from nautilus.telemetry.catalog import (
    BANNED_ANALYSIS_WORDS,
    EVENT_SPECS,
    METRIC_SPECS,
    STRUCTURAL_METRICS,
    MetricKind,
)

_WORD = re.compile(r"[a-z_]+")


def _human_strings() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for spec in METRIC_SPECS.values():
        out.append((spec.name, spec.meaning))
        if spec.derivation:
            out.append((spec.name, spec.derivation))
    for spec in EVENT_SPECS.values():
        out.append((spec.name, spec.meaning))
    return out


def test_no_analysis_words_in_catalog():
    """nautilus ships data, not verdicts: no causal/diagnostic words may appear in any catalog text."""
    offenders = []
    for owner, text in _human_strings():
        tokens = set(_WORD.findall(text.lower()))
        bad = tokens & BANNED_ANALYSIS_WORDS
        if bad:
            offenders.append((owner, sorted(bad), text))
    assert not offenders, f"analysis words leaked into the catalog: {offenders}"


def test_micros_metrics_have_micros_unit():
    for name, spec in METRIC_SPECS.items():
        if name.endswith("_micros"):
            assert "micros" in spec.unit, f"{name} unit {spec.unit!r} should mention micros"


def test_every_metric_has_a_reduction():
    for spec in METRIC_SPECS.values():
        assert spec.reduction is not None


def test_histograms_have_boundaries_and_others_do_not():
    for spec in METRIC_SPECS.values():
        if spec.kind is MetricKind.HISTOGRAM:
            assert spec.boundaries, f"{spec.name} histogram needs boundaries"
            assert list(spec.boundaries) == sorted(spec.boundaries)
        else:
            assert spec.boundaries == (), f"{spec.name} is not a histogram but has boundaries"


def test_relates_to_references_exist():
    for spec in METRIC_SPECS.values():
        for ref in spec.relates_to:
            assert ref in METRIC_SPECS, f"{spec.name} relates_to unknown metric {ref}"


def test_structural_metrics_are_declared_and_deterministic():
    for name in STRUCTURAL_METRICS:
        assert name in METRIC_SPECS, f"structural metric {name} not in catalog"
        assert METRIC_SPECS[name].deterministic, f"structural metric {name} must be deterministic"


@pytest.mark.parametrize("name", ["operator.bytes_in", "operator.bytes_out"])
def test_byte_metrics_are_full_tier(name):
    from nautilus.telemetry.catalog import Tier

    assert METRIC_SPECS[name].min_tier is Tier.FULL
