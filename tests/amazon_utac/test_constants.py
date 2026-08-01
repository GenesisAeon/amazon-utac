"""Tests for constants.py -- in particular the 2026-08-01 PRODES-based update."""

from __future__ import annotations

import pytest

from amazon_utac.constants import (
    AMAZON_TARGETS,
    ANNUAL_DEFORESTATION_RATE_OBSERVED_2025,
    CURRENT_DEFORESTATION_FRACTION,
    CURRENT_FOREST_FRACTION,
    PRODES_DEFORESTED_2024_KM2,
    PRODES_DEFORESTED_2025_KM2,
    PRODES_ORIGINAL_AREA_KM2,
)


def test_current_fraction_consistent_with_prodes_2025() -> None:
    expected = PRODES_DEFORESTED_2025_KM2 / PRODES_ORIGINAL_AREA_KM2
    assert pytest.approx(expected, abs=1e-3) == CURRENT_DEFORESTATION_FRACTION


def test_forest_and_deforestation_fractions_sum_to_one() -> None:
    assert pytest.approx(1.0) == CURRENT_FOREST_FRACTION + CURRENT_DEFORESTATION_FRACTION


def test_2025_area_is_larger_than_2024() -> None:
    """Real, well-documented: deforestation continued in 2025, just far slower."""
    assert PRODES_DEFORESTED_2025_KM2 > PRODES_DEFORESTED_2024_KM2


def test_observed_2025_rate_is_far_below_legacy_default() -> None:
    """The real 2024/2025 PRODES rate is roughly 6x below the model's legacy
    1%/yr default -- both are kept, not silently merged (see constants.py)."""
    from amazon_utac.constants import ANNUAL_DEFORESTATION_RATE

    assert ANNUAL_DEFORESTATION_RATE_OBSERVED_2025 < ANNUAL_DEFORESTATION_RATE / 3


def test_benchmark_target_matches_updated_fraction() -> None:
    target, _ = AMAZON_TARGETS["current_deforestation_pct"]
    assert target == pytest.approx(CURRENT_DEFORESTATION_FRACTION * 100, abs=0.1)
