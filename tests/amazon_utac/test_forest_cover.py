"""Tests for ForestCoverLoader and ProdesDeforestation."""

from __future__ import annotations

import numpy as np
import pytest

from amazon_utac.forest_cover import ForestCoverLoader
from amazon_utac.deforestation import ProdesDeforestation
from amazon_utac.constants import (
    ANNUAL_DEFORESTATION_RATE,
    CURRENT_DEFORESTATION_FRACTION,
    K_FOREST,
)


class TestForestCoverLoader:
    @pytest.fixture
    def loader(self) -> ForestCoverLoader:
        return ForestCoverLoader(start_year=1988, end_year=2024, seed=42)

    def test_years_length(self, loader: ForestCoverLoader) -> None:
        assert len(loader.years) == 2024 - 1988 + 1

    def test_cover_in_range(self, loader: ForestCoverLoader) -> None:
        assert np.all(loader.cover >= 0.0)
        assert np.all(loader.cover <= K_FOREST)

    def test_cover_is_declining(self, loader: ForestCoverLoader) -> None:
        cover = loader.cover
        # Overall trend must be negative (forest is lost)
        trend = np.polyfit(np.arange(len(cover)), cover, 1)[0]
        assert trend < 0, f"Forest cover trend is positive ({trend:.4f}) — should be declining"

    def test_current_cover_approx_2024(self, loader: ForestCoverLoader) -> None:
        # Current cover should be near 0.84 (84% of original Amazon)
        assert 0.70 < loader.current_cover() < 0.95

    def test_deforestation_fraction(self, loader: ForestCoverLoader) -> None:
        defor = loader.deforestation_fraction()
        assert len(defor) == len(loader.cover)
        assert np.allclose(defor, 1.0 - loader.cover)

    def test_cover_at_interpolation(self, loader: ForestCoverLoader) -> None:
        c2000 = loader.cover_at(2000)
        assert 0.7 < c2000 < 1.0

    def test_projection_zero_deforestation(self, loader: ForestCoverLoader) -> None:
        proj_years, proj_cover = loader.project(horizon_years=20, scenario="zero-deforestation")
        assert len(proj_years) == 20
        # Zero deforestation: cover should be roughly stable
        delta = abs(proj_cover[-1] - proj_cover[0])
        assert delta < 0.05

    def test_projection_accelerated_declines(self, loader: ForestCoverLoader) -> None:
        _, cover_current = loader.project(horizon_years=50, scenario="current-rate")
        _, cover_accel = loader.project(horizon_years=50, scenario="accelerated")
        assert cover_accel[-1] < cover_current[-1]

    def test_unknown_scenario_raises(self, loader: ForestCoverLoader) -> None:
        with pytest.raises(ValueError, match="Unknown scenario"):
            loader.project(scenario="fantasy-scenario")

    def test_reproducibility(self) -> None:
        loader1 = ForestCoverLoader(seed=42)
        loader2 = ForestCoverLoader(seed=42)
        assert np.allclose(loader1.cover, loader2.cover)

    def test_different_seeds_differ(self) -> None:
        loader1 = ForestCoverLoader(seed=42)
        loader2 = ForestCoverLoader(seed=99)
        assert not np.allclose(loader1.cover, loader2.cover)


class TestProdesDeforestation:
    @pytest.fixture
    def prodes(self) -> ProdesDeforestation:
        return ProdesDeforestation(seed=42)

    def test_rate_series_length(self, prodes: ProdesDeforestation) -> None:
        years, rates = prodes.rate_series(1988, 2024)
        assert len(years) == len(rates) == 2024 - 1988 + 1

    def test_rates_positive(self, prodes: ProdesDeforestation) -> None:
        _, rates = prodes.rate_series()
        assert np.all(rates >= 0)

    def test_fraction_per_year_small(self, prodes: ProdesDeforestation) -> None:
        _, fracs = prodes.fraction_per_year()
        # Each year's fraction should be < 5% (plausible range)
        assert np.all(fracs < 0.05)

    def test_cumulative_deforestation_growing(self, prodes: ProdesDeforestation) -> None:
        _, cumulative = prodes.cumulative_deforestation()
        assert np.all(np.diff(cumulative) >= 0)

    def test_delta_H_negative(self, prodes: ProdesDeforestation) -> None:
        _, delta_H = prodes.delta_H_per_year()
        assert np.all(delta_H <= 0)  # forest is lost → H decreases

    def test_years_to_threshold(self, prodes: ProdesDeforestation) -> None:
        years = prodes.years_to_threshold(
            current_deforestation_fraction=0.16,
            annual_rate_fraction=0.01,
            threshold=0.20,
        )
        assert years > 0
        # At 1%/yr of remaining 84% forest → ~0.84%/yr → 4/0.84% → ~4.8 years
        assert years < 20

    def test_years_to_threshold_already_crossed(self, prodes: ProdesDeforestation) -> None:
        years = prodes.years_to_threshold(
            current_deforestation_fraction=0.30,
            annual_rate_fraction=0.01,
            threshold=0.20,
        )
        assert years == 0.0

    def test_years_to_threshold_zero_rate(self, prodes: ProdesDeforestation) -> None:
        years = prodes.years_to_threshold(
            current_deforestation_fraction=0.10,
            annual_rate_fraction=0.0,
            threshold=0.20,
        )
        assert years == float("inf")

    def test_tipping_year_keys(self, prodes: ProdesDeforestation) -> None:
        result = prodes.tipping_year()
        assert "lower" in result
        assert "midpoint" in result
        assert "upper" in result

    def test_tipping_year_ordering(self, prodes: ProdesDeforestation) -> None:
        result = prodes.tipping_year()
        assert result["lower"] <= result["midpoint"] <= result["upper"]

    def test_scenario_rates(self, prodes: ProdesDeforestation) -> None:
        for scenario in ("current-rate", "zero-deforestation", "accelerated", "recovery"):
            years, rates = prodes.scenario_rates(scenario, horizon_years=20)
            assert len(years) == len(rates) == 20

    def test_unknown_scenario_raises(self, prodes: ProdesDeforestation) -> None:
        with pytest.raises(ValueError, match="Unknown scenario"):
            prodes.scenario_rates("does-not-exist")
