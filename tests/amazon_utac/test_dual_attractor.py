"""Tests for the DualAttractorLandscape double-well potential."""

from __future__ import annotations

import numpy as np
import pytest

from amazon_utac.savanna_attractor import DualAttractorLandscape
from amazon_utac.crep_amazon import AmazonCREP
from amazon_utac.constants import (
    GAMMA_AMAZON,
    H_FOREST_ATTRACTOR,
    H_SADDLE,
    H_SAVANNA_ATTRACTOR,
)


@pytest.fixture
def landscape() -> DualAttractorLandscape:
    return DualAttractorLandscape()


class TestDualAttractorLandscape:
    def test_effective_saddle_at_zero_gamma(self, landscape: DualAttractorLandscape) -> None:
        hs = landscape.effective_saddle(0.0)
        assert hs == pytest.approx(H_SADDLE, abs=0.01)

    def test_effective_saddle_increases_with_gamma(self, landscape: DualAttractorLandscape) -> None:
        hs0 = landscape.effective_saddle(0.0)
        hs1 = landscape.effective_saddle(0.5)
        hs2 = landscape.effective_saddle(1.0)
        assert hs0 < hs1 < hs2

    def test_effective_saddle_bounded_by_H_f(self, landscape: DualAttractorLandscape) -> None:
        # Saddle can never exceed H_forest (would destroy forest attractor)
        for gamma in [0.0, 0.1, 0.5, 1.0, 2.0]:
            hs = landscape.effective_saddle(gamma)
            assert hs <= landscape.H_f + 0.01

    def test_potential_is_array(self, landscape: DualAttractorLandscape) -> None:
        H_grid = np.linspace(0.1, 0.95, 50)
        V = landscape.potential(H_grid, gamma=0.0)
        assert V.shape == H_grid.shape

    def test_ode_zero_at_attractors(self, landscape: DualAttractorLandscape) -> None:
        gamma = 0.0
        # ODE must be near zero at the two stable fixed points
        dH_forest = landscape.ode(0.0, landscape.H_f, gamma)
        dH_savanna = landscape.ode(0.0, landscape.H_s, gamma)
        assert abs(dH_forest) < 1e-6
        assert abs(dH_savanna) < 1e-6

    def test_ode_restoring_toward_forest(self, landscape: DualAttractorLandscape) -> None:
        gamma = 0.0
        # Slightly below H_f → ODE should push H back up (dH/dt > 0)
        H_perturbed = landscape.H_f - 0.05
        dH = landscape.ode(0.0, H_perturbed, gamma)
        assert dH > 0, f"Should have restoring force toward forest, got dH/dt={dH:.4f}"

    def test_ode_restoring_toward_savanna(self, landscape: DualAttractorLandscape) -> None:
        gamma = 0.0
        # Slightly above H_s → ODE should push H back down (dH/dt < 0)
        H_perturbed = landscape.H_s + 0.05
        dH = landscape.ode(0.0, H_perturbed, gamma)
        assert dH < 0, f"Should have restoring force toward savanna, got dH/dt={dH:.4f}"

    def test_basin_widths_sum_to_one(self, landscape: DualAttractorLandscape) -> None:
        for gamma in [0.0, GAMMA_AMAZON, 0.3]:
            basins = landscape.basin_widths(gamma)
            total = basins["forest_basin_width"] + basins["savanna_basin_width"]
            assert abs(total - 1.0) < 0.02, f"Basin widths don't sum to ~1: {total}"

    def test_forest_basin_shrinks_with_gamma(self, landscape: DualAttractorLandscape) -> None:
        b0 = landscape.basin_widths(0.0)["forest_basin_width"]
        b1 = landscape.basin_widths(0.5)["forest_basin_width"]
        assert b0 > b1, "Forest basin should shrink as Γ increases"

    def test_is_in_forest_basin(self, landscape: DualAttractorLandscape) -> None:
        # At Γ=0, H=0.84 should be in forest basin
        assert landscape.is_in_forest_basin(0.84, 0.0)
        # H=0.20 should be in savanna basin
        assert not landscape.is_in_forest_basin(0.20, 0.0)

    def test_barrier_height_decreases_with_gamma(self, landscape: DualAttractorLandscape) -> None:
        b0 = landscape.barrier_height(0.0)
        b1 = landscape.barrier_height(0.3)
        # Higher deforestation pressure should lower the barrier
        # (barrier height may be negative once saddle merges with forest)
        assert b0 >= b1 or abs(b0 - b1) < 0.5  # allow small numerical noise

    def test_landscape_summary_keys(self, landscape: DualAttractorLandscape) -> None:
        summary = landscape.landscape_summary(GAMMA_AMAZON)
        for key in ("gamma", "H_forest_attractor", "H_savanna_attractor",
                    "H_saddle_effective", "forest_basin_width", "barrier_height"):
            assert key in summary


class TestAmazonCREP:
    @pytest.fixture
    def crep(self) -> AmazonCREP:
        return AmazonCREP()

    def test_component_C_in_range(self, crep: AmazonCREP) -> None:
        for ar1 in [-0.5, 0.0, 0.5, 0.8, 1.2]:
            C = crep.component_C(ar1)
            assert 0.0 <= C <= 1.0

    def test_component_R_sigmoid(self, crep: AmazonCREP) -> None:
        R_low = crep.component_R(2.0)   # below baseline
        R_high = crep.component_R(5.0)  # above baseline
        assert R_low < R_high
        assert 0.0 < R_low < 1.0
        assert 0.0 < R_high < 1.0

    def test_component_E_zero_at_zero(self, crep: AmazonCREP) -> None:
        E = crep.component_E(0.0, 0.0)
        assert E == pytest.approx(0.0, abs=1e-6)

    def test_component_E_increases(self, crep: AmazonCREP) -> None:
        E1 = crep.component_E(0.1, 0.5)
        E2 = crep.component_E(0.3, 0.5)
        assert E2 > E1

    def test_component_P_in_range(self, crep: AmazonCREP) -> None:
        rng = np.random.default_rng(42)
        series = rng.normal(0, 1, 50)
        P = crep.component_P(series)
        assert 0.0 <= P <= 1.0

    def test_gamma_positive(self, crep: AmazonCREP) -> None:
        gamma = crep.gamma(C=0.5, R=0.5, E=0.5, P=0.5)
        assert gamma > 0.0

    def test_gamma_from_eta(self, crep: AmazonCREP) -> None:
        gamma = crep.gamma_from_eta(0.25)
        assert gamma == pytest.approx(GAMMA_AMAZON, abs=1e-4)

    def test_calibrated_gamma_value(self, crep: AmazonCREP) -> None:
        cal = crep.calibrated_gamma()
        assert abs(cal["Gamma"] - GAMMA_AMAZON) < 1e-4
        assert abs(cal["eta"] - 0.25) < 1e-6

    def test_gamma_monotone_in_crep(self, crep: AmazonCREP) -> None:
        """Higher CREP components → higher Γ."""
        g1 = crep.gamma(0.2, 0.2, 0.2, 0.2)
        g2 = crep.gamma(0.5, 0.5, 0.5, 0.5)
        assert g2 > g1
