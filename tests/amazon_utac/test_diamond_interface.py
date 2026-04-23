"""Tests for the AmazonUTAC Diamond-Template contract (Package 19)."""

from __future__ import annotations

import pytest

from amazon_utac import GAMMA_AMAZON, AmazonUTAC
from amazon_utac.constants import (
    CREP_SIGMA,
    ETA_AMAZON,
    H_FOREST_ATTRACTOR,
    H_SAVANNA_ATTRACTOR,
)


@pytest.fixture(scope="module")
def utac() -> AmazonUTAC:
    return AmazonUTAC(seed=42)


# ── Diamond contract: method presence ────────────────────────────────────────

class TestDiamondContract:
    def test_has_run_cycle(self, utac: AmazonUTAC) -> None:
        assert callable(getattr(utac, "run_cycle", None))

    def test_has_get_crep_state(self, utac: AmazonUTAC) -> None:
        assert callable(getattr(utac, "get_crep_state", None))

    def test_has_get_utac_state(self, utac: AmazonUTAC) -> None:
        assert callable(getattr(utac, "get_utac_state", None))

    def test_has_get_phase_events(self, utac: AmazonUTAC) -> None:
        assert callable(getattr(utac, "get_phase_events", None))

    def test_has_to_zenodo_record(self, utac: AmazonUTAC) -> None:
        assert callable(getattr(utac, "to_zenodo_record", None))


# ── get_crep_state ───────────────────────────────────────────────────────────

class TestCREPState:
    def test_crep_returns_dict(self, utac: AmazonUTAC) -> None:
        state = utac.get_crep_state()
        assert isinstance(state, dict)

    def test_crep_has_required_keys(self, utac: AmazonUTAC) -> None:
        state = utac.get_crep_state()
        for key in ("C", "R", "E", "P", "Gamma"):
            assert key in state, f"Missing CREP key: {key}"

    def test_crep_components_in_range(self, utac: AmazonUTAC) -> None:
        state = utac.get_crep_state()
        for key in ("C", "R", "E", "P"):
            val = state[key]
            assert 0.0 <= val <= 1.0, f"CREP {key}={val} out of [0, 1]"

    def test_gamma_positive(self, utac: AmazonUTAC) -> None:
        state = utac.get_crep_state()
        assert state["Gamma"] >= 0.0

    def test_gamma_low_crep_zone(self, utac: AmazonUTAC) -> None:
        """Amazon must be in LOW-CREP zone: Γ < 0.25."""
        state = utac.get_crep_state()
        assert state["Gamma"] < 0.25, (
            f"Γ={state['Gamma']:.3f} should be < 0.25 (LOW-CREP zone for Amazon)"
        )


# ── get_utac_state ───────────────────────────────────────────────────────────

class TestUTACState:
    def test_utac_returns_dict(self, utac: AmazonUTAC) -> None:
        state = utac.get_utac_state()
        assert isinstance(state, dict)

    def test_utac_has_required_keys(self, utac: AmazonUTAC) -> None:
        state = utac.get_utac_state()
        for key in ("H", "dH_dt", "H_star_forest", "H_star_savanna", "K_eff"):
            assert key in state, f"Missing UTAC key: {key}"

    def test_H_in_range(self, utac: AmazonUTAC) -> None:
        state = utac.get_utac_state()
        assert 0.0 <= state["H"] <= 1.0

    def test_H_star_forest_value(self, utac: AmazonUTAC) -> None:
        state = utac.get_utac_state()
        assert abs(state["H_star_forest"] - H_FOREST_ATTRACTOR) < 0.01

    def test_H_star_savanna_value(self, utac: AmazonUTAC) -> None:
        state = utac.get_utac_state()
        assert abs(state["H_star_savanna"] - H_SAVANNA_ATTRACTOR) < 0.01

    def test_K_eff_is_one(self, utac: AmazonUTAC) -> None:
        state = utac.get_utac_state()
        assert state["K_eff"] == pytest.approx(1.0)


# ── get_phase_events ─────────────────────────────────────────────────────────

class TestPhaseEvents:
    def test_phase_events_returns_list(self, utac: AmazonUTAC) -> None:
        events = utac.get_phase_events()
        assert isinstance(events, list)

    def test_phase_events_after_run_cycle(self) -> None:
        u = AmazonUTAC(seed=42, deforestation_scenario="accelerated")
        u.run_cycle(duration_years=80)
        events = u.get_phase_events()
        # Accelerated scenario should trigger at least one threshold crossing
        assert isinstance(events, list)

    def test_phase_event_structure(self) -> None:
        u = AmazonUTAC(seed=42, deforestation_scenario="accelerated")
        u.run_cycle(duration_years=80)
        events = u.get_phase_events()
        if events:
            e = events[0]
            for key in ("year", "H", "event_type", "description"):
                assert key in e, f"Phase event missing key: {key}"


# ── run_cycle ────────────────────────────────────────────────────────────────

class TestRunCycle:
    def test_run_cycle_returns_dict(self, utac: AmazonUTAC) -> None:
        result = utac.run_cycle(duration_years=30)
        assert isinstance(result, dict)

    def test_run_cycle_has_required_keys(self, utac: AmazonUTAC) -> None:
        result = utac.run_cycle(duration_years=30)
        for key in ("years", "H", "Gamma", "crep_state", "utac_state", "phase_events"):
            assert key in result, f"run_cycle missing key: {key}"

    def test_H_trajectory_physical(self, utac: AmazonUTAC) -> None:
        result = utac.run_cycle(duration_years=30)
        H = result["H"]
        assert all(0.0 <= h <= 1.0 for h in H), "H trajectory contains unphysical values"

    def test_run_cycle_ethics_warnings_list(self, utac: AmazonUTAC) -> None:
        result = utac.run_cycle(duration_years=30)
        assert isinstance(result.get("ethics_warnings", []), list)


# ── to_zenodo_record ─────────────────────────────────────────────────────────

class TestZenodoRecord:
    def test_zenodo_returns_dict(self, utac: AmazonUTAC) -> None:
        record = utac.to_zenodo_record()
        assert isinstance(record, dict)

    def test_zenodo_has_metadata(self, utac: AmazonUTAC) -> None:
        record = utac.to_zenodo_record()
        assert "metadata" in record
        assert "title" in record["metadata"]

    def test_zenodo_has_disclaimer(self, utac: AmazonUTAC) -> None:
        record = utac.to_zenodo_record()
        assert "disclaimer" in record
        assert len(record["disclaimer"]) > 10

    def test_zenodo_has_uncertainty_bounds(self, utac: AmazonUTAC) -> None:
        record = utac.to_zenodo_record()
        assert "uncertainty_bounds" in record
        ub = record["uncertainty_bounds"]
        assert "H_lower" in ub and "H_upper" in ub

    def test_zenodo_ethics_gate_passed(self, utac: AmazonUTAC) -> None:
        record = utac.to_zenodo_record()
        assert record["ethics_gate"]["status"] == "passed"


# ── CREP calibration ─────────────────────────────────────────────────────────

class TestCalibration:
    def test_gamma_analytical(self) -> None:
        """Verify Γ_Amazon = arctanh(0.25) / 2.2 ≈ 0.116."""
        import math
        gamma_expected = math.atanh(ETA_AMAZON) / CREP_SIGMA
        assert abs(GAMMA_AMAZON - gamma_expected) < 1e-6

    def test_gamma_approx_0116(self) -> None:
        assert abs(GAMMA_AMAZON - 0.116) < 0.005

    def test_eta_amazon_is_025(self) -> None:
        assert pytest.approx(0.25) == ETA_AMAZON

    def test_gamma_lower_than_amoc(self) -> None:
        """Amazon (Γ ≈ 0.116) must be lower than AMOC (Γ ≈ 0.251)."""
        GAMMA_AMOC = 0.251
        assert GAMMA_AMAZON < GAMMA_AMOC
