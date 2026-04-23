"""Tests for benchmark validation and Ethics-Gate Light."""

from __future__ import annotations

import pytest

from amazon_utac.benchmark import (
    BenchmarkResult,
    benchmark_summary,
    run_benchmarks,
    validate_gamma_spectrum,
)
from amazon_utac.constants import AMAZON_TARGETS, GAMMA_AMAZON
from amazon_utac.ethics_gate import EthicsGate, TensionMetric


class TestBenchmarks:
    def test_run_benchmarks_returns_list(self) -> None:
        results = run_benchmarks()
        assert isinstance(results, list)
        assert len(results) > 0

    def test_all_results_are_benchmark_result(self) -> None:
        results = run_benchmarks()
        for r in results:
            assert isinstance(r, BenchmarkResult)

    def test_benchmark_names_match_targets(self) -> None:
        results = run_benchmarks()
        names = {r.name for r in results}
        for key in AMAZON_TARGETS:
            assert key in names, f"Missing benchmark: {key}"

    def test_gamma_benchmark_passes(self) -> None:
        results = run_benchmarks()
        gamma_result = next(r for r in results if r.name == "gamma_amazon")
        assert gamma_result.passed, (
            f"Gamma benchmark failed: actual={gamma_result.actual}, "
            f"target={gamma_result.target}"
        )

    def test_deforestation_threshold_passes(self) -> None:
        results = run_benchmarks()
        thresh = next(r for r in results if r.name == "deforestation_threshold_pct")
        assert thresh.passed

    def test_ar1_trend_passes(self) -> None:
        results = run_benchmarks()
        ar1 = next(r for r in results if r.name == "resilience_loss_ar1_trend")
        assert ar1.passed

    def test_summary_structure(self) -> None:
        results = run_benchmarks()
        summary = benchmark_summary(results)
        assert "total" in summary
        assert "passed" in summary
        assert "failed" in summary
        assert "pass_rate" in summary
        assert "details" in summary

    def test_pass_rate_between_0_and_1(self) -> None:
        results = run_benchmarks()
        summary = benchmark_summary(results)
        assert 0.0 <= summary["pass_rate"] <= 1.0

    def test_majority_benchmarks_pass(self) -> None:
        results = run_benchmarks()
        summary = benchmark_summary(results)
        assert summary["pass_rate"] >= 0.80, (
            f"Only {summary['pass_rate']:.0%} of benchmarks pass — "
            f"expected ≥ 80%\n{summary['details']}"
        )


class TestGammaSpectrum:
    def test_validate_spectrum_monotonic(self) -> None:
        result = validate_gamma_spectrum({})
        assert result["is_monotonic"], (
            f"CREP spectrum is not monotonic: {result['spectrum']}"
        )

    def test_amazon_gamma_in_spectrum(self) -> None:
        result = validate_gamma_spectrum({})
        assert abs(result["amazon_gamma"] - GAMMA_AMAZON) < 1e-4

    def test_cross_domain_universality_amoc_brain(self) -> None:
        result = validate_gamma_spectrum({})
        assert result["cross_domain_universality"], (
            "Γ_AMOC ≠ Γ_brain — cross-domain universality check failed"
        )


class TestEthicsGate:
    @pytest.fixture
    def gate(self) -> EthicsGate:
        return EthicsGate()

    def test_valid_state_allowed(self, gate: EthicsGate) -> None:
        state = {"H": 0.84, "Gamma": 0.116}
        result = gate.check(state, tension=0.0)
        assert result["allowed"] is True

    def test_H_out_of_range_blocked(self, gate: EthicsGate) -> None:
        state = {"H": -0.1, "Gamma": 0.116}
        result = gate.check(state, tension=0.0)
        assert result["allowed"] is False
        assert "outside physical bounds" in result["reason"]

    def test_H_above_one_blocked(self, gate: EthicsGate) -> None:
        state = {"H": 1.5, "Gamma": 0.116}
        result = gate.check(state, tension=0.0)
        assert result["allowed"] is False

    def test_gamma_too_high_blocked(self, gate: EthicsGate) -> None:
        state = {"H": 0.84, "Gamma": 5.0}
        result = gate.check(state, tension=0.0)
        assert result["allowed"] is False
        assert "exceeds maximum" in result["reason"]

    def test_gamma_negative_blocked(self, gate: EthicsGate) -> None:
        state = {"H": 0.84, "Gamma": -0.1}
        result = gate.check(state, tension=0.0)
        assert result["allowed"] is False
        assert "negative" in result["reason"]

    def test_high_tension_blocked(self, gate: EthicsGate) -> None:
        state = {"H": 0.84, "Gamma": 0.116}
        result = gate.check(state, tension=0.99)
        assert result["allowed"] is False
        assert "tension" in result["reason"].lower()

    def test_moderate_tension_warns(self, gate: EthicsGate) -> None:
        state = {"H": 0.84, "Gamma": 0.116}
        result = gate.check(state, tension=0.75)
        # Should be allowed but with warning
        assert result["allowed"] is True
        assert any("tension" in w.lower() for w in result.get("warnings", []))

    def test_irresponsible_tipping_year_blocked(self, gate: EthicsGate) -> None:
        state = {"H": 0.84, "Gamma": 0.116, "tipping_year": 1800.0}
        result = gate.check(state, tension=0.0)
        assert result["allowed"] is False

    def test_zenodo_requires_uncertainty_bounds(self, gate: EthicsGate) -> None:
        state = {"H": 0.84, "Gamma": 0.116, "is_zenodo_record": True}
        result = gate.check(state, tension=0.0)
        assert result["allowed"] is False
        assert "uncertainty" in result["reason"].lower()

    def test_zenodo_with_bounds_allowed(self, gate: EthicsGate) -> None:
        state = {
            "H": 0.84,
            "Gamma": 0.116,
            "is_zenodo_record": True,
            "uncertainty_bounds": {"H_lower": 0.79, "H_upper": 0.89},
        }
        result = gate.check(state, tension=0.0)
        assert result["allowed"] is True

    def test_disclaimer_is_string(self, gate: EthicsGate) -> None:
        disclaimer = gate.zenodo_disclaimer()
        assert isinstance(disclaimer, str)
        assert len(disclaimer) > 50


class TestTensionMetric:
    def test_initial_tension_zero(self) -> None:
        tm = TensionMetric()
        assert tm.get_current_tension() == 0.0

    def test_update_increases_tension(self) -> None:
        tm = TensionMetric()
        tm.update(max_step_ratio=0.01, n_failures=50)
        assert tm.get_current_tension() > 0.0

    def test_reset_clears_tension(self) -> None:
        tm = TensionMetric()
        tm.update(0.01, 50)
        tm.reset()
        assert tm.get_current_tension() == 0.0

    def test_tension_bounded(self) -> None:
        tm = TensionMetric()
        tm.update(0.0, 1000)
        assert 0.0 <= tm.get_current_tension() <= 1.0
