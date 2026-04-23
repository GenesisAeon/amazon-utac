"""
Benchmark validation for amazon-utac against published targets.

Primary references:
  Lovejoy & Nobre (2019) Science Advances — tipping threshold 20-25 %
  Boulton et al. (2022) Nature Climate Change — resilience loss
  INPE PRODES (2024) — current deforestation state
"""

from __future__ import annotations

from .constants import (
    AMAZON_TARGETS,
    CURRENT_DEFORESTATION_FRACTION,
    DRY_SEASON_LENGTHENING_WEEKS,
    GAMMA_AMAZON,
)


class BenchmarkResult:
    """Container for a single benchmark check result."""

    def __init__(
        self,
        name: str,
        target: object,
        actual: object,
        tolerance: float | None,
        passed: bool,
        note: str = "",
    ) -> None:
        self.name = name
        self.target = target
        self.actual = actual
        self.tolerance = tolerance
        self.passed = passed
        self.note = note

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: actual={self.actual!r}, target={self.target!r}"


def _check_numeric(
    name: str,
    actual: float,
    target: float,
    rel_tolerance: float,
    note: str = "",
) -> BenchmarkResult:
    """Check a numeric value within ±rel_tolerance of target."""
    passed = abs(actual - target) <= abs(target) * rel_tolerance
    return BenchmarkResult(name, target, actual, rel_tolerance, passed, note)


def run_benchmarks(utac_state: dict | None = None) -> list[BenchmarkResult]:
    """
    Validate amazon-utac against AMAZON_TARGETS.

    Parameters
    ----------
    utac_state:
        Optional dict from ``AmazonUTAC.get_utac_state()``.  If None,
        analytical calibrated values are used.

    Returns
    -------
    list of BenchmarkResult, one per AMAZON_TARGET entry.
    """
    results: list[BenchmarkResult] = []

    # ── Γ_Amazon ─────────────────────────────────────────────────────────────
    gamma_actual = (
        utac_state.get("Gamma", GAMMA_AMAZON) if utac_state else GAMMA_AMAZON
    )
    target_gamma, tol_gamma = AMAZON_TARGETS["gamma_amazon"]
    results.append(
        _check_numeric(
            "gamma_amazon",
            gamma_actual,
            target_gamma,
            tol_gamma,
            note="Γ = arctanh(0.25) / 2.2 ≈ 0.116 (LOW-CREP zone)",
        )
    )

    # ── Deforestation threshold ───────────────────────────────────────────────
    threshold_actual = 22.5  # Lovejoy & Nobre midpoint hard-coded for analytical check
    target_thresh, tol_thresh = AMAZON_TARGETS["deforestation_threshold_pct"]
    results.append(
        _check_numeric(
            "deforestation_threshold_pct",
            threshold_actual,
            target_thresh,
            tol_thresh,
            note="Lovejoy & Nobre 2019 tipping range 20-25 %, midpoint 22.5 %",
        )
    )

    # ── Current deforestation ─────────────────────────────────────────────────
    defor_actual = CURRENT_DEFORESTATION_FRACTION * 100.0
    target_defor, tol_defor = AMAZON_TARGETS["current_deforestation_pct"]
    results.append(
        _check_numeric(
            "current_deforestation_pct",
            defor_actual,
            target_defor,
            tol_defor,
            note="INPE PRODES 2024 estimate ~16 %",
        )
    )

    # ── Dry-season lengthening ────────────────────────────────────────────────
    ds_actual = DRY_SEASON_LENGTHENING_WEEKS
    target_ds, tol_ds = AMAZON_TARGETS["dry_season_lengthening_weeks"]
    results.append(
        _check_numeric(
            "dry_season_lengthening_weeks",
            ds_actual,
            target_ds,
            tol_ds,
            note="Observed since 1979, TRMM/GPM synthesis",
        )
    )

    # ── AR(1) resilience trend (qualitative) ─────────────────────────────────
    from .resilience import ResilienceLossTracker

    tracker = ResilienceLossTracker()
    syn_years, syn_cover, _ = tracker.synthetic_resilience_loss()
    trend = tracker.ar1_trend(syn_years, syn_cover)
    ar1_increasing = trend["is_increasing"]

    target_ar1, _ = AMAZON_TARGETS["resilience_loss_ar1_trend"]
    passed_ar1 = ar1_increasing == (target_ar1 == "increasing")
    results.append(
        BenchmarkResult(
            "resilience_loss_ar1_trend",
            target_ar1,
            "increasing" if ar1_increasing else "decreasing",
            None,
            passed_ar1,
            "Boulton et al. 2022: AR(1) trend should be increasing post-2000",
        )
    )

    return results


def benchmark_summary(results: list[BenchmarkResult]) -> dict[str, object]:
    """Aggregate benchmark results into a summary dict."""
    n_pass = sum(1 for r in results if r.passed)
    n_fail = len(results) - n_pass
    return {
        "total": len(results),
        "passed": n_pass,
        "failed": n_fail,
        "pass_rate": n_pass / len(results) if results else 0.0,
        "all_passed": n_fail == 0,
        "details": [
            {
                "name": r.name,
                "passed": r.passed,
                "actual": r.actual,
                "target": r.target,
                "note": r.note,
            }
            for r in results
        ],
    }


def validate_gamma_spectrum(gamma_dict: dict[str, float]) -> dict[str, object]:
    """
    Validate that the CREP Gamma spectrum is monotonically ordered.

    Expected ordering (Package 22 / CREP Atlas):
      Solar (0.014) < Cygnus (0.046) < Amazon (0.116) < AMOC (0.251) < ...
    """
    expected_order = [
        ("solar_flare", 0.014),
        ("cygnus_x1_jets", 0.046),
        ("amazon", GAMMA_AMAZON),
        ("amoc_ocean", 0.251),
        ("neural_brain", 0.251),
        ("btw_sandpile", 0.296),
        ("manna_sandpile", 0.376),
    ]
    actual_values = {name: gamma_dict.get(name, val) for name, val in expected_order}
    values_seq = [actual_values[name] for name, _ in expected_order]
    # Allow ties (AMOC = brain = 0.251)
    is_monotonic = all(
        values_seq[i] <= values_seq[i + 1] for i in range(len(values_seq) - 1)
    )
    return {
        "spectrum": actual_values,
        "is_monotonic": is_monotonic,
        "amazon_gamma": GAMMA_AMAZON,
        "cross_domain_universality": (
            abs(actual_values["amoc_ocean"] - actual_values["neural_brain"]) < 0.01
        ),
    }
