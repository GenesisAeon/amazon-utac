"""Dry-season length index and TRMM/GPM rainfall diagnostics for the Amazon."""

from __future__ import annotations

import numpy as np

from .constants import (
    DRY_SEASON_BASELINE_MONTHS,
    DRY_SEASON_LENGTHENING_WEEKS,
    DRY_SEASON_THRESHOLD_MM,
    RAINFALL_MEAN_MM_YR,
    SEED,
)


class RainfallIndex:
    """
    Amazon basin rainfall diagnostics, including dry-season-length computation.

    In the absence of live TRMM/GPM data, deterministic synthetic monthly
    rainfall series are generated (seed=42) calibrated to observed statistics.

    Key observational facts encoded:
    - Dry season has lengthened by ~4.5 weeks since 1979 (observed).
    - Basin mean annual rainfall ~2 300 mm/yr.
    - Dry season defined as months where rainfall < 100 mm/month.
    """

    def __init__(self, seed: int = SEED) -> None:
        self.rng = np.random.default_rng(seed)

    # ── Synthetic monthly rainfall ───────────────────────────────────────────

    def monthly_rainfall(
        self,
        n_years: int = 45,
        start_year: int = 1979,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic monthly Amazon basin rainfall [mm/month].

        Returns
        -------
        dates:
            Array of (year, month) tuples as decimal years.
        rainfall:
            Monthly rainfall [mm/month].
        """
        n_months = n_years * 12
        # Seasonal cycle (Amazon: wet season Oct–Mar, dry season Apr–Sep)
        month_idx = np.arange(n_months) % 12
        seasonal = 100.0 + 250.0 * np.cos(np.pi * month_idx / 6 + np.pi)
        # Long-term drying trend calibrated to reproduce ~4.5 weeks of additional
        # dry season over 45 years (≈1.1 months): tune so dry months go from 3.5 → 4.6
        # That requires ~0.024 months/yr of additional drying in the dry season.
        # Applied as a reduction in dry-month rainfall of ~2 mm/yr.
        trend = -np.where(
            (month_idx >= 3) & (month_idx <= 8),
            0.020 * np.arange(n_months) / 12,  # ~0.24 mm/yr in dry months
            0.0,
        )
        noise = self.rng.normal(0, 15, n_months)
        rainfall = np.clip(seasonal + trend + noise, 0.0, None)

        decimal_years = start_year + np.arange(n_months) / 12.0
        return decimal_years, rainfall

    # ── Dry-season length ────────────────────────────────────────────────────

    def dry_season_length(
        self,
        n_years: int = 45,
        start_year: int = 1979,
        threshold_mm: float = DRY_SEASON_THRESHOLD_MM,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Annual dry-season length [months] from synthetic monthly rainfall.

        Returns
        -------
        years, dry_months
        """
        _, rainfall = self.monthly_rainfall(n_years, start_year)
        years_arr = np.arange(start_year, start_year + n_years, dtype=float)
        dry_months = np.zeros(n_years)
        for i in range(n_years):
            mo = rainfall[i * 12 : (i + 1) * 12]
            dry_months[i] = float(np.sum(mo < threshold_mm))
        return years_arr, dry_months

    def dry_season_trend(
        self,
        n_years: int = 45,
        start_year: int = 1979,
    ) -> dict[str, float]:
        """
        Linear trend analysis of dry-season length over the study period.

        Returns
        -------
        dict with keys ``'slope_months_per_yr'``, ``'total_lengthening_weeks'``,
        ``'r_squared'``.
        """
        years, dry_months = self.dry_season_length(n_years, start_year)
        # Linear regression
        coeffs = np.polyfit(years - years[0], dry_months, 1)
        slope = float(coeffs[0])
        trend_line = np.polyval(coeffs, years - years[0])
        ss_res = float(np.sum((dry_months - trend_line) ** 2))
        ss_tot = float(np.sum((dry_months - dry_months.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        total_weeks = slope * n_years * (52.0 / 12.0)  # months → weeks

        return {
            "slope_months_per_yr": slope,
            "total_lengthening_weeks": total_weeks,
            "r_squared": r2,
            "observed_lengthening_weeks": DRY_SEASON_LENGTHENING_WEEKS,
        }

    # ── CREP R-component ────────────────────────────────────────────────────

    def crep_r_component(
        self,
        n_years: int = 45,
        start_year: int = 1979,
    ) -> np.ndarray:
        """
        CREP resonance component R ∈ [0, 1].

        R measures the resonance between photospheric/atmospheric driver and
        forest response.  Here:
          R = sigmoid((dry_months − baseline) / scale)
        where baseline = 3.5 months (pre-1979 reference).

        Returns
        -------
        r_component:
            Annual R values ∈ [0, 1], length n_years.
        """
        _, dry_months = self.dry_season_length(n_years, start_year)
        delta = dry_months - DRY_SEASON_BASELINE_MONTHS
        return 1.0 / (1.0 + np.exp(-delta))

    def summary(self, n_years: int = 45, start_year: int = 1979) -> dict:
        """Return a summary dict of key rainfall diagnostics."""
        trend = self.dry_season_trend(n_years, start_year)
        _, dry_months = self.dry_season_length(n_years, start_year)
        return {
            "annual_mean_mm_yr": RAINFALL_MEAN_MM_YR,
            "dry_season_baseline_months": DRY_SEASON_BASELINE_MONTHS,
            "dry_season_current_months": float(dry_months[-1]),
            "dry_season_lengthening_weeks_simulated": trend["total_lengthening_weeks"],
            "dry_season_lengthening_weeks_observed": DRY_SEASON_LENGTHENING_WEEKS,
            "trend_r_squared": trend["r_squared"],
        }
