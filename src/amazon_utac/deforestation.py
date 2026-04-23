"""PRODES-calibrated annual deforestation rate model and scenario engine."""

from __future__ import annotations

import numpy as np

from .constants import (
    ANNUAL_DEFORESTATION_RATE,
    DEFORESTATION_THRESHOLD_HIGH,
    DEFORESTATION_THRESHOLD_LOW,
    PRODES_ORIGINAL_AREA_KM2,
    SEED,
)

# Historical PRODES-derived rates (km²/yr), selected key years
# Source: INPE PRODES, publicly available annual reports
_PRODES_HISTORICAL: dict[int, float] = {
    1988: 21_050,
    1994: 14_896,
    1995: 29_059,
    2000: 18_226,
    2004: 27_772,  # peak
    2005: 19_014,
    2009: 7_464,   # post-PPCDAm minimum
    2012: 4_571,
    2016: 7_893,
    2019: 11_088,
    2020: 11_568,
    2021: 13_038,
    2022: 11_568,
    2023: 11_568,
}


class ProdesDeforestation:
    """
    Deforestation rate model calibrated to INPE PRODES annual data.

    Provides:
    - Historical observed rates (km²/yr and fraction/yr)
    - Smooth interpolated rate series from 1988–2024
    - Scenario projections for 2024–2100
    - Conversion between km², fraction of original, and ΔH
    """

    def __init__(self, seed: int = SEED) -> None:
        self.rng = np.random.default_rng(seed)
        self._historical = _PRODES_HISTORICAL.copy()

    # ── Observed data ────────────────────────────────────────────────────────

    def observed_km2(self) -> dict[int, float]:
        """Return observed deforestation rates [km²/yr] for key PRODES years."""
        return self._historical.copy()

    def rate_series(
        self, start: int = 1988, end: int = 2024
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Smoothly interpolated annual deforestation rate series.

        Returns
        -------
        years, rates:
            Year array and deforestation rate [km²/yr] for each year.
        """
        ref_years = np.array(sorted(self._historical.keys()), dtype=float)
        ref_rates = np.array([self._historical[y] for y in ref_years.astype(int)])
        years = np.arange(start, end + 1, dtype=float)
        rates = np.interp(years, ref_years, ref_rates)
        return years, rates

    def fraction_per_year(
        self, start: int = 1988, end: int = 2024
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Annual deforestation as fraction of original Amazon area.

        Returns
        -------
        years, fractions:
            Year array and deforestation rate [fraction/yr].
        """
        years, rates_km2 = self.rate_series(start, end)
        return years, rates_km2 / PRODES_ORIGINAL_AREA_KM2

    def cumulative_deforestation(
        self, start: int = 1988, end: int = 2024
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Cumulative deforestation fraction from ``start`` to ``end``.

        Returns
        -------
        years, cumulative_fraction
        """
        years, fracs = self.fraction_per_year(start, end)
        cumulative = np.cumsum(fracs)
        return years, cumulative

    # ── Rate → UTAC ΔH conversion ───────────────────────────────────────────

    def delta_H_per_year(
        self, start: int = 1988, end: int = 2024
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Annual change in UTAC state variable H = forest cover fraction.

        dH/dt = −rate_fraction  (H decreases as forests are cleared)
        """
        years, fracs = self.fraction_per_year(start, end)
        return years, -fracs

    # ── Threshold analysis ───────────────────────────────────────────────────

    def years_to_threshold(
        self,
        current_deforestation_fraction: float = 0.16,
        annual_rate_fraction: float = ANNUAL_DEFORESTATION_RATE,
        threshold: float = DEFORESTATION_THRESHOLD_LOW,
    ) -> float:
        """
        Years until cumulative deforestation exceeds the Lovejoy & Nobre
        tipping threshold, given a constant annual rate.

        Parameters
        ----------
        current_deforestation_fraction:
            Current cumulative deforestation as fraction of original area.
        annual_rate_fraction:
            Annual deforestation rate as fraction of remaining forest.
        threshold:
            Tipping threshold (default: 20 % = Lovejoy & Nobre lower bound).
        """
        if current_deforestation_fraction >= threshold:
            return 0.0

        remaining = threshold - current_deforestation_fraction
        forest_fraction = 1.0 - current_deforestation_fraction
        # Solve: remaining = annual_rate * forest * t  (linear approx)
        if annual_rate_fraction <= 0:
            return float("inf")
        return remaining / (annual_rate_fraction * forest_fraction)

    def tipping_year(
        self,
        base_year: int = 2024,
        current_deforestation_fraction: float = 0.16,
        annual_rate_fraction: float = ANNUAL_DEFORESTATION_RATE,
    ) -> dict[str, float]:
        """
        UTAC tipping year estimates under current deforestation trajectory.

        Returns
        -------
        dict with keys ``'lower'``, ``'midpoint'``, ``'upper'``
        corresponding to the 20 %, 22.5 %, and 25 % thresholds.
        """
        return {
            "lower": base_year
            + self.years_to_threshold(
                current_deforestation_fraction,
                annual_rate_fraction,
                DEFORESTATION_THRESHOLD_LOW,
            ),
            "midpoint": base_year
            + self.years_to_threshold(
                current_deforestation_fraction,
                annual_rate_fraction,
                (DEFORESTATION_THRESHOLD_LOW + DEFORESTATION_THRESHOLD_HIGH) / 2,
            ),
            "upper": base_year
            + self.years_to_threshold(
                current_deforestation_fraction,
                annual_rate_fraction,
                DEFORESTATION_THRESHOLD_HIGH,
            ),
        }

    # ── Scenario projections ─────────────────────────────────────────────────

    def scenario_rates(
        self,
        scenario: str = "current-rate",
        horizon_years: int = 76,
        base_year: int = 2024,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Annual deforestation rates under different policy scenarios.

        Scenarios
        ---------
        ``'current-rate'``
            1 %/yr of remaining forest, continuing indefinitely.
        ``'zero-deforestation'``
            Paris-aligned zero deforestation from base_year.
        ``'accelerated'``
            2 %/yr of remaining forest (pessimistic).
        ``'recovery'``
            Net negative — reforestation at 0.5 %/yr.
        """
        years = np.arange(base_year, base_year + horizon_years, dtype=float)
        rate_map = {
            "current-rate": ANNUAL_DEFORESTATION_RATE,
            "zero-deforestation": 0.0,
            "accelerated": ANNUAL_DEFORESTATION_RATE * 2.0,
            "recovery": -ANNUAL_DEFORESTATION_RATE * 0.5,
        }
        if scenario not in rate_map:
            raise ValueError(f"Unknown scenario '{scenario}'. Choose from {list(rate_map)}")
        rates = np.full(horizon_years, rate_map[scenario])
        return years, rates

    def __repr__(self) -> str:
        return f"ProdesDeforestation(n_years={len(self._historical)})"
