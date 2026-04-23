"""MODIS/PRODES forest cover loader and synthetic time-series generator."""

from __future__ import annotations

import numpy as np

from .constants import (
    ANNUAL_DEFORESTATION_RATE,
    K_FOREST,
    PRODES_ORIGINAL_AREA_KM2,
    SEED,
)


class ForestCoverLoader:
    """
    Generates and manages Amazon forest cover time series.

    In the absence of live MODIS/PRODES data this class produces
    deterministic synthetic series calibrated to observed statistics
    (seed=42).  When real data files are provided via ``load_csv``
    the synthetic series is replaced by the observations.

    H(t) is the normalised forest cover fraction ∈ [0, 1].
    H = 1 corresponds to the pre-1970 intact Amazon (reference state K).
    """

    def __init__(
        self,
        start_year: int = 1988,
        end_year: int = 2024,
        seed: int = SEED,
    ) -> None:
        self.start_year = start_year
        self.end_year = end_year
        self.rng = np.random.default_rng(seed)
        self._years: np.ndarray | None = None
        self._cover: np.ndarray | None = None

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def years(self) -> np.ndarray:
        if self._years is None:
            self._generate()
        return self._years  # type: ignore[return-value]

    @property
    def cover(self) -> np.ndarray:
        """Normalised forest cover H(t) ∈ [0, 1]."""
        if self._cover is None:
            self._generate()
        return self._cover  # type: ignore[return-value]

    def forest_fraction(self) -> np.ndarray:
        """Alias for ``cover`` — normalised forest cover H(t)."""
        return self.cover

    def deforestation_fraction(self) -> np.ndarray:
        """Cumulative deforestation = 1 − H(t)."""
        return 1.0 - self.cover

    def current_cover(self) -> float:
        """Most recent H value (end of series)."""
        return float(self.cover[-1])

    def cover_at(self, year: int) -> float:
        """Interpolated forest cover at a given year."""
        return float(np.interp(year, self.years, self.cover))

    def load_prodes_summary(self, deforested_km2_by_year: dict[int, float]) -> None:
        """
        Replace the synthetic series with PRODES-derived observations.

        Parameters
        ----------
        deforested_km2_by_year:
            Mapping from year → cumulative deforested area [km²].
        """
        years = np.array(sorted(deforested_km2_by_year.keys()), dtype=float)
        deforested = np.array([deforested_km2_by_year[y] for y in years.astype(int)])
        cover = 1.0 - deforested / PRODES_ORIGINAL_AREA_KM2
        self._years = years
        self._cover = np.clip(cover, 0.0, K_FOREST)

    # ── Synthetic generation ────────────────────────────────────────────────

    def _generate(self) -> None:
        """Build a deterministic synthetic forest-cover time series."""
        n = self.end_year - self.start_year + 1
        years = np.arange(self.start_year, self.end_year + 1, dtype=float)

        # PRODES: ~383 000 km² deforested by 1988 out of 4 153 741 km²
        # → H_1988 ≈ 0.908
        cover_1988 = 1.0 - 383_000.0 / PRODES_ORIGINAL_AREA_KM2  # ≈ 0.908

        # Target: H_2024 ≈ 0.840 (PRODES 664 450 km² cumulative)
        # Total decline over 36 years ≈ 0.068 → mean rate ≈ 0.0019/yr as fraction of K
        # Apply policy-era rates scaled to hit observed endpoint:
        #   pre-PPCDAm 1988-2004: higher rate (~60 % of total loss)
        #   2004-2019 slowdown: lower rate (~25 % of total loss)
        #   2019-2024 increase:  medium rate (~15 % of total loss)
        total_loss = cover_1988 - 0.840  # ≈ 0.068
        n_pre = max(1, 2004 - self.start_year)   # years before PPCDAm
        n_slow = max(1, 2019 - max(2004, self.start_year))  # slowdown years
        n_post = max(1, self.end_year - max(2019, self.start_year) + 1)  # post-2019

        rate_pre = (0.60 * total_loss) / (n_pre * cover_1988) if n_pre else 0.0
        rate_slow = (0.25 * total_loss) / (n_slow * cover_1988) if n_slow else 0.0
        rate_post = (0.15 * total_loss) / (n_post * cover_1988) if n_post else 0.0

        deforestation_rates = np.where(
            years < 2004,
            rate_pre,
            np.where(years < 2019, rate_slow, rate_post),
        )

        # Integrate cover decline
        cover = np.empty(n)
        cover[0] = cover_1988
        for i in range(1, n):
            noise = self.rng.normal(0, 0.0005)
            cover[i] = cover[i - 1] - deforestation_rates[i] * cover[i - 1] + noise
            cover[i] = np.clip(cover[i], 0.0, K_FOREST)

        self._years = years
        self._cover = cover

    # ── Scenario projections ────────────────────────────────────────────────

    def project(
        self,
        horizon_years: int = 80,
        scenario: str = "current-rate",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Project forest cover beyond the observed series.

        Parameters
        ----------
        horizon_years:
            Number of years to project into the future.
        scenario:
            One of ``'current-rate'``, ``'zero-deforestation'``,
            ``'accelerated'`` (2× current rate).

        Returns
        -------
        years, cover:
            Projected year array and normalised forest cover.
        """
        rate_map = {
            "current-rate": ANNUAL_DEFORESTATION_RATE,
            "zero-deforestation": 0.0,
            "accelerated": ANNUAL_DEFORESTATION_RATE * 2.0,
        }
        if scenario not in rate_map:
            raise ValueError(f"Unknown scenario '{scenario}'. Choose from {list(rate_map)}")

        rate = rate_map[scenario]
        start = self.end_year + 1
        proj_years = np.arange(start, start + horizon_years, dtype=float)
        proj_cover = np.empty(horizon_years)
        proj_cover[0] = self.current_cover()

        for i in range(1, horizon_years):
            noise = self.rng.normal(0, 0.001)
            proj_cover[i] = proj_cover[i - 1] - rate * proj_cover[i - 1] + noise
            proj_cover[i] = np.clip(proj_cover[i], 0.0, K_FOREST)

        return proj_years, proj_cover

    def __repr__(self) -> str:
        return (
            f"ForestCoverLoader(years={self.start_year}–{self.end_year}, "
            f"current_cover={self.current_cover():.3f})"
        )
