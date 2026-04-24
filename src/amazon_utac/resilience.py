"""
Resilience loss tracker — replicates Boulton et al. (2022) autocorrelation analysis.

Boulton, C.A., Lenton, T.M. & Boers, N. (2022).
"Pronounced loss of Amazon rainforest resilience since the early 2000s."
Nature Climate Change 12, 271-278. DOI: 10.1038/s41558-022-01287-8
"""

from __future__ import annotations

import numpy as np

from .constants import AR1_CRITICAL_THRESHOLD, AR1_TREND_WINDOW_YEARS, SEED


class ResilienceLossTracker:
    """
    Replicates Boulton et al. (2022) autocorrelation-based resilience loss.

    Computes the AR(1) coefficient on rolling windows of a forest-cover
    time series.  As the system approaches a critical transition, the AR(1)
    coefficient rises toward 1 (critical slowing down).

    This quantity maps to the **CREP C-component** (coherence): high AR(1)
    near 1 signals high spatial autocorrelation and low resilience.
    """

    def __init__(
        self,
        window_years: int = AR1_TREND_WINDOW_YEARS,
        seed: int = SEED,
    ) -> None:
        self.window_years = window_years
        self.rng = np.random.default_rng(seed)

    # ── AR(1) estimation ─────────────────────────────────────────────────────

    @staticmethod
    def ar1_coefficient(x: np.ndarray) -> float:
        """
        Harris (1963) / Pearson AR(1) estimator on a 1-D time series.

        AR(1): x[t] = φ·x[t-1] + ε[t]
        φ̂ = Σ(x[t]·x[t-1]) / Σ(x[t-1]²)
        """
        x = np.asarray(x, dtype=float)
        if len(x) < 3:
            return float("nan")
        x_c = x - x.mean()
        num = float(np.dot(x_c[1:], x_c[:-1]))
        denom = float(np.dot(x_c[:-1], x_c[:-1]))
        return num / denom if denom != 0 else float("nan")

    def rolling_ar1(
        self,
        years: np.ndarray,
        cover: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Rolling AR(1) coefficient over a moving window.

        Parameters
        ----------
        years:
            Year array (annual resolution).
        cover:
            Normalised forest cover H(t).

        Returns
        -------
        window_years, ar1_values
        """
        n = len(cover)
        w = self.window_years
        if n < w:
            raise ValueError(f"Series too short ({n}) for window={w} years.")
        out_years = years[w - 1 :]
        ar1 = np.array(
            [self.ar1_coefficient(cover[i : i + w]) for i in range(n - w + 1)]
        )
        return out_years, ar1

    # ── Trend in AR(1) ───────────────────────────────────────────────────────

    def ar1_trend(
        self,
        years: np.ndarray,
        cover: np.ndarray,
    ) -> dict[str, object]:
        """
        Linear trend in the rolling AR(1) coefficient.

        Returns
        -------
        dict with keys:
            ``'slope'``         – AR(1) change per year,
            ``'is_increasing'`` – True if slope > 0 (resilience loss),
            ``'final_ar1'``     – most recent AR(1) value,
            ``'r_squared'``     – goodness of fit,
            ``'near_critical'`` – True if final_ar1 > threshold.
        """
        w_years, ar1 = self.rolling_ar1(years, cover)
        valid = ~np.isnan(ar1)
        if valid.sum() < 3:
            return {
                "slope": float("nan"),
                "is_increasing": False,
                "final_ar1": float("nan"),
                "r_squared": float("nan"),
                "near_critical": False,
            }
        x = w_years[valid] - w_years[valid][0]
        y = ar1[valid]
        coeffs = np.polyfit(x, y, 1)
        slope = float(coeffs[0])
        trend_line = np.polyval(coeffs, x)
        ss_res = float(np.sum((y - trend_line) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        final_ar1 = float(ar1[valid][-1])

        return {
            "slope": slope,
            "is_increasing": slope > 0,
            "final_ar1": final_ar1,
            "r_squared": r2,
            "near_critical": final_ar1 > AR1_CRITICAL_THRESHOLD,
        }

    # ── CREP C-component ────────────────────────────────────────────────────

    def crep_c_component(
        self,
        years: np.ndarray,
        cover: np.ndarray,
    ) -> np.ndarray:
        """
        CREP coherence component C ∈ [0, 1].

        C = clip(AR(1), 0, 1) — directly maps the rolling AR(1) coefficient
        to the [0,1] CREP domain.  C → 1 signals critical slowing down.

        Returns
        -------
        c_component:
            Array aligned with ``rolling_ar1`` output (length n − window + 1).
        """
        _, ar1 = self.rolling_ar1(years, cover)
        return np.clip(ar1, 0.0, 1.0)

    # ── Synthetic Boulton-like series ────────────────────────────────────────

    def synthetic_resilience_loss(
        self,
        n_years: int = 35,
        start_year: int = 1990,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a synthetic forest cover series that reproduces the
        Boulton et al. (2022) resilience loss signature.

        The AR(1) trend is positive (increasing) after ~2000, replicating
        the observed signal of declining forest resilience.

        Returns
        -------
        years, cover, ar1_series
        """
        years = np.arange(start_year, start_year + n_years, dtype=float)
        # Slowly declining cover with increasing autocorrelation post-2000
        cover = np.empty(n_years)
        cover[0] = 0.84
        phi_base = 0.55
        mu = 0.84  # mean-reversion level, drifts slowly downward
        for i in range(1, n_years):
            # AR(1) parameter increases after 2000 (resilience loss)
            year = start_year + i
            phi = phi_base + max(0.0, (year - 2000) * 0.008)
            phi = min(phi, 0.97)
            noise = self.rng.normal(0, 0.005)
            drift = -0.001 - max(0.0, (year - 2000) * 0.0003)  # accelerating loss
            mu = max(0.0, mu + drift)
            # Mean-reverting AR(1): cover[i] = mu + phi*(cover[i-1] - mu) + noise
            cover[i] = mu + phi * (cover[i - 1] - mu) + noise
            cover[i] = np.clip(cover[i], 0.0, 1.0)

        _, ar1 = self.rolling_ar1(years, cover)
        # Pad to full length (NaN for the first window-1 entries)
        ar1_full = np.full(n_years, float("nan"))
        ar1_full[self.window_years - 1 :] = ar1
        return years, cover, ar1_full

    def summary(
        self, years: np.ndarray, cover: np.ndarray
    ) -> dict[str, object]:
        """Return a summary dict combining trend and current AR(1) state."""
        trend = self.ar1_trend(years, cover)
        return {
            "window_years": self.window_years,
            "ar1_trend": trend,
            "resilience_loss_detected": trend["is_increasing"],
        }
