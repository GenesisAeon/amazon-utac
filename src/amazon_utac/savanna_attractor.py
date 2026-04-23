"""
Dual-attractor landscape for the Amazon forest–savanna transition.

Implements the double-well potential V(H) and the associated UTAC ODE with
two stable fixed points (forest and savanna) separated by an unstable saddle.
Maps directly to Φ(H) in the GenesisAeon Unified Lagrangian.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    CREP_SIGMA,
    H_FOREST_ATTRACTOR,
    H_SADDLE,
    H_SAVANNA_ATTRACTOR,
    R_FOREST,
)


class DualAttractorLandscape:
    """
    Double-well potential V(H) for the Amazon forest–savanna system.

    Two stable attractors:
      - Forest:  H_forest  ≈ 0.80
      - Savanna: H_savanna ≈ 0.15

    Separated by an unstable saddle at H_saddle ≈ 0.50.

    The potential barrier between them is modulated by Γ_degrade: as
    deforestation + drought pressure increases, the saddle point drifts
    upward toward H_forest, eventually destroying the forest attractor
    in a fold (saddle-node) bifurcation — the tipping point.
    """

    def __init__(
        self,
        H_s: float = H_SAVANNA_ATTRACTOR,
        H_saddle0: float = H_SADDLE,
        H_f: float = H_FOREST_ATTRACTOR,
        r: float = R_FOREST,
        sigma: float = CREP_SIGMA,
    ) -> None:
        self.H_s = H_s
        self.H_saddle0 = H_saddle0
        self.H_f = H_f
        self.r = r
        self.sigma = sigma

    # ── Effective saddle point ───────────────────────────────────────────────

    def effective_saddle(self, gamma: float) -> float:
        """
        Deforestation-shifted saddle point.

        H_saddle(Γ) = H_saddle0 + (H_f − H_saddle0) · tanh(σ · Γ)

        As Γ increases, H_saddle → H_f and the forest attractor shrinks.
        At the tipping point, H_saddle = H_f → bifurcation.
        """
        shift = (self.H_f - self.H_saddle0) * np.tanh(self.sigma * gamma)
        return self.H_saddle0 + shift

    def tipping_gamma(self) -> float:
        """
        Critical Γ at which H_saddle = H_f (forest attractor destroyed).

        Solving: H_saddle0 + (H_f − H_saddle0)·tanh(σ·Γ_tip) = H_f
          → tanh(σ·Γ_tip) = 1  →  Γ_tip → ∞
        In practice the saddle reaches 95 % of H_f at:
          tanh(σ·Γ_95) = 0.95  →  Γ_95 = arctanh(0.95) / σ
        """
        return float(np.arctanh(0.95)) / self.sigma

    # ── Double-well potential ────────────────────────────────────────────────

    def potential(self, H: np.ndarray, gamma: float = CREP_SIGMA * 0.0) -> np.ndarray:
        """
        V(H; Γ) = ∫₀ᴴ dH/dt dH  (the negative potential landscape).

        dH/dt = −r · (H − H_s) · (H − H_saddle(Γ)) · (H − H_f)

        The potential is computed by numerical integration over a fine grid.

        Parameters
        ----------
        H:
            Array of forest cover values at which to evaluate V.
        gamma:
            CREP Γ_degrade value (shifts saddle point).
        """
        H_arr = np.asarray(H, dtype=float)
        H_saddle_eff = self.effective_saddle(gamma)

        # dV/dH = r · (H − H_s)(H − H_saddle)(H − H_f)
        # This is the force on H — negative of what drives H
        dV_dH = self.r * (H_arr - self.H_s) * (H_arr - H_saddle_eff) * (H_arr - self.H_f)

        # Integrate dV/dH from H_s to H using cumulative trapezoid
        # Sort then integrate for a smooth curve
        sort_idx = np.argsort(H_arr)
        H_sorted = H_arr[sort_idx]
        dV_sorted = dV_dH[sort_idx]

        V_sorted = np.zeros_like(H_sorted)
        V_sorted[1:] = np.cumsum(
            0.5 * (dV_sorted[:-1] + dV_sorted[1:]) * np.diff(H_sorted)
        )
        # Unsort
        V = np.empty_like(H_arr)
        V[sort_idx] = V_sorted
        return V

    # ── UTAC ODE ─────────────────────────────────────────────────────────────

    def ode(self, _t: float, H: float, gamma: float) -> float:
        """
        UTAC ODE for the Amazon forest cover.

        dH/dt = −r · (H − H_s)(H − H_saddle(Γ))(H − H_f)

        This is the negative gradient of the double-well potential, giving:
        - Restoring force toward H_f when H is slightly below H_f
        - Restoring force toward H_s when H is slightly above H_s
        - Repulsive force away from H_saddle (unstable equilibrium)
        """
        H_saddle_eff = self.effective_saddle(gamma)
        return -self.r * (H - self.H_s) * (H - H_saddle_eff) * (H - self.H_f)

    # ── Basin analysis ───────────────────────────────────────────────────────

    def basin_widths(self, gamma: float) -> dict[str, float]:
        """
        Width of the forest and savanna attraction basins at Γ.

        Forest basin:  [H_saddle(Γ), 1.0]
        Savanna basin: [0.0, H_saddle(Γ)]
        """
        hs = self.effective_saddle(gamma)
        return {
            "forest_basin_width": max(0.0, 1.0 - hs),
            "savanna_basin_width": max(0.0, hs),
            "effective_saddle": hs,
            "forest_basin_fraction": max(0.0, 1.0 - hs),
        }

    def is_in_forest_basin(self, H: float, gamma: float) -> bool:
        """True if H is in the forest attraction basin at this Γ."""
        return self.effective_saddle(gamma) < H

    def barrier_height(self, gamma: float) -> float:
        """
        Potential energy barrier separating forest from savanna attractor.

        A shrinking barrier → higher tipping risk.  At the tipping point,
        barrier_height → 0.
        """
        H_saddle_eff = self.effective_saddle(gamma)
        H_grid = np.array([self.H_s, H_saddle_eff, self.H_f])
        V = self.potential(H_grid, gamma)
        # Barrier = V(saddle) − V(forest)
        return float(V[1] - V[2])

    # ── Summary ──────────────────────────────────────────────────────────────

    def landscape_summary(self, gamma: float) -> dict[str, float]:
        """Return key landscape metrics at a given Γ."""
        basins = self.basin_widths(gamma)
        return {
            "gamma": gamma,
            "H_savanna_attractor": self.H_s,
            "H_forest_attractor": self.H_f,
            "H_saddle_effective": basins["effective_saddle"],
            "forest_basin_width": basins["forest_basin_width"],
            "savanna_basin_width": basins["savanna_basin_width"],
            "barrier_height": self.barrier_height(gamma),
            "tipping_gamma_95pct": self.tipping_gamma(),
        }
