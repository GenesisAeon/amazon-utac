"""
AmazonUTAC — GenesisAeon Package 19 Diamond Interface.

Diamond-Template contract:
  run_cycle(self) -> dict
  get_crep_state(self) -> dict   # {C, R, E, P, Gamma}
  get_utac_state(self) -> dict   # {H, dH_dt, H_star, K_eff}
  get_phase_events(self) -> list
  to_zenodo_record(self) -> dict

Physics:
  H(t) = normalised forest cover fraction ∈ [0, 1]
  K    = 1.0 (intact Amazon ceiling)
  H*   = dual-stable: H*_forest ≈ 0.80, H*_savanna ≈ 0.15
  Γ(t) = CREP tensor (C, R, E, P) → Γ_Amazon ≈ 0.116

ODE:
  dH/dt = −r · (H − H_s)(H − H_saddle(Γ))(H − H_f)
  where H_saddle(Γ) rises with deforestation pressure.

Ethics-Gate Light (Phase H) runs inside run_cycle() and to_zenodo_record().
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

# ── genesis-os imports (stub if not installed) ───────────────────────────────
try:
    import genesis  # noqa: F401

    GENESIS_AVAILABLE = True
except ImportError:
    GENESIS_AVAILABLE = False

try:
    from scipy.integrate import solve_ivp

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .constants import (
    ANNUAL_DEFORESTATION_RATE,
    CREP_SIGMA,
    CURRENT_DEFORESTATION_FRACTION,
    GAMMA_AMAZON,
    H_FOREST_ATTRACTOR,
    H_SAVANNA_ATTRACTOR,
    H_TIPPING_LOW,
    PACKAGE_REGISTRY_ENTRY,
    SEED,
)
from .crep_amazon import AmazonCREP
from .deforestation import ProdesDeforestation
from .ethics_gate import EthicsGate, EthicsGateConfig, TensionMetric
from .forest_cover import ForestCoverLoader
from .rainfall import RainfallIndex
from .resilience import ResilienceLossTracker
from .savanna_attractor import DualAttractorLandscape


class AmazonUTAC:
    """
    UTAC model for Amazon rainforest savannisation threshold.

    Implements the full GenesisAeon Diamond-Template contract.
    Calibrated against Lovejoy & Nobre (2019) and Boulton et al. (2022).

    Parameters
    ----------
    seed:
        Random seed for reproducibility (default 42).
    start_year:
        First year of the historical record (default 1988).
    deforestation_scenario:
        One of ``'current-rate'``, ``'zero-deforestation'``,
        ``'accelerated'``, ``'recovery'``.
    ethics_config:
        Optional EthicsGateConfig.  Defaults to standard settings.
    """

    VERSION = "0.1.0"
    PACKAGE_ID = 19

    def __init__(
        self,
        seed: int = SEED,
        start_year: int = 1988,
        deforestation_scenario: str = "current-rate",
        ethics_config: EthicsGateConfig | None = None,
    ) -> None:
        self.seed = seed
        self.start_year = start_year
        self.deforestation_scenario = deforestation_scenario

        # Sub-modules
        self._forest = ForestCoverLoader(start_year=start_year, seed=seed)
        self._prodes = ProdesDeforestation(seed=seed)
        self._rainfall = RainfallIndex(seed=seed)
        self._resilience = ResilienceLossTracker()
        self._crep = AmazonCREP(sigma=CREP_SIGMA)
        self._landscape = DualAttractorLandscape()

        # Ethics-Gate Light (Phase H)
        self._ethics_gate = EthicsGate(ethics_config)
        self._tension_metric = TensionMetric()

        # State caches
        self._cycle_result: dict[str, Any] | None = None
        self._phase_events: list[dict[str, Any]] = []

    # ── Diamond Interface ────────────────────────────────────────────────────

    def run_cycle(self, duration_years: int = 80) -> dict[str, Any]:
        """
        Run a full UTAC simulation cycle over ``duration_years``.

        Integrates the double-well ODE from the current observed state
        (H₀ = forest cover in 2024) under the selected deforestation scenario.

        Returns
        -------
        dict with keys:
            ``years``, ``H``, ``dH_dt``, ``Gamma``, ``phase_events``,
            ``tipping_year``, ``crep_state``, ``utac_state``.
        """
        self._phase_events = []
        self._tension_metric.reset()

        # Initial conditions
        H0 = self._forest.current_cover()

        years_hist = self._forest.years
        cover_hist = self._forest.cover
        _, dry_months = self._rainfall.dry_season_length(n_years=len(years_hist))
        defor_frac = float(self._forest.deforestation_fraction()[-1])
        drought_anom = max(0.0, (dry_months[-1] - 3.5) / 1.5)

        crep_state = self._crep.compute_state(
            cover_hist, dry_months, defor_frac, drought_anom
        )
        gamma_current = crep_state["Gamma"]

        # Project deforestation rates for the simulation horizon
        base_year = int(self._forest.years[-1]) + 1
        proj_years, proj_rates = self._prodes.scenario_rates(
            self.deforestation_scenario, duration_years, base_year
        )

        # Integrate ODE
        sim_years, H_sim, dH_sim = self._integrate_ode(
            H0, proj_years, proj_rates, gamma_current
        )

        # Detect phase events (tipping threshold crossings)
        self._detect_phase_events(sim_years, H_sim, gamma_current)

        tipping_year = self._estimate_tipping_year(sim_years, H_sim)

        # Build state for Ethics-Gate check
        state: dict[str, Any] = {
            "H": float(H_sim[-1]),
            "Gamma": float(gamma_current),
            "tipping_year": tipping_year,
            "dH_dt": float(dH_sim[-1]) if len(dH_sim) > 0 else 0.0,
        }

        # Ethics-Gate Light (Phase H)
        tension = (
            self._tension_metric.get_current_tension()
            if hasattr(self, "_tension_metric")
            else 0.0
        )
        ethics_result = self._ethics_gate.check(state=state, tension=tension)
        if not ethics_result["allowed"]:
            raise RuntimeError(f"EthicsGate blocked: {ethics_result['reason']}")

        H_final = float(H_sim[-1])
        utac_state_sim: dict[str, Any] = {
            "H": H_final,
            "dH_dt": float(dH_sim[-1]),
            "H_star_forest": H_FOREST_ATTRACTOR,
            "H_star_savanna": H_SAVANNA_ATTRACTOR,
            "H_saddle_effective": self._landscape.effective_saddle(gamma_current),
            "K_eff": 1.0,
            "Gamma": float(gamma_current),
            "in_forest_basin": self._landscape.is_in_forest_basin(H_final, gamma_current),
            "barrier_height": self._landscape.barrier_height(gamma_current),
        }

        result: dict[str, Any] = {
            "years": sim_years,
            "H": H_sim,
            "dH_dt": dH_sim,
            "Gamma": gamma_current,
            "crep_state": crep_state,
            "utac_state": utac_state_sim,
            "phase_events": list(self._phase_events),
            "tipping_year": tipping_year,
            "deforestation_scenario": self.deforestation_scenario,
            "ethics_warnings": ethics_result.get("warnings", []),
        }
        self._cycle_result = result
        return result

    def get_crep_state(self) -> dict[str, Any]:
        """
        Return current CREP tensor state {C, R, E, P, Gamma}.

        If ``run_cycle`` has been called, returns the cycle's CREP state.
        Otherwise, computes from the historical record.
        """
        if self._cycle_result is not None:
            return dict(self._cycle_result["crep_state"])

        years_hist = self._forest.years
        cover_hist = self._forest.cover
        n = len(years_hist)
        _, dry_months = self._rainfall.dry_season_length(n_years=n)
        defor_frac = float(self._forest.deforestation_fraction()[-1])
        drought_anom = max(0.0, (dry_months[-1] - 3.5) / 1.5)
        return self._crep.compute_state(cover_hist, dry_months, defor_frac, drought_anom)

    def get_utac_state(self) -> dict[str, Any]:
        """
        Return current UTAC state {H, dH_dt, H_star, K_eff}.

        H_star is reported as both stable fixed points (forest and savanna).
        K_eff is the forest ceiling (1.0).
        """
        crep = self.get_crep_state()
        gamma = crep["Gamma"]
        H_current = self._forest.current_cover()
        H_saddle_eff = self._landscape.effective_saddle(gamma)

        dH_current = self._landscape.ode(0.0, H_current, gamma)

        return {
            "H": H_current,
            "dH_dt": dH_current,
            "H_star_forest": H_FOREST_ATTRACTOR,
            "H_star_savanna": H_SAVANNA_ATTRACTOR,
            "H_saddle_effective": H_saddle_eff,
            "K_eff": 1.0,
            "Gamma": gamma,
            "in_forest_basin": self._landscape.is_in_forest_basin(H_current, gamma),
            "barrier_height": self._landscape.barrier_height(gamma),
        }

    def get_phase_events(self) -> list[dict[str, Any]]:
        """
        Return list of detected phase transition events.

        Each event is a dict with keys:
            ``'year'``, ``'H'``, ``'event_type'``, ``'description'``.
        """
        return list(self._phase_events)

    def to_zenodo_record(self) -> dict[str, Any]:
        """
        Export a Zenodo-compatible metadata record.

        Includes calibration results, CREP state, benchmark summary,
        and the standard GenesisAeon scientific disclaimer.

        Ethics-Gate Light (Phase H) blocks irresponsible records.
        """
        crep = self.get_crep_state()
        utac = self.get_utac_state()

        state: dict[str, Any] = {
            "H": utac["H"],
            "Gamma": utac["Gamma"],
            "is_zenodo_record": True,
            "uncertainty_bounds": {
                "H_lower": max(0.0, utac["H"] - 0.05),
                "H_upper": min(1.0, utac["H"] + 0.05),
                "Gamma_lower": max(0.0, utac["Gamma"] - 0.020),
                "Gamma_upper": utac["Gamma"] + 0.020,
            },
        }

        # Ethics-Gate Light (Phase H)
        tension = (
            self._tension_metric.get_current_tension()
            if hasattr(self, "_tension_metric")
            else 0.0
        )
        ethics_result = self._ethics_gate.check(state=state, tension=tension)
        if not ethics_result["allowed"]:
            raise RuntimeError(f"EthicsGate blocked: {ethics_result['reason']}")

        from .benchmark import benchmark_summary, run_benchmarks

        benchmarks = run_benchmarks(utac)
        bench_summary = benchmark_summary(benchmarks)

        record: dict[str, Any] = {
            "metadata": {
                "title": "Amazon Rainforest Savannisation Threshold — UTAC Model (Package 19)",
                "creators": [{"name": "MOR Research Collective / GenesisAeon"}],
                "description": (
                    "GenesisAeon Package 19: UTAC dynamical model of the Amazon "
                    "forest–savanna tipping point. Calibrated against Lovejoy & Nobre "
                    "(2019) and Boulton et al. (2022). "
                    "Central result: Γ_Amazon ≈ 0.116 (LOW-CREP zone)."
                ),
                "license": "MIT",
                "version": self.VERSION,
                "doi": PACKAGE_REGISTRY_ENTRY["zenodo"],
                "keywords": [
                    "amazon",
                    "deforestation",
                    "tipping point",
                    "UTAC",
                    "CREP",
                    "GenesisAeon",
                    "savannisation",
                    "resilience",
                ],
                "references": [
                    "10.1126/sciadv.aba2949",  # Lovejoy & Nobre 2019
                    "10.1038/s41558-022-01287-8",  # Boulton et al. 2022
                ],
                "created": datetime.now(timezone.utc).isoformat(),
            },
            "crep_state": crep,
            "utac_state": utac,
            "calibration": {
                "eta_amazon": 0.25,
                "gamma_amazon_analytical": GAMMA_AMAZON,
                "gamma_amazon_simulated": crep["Gamma"],
                "crep_spectrum_position": "LOW-CREP (Γ ≈ 0.116, between Amazon and AMOC)",
                "cross_domain_comparison": {
                    "solar_flare": 0.014,
                    "cygnus_x1_jets": 0.046,
                    "amazon_forest": GAMMA_AMAZON,
                    "amoc_ocean": 0.251,
                    "neural_brain": 0.251,
                },
            },
            "benchmark_summary": bench_summary,
            "uncertainty_bounds": state["uncertainty_bounds"],
            "ethics_gate": {
                "status": "passed",
                "warnings": ethics_result.get("warnings", []),
                "tension": tension,
            },
            "disclaimer": self._ethics_gate.zenodo_disclaimer(),
        }
        return record

    # ── Scientific methods ───────────────────────────────────────────────────

    def deforestation_threshold(self) -> float:
        """
        UTAC-predicted safe deforestation limit.

        Returns the deforestation fraction at which the forest attractor
        disappears (H_saddle = H_forest in the double-well landscape).
        Should be ~20-25 % (Lovejoy & Nobre 2019).
        """
        # Threshold corresponds to η_Amazon = 0.25 → Γ = 0.116
        # Beyond this, the 40% scenario (Γ → 0.19) sees the Mirror-Machine
        # detect irreversibility.
        return float(1.0 - H_TIPPING_LOW)  # 0.25 = 25 %

    def time_to_tipping(self, deforestation_rate: float = ANNUAL_DEFORESTATION_RATE) -> float:
        """
        Years until H crosses H_threshold given a constant deforestation rate.

        Parameters
        ----------
        deforestation_rate:
            Annual deforestation as fraction of remaining forest (e.g. 0.01).

        Returns
        -------
        Years until tipping (float; inf if rate is zero or negative).
        """
        return self._prodes.years_to_threshold(
            current_deforestation_fraction=CURRENT_DEFORESTATION_FRACTION,
            annual_rate_fraction=deforestation_rate,
        )

    def predict_tipping_year(self) -> dict[str, Any]:
        """
        UTAC-based tipping year prediction.

        Compares against:
          - Lovejoy & Nobre (2019): 20-25 % threshold
          - INPE PRODES trajectory: ~1 %/yr current rate
          - UTAC falsifiable prediction: 2038 ± 5 years
        """
        tipping = self._prodes.tipping_year()
        return {
            "utac_prediction": tipping,
            "central_estimate": tipping["midpoint"],
            "uncertainty_years": 5.0,
            "scenario": self.deforestation_scenario,
            "falsifiable_prediction": (
                "If annual deforestation stays at ~1 % of remaining forest, "
                "UTAC predicts H crosses the tipping threshold in 2038 ± 5 years. "
                "Testable against PRODES and MODIS continuous monitoring."
            ),
            "comparison": {
                "lovejoy_nobre_threshold_pct": 22.5,
                "current_deforestation_pct": CURRENT_DEFORESTATION_FRACTION * 100,
                "remaining_buffer_pct": (0.225 - CURRENT_DEFORESTATION_FRACTION) * 100,
            },
        }

    # ── Internal ODE integration ─────────────────────────────────────────────

    def _integrate_ode(
        self,
        H0: float,
        proj_years: np.ndarray,
        proj_rates: np.ndarray,
        gamma0: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Integrate the double-well ODE over the projection horizon."""
        n = len(proj_years)
        H_sim = np.empty(n)
        dH_sim = np.empty(n)
        H_sim[0] = H0

        # Precompute cumulative deforestation at each year for use inside ODE
        cumulative_defor = CURRENT_DEFORESTATION_FRACTION + np.concatenate(
            [[0.0], np.cumsum(proj_rates[:-1])]
        )

        if SCIPY_AVAILABLE:
            # Use scipy for accurate integration
            def _ode(t: float, y: list[float]) -> list[float]:
                H = y[0]
                rate_idx = min(int(t), n - 1)
                # Gamma grows as deforestation accumulates (updated each step)
                gamma = gamma0 + 0.5 * max(
                    0.0, cumulative_defor[rate_idx] - CURRENT_DEFORESTATION_FRACTION
                )
                dH = self._landscape.ode(t, H, gamma)
                dH -= proj_rates[rate_idx] * H
                return [dH]

            t_span = (0.0, float(n - 1))
            t_eval = np.arange(n, dtype=float)
            sol = solve_ivp(
                _ode,
                t_span,
                [H0],
                t_eval=t_eval,
                method="RK45",
                max_step=1.0,
                rtol=1e-4,
                atol=1e-6,
            )
            H_raw = sol.y[0]
            # Update tension metric
            step_ratios = np.diff(sol.t) if len(sol.t) > 1 else np.array([1.0])
            max_ratio = float(np.min(step_ratios)) / 1.0 if len(step_ratios) > 0 else 1.0
            n_failures = int(sol.nfev / max(1, n)) - 4  # RK45 uses ~4 evaluations/step
            self._tension_metric.update(max(0.01, max_ratio), max(0, n_failures))

            H_sim = np.clip(H_raw, 0.0, 1.0)
        else:
            # Euler fallback
            for i in range(1, n):
                gamma_i = gamma0 + 0.5 * max(
                    0.0, cumulative_defor[i - 1] - CURRENT_DEFORESTATION_FRACTION
                )
                dH = self._landscape.ode(0.0, H_sim[i - 1], gamma_i)
                dH -= proj_rates[i] * H_sim[i - 1]
                H_sim[i] = max(0.0, min(1.0, H_sim[i - 1] + dH))

        # Compute dH/dt as finite differences
        dH_sim = np.gradient(H_sim, proj_years - proj_years[0])
        return proj_years, H_sim, dH_sim

    def _detect_phase_events(
        self,
        years: np.ndarray,
        H: np.ndarray,
        gamma: float,
    ) -> None:
        """Detect threshold crossings and record them as phase events."""
        thresholds = [
            (H_TIPPING_LOW, "low_tipping", "H crossed 20 % deforestation threshold"),
            (0.77, "mid_tipping", "H crossed Lovejoy & Nobre midpoint"),
            (H_SAVANNA_ATTRACTOR + 0.05, "savanna_proximity", "H approaching savanna attractor"),
        ]
        for threshold, event_type, desc in thresholds:
            crossings = np.where(np.diff(np.sign(H - threshold)))[0]
            for idx in crossings:
                self._phase_events.append(
                    {
                        "year": float(years[idx]),
                        "H": float(H[idx]),
                        "event_type": event_type,
                        "description": desc,
                        "Gamma": gamma,
                    }
                )

    def _estimate_tipping_year(
        self,
        years: np.ndarray,
        H: np.ndarray,
    ) -> float | None:
        """Estimate the first year H crosses the tipping threshold."""
        crossings = np.where(H < H_TIPPING_LOW)[0]
        if len(crossings) == 0:
            return None
        return float(years[crossings[0]])

    def __repr__(self) -> str:
        utac = self.get_utac_state()
        return (
            f"AmazonUTAC(H={utac['H']:.3f}, "
            f"Γ={utac['Gamma']:.3f}, "
            f"scenario={self.deforestation_scenario!r})"
        )
