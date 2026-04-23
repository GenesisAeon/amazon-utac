"""
Ethics-Gate Light — Phase H safety check for amazon-utac.

Implements the GenesisAeon Ethics-Gate Light pattern used in run_cycle()
and to_zenodo_record().  The gate blocks outputs that:
  - Contain irresponsible or uncalibrated tipping-year predictions
  - Lack uncertainty bounds
  - Exceed biophysically plausible parameter ranges
  - Are produced under excessive tension (instability in the ODE solution)

The gate is intentionally lightweight ("Light") — it does not perform
full ethical review but catches the most common failure modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class EthicsGateConfig:
    """Configuration for the Ethics-Gate Light checks."""

    # Maximum permissible CREP Gamma (values above suggest model miscalibration)
    max_gamma: float = 2.0
    # Minimum/maximum physical forest cover fraction
    min_H: float = 0.0
    max_H: float = 1.0
    # Maximum allowed ODE tension (normalised instability metric)
    max_tension: float = 0.95
    # Require uncertainty bounds in zenodo records
    require_uncertainty_bounds: bool = True
    # Tipping year must be within [2024, 2200] to be considered responsible
    min_tipping_year: float = 2024.0
    max_tipping_year: float = 2200.0
    # Scientific disclaimer appended to all zenodo records
    zenodo_disclaimer: str = (
        "DISCLAIMER: This record is produced by a physics-informed model "
        "(GenesisAeon Package 19, amazon-utac). Predictions represent "
        "model-based scenarios, not guaranteed outcomes. Uncertainty bounds "
        "are provided and should be interpreted in the context of INPE PRODES "
        "observations and peer-reviewed literature (Lovejoy & Nobre 2019; "
        "Boulton et al. 2022). The model is falsifiable and open-source."
    )


class TensionMetric:
    """
    Tracks ODE solution instability (tension) during run_cycle.

    Tension ∈ [0, 1]:
      0.0 = perfectly stable integration
      1.0 = near-divergent ODE (step-size crisis, physically implausible)
    """

    def __init__(self) -> None:
        self._tension: float = 0.0

    def update(self, max_step_ratio: float, n_failures: int = 0) -> None:
        """
        Update tension from ODE solver diagnostics.

        Parameters
        ----------
        max_step_ratio:
            Ratio of largest accepted step to initial step.  Values < 0.1
            indicate step-size reduction (stiffness / near-singularity).
        n_failures:
            Number of rejected ODE steps.
        """
        step_tension = max(0.0, 1.0 - max_step_ratio) * 0.5
        failure_tension = min(1.0, n_failures / 100.0) * 0.5
        self._tension = min(1.0, step_tension + failure_tension)

    def get_current_tension(self) -> float:
        """Return current tension ∈ [0, 1]."""
        return self._tension

    def reset(self) -> None:
        """Reset tension to 0."""
        self._tension = 0.0


@dataclass
class EthicsCheckResult:
    """Result of an Ethics-Gate Light check."""

    allowed: bool
    reason: str
    warnings: list[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


class EthicsGate:
    """
    Ethics-Gate Light (Phase H) for GenesisAeon Package 19.

    Usage (as specified in the GenesisAeon contract):

        # Ethics-Gate Light (Phase H)
        tension = (
            self._tension_metric.get_current_tension()
            if hasattr(self, "_tension_metric")
            else 0.0
        )
        ethics_result = self._ethics_gate.check(state=state, tension=tension)
        if not ethics_result["allowed"]:
            raise RuntimeError(f"EthicsGate blocked: {ethics_result['reason']}")
    """

    def __init__(self, config: EthicsGateConfig | None = None) -> None:
        self.config = config or EthicsGateConfig()

    def check(self, state: dict, tension: float = 0.0) -> dict:
        """
        Run all Ethics-Gate Light checks on a model state dict.

        Parameters
        ----------
        state:
            Model state dict containing at minimum:
              ``'H'`` – current forest cover fraction,
              ``'Gamma'`` – current CREP Γ value.
            Optionally: ``'tipping_year'``, ``'uncertainty_bounds'``.
        tension:
            Current ODE tension metric ∈ [0, 1].

        Returns
        -------
        dict with keys ``'allowed'`` (bool), ``'reason'`` (str),
        ``'warnings'`` (list[str]).
        """
        warnings: list[str] = []
        cfg = self.config

        # Check 1: physical state bounds
        H = state.get("H", 0.5)
        if not (cfg.min_H <= H <= cfg.max_H):
            return {
                "allowed": False,
                "reason": (
                    f"H={H:.4f} outside physical bounds "
                    f"[{cfg.min_H}, {cfg.max_H}]. ODE solution invalid."
                ),
                "warnings": warnings,
            }

        # Check 2: Gamma plausibility
        gamma = state.get("Gamma", 0.0)
        if gamma > cfg.max_gamma:
            return {
                "allowed": False,
                "reason": (
                    f"Γ={gamma:.3f} exceeds maximum plausible value "
                    f"{cfg.max_gamma}. Model likely miscalibrated."
                ),
                "warnings": warnings,
            }
        if gamma < 0.0:
            return {
                "allowed": False,
                "reason": f"Γ={gamma:.3f} is negative — physically undefined.",
                "warnings": warnings,
            }

        # Check 3: ODE tension
        if tension > cfg.max_tension:
            return {
                "allowed": False,
                "reason": (
                    f"ODE tension={tension:.3f} exceeds threshold "
                    f"{cfg.max_tension}. Integration unstable — "
                    "results not scientifically responsible to publish."
                ),
                "warnings": warnings,
            }
        if tension > 0.7:
            warnings.append(
                f"High ODE tension ({tension:.2f}). "
                "Consider reducing time step or reviewing parameters."
            )

        # Check 4: tipping year bounds (if present)
        tipping_year = state.get("tipping_year")
        if tipping_year is not None and not (
            cfg.min_tipping_year <= tipping_year <= cfg.max_tipping_year
        ):
            return {
                "allowed": False,
                "reason": (
                    f"Predicted tipping year {tipping_year:.0f} outside "
                    f"responsible range [{cfg.min_tipping_year:.0f}, "
                    f"{cfg.max_tipping_year:.0f}]. "
                    "Prediction is not scientifically defensible."
                ),
                "warnings": warnings,
            }

        # Check 5: uncertainty bounds required for zenodo records
        if (
            cfg.require_uncertainty_bounds
            and state.get("is_zenodo_record", False)
            and "uncertainty_bounds" not in state
        ):
            return {
                "allowed": False,
                "reason": (
                    "Zenodo record requires uncertainty bounds. "
                    "Provide 'uncertainty_bounds' in the state dict."
                ),
                "warnings": warnings,
            }

        # Gamma advisory warnings
        if gamma < 0.05:
            warnings.append(
                f"Very low Γ={gamma:.3f} — system is ultra-sensitive. "
                "Small perturbations may trigger large transitions."
            )
        if gamma > 0.5:
            warnings.append(
                f"High Γ={gamma:.3f} — system approaching saturation. "
                "Verify calibration against observational data."
            )

        return {
            "allowed": True,
            "reason": "All Ethics-Gate Light checks passed.",
            "warnings": warnings,
        }

    def zenodo_disclaimer(self) -> str:
        """Return the standard GenesisAeon scientific disclaimer for records."""
        return self.config.zenodo_disclaimer
