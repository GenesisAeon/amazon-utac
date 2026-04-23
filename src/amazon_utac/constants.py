"""Physical and model constants for amazon-utac (GenesisAeon Package 19)."""

import math

# ── CREP / UTAC core ────────────────────────────────────────────────────────
CREP_SIGMA: float = 2.2          # GenesisAeon standard coupling constant
SEED: int = 42

# ── Amazon biome state space ────────────────────────────────────────────────
K_FOREST: float = 1.0            # intact-forest ceiling (H=1 → full Amazon)
H_FOREST_ATTRACTOR: float = 0.80  # stable forest fixed point
H_SAVANNA_ATTRACTOR: float = 0.15 # stable savanna fixed point
H_SADDLE: float = 0.50            # unstable equilibrium (double-well saddle)
H_TIPPING_LOW: float = 0.75      # lower tipping bound (20 % deforestation)
H_TIPPING_HIGH: float = 0.80     # upper tipping bound (25 % deforestation)
H_TIPPING_MIDPOINT: float = 0.775 # Lovejoy & Nobre midpoint

# ── Current observed state (2024) ──────────────────────────────────────────
CURRENT_FOREST_FRACTION: float = 0.84   # ~84 % of original Amazon remains
CURRENT_DEFORESTATION_FRACTION: float = 0.16  # ~16 % deforested as of 2024
DRY_SEASON_LENGTHENING_WEEKS: float = 4.5     # observed since 1979
ANNUAL_DEFORESTATION_RATE: float = 0.01        # ~1 %/yr of remaining forest

# ── Deforestation threshold (Lovejoy & Nobre 2019) ─────────────────────────
DEFORESTATION_THRESHOLD_LOW: float = 0.20    # 20 % → first tipping signals
DEFORESTATION_THRESHOLD_HIGH: float = 0.25   # 25 % → irreversible savannisation
DEFORESTATION_THRESHOLD_MID: float = 0.225   # benchmark midpoint

# ── CREP calibration ────────────────────────────────────────────────────────
# Γ_Amazon = arctanh(1 − H_threshold) / σ
#           = arctanh(0.25) / 2.2 ≈ 0.116
# This is the LOW-CREP zone: Amazon is more vulnerable per unit forcing than
# AMOC (Γ ≈ 0.251) but less sensitive than solar flares (Γ ≈ 0.014).
ETA_AMAZON: float = 0.25          # fractional efficiency = 1 − H_threshold
GAMMA_AMAZON: float = math.atanh(ETA_AMAZON) / CREP_SIGMA  # ≈ 0.116

# Degraded scenario (40 % deforestation, Cano et al. 2022)
ETA_AMAZON_40PCT: float = 0.40
GAMMA_AMAZON_40PCT: float = math.atanh(ETA_AMAZON_40PCT) / CREP_SIGMA  # ≈ 0.189

# ── Forest dynamics rates ───────────────────────────────────────────────────
R_FOREST: float = 0.05    # forest recovery / degradation magnitude [yr⁻¹]
R_SLOW: float = 0.02      # slow climate-driven change rate [yr⁻¹]

# ── Rainfall / dry season ───────────────────────────────────────────────────
RAINFALL_MEAN_MM_YR: float = 2300.0   # Amazon basin annual mean [mm/yr]
DRY_SEASON_THRESHOLD_MM: float = 100.0 # monthly threshold for "dry" [mm/mo]
DRY_SEASON_BASELINE_MONTHS: float = 3.5  # pre-1979 dry season length [months]
DRY_SEASON_CURRENT_MONTHS: float = 4.6   # current dry season length [months]

# ── AR(1) resilience thresholds (Boulton et al. 2022) ─────────────────────
AR1_CRITICAL_THRESHOLD: float = 0.90  # AR(1) → 1 signals tipping proximity
AR1_TREND_WINDOW_YEARS: int = 10      # rolling window for trend detection

# ── PRODES reference data (INPE, as of 2024) ───────────────────────────────
PRODES_REFERENCE_YEAR: int = 1988    # PRODES monitoring start year
PRODES_ORIGINAL_AREA_KM2: float = 4_153_741.0  # original Amazon area [km²]
PRODES_DEFORESTED_2024_KM2: float = 664_450.0  # cumulative deforested [km²]

# ── Benchmark targets ───────────────────────────────────────────────────────
AMAZON_TARGETS: dict = {
    "deforestation_threshold_pct": (22.5, 0.10),   # Lovejoy & Nobre midpoint
    "current_deforestation_pct":   (16.0, 0.10),   # as of 2024
    "gamma_amazon":                (0.116, 0.05),
    "dry_season_lengthening_weeks": (4.5, 0.20),   # observed since 1979
    "resilience_loss_ar1_trend":   ("increasing", None),  # Boulton 2022
}

# ── GenesisAeon package registry entry ────────────────────────────────────
PACKAGE_REGISTRY_ENTRY: dict = {
    "name": "amazon-utac",
    "class": "AmazonUTAC",
    "domain": "ecology",
    "scale": "continental",
    "zenodo": "10.5281/zenodo.19645351",
    "reference": "10.1038/s41558-022-01287-8",
    "package_id": 19,
}
