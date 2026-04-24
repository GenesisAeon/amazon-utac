"""
amazon-utac template — GenesisAeon Package 19 scaffold.

Generates a project skeleton for the Amazon Rainforest Savannisation
Threshold UTAC model, following the GenesisAeon Diamond-Template contract.
"""

from diamond_setup._types import TemplateDict

TEMPLATE: TemplateDict = {
    "name": "amazon-utac",
    "description": "GenesisAeon Package 19: Amazon savannisation UTAC model (Diamond-Template)",
    "variables": ["name", "description", "author", "python_version"],
    "defaults": {
        "description": "Amazon Rainforest Savannisation Threshold UTAC Model",
        "author": "MOR Research Collective",
        "python_version": "3.11",
    },
    "files": {
        "pyproject.toml": """\
[project]
name = "${name}"
version = "0.1.0"
description = "${description}"
readme = "README.md"
license = { text = "MIT" }
authors = [{ name = "${author}" }]
requires-python = ">=${python_version}"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.12",
    "matplotlib>=3.8",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["ruff>=0.6.0", "pytest>=8.0.0", "pytest-cov>=5.0.0"]
notebooks = ["jupyter>=1.0", "ipykernel>=6.0"]

[project.scripts]
amazon-utac = "${name_snake}.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/${name_snake}"]

[tool.ruff]
line-length = 100
target-version = "py${python_version_nodot}"

[tool.ruff.lint]
select = ["E", "F", "B", "I", "W", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=${name_snake} --cov-report=term-missing -v"
""",
        "README.md": """\
# ${name}

${description}

GenesisAeon Package 19 — Amazon Rainforest Savannisation Threshold UTAC Model.

## Physics

Maps the Amazon forest→savanna transition to a UTAC dynamical system:

- **H(t)** — normalised forest cover fraction ∈ [0, 1]
- **K = 1.0** — intact Amazon ceiling
- **H\\* = 0.80** (forest) / **0.15** (savanna) — dual stable attractors
- **Γ_Amazon ≈ 0.116** — LOW-CREP zone
- ODE: `dH/dt = −r·(H−H_s)(H−H_saddle(Γ))(H−H_f)`

Central result:
> Γ_Amazon = arctanh(0.25) / 2.2 ≈ **0.116** (LOW-CREP zone)

## Quickstart

```bash
uv sync
uv run python -c "from ${name_snake} import AmazonUTAC; m = AmazonUTAC(); print(m)"
uv run pytest
```

## Diamond Interface

```python
from ${name_snake} import AmazonUTAC

m = AmazonUTAC(seed=42)
cycle   = m.run_cycle(duration_years=80)
crep    = m.get_crep_state()   # {C, R, E, P, Gamma}
utac    = m.get_utac_state()   # {H, dH_dt, H_star, K_eff}
events  = m.get_phase_events() # threshold crossings
record  = m.to_zenodo_record() # Zenodo-compatible metadata
```

## References

- Lovejoy & Nobre (2019) Science Advances — tipping threshold 20-25 %
- Boulton et al. (2022) Nature Climate Change — resilience loss
- INPE PRODES — Amazon deforestation monitoring
""",
        "src/${name_snake}/__init__.py": """\
\"\"\"${name} — GenesisAeon Package 19 UTAC Model.\"\"\"

from .system import AmazonUTAC
from .constants import AMAZON_TARGETS, GAMMA_AMAZON

__version__ = "0.1.0"
__all__ = ["AmazonUTAC", "AMAZON_TARGETS", "GAMMA_AMAZON"]
""",
        "src/${name_snake}/constants.py": """\
\"\"\"Physical and model constants for ${name}.\"\"\"
import math

CREP_SIGMA: float = 2.2
SEED: int = 42
K_FOREST: float = 1.0
H_FOREST_ATTRACTOR: float = 0.80
H_SAVANNA_ATTRACTOR: float = 0.15
H_SADDLE: float = 0.50
ETA_AMAZON: float = 0.25
GAMMA_AMAZON: float = math.atanh(ETA_AMAZON) / CREP_SIGMA  # ≈ 0.116
ANNUAL_DEFORESTATION_RATE: float = 0.01
AMAZON_TARGETS: dict = {
    "deforestation_threshold_pct": (22.5, 0.10),
    "current_deforestation_pct": (16.0, 0.10),
    "gamma_amazon": (0.116, 0.05),
    "dry_season_lengthening_weeks": (4.5, 0.20),
    "resilience_loss_ar1_trend": ("increasing", None),
}
""",
        "src/${name_snake}/system.py": """\
\"\"\"AmazonUTAC — Diamond-Template interface for ${name}.\"\"\"
from __future__ import annotations
from .constants import GAMMA_AMAZON, K_FOREST, H_FOREST_ATTRACTOR, H_SAVANNA_ATTRACTOR


class AmazonUTAC:
    \"\"\"
    UTAC model for Amazon rainforest savannisation.

    Diamond-Template contract:
      run_cycle() -> dict
      get_crep_state() -> dict   # {C, R, E, P, Gamma}
      get_utac_state() -> dict   # {H, dH_dt, H_star, K_eff}
      get_phase_events() -> list
      to_zenodo_record() -> dict
    \"\"\"

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def run_cycle(self, duration_years: int = 80) -> dict:
        # Ethics-Gate Light (Phase H)
        state = self.get_utac_state()
        tension = 0.0
        # ethics_result = self._ethics_gate.check(state=state, tension=tension)
        # if not ethics_result["allowed"]:
        #     raise RuntimeError(f"EthicsGate blocked: {ethics_result['reason']}")
        return {"years": [], "H": [], "Gamma": GAMMA_AMAZON, "phase_events": []}

    def get_crep_state(self) -> dict:
        return {"C": 0.5, "R": 0.5, "E": 0.5, "P": 0.5, "Gamma": GAMMA_AMAZON}

    def get_utac_state(self) -> dict:
        return {
            "H": 0.84,
            "dH_dt": -0.001,
            "H_star_forest": H_FOREST_ATTRACTOR,
            "H_star_savanna": H_SAVANNA_ATTRACTOR,
            "K_eff": K_FOREST,
            "Gamma": GAMMA_AMAZON,
        }

    def get_phase_events(self) -> list:
        return []

    def to_zenodo_record(self) -> dict:
        # Ethics-Gate Light (Phase H)
        state = {**self.get_utac_state(), "is_zenodo_record": True,
                 "uncertainty_bounds": {"H_lower": 0.79, "H_upper": 0.89}}
        tension = 0.0
        # ethics_result = self._ethics_gate.check(state=state, tension=tension)
        # if not ethics_result["allowed"]:
        #     raise RuntimeError(f"EthicsGate blocked: {ethics_result['reason']}")
        return {
            "metadata": {"title": "${name}", "version": "0.1.0"},
            "crep_state": self.get_crep_state(),
            "utac_state": self.get_utac_state(),
        }
""",
        "tests/__init__.py": "",
        "tests/test_diamond_interface.py": """\
\"\"\"Tests for the AmazonUTAC Diamond-Template contract.\"\"\"
from ${name_snake} import AmazonUTAC


def test_has_diamond_methods():
    m = AmazonUTAC()
    assert callable(m.run_cycle)
    assert callable(m.get_crep_state)
    assert callable(m.get_utac_state)
    assert callable(m.get_phase_events)
    assert callable(m.to_zenodo_record)


def test_crep_state_keys():
    m = AmazonUTAC()
    state = m.get_crep_state()
    for key in ("C", "R", "E", "P", "Gamma"):
        assert key in state


def test_utac_state_keys():
    m = AmazonUTAC()
    state = m.get_utac_state()
    for key in ("H", "dH_dt", "H_star_forest", "H_star_savanna", "K_eff"):
        assert key in state


def test_gamma_approx_0116():
    from ${name_snake}.constants import GAMMA_AMAZON
    assert abs(GAMMA_AMAZON - 0.116) < 0.005
""",
        ".gitignore": """\
__pycache__/
*.py[cod]
*.egg-info/
.eggs/
dist/
build/
.venv/
.uv/
.coverage
htmlcov/
.pytest_cache/
site/
""",
    },
}
