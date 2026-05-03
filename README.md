# amazon-utac

> GenesisAeon Package 19 — Amazon Rainforest Savannisation as UTAC System

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.19645351"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.19645351.svg" alt="DOI (GenesisAeon Whitepaper)"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="GPLv3 License"/></a>
  <a href="https://creativecommons.org/licenses/by/4.0/"><img src="https://img.shields.io/badge/docs-CC%20BY%204.0-lightblue.svg" alt="CC BY 4.0"/></a>
  <a href="https://github.com/GenesisAeon/genesis-os"><img src="https://img.shields.io/badge/part%20of-genesis--os-blueviolet" alt="Part of genesis-os"/></a>
  <img src="https://img.shields.io/badge/UTAC-package%2019-orange" alt="Package 19"/>
</p>

**Amazon rainforest → savanna transition modelled as reversed UTAC system.**

**Key result**: Γ_Amazon ≈ 0.116 (low-CREP, fragile regime).

## Installation

```bash
pip install -e ".[dev]"
```

## Quickstart

```bash
amazon-utac run --scenario current-rate
amazon-utac threshold --deforestation-rate 0.01
```

## Integration in genesis-os

```python
from genesis_os import GenesisOS
os = GenesisOS()
amazon = os.load_package(19)
results = amazon.run_cycle(duration_years=80)
```

## Benchmark

Validated against Lovejoy & Nobre (2019) and Boulton et al. (2022).

## Falsifiable Prediction

With 1 % annual deforestation, tipping threshold crossed in 2038 ± 5 years.

## License

Code: MIT • Docs & Data: CC BY 4.0
