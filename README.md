# amazon-utac

> GenesisAeon Package 19 — Amazon Rainforest Savannisation as UTAC System

[![GenesisAeon](https://img.shields.io/badge/GenesisAeon-Package%2019-blueviolet)](https://github.com/GenesisAeon)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19645351.svg)](https://doi.org/10.5281/zenodo.19645351)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Reference](https://img.shields.io/badge/Ref-Nature%20Climate%20Change%202022-red)](https://doi.org/10.1038/s41558-022-01287-8)

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
