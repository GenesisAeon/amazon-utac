# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.1.0] - 2026-08-01
### Changed
- `CURRENT_DEFORESTATION_FRACTION` updated from 0.16 (2024) to 0.161 (PRODES,
  Aug2024-Jul2025 monitoring year: 670,246 km² / 4,153,741 km²), per the
  KlimaAktuell citation-accuracy review. Note: a naive forward-extrapolation
  of the old ~1%/yr rate assumption would have suggested ~17-18% by 2026 --
  the real PRODES-consistent figure is barely higher than 2024's, because
  INPE recorded an 11.08% y/y drop in cleared area (5,796 km², the lowest
  annual PRODES figure in 11 years).
- Added `PRODES_DEFORESTED_2025_KM2` and
  `ANNUAL_DEFORESTATION_RATE_OBSERVED_2025` (~0.17%/yr, roughly 6x below the
  legacy `ANNUAL_DEFORESTATION_RATE` default) as separate, clearly-labeled
  fields alongside the existing ones, rather than silently replacing them.
### Fixed
- CI: dropped Python 3.10 from the test matrix and bumped `requires-python`
  to `>=3.11` -- the package already depends on `diamond-setup>=2.2.0`,
  which itself requires Python >=3.11, so 3.10 could never actually
  install in CI (this had been silently broken since the "declare
  diamond-setup as a real dependency" change on 2026-07-17).
### Known issue (not fixed here, out of scope for this pass)
- `tests/test_cli.py`, `tests/test_preset.py`, `tests/test_validator.py`
  appear to be leftover `diamond-setup` scaffold-template tests (they
  import and test `diamond_setup`'s own CLI, not `amazon_utac`'s) --
  `test_cli.py::test_version` fails locally due to Rich ANSI-code
  splitting, unrelated to this package's own logic. Needs a follow-up
  cleanup pass to remove or properly replace these files.

## [1.0.0] - 2026
### Added
- Initial v1.0.0 release as part of the GenesisAeon ecosystem-wide 1.0.0
  milestone.
- Standardized release tooling: `.zenodo.json`, `RELEASE_GUIDE.md`,
  `CONTRIBUTING.md`, issue/PR templates (CI and release workflows already
  existed in this repo).

### Changed
- `.zenodo.json` metadata normalized to match the package's actual MIT
  license and 1.0.0 version, and aligned with the GenesisAeon ecosystem
  metadata template.
