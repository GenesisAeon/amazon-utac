# Contributing

Thanks for your interest in contributing to this GenesisAeon ecosystem
package!

## Getting started

1. Fork and clone the repository.
2. Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`
   (or `.venv\Scripts\activate` on Windows).
3. Install in editable mode with dev dependencies:
   `pip install -e ".[dev]"` (or `.[test]` if `dev` extras don't exist).
4. Run the test suite: `pytest`.

## Code style

- Format with `ruff format` and lint with `ruff check`.
- Type-check with `mypy` where the package uses type hints.
- Keep functions documented with docstrings (Google or NumPy style,
  whichever this repo already uses).

## Diamond Interface packages

If this package implements the GenesisAeon Diamond Interface
(`run_cycle`, `get_crep_state`, `get_utac_state`, `get_phase_events`,
`to_zenodo_record`), any change to these methods' signatures or return
shapes is a **breaking change** and requires a MAJOR version bump (see
`RELEASE_GUIDE.md`).

## Pull requests

- One logical change per PR.
- Add or update tests for any behavioral change.
- Update `CHANGELOG.md` under an `## [Unreleased]` section.
- Fill out the PR template (`.github/PULL_REQUEST_TEMPLATE.md`).

## Reporting issues

Please use the issue templates in `.github/ISSUE_TEMPLATE/` — they help us
triage bug reports vs. feature requests quickly.

## Scientific claims

This is part of a research framework. If your contribution touches any
scientific model, prediction, or benchmark (e.g. CREP `Γ` values, UTAC
parameters, falsifiable predictions), please:
- Cite the source (paper, dataset, or prior GenesisAeon Zenodo record).
- Clearly mark speculative vs. validated claims, consistent with this
  repo's existing `DISCLAIMER.md` (if present).
