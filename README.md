# diamond-setup

**Universal Python project scaffold** — generate professional, CI-ready skeletons in seconds.

[![CI](https://github.com/GenesisAeon/diamond-setup/actions/workflows/ci.yml/badge.svg)](https://github.com/GenesisAeon/diamond-setup/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

No cookiecutter, no Jinja2, no magic. Just a clean CLI that produces a fully working project — `uv sync`, `pytest`, ruff, pre-commit and CI all wired up from second one.

---

## Install

```bash
pip install diamond-setup
# or
uv tool install diamond-setup
```

## Usage

```bash
# New project with the minimal template (default)
diamond scaffold my-lib

# GenesisAeon preset (adds domains.yaml + entropy-table bridge)
diamond scaffold my-physics-tool --template genesis --author "Ada Lovelace"

# Preview what would be generated (no files written)
diamond scaffold my-lib --dry-run

# See all templates
diamond list-templates

# Validate any project directory
diamond validate path/to/my-project
diamond validate          # validates the current directory
```

## What you get

Running `diamond scaffold my-lib` produces:

```
my-lib/
├── src/
│   └── my_lib/
│       └── __init__.py       # __version__ = "0.1.0"
├── tests/
│   ├── __init__.py
│   └── test_main.py
├── .github/
│   └── workflows/
│       └── ci.yml            # matrix: 3.11 + 3.12
├── pyproject.toml            # hatchling, ruff, pytest configured
├── README.md
├── .gitignore
└── .pre-commit-config.yaml   # ruff + standard hooks
```

Then just:

```bash
cd my-lib
uv sync --dev
pre-commit install
uv run pytest
```

## Templates

| Template | Description |
|----------|-------------|
| `minimal` | Clean Python package for everyone |
| `genesis` | Adds `domains.yaml` + entropy-table bridge (GenesisAeon preset) |

## Extending

Adding a new template is one Python file. See [docs/templates.md](docs/templates.md).

---

Built with [uv](https://docs.astral.sh/uv/) · [Typer](https://typer.tiangolo.com/) · [Rich](https://rich.readthedocs.io/)
