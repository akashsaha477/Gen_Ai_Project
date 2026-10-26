"""Repository paths, resolved at call time so tests and smoke runs can redirect them.

Set ``ECGAUG_ROOT`` to point data, runs, results and docs at another directory.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the project root: ``ECGAUG_ROOT`` if set, else the directory holding pyproject.toml."""
    env = os.environ.get("ECGAUG_ROOT")
    if env:
        return Path(env).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Cannot locate the repository root; set ECGAUG_ROOT.")


def source_root() -> Path:
    """The checkout holding this package's source (ignores ``ECGAUG_ROOT``): where the committed configs live."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Cannot locate the source checkout")


def raw_dir() -> Path:
    return repo_root() / "data" / "raw" / "mitdb"


def processed_dir() -> Path:
    return repo_root() / "data" / "processed"


def runs_dir() -> Path:
    return repo_root() / "runs"


def results_dir() -> Path:
    return repo_root() / "results"


def docs_dir() -> Path:
    return repo_root() / "docs"


def configs_dir() -> Path:
    return repo_root() / "configs"


def results_for(name: str) -> Path:
    """``results/<name>/``, where name is an experiment or phase folder such as phase1_data."""
    return results_dir() / name


def figures_for(name: str) -> Path:
    """``figures/<name>/``, named like the results folder of the same experiment or phase."""
    return repo_root() / "figures" / name
