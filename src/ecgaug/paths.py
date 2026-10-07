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
