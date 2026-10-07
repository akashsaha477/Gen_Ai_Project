"""Guarded dataset loaders. ``load_test`` is the only way to read ``test.npz`` (docs/RULES.md rule 2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ecgaug import paths
from ecgaug.data.access import log_test_access, require_test_access
from ecgaug.data.contract import TEST_FILE, TRAINVAL_FILE, verify_file


def _load(dataset_dir: Path, filename: str) -> dict[str, np.ndarray]:
    verify_file(dataset_dir, filename)
    with np.load(dataset_dir / filename, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def load_trainval(version: str = "v1") -> dict[str, np.ndarray]:
    """Training and validation arrays of a frozen contract, after verifying the file hash."""
    return _load(paths.processed_dir() / version, TRAINVAL_FILE)


def load_candidate(name: str) -> dict[str, np.ndarray]:
    """Training and validation arrays of a Module 1 candidate dataset (DS1 only)."""
    return _load(paths.processed_dir() / "candidates" / name, TRAINVAL_FILE)


def load_test(version: str = "v1", run_ids: str = "", approved_by: str = "") -> dict[str, np.ndarray]:
    """Test arrays (DS2). Raises unless ``ALLOW_TEST=1``, which only ``/final-eval`` sets.

    The permission is checked before anything is read, and every successful call is appended to
    ``docs/TEST_ACCESS_LOG.md``.
    """
    require_test_access("evaluation")
    arrays = _load(paths.processed_dir() / version, TEST_FILE)
    log_test_access("evaluation", f"load_test('{version}')", run_ids=run_ids, approved_by=approved_by)
    return arrays
