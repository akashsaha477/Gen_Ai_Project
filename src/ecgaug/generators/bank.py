"""Synthetic banks: S, V and F beats from a trained generator, in the data-contract format (CLAUDE.md §5).

    data/synthetic/<generator_id>/seed<s>.npz    the beats, with every contract field
    data/synthetic/<generator_id>/bank.json      how they were made and the SHA-256 of each seed's file

Synthetic beats carry ``is_synthetic=True``, ``source=<generator_id>``, ``split='train'``,
``record=-1`` and ``r_idx=-1``. Their RR features are copied from randomly drawn real training beats
of the same class: the class-conditional empirical distribution (PREREGISTRATION §5). The three RR
values of one real beat are copied together, so their joint distribution is the real one; how they
relate to the waveform is not modelled, which is a stated limitation. The waveform is per-beat
z-scored again after sampling, as the contract requires.

A bank is never overwritten, and a bank under a path containing ``audit`` is refused (PLAN Phase 4:
leakage-audit banks never enter the main grid).
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np

from ecgaug import paths
from ecgaug.data.contract import FIELDS, content_sha256, save_npz_deterministic, sha256_file
from ecgaug.data.records import CLASS_INDEX
from ecgaug.data.representation import zscore_per_beat
from ecgaug.generators.base import GENERATED_CLASSES, BeatGenerator

RR_FIELDS = ("rr_prev", "rr_next", "rr_ratio")
BANK_FILE = "bank.json"
RR_RULE = (
    "RR features copied from real training beats of the same class, drawn uniformly with replacement; "
    "the three values of one beat are copied together (PREREGISTRATION §5)"
)


class BankError(RuntimeError):
    """A bank is missing, too small, changed since export, or not allowed here."""


def synthetic_dir() -> Path:
    return paths.repo_root() / "data" / "synthetic"


def bank_dir(generator_id: str) -> Path:
    if "audit" in generator_id.lower():
        raise BankError(f"{generator_id}: leakage-audit banks never enter the main pipeline")
    return synthetic_dir() / generator_id


def empty_part() -> dict[str, np.ndarray]:
    """Zero beats with the contract's fields and dtypes."""
    return {
        "X": np.zeros((0, 0), dtype=np.float32), "y": np.zeros(0, dtype=np.int8), "record": np.zeros(0, dtype=np.int16),
        "split": np.zeros(0, dtype="<U5"), "r_idx": np.zeros(0, dtype=np.int32), "rr_prev": np.zeros(0, dtype=np.float32),
        "rr_next": np.zeros(0, dtype=np.float32), "rr_ratio": np.zeros(0, dtype=np.float32),
        "is_synthetic": np.zeros(0, dtype=bool), "source": np.zeros(0, dtype="<U4"),
    }  # fmt: skip


def concat_parts(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate contract-format parts field by field (strings keep the widest width)."""
    parts = [p for p in parts if len(p["y"])]
    if not parts:
        return empty_part()
    return {name: np.concatenate([p[name] for p in parts]) for name in FIELDS}


def rr_from_training(train: dict[str, np.ndarray], cls: str, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """RR features of ``n`` real training beats of class ``cls`` drawn with replacement."""
    pool = np.flatnonzero(np.asarray(train["y"]) == CLASS_INDEX[cls])
    if not pool.size:
        raise BankError(f"No training beats of class {cls} to take RR features from")
    pick = rng.choice(pool, size=n, replace=True)
    return {name: np.asarray(train[name])[pick].astype(np.float32) for name in RR_FIELDS}


def synthetic_part(x: np.ndarray, cls: str, rr: dict[str, np.ndarray], source: str, is_synthetic: bool = True) -> dict[str, np.ndarray]:
    """Contract-format arrays for added beats of one class (no record, no R-peak index)."""
    n = len(x)
    return {
        "X": np.asarray(x, dtype=np.float32),
        "y": np.full(n, CLASS_INDEX[cls], dtype=np.int8),
        "record": np.full(n, -1, dtype=np.int16),
        "split": np.full(n, "train", dtype="<U5"),
        "r_idx": np.full(n, -1, dtype=np.int32),
        **{name: np.asarray(rr[name], dtype=np.float32) for name in RR_FIELDS},
        "is_synthetic": np.full(n, is_synthetic, dtype=bool),
        "source": np.full(n, source),
    }


def sample_bank(generator: BeatGenerator, train: dict[str, np.ndarray], counts: dict[str, int], seed: int, source: str) -> dict[str, np.ndarray]:
    """``counts[c]`` synthetic beats of each class ``c`` in contract format; the same seed gives the same beats."""
    rng = np.random.default_rng(seed)
    parts = []
    for cls in GENERATED_CLASSES:
        n = int(counts.get(cls, 0))
        if n <= 0:
            continue
        x = generator.sample(cls, n, seed=seed * 100 + CLASS_INDEX[cls])
        x = zscore_per_beat(x.astype(np.float64)).astype(np.float32)
        parts.append(synthetic_part(x, cls, rr_from_training(train, cls, n, rng), source))
    return concat_parts(parts)


def bank_counts(train_y: np.ndarray, rule: str = "balance_to_n") -> dict[str, int]:
    """Beats per class to export: enough to bring each minority class up to the training N count.

    That is the most any augmentation policy can use (PLAN Phase 4, policy A), so one bank serves all.
    """
    if rule != "balance_to_n":
        raise ValueError(f"Unknown bank size rule {rule!r}")
    y = np.asarray(train_y)
    n_normal = int((y == CLASS_INDEX["N"]).sum())
    return {c: max(0, n_normal - int((y == CLASS_INDEX[c]).sum())) for c in GENERATED_CLASSES}


def export_bank(arrays: dict[str, np.ndarray], generator_id: str, seed: int, info: dict) -> dict:
    """Write ``seed<seed>.npz`` and add its entry to ``bank.json``; return that entry."""
    folder = bank_dir(generator_id)
    path = folder / f"seed{seed}.npz"
    if path.exists():
        raise BankError(f"{path} exists; a bank is never overwritten")
    save_npz_deterministic(path, arrays)
    y = np.asarray(arrays["y"])
    entry = {
        "file": path.name,
        "sha256": sha256_file(path),
        "content_sha256": content_sha256(arrays),
        "n_beats": int(len(y)),
        "counts": {c: int((y == CLASS_INDEX[c]).sum()) for c in GENERATED_CLASSES},
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **info,
    }
    meta_path = folder / BANK_FILE
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {
        "generator_id": generator_id, "format": "data contract (CLAUDE.md §5)", "classes": list(GENERATED_CLASSES),
        "rr_features": RR_RULE, "normalization": "per-beat z-score applied after sampling", "seeds": {},
    }  # fmt: skip
    meta["seeds"][str(seed)] = entry
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return entry


def load_bank(generator_id: str, seed: int, expected_sha256: str | None = None) -> dict[str, np.ndarray]:
    """A bank's arrays after checking the file against ``bank.json`` (and ``expected_sha256`` if given)."""
    folder = bank_dir(generator_id)
    meta_path, path = folder / BANK_FILE, folder / f"seed{seed}.npz"
    if not meta_path.is_file() or not path.is_file():
        raise BankError(f"Bank {generator_id} seed {seed} is missing ({path})")
    entry = json.loads(meta_path.read_text())["seeds"].get(str(seed))
    if entry is None:
        raise BankError(f"{meta_path} has no entry for seed {seed}")
    actual = sha256_file(path)
    for label, wanted in (("bank.json", entry["sha256"]), ("the frozen manifest", expected_sha256)):
        if wanted is not None and actual != wanted:
            raise BankError(f"{path}: SHA-256 {actual[:12]}... does not match {label}")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    if set(arrays) != set(FIELDS):
        raise BankError(f"{path} does not have the contract fields")
    if not (np.asarray(arrays["split"]) == "train").all():
        raise BankError(f"{path}: synthetic beats must all be marked as training beats")
    return arrays
