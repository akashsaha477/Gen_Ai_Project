"""Dataset builder and data contract (CLAUDE.md §5).

Builds contract-format arrays for a representation, writes them as deterministic ``.npz`` files with
a ``contract.json`` that records how they were made and the SHA-256 of each file, and verifies them.

Known nondeterminism: filtering and resampling can differ in the last bit between platforms, so a
frozen contract is built once, hashed and distributed as files; other machines verify the hashes
instead of rebuilding.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from ecgaug import paths
from ecgaug.data import raw
from ecgaug.data.access import log_test_access
from ecgaug.data.beats import RR_LOCAL_BEATS, UNMAPPED, extract_beats
from ecgaug.data.records import AAMI, CLASSES, DS2, FS, LEAD
from ecgaug.data.representation import USABLE_POST_S, USABLE_PRE_S, RepresentationSpec, transform_record, usable_mask
from ecgaug.data.splits import Split, check_split
from ecgaug.runs import git_dirty, git_sha

FIELDS: tuple[str, ...] = (
    "X", "y", "record", "split", "r_idx", "rr_prev", "rr_next", "rr_ratio", "is_synthetic", "source",
)  # fmt: skip
TRAINVAL_FILE = "trainval.npz"
TEST_FILE = "test.npz"
CONTRACT_FILE = "contract.json"

# R-relative segment windows in milliseconds (PLAN.md phase5_explainability), used by the saliency summaries.
SEGMENT_WINDOWS_MS: dict[str, tuple[int, int]] = {
    "P": (-250, -100),
    "QRS": (-50, 50),
    "ST": (50, 150),
    "T": (150, 400),
}


class ContractError(RuntimeError):
    """A dataset does not match its contract (missing file or hash mismatch)."""


def build_arrays(
    records: list[int] | tuple[int, ...],
    spec: RepresentationSpec,
    split_of: dict[int, str],
    access: str | None = None,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Contract-format arrays for ``records`` plus a per-record report of kept and dropped beats.

    Q-class and unmapped QRS annotations take part in the RR computation and are then dropped.
    Unusable beats (window past the record edge, or a missing RR interval) are dropped by the same
    rule for every representation.
    """
    parts: dict[str, list[np.ndarray]] = {name: [] for name in FIELDS}
    report: list[dict] = []
    for record in records:
        beats = extract_beats(record, access)
        signal = raw.read_signal(record, access)
        usable = usable_mask(
            beats["r_idx"].to_numpy(), signal.size, beats["rr_prev"].to_numpy(), beats["rr_next"].to_numpy()
        )
        classified = (beats["y"] >= 0).to_numpy()
        keep = usable & classified
        kept = beats[keep]
        n = int(keep.sum())

        parts["X"].append(transform_record(signal, kept["r_idx"].to_numpy(), spec))
        parts["y"].append(kept["y"].to_numpy(dtype=np.int8))
        parts["record"].append(np.full(n, int(record), dtype=np.int16))
        parts["split"].append(np.full(n, split_of[int(record)], dtype="<U5"))
        parts["r_idx"].append(kept["r_idx"].to_numpy(dtype=np.int32))
        for name in ("rr_prev", "rr_next", "rr_ratio"):
            parts[name].append(kept[name].to_numpy(dtype=np.float32))
        parts["is_synthetic"].append(np.zeros(n, dtype=bool))
        parts["source"].append(np.full(n, "real", dtype="<U4"))

        tally = beats["aami"].value_counts()
        dropped = beats[classified & ~usable]["aami"].value_counts()
        report.append(
            {
                "record": int(record),
                "split": split_of[int(record)],
                "qrs_annotations": int(len(beats)),
                "annotated": {c: int(tally.get(c, 0)) for c in CLASSES},
                "q_beats_excluded": int(tally.get("Q", 0)),
                "unmapped_qrs_excluded": int(tally.get(UNMAPPED, 0)),
                "dropped_unusable": {c: int(dropped.get(c, 0)) for c in CLASSES},
                "kept": {c: int((kept["aami"] == c).sum()) for c in CLASSES},
            }
        )
    arrays = {name: np.concatenate(chunks) for name, chunks in parts.items()}
    if not np.isfinite(arrays["X"]).all():
        raise ValueError("Non-finite values in X")
    return arrays, report


def save_npz_deterministic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an uncompressed ``.npz`` whose bytes depend only on the arrays (fixed zip timestamps)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            with zf.open(info, "w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, np.asanyarray(arrays[name]), allow_pickle=False)


def sha256_file(path: Path) -> str:
    return raw.file_sha256(path)


def content_sha256(arrays: dict[str, np.ndarray]) -> str:
    """Hash of array names, dtypes, shapes and bytes; independent of the container format."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _class_counts(arrays: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, int]:
    y = arrays["y"][mask]
    return {c: int((y == i).sum()) for i, c in enumerate(CLASSES)}


def write_dataset(
    out_dir: Path,
    name: str,
    kind: str,
    spec: RepresentationSpec,
    split: Split,
    trainval: tuple[dict[str, np.ndarray], list[dict]],
    test: tuple[dict[str, np.ndarray], list[dict]] | None = None,
) -> dict:
    """Write ``trainval.npz`` (and ``test.npz``) with their ``contract.json``; return the contract."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files, counts, per_record = {}, {}, {}
    tv_arrays, tv_report = trainval
    save_npz_deterministic(out_dir / TRAINVAL_FILE, tv_arrays)
    files[TRAINVAL_FILE] = {
        "sha256": sha256_file(out_dir / TRAINVAL_FILE),
        "content_sha256": content_sha256(tv_arrays),
        "n_beats": int(len(tv_arrays["y"])),
    }
    for part in ("train", "val"):
        counts[part] = _class_counts(tv_arrays, tv_arrays["split"] == part)
    per_record["trainval"] = tv_report
    if test is not None:
        test_arrays, test_report = test
        save_npz_deterministic(out_dir / TEST_FILE, test_arrays)
        files[TEST_FILE] = {
            "sha256": sha256_file(out_dir / TEST_FILE),
            "content_sha256": content_sha256(test_arrays),
            "n_beats": int(len(test_arrays["y"])),
        }
        counts["test"] = _class_counts(test_arrays, np.ones(len(test_arrays["y"]), dtype=bool))
        per_record["test"] = test_report

    contract = {
        "name": name,
        "kind": kind,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "source": {
            "database": "MIT-BIH Arrhythmia Database 1.0.0 (PhysioNet)",
            "fs": FS,
            "lead": f"{LEAD}, selected by signal name",
        },
        "representation": spec.to_dict(),
        "labels": {
            "classes": list(CLASSES),
            "aami_map": {k: list(v) for k, v in AAMI.items()},
            "q_policy": "Q beats are excluded from the arrays and from all metrics; counts are reported",
            "unmapped_policy": "QRS annotations outside the AAMI map (e.g. '!') are excluded; counts are reported",
        },
        "rr": {
            "definition": "seconds between consecutive QRS-type annotations (WFDB isqrs), computed "
            "before Q and unmapped annotations are dropped",
            "rr_ratio": f"rr_prev / mean of the up to {RR_LOCAL_BEATS} preceding RR intervals (1 if none)",
        },
        "usable_beat_rule": {
            "window_pre_s": USABLE_PRE_S,
            "window_post_s": USABLE_POST_S,
            "requires": "window inside the record at 360 Hz and finite rr_prev and rr_next",
            "note": "identical for every representation",
        },
        "segment_windows_ms": {k: list(v) for k, v in SEGMENT_WINDOWS_MS.items()},
        "split": {
            "name": split.name,
            "train": list(split.train),
            "val": list(split.val),
            "test": list(split.test),
            "selection_metric": split.selection_metric,
        },
        "counts": counts,
        "per_record": per_record,
        "files": files,
    }
    (out_dir / CONTRACT_FILE).write_text(json.dumps(contract, indent=2) + "\n")
    return contract


def verify_file(dataset_dir: Path, filename: str) -> dict:
    """Check ``filename`` against the hash in ``contract.json``; return the contract."""
    contract_path = dataset_dir / CONTRACT_FILE
    if not contract_path.is_file():
        raise ContractError(f"{contract_path} is missing")
    contract = json.loads(contract_path.read_text())
    if filename not in contract["files"]:
        raise ContractError(f"{filename} is not listed in {contract_path}")
    path = dataset_dir / filename
    if not path.is_file():
        raise ContractError(f"{path} is missing")
    actual = sha256_file(path)
    if actual != contract["files"][filename]["sha256"]:
        raise ContractError(f"{path}: SHA-256 {actual[:12]}... does not match the contract")
    return contract


def build_candidate(spec: RepresentationSpec, split: Split, require_all_ds1: bool = True) -> Path:
    """Build a phase1_input_representation candidate dataset from DS1 records only (no test data is read)."""
    check_split(split, require_all_ds1=require_all_ds1)
    records = sorted(split.train + split.val)
    out_dir = paths.processed_dir() / "candidates" / spec.name
    write_dataset(out_dir, spec.name, "candidate", spec, split, build_arrays(records, spec, split.split_of()))
    return out_dir


def build_frozen(version: str, spec: RepresentationSpec, split: Split) -> Path:
    """Build the frozen contract ``data/processed/<version>/`` including ``test.npz``.

    Reading DS2 here is a construction access: it needs ``ALLOW_TEST_BUILD=1`` and is logged. Run it
    once, at the phase 1 gate freeze, for the representation selected by rule.
    """
    check_split(split)
    out_dir = paths.processed_dir() / version
    if (out_dir / CONTRACT_FILE).exists():
        raise ContractError(f"{out_dir} already holds a contract; frozen artifacts are never overwritten")
    trainval = build_arrays(sorted(split.train + split.val), spec, split.split_of())
    test = build_arrays(sorted(DS2), spec, {r: "test" for r in DS2}, access="construction")
    log_test_access("construction", f"built {version}/{TEST_FILE} for representation {spec.name}")
    write_dataset(out_dir, version, "frozen", spec, split, trainval, test)
    return out_dir
