"""PTB-XL preprocessing for the pretraining arm (DECISIONS A12).

PTB-XL has 10-second 12-lead recordings with one set of diagnostic statements per recording and no
beat annotations. This module turns it into beats that look like the MIT-BIH beats:

1. Metadata: keep recordings with at least one diagnostic superclass; drop pacemaker recordings
   (MIT-BIH paced records are excluded too) and recordings flagged with electrode problems.
2. Signal: lead II by name (closest to MLII), checked for length, finite values, flat lines and
   implausible amplitude, then resampled from 500 Hz to 360 Hz with a polyphase filter.
3. Beats: R peaks from the detector validated on MIT-BIH DS1 (``ecgaug.data.rpeaks``); recordings
   with an implausible number of peaks, or peaks that do not stand out of the noise, are dropped; RR features are computed as for MIT-BIH.
4. Beat filter: the same usable-beat rule as MIT-BIH, plus both RR intervals inside 0.3-2.0 s.
5. Labels: every beat gets its recording's superclasses (multi-label). Noisy at beat level by design.
6. Split: PTB-XL's patient-wise folds, 1-8 train, 9 validation, 10 unused.

Everything up to here is independent of the input representation. ``beat_arrays`` then cuts windows
with any ``RepresentationSpec``, through the same ``transform_record`` used for MIT-BIH.

Usage: ``python -m ecgaug.data.ptbxl --root <folder with ptbxl_database.csv>``
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps

from ecgaug import paths
from ecgaug.data import raw
from ecgaug.data.beats import rr_intervals
from ecgaug.data.contract import content_sha256, save_npz_deterministic, sha256_file
from ecgaug.data.records import FS
from ecgaug.data.representation import RepresentationSpec, transform_record, usable_mask
from ecgaug.data.rpeaks import detect_r_peaks, qrs_contrast

SUPERCLASSES: tuple[str, ...] = ("NORM", "MI", "STTC", "CD", "HYP")
LEAD = "II"
SOURCE_FS = 500
SECONDS = 10
N_SAMPLES = SECONDS * FS  # 3600 samples per recording after resampling
TRAIN_FOLDS, VAL_FOLDS, UNUSED_FOLDS = (1, 2, 3, 4, 5, 6, 7, 8), (9,), (10,)

MIN_PEAK_TO_PEAK_MV = 0.05
MAX_ABS_MV = 10.0
MAX_FLAT_S = 1.0
MIN_PEAKS, MAX_PEAKS = 5, 33  # 30 to 200 beats per minute over 10 seconds
RR_RANGE_S = (0.3, 2.0)
# Noise without QRS complexes scores below 3; the lowest 1% of MIT-BIH DS1 10-second segments score 6.6.
MIN_QRS_CONTRAST = 4.0
OUT_NAME = "records.npz"


def superclass_map(scp_statements: pd.DataFrame) -> dict[str, str]:
    """SCP code -> diagnostic superclass, for diagnostic statements only."""
    diagnostic = scp_statements[scp_statements["diagnostic"] == 1]
    return {code: cls for code, cls in diagnostic["diagnostic_class"].items() if cls in SUPERCLASSES}


def record_labels(scp_codes: str, code_to_class: dict[str, str]) -> np.ndarray:
    """Multi-hot superclass vector of one recording.

    Every diagnostic code in the recording counts, whatever its likelihood (0 means "likelihood not
    given" in PTB-XL), which is the convention of the PTB-XL benchmark.
    """
    present = {code_to_class[code] for code in ast.literal_eval(scp_codes) if code in code_to_class}
    return np.array([cls in present for cls in SUPERCLASSES], dtype=bool)


def load_metadata(root: Path) -> tuple[pd.DataFrame, dict]:
    """Recordings eligible for pretraining, plus counts of what was excluded and why."""
    root = Path(root)
    meta = pd.read_csv(root / "ptbxl_database.csv")
    code_to_class = superclass_map(pd.read_csv(root / "scp_statements.csv", index_col=0))
    labels = np.stack([record_labels(codes, code_to_class) for codes in meta["scp_codes"]])
    flagged = lambda col: meta[col].fillna("").astype(str).str.strip().ne("")  # noqa: E731
    reasons = pd.Series("", index=meta.index)
    reasons[~labels.any(axis=1)] = "no_diagnostic_superclass"
    reasons[flagged("electrodes_problems") & reasons.eq("")] = "electrodes_problems"
    reasons[flagged("pacemaker") & reasons.eq("")] = "pacemaker"
    reasons[meta["patient_id"].isna() & reasons.eq("")] = "no_patient_id"
    keep = reasons.eq("")
    out = meta.loc[keep, ["ecg_id", "patient_id", "strat_fold", "filename_hr"]].copy()
    out["patient_id"] = out["patient_id"].astype(np.int64)
    out["labels"] = list(labels[keep.to_numpy()])
    report = {"in_metadata": int(len(meta)), "eligible": int(keep.sum())}
    report["excluded_by_metadata"] = {k: int(v) for k, v in Counter(reasons[~keep]).items()}
    return out.reset_index(drop=True), report


def longest_flat_run(x: np.ndarray) -> int:
    """Length in samples of the longest run of identical consecutive values."""
    if x.size < 2:
        return x.size
    change = np.flatnonzero(np.diff(x) != 0)
    edges = np.concatenate([[-1], change, [x.size - 1]])
    return int(np.diff(edges).max())


def signal_problem(x: np.ndarray, fs: int) -> str | None:
    """Why a raw lead is unusable, or None if it passes every check."""
    if fs != SOURCE_FS or x.size != SOURCE_FS * SECONDS:
        return "unexpected_length_or_rate"
    if not np.isfinite(x).all():
        return "non_finite"
    if np.ptp(x) < MIN_PEAK_TO_PEAK_MV:
        return "flat_signal"
    if np.abs(x - np.median(x)).max() > MAX_ABS_MV:
        return "implausible_amplitude"
    if longest_flat_run(x) > MAX_FLAT_S * fs:
        return "flat_segment"
    return None


def resample_to_fs(x: np.ndarray, fs: int) -> np.ndarray:
    """Polyphase resampling to 360 Hz; a 10-second recording becomes 3600 samples."""
    divisor = np.gcd(FS, fs)
    return sps.resample_poly(x, FS // divisor, fs // divisor)


def process_record(x: np.ndarray, fs: int) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """Quality-check, resample and detect: ``(signal_360hz, r_peaks, problem)``."""
    problem = signal_problem(x, fs)
    if problem:
        return None, None, problem
    y = resample_to_fs(x, fs)
    peaks = detect_r_peaks(y)
    if not MIN_PEAKS <= peaks.size <= MAX_PEAKS:
        return None, None, "implausible_peak_count"
    if qrs_contrast(y, peaks) < MIN_QRS_CONTRAST:
        return None, None, "low_qrs_contrast"
    return y.astype(np.float32), peaks, None


def beat_table(peaks: np.ndarray, n_signal: int = N_SAMPLES) -> tuple[np.ndarray, dict[str, np.ndarray], Counter]:
    """Usable beats of one recording: indices into ``peaks``, their RR features, and drop counts."""
    rr_prev, rr_next, rr_ratio = rr_intervals(peaks)
    usable = usable_mask(peaks, n_signal, rr_prev, rr_next)
    lo, hi = RR_RANGE_S
    with np.errstate(invalid="ignore"):
        plausible = (rr_prev >= lo) & (rr_prev <= hi) & (rr_next >= lo) & (rr_next <= hi)
    keep = usable & plausible
    dropped = Counter(edge_or_missing_rr=int((~usable).sum()), implausible_rr=int((usable & ~plausible).sum()))
    features = {"rr_prev": rr_prev[keep], "rr_next": rr_next[keep], "rr_ratio": rr_ratio[keep]}
    return np.flatnonzero(keep), features, dropped


def fold_split(fold: int) -> str:
    if fold in TRAIN_FOLDS:
        return "train"
    return "val" if fold in VAL_FOLDS else "unused"


def build(root: Path, limit: int | None = None, allow_missing: bool = False) -> tuple[dict[str, np.ndarray], dict]:
    """Run the whole pipeline. Returns record- and beat-level arrays and a stage-by-stage report."""
    root = Path(root)
    meta, report = load_metadata(root)
    if limit:
        meta = meta.head(limit)
    signals, rec_rows, beat_cols = [], [], {k: [] for k in ("beat_record", "r_idx", "rr_prev", "rr_next", "rr_ratio")}
    excluded: Counter = Counter()
    dropped: Counter = Counter()
    for row in meta.itertuples(index=False):
        if not (root / f"{row.filename_hr}.hea").is_file():
            if allow_missing:
                excluded["file_missing"] += 1
                continue
            raise FileNotFoundError(f"{root / row.filename_hr}.hea (use --allow-missing for a partial download)")
        x, fs = raw.read_ptbxl_lead(root, row.filename_hr, LEAD)
        y, peaks, problem = process_record(x, fs)
        if problem:
            excluded[problem] += 1
            continue
        keep, features, drop = beat_table(peaks)
        dropped.update(drop)
        if keep.size == 0:
            excluded["no_usable_beat"] += 1
            continue
        index = len(signals)
        signals.append(y)
        rec_rows.append((row.ecg_id, row.patient_id, row.strat_fold, row.labels))
        beat_cols["beat_record"].append(np.full(keep.size, index, dtype=np.int32))
        beat_cols["r_idx"].append(peaks[keep].astype(np.int32))
        for name in ("rr_prev", "rr_next", "rr_ratio"):
            beat_cols[name].append(features[name].astype(np.float32))

    if not signals:
        raise RuntimeError("No PTB-XL recording survived preprocessing")
    arrays = {
        "signal": np.stack(signals),
        "ecg_id": np.array([r[0] for r in rec_rows], dtype=np.int32),
        "patient_id": np.array([r[1] for r in rec_rows], dtype=np.int64),
        "fold": np.array([r[2] for r in rec_rows], dtype=np.int8),
        "labels": np.stack([r[3] for r in rec_rows]),
        **{name: np.concatenate(chunks) for name, chunks in beat_cols.items()},
    }
    arrays["split"] = np.array([fold_split(int(f)) for f in arrays["fold"]], dtype="<U6")
    check_patient_split(arrays)

    beat_split = arrays["split"][arrays["beat_record"]]
    per_split = {}
    for part in ("train", "val", "unused"):
        rec_mask, beat_mask = arrays["split"] == part, beat_split == part
        per_split[part] = {
            "recordings": int(rec_mask.sum()),
            "patients": int(np.unique(arrays["patient_id"][rec_mask]).size),
            "beats": int(beat_mask.sum()),
            "recordings_per_class": {c: int(arrays["labels"][rec_mask, i].sum()) for i, c in enumerate(SUPERCLASSES)},
        }
    beats_per_record = np.bincount(arrays["beat_record"], minlength=len(signals))
    report.update(
        {
            "processed_from_metadata": int(len(meta)),
            "excluded_by_signal": dict(excluded),
            "recordings_kept": int(len(signals)),
            "beats_kept": int(arrays["r_idx"].size),
            "beats_dropped": dict(dropped),
            "beats_per_recording": {
                "min": int(beats_per_record.min()),
                "median": float(np.median(beats_per_record)),
                "max": int(beats_per_record.max()),
            },
            "per_split": per_split,
            "settings": {
                "lead": LEAD, "source_fs": SOURCE_FS, "fs": FS, "superclasses": list(SUPERCLASSES),
                "train_folds": list(TRAIN_FOLDS), "val_folds": list(VAL_FOLDS), "unused_folds": list(UNUSED_FOLDS),
                "min_peak_to_peak_mv": MIN_PEAK_TO_PEAK_MV, "max_abs_mv": MAX_ABS_MV, "max_flat_s": MAX_FLAT_S,
                "peak_count_range": [MIN_PEAKS, MAX_PEAKS], "rr_range_s": list(RR_RANGE_S),
                "min_qrs_contrast": MIN_QRS_CONTRAST,
            },  # fmt: skip
            "content_sha256": content_sha256(arrays),
        }
    )
    return arrays, report


def check_patient_split(arrays: dict[str, np.ndarray]) -> None:
    """Raise if any patient has recordings in more than one of train, validation and unused."""
    table = pd.DataFrame({"patient": arrays["patient_id"], "split": arrays["split"]})
    leaking = table.groupby("patient")["split"].nunique()
    if (leaking > 1).any():
        raise ValueError(f"PTB-XL patients in more than one split: {leaking[leaking > 1].index.tolist()[:5]}")


def out_dir() -> Path:
    return paths.processed_dir() / "ptbxl"


def load() -> dict[str, np.ndarray]:
    """The preprocessed PTB-XL arrays, after checking the file hash recorded at build time."""
    meta = json.loads((out_dir() / "report.json").read_text())
    path = out_dir() / OUT_NAME
    if sha256_file(path) != meta["file_sha256"]:
        raise RuntimeError(f"{path} does not match the hash recorded when it was built")
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def beat_arrays(arrays: dict[str, np.ndarray], spec: RepresentationSpec, split: str) -> dict[str, np.ndarray]:
    """Windows and labels for one split, cut with the same code path as the MIT-BIH beats.

    Returns ``X`` [n, L], ``y`` [n, 5] multi-hot (the recording's superclasses), ``side`` [n, 3] RR
    features, and ``patient`` and ``ecg_id`` per beat.
    """
    records = np.flatnonzero(arrays["split"] == split)
    xs, rows = [], []
    order = np.argsort(arrays["beat_record"], kind="stable")
    starts = np.searchsorted(arrays["beat_record"][order], records, side="left")
    stops = np.searchsorted(arrays["beat_record"][order], records, side="right")
    for index, start, stop in zip(records, starts, stops):
        beat_ids = order[start:stop]
        if beat_ids.size == 0:
            continue
        xs.append(transform_record(arrays["signal"][index], arrays["r_idx"][beat_ids], spec))
        rows.append(beat_ids)
    beat_ids = np.concatenate(rows)
    rec = arrays["beat_record"][beat_ids]
    return {
        "X": np.concatenate(xs),
        "y": arrays["labels"][rec].astype(np.float32),
        "side": np.stack([arrays[f][beat_ids] for f in ("rr_prev", "rr_next", "rr_ratio")], axis=1),
        "patient": arrays["patient_id"][rec],
        "ecg_id": arrays["ecg_id"][rec],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess PTB-XL into beats for the pretraining arm.")
    parser.add_argument("--root", default=str(paths.repo_root() / "data" / "raw" / "ptbxl"))
    parser.add_argument("--limit", type=int, default=None, help="process only the first N eligible recordings")
    parser.add_argument("--allow-missing", action="store_true", help="skip recordings whose files are absent")
    args = parser.parse_args(argv)

    arrays, report = build(Path(args.root), limit=args.limit, allow_missing=args.allow_missing)
    partial = bool(args.limit) or report["excluded_by_signal"].get("file_missing", 0) > 0
    report["partial_run"] = partial
    out_dir().mkdir(parents=True, exist_ok=True)
    save_npz_deterministic(out_dir() / OUT_NAME, arrays)
    report["file_sha256"] = sha256_file(out_dir() / OUT_NAME)
    (out_dir() / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    name = "ptbxl_preprocessing_sample.json" if partial else "ptbxl_preprocessing.json"
    summary = paths.results_dir() / "data" / name
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "settings"}, indent=2))
    print(f"wrote {out_dir() / OUT_NAME} and {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
