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

Usage: ``python -m ecgaug.data.ptbxl --root <folder that holds ptbxl_database.csv> [--workers 4]``
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
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
REFERENCE_SHA256_URL = "https://physionet.org/files/ptb-xl/1.0.3/SHA256SUMS.txt"
POWERLINE_HZ = 50.0  # PTB-XL was recorded in Germany
# Mirror padding before filtering a 10-second recording. MIT-BIH records are 30 minutes long, so their
# beats never see a filter's start-up transient; PTB-XL beats near the ends of a recording would.
# Mirror (even) reflection was the best of four treatments measured on MIT-BIH DS1
# (results/data/ptbxl_method_checks_ds1.json, from ``python -m ecgaug.data.checks``).
EDGE_PAD_S = 2.0
EDGE_PAD_MODE = "reflect"


def find_root(path: Path) -> Path:
    """The folder that holds ``ptbxl_database.csv``, looked for in ``path`` and below it."""
    path = Path(path)
    if (path / "ptbxl_database.csv").is_file():
        return path
    hits = sorted(path.rglob("ptbxl_database.csv"))
    if not hits:
        raise FileNotFoundError(f"no ptbxl_database.csv in or below {path}")
    return hits[0].parent


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


def powerline_ratio(x: np.ndarray, fs: int, f0: float = POWERLINE_HZ) -> float:
    """Spectral peak within 1 Hz of ``f0`` divided by the median power 2-6 Hz to either side.

    About 1-3 for a signal without mains interference; above 10 when it is clearly present.
    """
    freqs, power = sps.welch(np.asarray(x, dtype=np.float64), fs=fs, nperseg=min(len(x), 4 * fs))
    band = np.abs(freqs - f0) < 1
    side = (np.abs(freqs - f0) > 2) & (np.abs(freqs - f0) < 6)
    return float(power[band].max() / np.median(power[side]))


def process_record(x: np.ndarray, fs: int) -> dict:
    """Quality-check, resample and detect peaks in one raw lead.

    Returns ``{"problem": reason}`` for an unusable recording, else the resampled signal, the R
    peaks, and the quality measures that go into the report.
    """
    problem = signal_problem(x, fs)
    if problem:
        return {"problem": problem}
    y = resample_to_fs(x, fs)
    peaks = detect_r_peaks(y)
    if not MIN_PEAKS <= peaks.size <= MAX_PEAKS:
        return {"problem": "implausible_peak_count"}
    contrast = qrs_contrast(y, peaks)
    if contrast < MIN_QRS_CONTRAST:
        return {"problem": "low_qrs_contrast"}
    return {
        "problem": None,
        "signal": y.astype(np.float32),
        "peaks": peaks,
        "qrs_contrast": contrast,
        "heart_rate_bpm": float(60.0 * FS / np.median(np.diff(peaks))),
        "powerline_ratio": powerline_ratio(x, fs),
    }


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


def _process_file(task: tuple[str, str, bool]) -> dict:
    """Everything that happens to one recording file. Top-level so worker processes can run it."""
    root, filename, with_hashes = task
    hea, dat = Path(root) / f"{filename}.hea", Path(root) / f"{filename}.dat"
    if not hea.is_file() or not dat.is_file():
        return {"problem": "file_missing"}
    x, fs = raw.read_ptbxl_lead(Path(root), filename, LEAD)
    out = process_record(x, fs)
    if with_hashes:
        out["sha256"] = {f"{filename}.hea": raw.file_sha256(hea), f"{filename}.dat": raw.file_sha256(dat)}
    if out["problem"]:
        return out
    peaks = out.pop("peaks")
    keep, features, dropped = beat_table(peaks)
    out["dropped"] = dict(dropped)
    if keep.size == 0:
        out["problem"] = "no_usable_beat"
        out.pop("signal")
        return out
    out["r_idx"] = peaks[keep]
    out["features"] = features
    return out


def _percentiles(values: list[float], points: tuple[float, ...]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {f"p{g:g}": float(np.percentile(array, g)) for g in points}


def verify_checksums(root: Path, hashes: dict[str, str], reference: dict[str, str] | None = None) -> dict:
    """Compare the files that were read with PhysioNet's published SHA-256 list for PTB-XL 1.0.3.

    A copy from Kaggle may be another version, so a mismatch is reported, not fatal.
    """
    root = Path(root)
    hashes = dict(hashes)
    for name in ("ptbxl_database.csv", "scp_statements.csv"):
        hashes[name] = raw.file_sha256(root / name)
    out: dict = {"reference_url": REFERENCE_SHA256_URL, "files_checked": len(hashes)}
    out["metadata_sha256"] = {name: hashes[name] for name in ("ptbxl_database.csv", "scp_statements.csv")}
    if reference is None:
        reference = raw.fetch_reference_checksums(REFERENCE_SHA256_URL)
    if reference is None:
        return {**out, "reference_available": False}
    mismatched = sorted(name for name, digest in hashes.items() if name in reference and reference[name] != digest)
    not_listed = sorted(name for name in hashes if name not in reference)
    out.update(
        reference_available=True,
        matched=len(hashes) - len(mismatched) - len(not_listed),
        mismatched=len(mismatched),
        not_in_reference=len(not_listed),
        examples_mismatched=mismatched[:10],
        metadata_matches_reference={
            name: reference.get(name) == hashes[name] for name in ("ptbxl_database.csv", "scp_statements.csv")
        },
    )
    return out


def build(
    root: Path,
    limit: int | None = None,
    allow_missing: bool = False,
    workers: int = 1,
    verify: bool = False,
) -> tuple[dict[str, np.ndarray], dict]:
    """Run the whole pipeline. Returns record- and beat-level arrays and a stage-by-stage report.

    The result does not depend on ``workers``. With ``verify`` the files that were read are checked
    against PhysioNet's checksum list.
    """
    root = Path(root)
    meta, report = load_metadata(root)
    if limit:
        meta = meta.head(limit)
    tasks = [(str(root), filename, verify) for filename in meta["filename_hr"]]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outputs = list(pool.map(_process_file, tasks, chunksize=64))
    else:
        outputs = [_process_file(task) for task in tasks]

    signals, rec_rows, beat_cols = [], [], {k: [] for k in ("beat_record", "r_idx", "rr_prev", "rr_next", "rr_ratio")}
    excluded: Counter = Counter()
    dropped: Counter = Counter()
    hashes: dict[str, str] = {}
    quality: dict[str, list[float]] = {"heart_rate_bpm": [], "qrs_contrast": [], "powerline_ratio": []}
    for row, out in zip(meta.itertuples(index=False), outputs):
        hashes.update(out.get("sha256", {}))
        if out["problem"] == "file_missing" and not allow_missing:
            raise FileNotFoundError(f"{root / row.filename_hr} (use --allow-missing for a partial download)")
        dropped.update(out.get("dropped", {}))
        if out["problem"]:
            excluded[out["problem"]] += 1
            continue
        index = len(signals)
        signals.append(out["signal"])
        rec_rows.append((row.ecg_id, row.patient_id, row.strat_fold, row.labels))
        beat_cols["beat_record"].append(np.full(out["r_idx"].size, index, dtype=np.int32))
        beat_cols["r_idx"].append(out["r_idx"].astype(np.int32))
        for name in ("rr_prev", "rr_next", "rr_ratio"):
            beat_cols[name].append(out["features"][name].astype(np.float32))
        for name in quality:
            quality[name].append(out[name])

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
    powerline = np.asarray(quality["powerline_ratio"])
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
            "quality_of_kept_recordings": {
                "heart_rate_bpm": _percentiles(quality["heart_rate_bpm"], (1, 5, 50, 95, 99)),
                "qrs_contrast": _percentiles(quality["qrs_contrast"], (1, 5, 50)),
                "powerline_50hz_ratio": {
                    **_percentiles(quality["powerline_ratio"], (50, 90, 99)),
                    "share_above_10": float((powerline > 10).mean()),
                },
            },
            "per_split": per_split,
            "settings": {
                "lead": LEAD, "source_fs": SOURCE_FS, "fs": FS, "superclasses": list(SUPERCLASSES),
                "train_folds": list(TRAIN_FOLDS), "val_folds": list(VAL_FOLDS), "unused_folds": list(UNUSED_FOLDS),
                "min_peak_to_peak_mv": MIN_PEAK_TO_PEAK_MV, "max_abs_mv": MAX_ABS_MV, "max_flat_s": MAX_FLAT_S,
                "peak_count_range": [MIN_PEAKS, MAX_PEAKS], "rr_range_s": list(RR_RANGE_S),
                "min_qrs_contrast": MIN_QRS_CONTRAST, "edge_pad_s": EDGE_PAD_S, "edge_pad_mode": EDGE_PAD_MODE,
            },  # fmt: skip
            "content_sha256": content_sha256(arrays),
        }
    )
    if verify:
        report["checksums"] = verify_checksums(root, hashes)
    return arrays, report


def check_patient_split(arrays: dict[str, np.ndarray]) -> None:
    """Raise if any patient has recordings in more than one of train, validation and unused."""
    table = pd.DataFrame({"patient": arrays["patient_id"], "split": arrays["split"]})
    leaking = table.groupby("patient")["split"].nunique()
    if (leaking > 1).any():
        raise ValueError(f"PTB-XL patients in more than one split: {leaking[leaking > 1].index.tolist()[:5]}")


def out_dir() -> Path:
    return paths.processed_dir() / "ptbxl"


def build_report() -> dict | None:
    """The report of the build on disk, or None if PTB-XL has not been preprocessed here."""
    path = out_dir() / "report.json"
    return json.loads(path.read_text()) if path.is_file() else None


def is_full_build() -> bool:
    """True if a complete (not sample or partial) build is on disk."""
    report = build_report()
    return bool(report) and not report.get("partial_run", True) and (out_dir() / OUT_NAME).is_file()


def load() -> dict[str, np.ndarray]:
    """The preprocessed PTB-XL arrays, after checking the file hash recorded at build time."""
    report = build_report()
    path = out_dir() / OUT_NAME
    if report is None or not path.is_file():
        raise FileNotFoundError(f"PTB-XL has not been preprocessed here ({path} is missing)")
    if sha256_file(path) != report["file_sha256"]:
        raise RuntimeError(f"{path} does not match the hash recorded when it was built")
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def prepare(
    root: Path, workers: int = 1, limit: int | None = None, allow_missing: bool = False, verify: bool = True
) -> dict:
    """Build, save and report. Writes ``data/processed/ptbxl/`` and a summary under ``results/data/``."""
    arrays, report = build(Path(root), limit=limit, allow_missing=allow_missing, workers=workers, verify=verify)
    partial = bool(limit) or report["excluded_by_signal"].get("file_missing", 0) > 0
    report["partial_run"] = partial
    out_dir().mkdir(parents=True, exist_ok=True)
    save_npz_deterministic(out_dir() / OUT_NAME, arrays)
    report["file_sha256"] = sha256_file(out_dir() / OUT_NAME)
    (out_dir() / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    name = "ptbxl_preprocessing_sample.json" if partial else "ptbxl_preprocessing.json"
    summary = paths.results_dir() / "data" / name
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(report, indent=2) + "\n")
    return report


def windows_for_record(signal: np.ndarray, r_idx: np.ndarray, spec: RepresentationSpec) -> np.ndarray:
    """Windows of one 10-second recording, cut by the MIT-BIH code after mirror padding.

    The padding keeps most of the start-up transient of a band-pass filter outside the recording,
    so beats near its ends are filtered almost as they would be inside a long record.
    """
    pad = round(EDGE_PAD_S * FS)
    padded = np.pad(np.asarray(signal, dtype=np.float64), pad, mode=EDGE_PAD_MODE)
    return transform_record(padded, np.asarray(r_idx, dtype=np.int64) + pad, spec)


def beat_arrays(arrays: dict[str, np.ndarray], spec: RepresentationSpec, split: str) -> dict[str, np.ndarray]:
    """Windows and labels for one split, cut with the same code path as the MIT-BIH beats.

    Returns ``X`` [n, L], ``y`` [n, 5] multi-hot (the recording's superclasses), ``side`` [n, 3] RR
    features, and ``patient`` and ``ecg_id`` per beat.
    """
    records = np.flatnonzero(arrays["split"] == split)
    order = np.argsort(arrays["beat_record"], kind="stable")
    starts = np.searchsorted(arrays["beat_record"][order], records, side="left")
    stops = np.searchsorted(arrays["beat_record"][order], records, side="right")
    xs, rows = [], []
    for index, start, stop in zip(records, starts, stops):
        beat_ids = order[start:stop]
        if beat_ids.size == 0:
            continue
        xs.append(windows_for_record(arrays["signal"][index], arrays["r_idx"][beat_ids], spec))
        rows.append(beat_ids)
    if not rows:
        raise ValueError(f"PTB-XL split {split!r} has no beats")
    beat_ids = np.concatenate(rows)
    rec = arrays["beat_record"][beat_ids]
    return {
        "X": np.concatenate(xs),
        "y": arrays["labels"][rec].astype(np.float32),
        "side": np.stack([arrays[f][beat_ids] for f in ("rr_prev", "rr_next", "rr_ratio")], axis=1),
        "patient": arrays["patient_id"][rec],
        "ecg_id": arrays["ecg_id"][rec],
    }


def alignment_report(ptbxl_x: np.ndarray, mitbih_x: np.ndarray, spec: RepresentationSpec) -> dict:
    """How alike two sets of beat windows are: mean-beat correlation and R-peak centring.

    "Centred" means the largest absolute value of a window lies within 6 ms of the R index. Call it
    with PTB-XL beats from normal recordings and MIT-BIH N beats of the training records.
    """
    pre = spec.window_samples[0]
    tolerance = max(1, round(0.006 * spec.effective_fs))

    def centred(x: np.ndarray) -> float:
        return float((np.abs(np.abs(x).argmax(axis=1) - pre) <= tolerance).mean())

    return {
        "n_ptbxl_beats": int(len(ptbxl_x)),
        "n_mitbih_beats": int(len(mitbih_x)),
        "mean_beat_correlation": float(np.corrcoef(ptbxl_x.mean(axis=0), mitbih_x.mean(axis=0))[0, 1]),
        "ptbxl_peak_at_centre": centred(ptbxl_x),
        "mitbih_peak_at_centre": centred(mitbih_x),
        "tolerance_samples": tolerance,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess PTB-XL into beats for the pretraining arm.")
    parser.add_argument("--root", default=str(paths.repo_root() / "data" / "raw" / "ptbxl"),
                        help="folder that holds ptbxl_database.csv, or any folder above it")  # fmt: skip
    parser.add_argument("--download", action="store_true",
                        help="download PTB-XL 1.0.3 from PhysioNet (1.8 GB) into data/raw/ptbxl first")  # fmt: skip
    parser.add_argument("--workers", type=int, default=1, help="worker processes (4 on Kaggle)")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N eligible recordings")
    parser.add_argument("--allow-missing", action="store_true", help="skip recordings whose files are absent")
    parser.add_argument("--skip-verify", action="store_true", help="do not compare files with PhysioNet checksums")
    args = parser.parse_args(argv)

    if args.download:
        raw.download_ptbxl(Path(args.root))
    report = prepare(
        find_root(Path(args.root)), workers=args.workers, limit=args.limit,
        allow_missing=args.allow_missing, verify=not args.skip_verify,
    )  # fmt: skip
    print(json.dumps({k: v for k, v in report.items() if k != "settings"}, indent=2))
    checks = report.get("checksums", {})
    if checks.get("mismatched"):
        print(f"WARNING: {checks['mismatched']} files differ from PhysioNet PTB-XL 1.0.3 (another version?)")
    print(f"wrote {out_dir() / OUT_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
