"""R-peak detection for recordings without beat annotations (PTB-XL), validated on MIT-BIH DS1.

MIT-BIH beats are cut around reference annotations. PTB-XL has none, so its beats are cut around
detected peaks. For the two beat sets to be comparable the detector has to find the same point of
the QRS complex that the MIT-BIH annotators marked. ``python -m ecgaug.data.rpeaks`` measures that
on DS1 (never DS2): sensitivity, positive predictivity and the timing offset to the annotations.

Method (Pan-Tompkins style, zero-phase): 5-20 Hz band-pass, derivative, squaring, 120 ms moving
average; candidates at least 250 ms apart; a candidate is kept if its height reaches a fraction of
the local 75th percentile of candidate heights; each kept peak is moved to the largest absolute
deflection of the band-passed signal within 80 ms.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import signal as sps

from ecgaug import paths
from ecgaug.names import PHASE2_PTBXL_DATA
from ecgaug.data.records import DS1, FS

BAND_HZ = (5.0, 20.0)
INTEGRATION_S = 0.12
REFRACTORY_S = 0.25
LOCAL_WINDOW_S = 5.0
HEIGHT_FRACTION = 0.15
REFINE_S = 0.08
MATCH_TOLERANCE_S = 0.15  # AAMI EC57 matching window for beat detection


def detect_r_peaks(x: np.ndarray, fs: int = FS) -> np.ndarray:
    """Sample indices of detected R peaks in a single-lead signal, sorted and unique."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < fs or not np.isfinite(x).all() or np.ptp(x) == 0:
        return np.empty(0, dtype=np.int64)
    sos = sps.butter(2, BAND_HZ, btype="bandpass", fs=fs, output="sos")
    band = sps.sosfiltfilt(sos, x)
    width = max(1, round(INTEGRATION_S * fs))
    envelope = np.convolve(np.gradient(band) ** 2, np.ones(width) / width, mode="same")

    candidates, _ = sps.find_peaks(envelope, distance=max(1, round(REFRACTORY_S * fs)))
    if candidates.size == 0:
        return np.empty(0, dtype=np.int64)
    heights = envelope[candidates]
    half = LOCAL_WINDOW_S * fs
    keep = np.zeros(candidates.size, dtype=bool)
    lo = hi = 0
    for i, c in enumerate(candidates):
        while candidates[lo] < c - half:
            lo += 1
        while hi < candidates.size and candidates[hi] <= c + half:
            hi += 1
        keep[i] = heights[i] >= HEIGHT_FRACTION * np.percentile(heights[lo:hi], 75)

    radius = round(REFINE_S * fs)
    peaks = []
    for c in candidates[keep]:
        start, stop = max(0, c - radius), min(x.size, c + radius + 1)
        peaks.append(start + int(np.argmax(np.abs(band[start:stop]))))
    peaks = np.unique(np.asarray(peaks, dtype=np.int64))
    # two candidates can refine to nearly the same point: keep the first of any pair inside the refractory time
    if peaks.size > 1:
        peaks = peaks[np.concatenate([[True], np.diff(peaks) >= round(REFRACTORY_S * fs * 0.8)])]
    return peaks


def qrs_contrast(x: np.ndarray, peaks: np.ndarray, fs: int = FS) -> float:
    """How far the detected peaks stand out of the signal: median band-passed amplitude at the peaks
    divided by a robust noise scale (1.4826 x median absolute deviation of the band-passed signal).

    Clean ECG scores well above 10; noise without QRS complexes scores around 3.
    """
    if len(peaks) == 0:
        return 0.0
    sos = sps.butter(2, BAND_HZ, btype="bandpass", fs=fs, output="sos")
    band = sps.sosfiltfilt(sos, np.asarray(x, dtype=np.float64))
    scale = 1.4826 * np.median(np.abs(band - np.median(band)))
    return float(np.median(np.abs(band[np.asarray(peaks)])) / scale) if scale > 0 else 0.0


def match_peaks(reference: np.ndarray, detected: np.ndarray, tolerance: int) -> tuple[int, int, int, np.ndarray]:
    """One-to-one matching within ``tolerance`` samples: (TP, FN, FP, signed offsets of the matches)."""
    reference, detected = np.asarray(reference), np.asarray(detected)
    offsets = []
    j = 0
    used = np.zeros(detected.size, dtype=bool)
    for r in reference:
        while j < detected.size and detected[j] < r - tolerance:
            j += 1
        best, k = None, j
        while k < detected.size and detected[k] <= r + tolerance:
            if not used[k] and (best is None or abs(detected[k] - r) < abs(detected[best] - r)):
                best = k
            k += 1
        if best is not None:
            used[best] = True
            offsets.append(int(detected[best] - r))
    tp = len(offsets)
    return tp, int(reference.size - tp), int(detected.size - tp), np.asarray(offsets, dtype=np.int64)


def validate_on_ds1() -> dict:
    """Run the detector on every DS1 record and compare with the reference QRS annotations."""
    from ecgaug.data import raw
    from ecgaug.data.beats import extract_beats

    tolerance = round(MATCH_TOLERANCE_S * FS)
    rows, all_offsets = [], []
    for record in DS1:
        reference = extract_beats(record)["r_idx"].to_numpy()
        detected = detect_r_peaks(raw.read_signal(record))
        tp, fn, fp, offsets = match_peaks(reference, detected, tolerance)
        all_offsets.append(offsets)
        rows.append(
            {
                "record": record,
                "reference": int(reference.size),
                "detected": int(detected.size),
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "sensitivity": tp / (tp + fn),
                "positive_predictivity": tp / (tp + fp) if tp + fp else float("nan"),
                "median_offset_ms": float(np.median(offsets) * 1000 / FS) if tp else float("nan"),
            }
        )
    offsets = np.concatenate(all_offsets) * 1000 / FS
    tp, fn, fp = (sum(r[k] for r in rows) for k in ("tp", "fn", "fp"))
    return {
        "dataset": "MIT-BIH DS1 (22 records), reference = all QRS-type annotations",
        "tolerance_ms": MATCH_TOLERANCE_S * 1000,
        "parameters": {
            "band_hz": BAND_HZ, "integration_s": INTEGRATION_S, "refractory_s": REFRACTORY_S,
            "local_window_s": LOCAL_WINDOW_S, "height_fraction": HEIGHT_FRACTION, "refine_s": REFINE_S,
        },  # fmt: skip
        "overall": {
            "tp": tp, "fn": fn, "fp": fp,
            "sensitivity": tp / (tp + fn),
            "positive_predictivity": tp / (tp + fp),
            "offset_ms": {
                "median": float(np.median(offsets)),
                "mean": float(offsets.mean()),
                "sd": float(offsets.std()),
                "abs_p95": float(np.percentile(np.abs(offsets), 95)),
                "within_10ms": float((np.abs(offsets) <= 10).mean()),
            },
        },  # fmt: skip
        "per_record": rows,
    }


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Validate the R-peak detector on MIT-BIH DS1.").parse_args(argv)
    report = validate_on_ds1()
    out = paths.results_for(PHASE2_PTBXL_DATA) / "rpeak_validation_ds1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    o = report["overall"]
    print(f"Se {o['sensitivity']:.4f}  +P {o['positive_predictivity']:.4f}  (TP {o['tp']}, FN {o['fn']}, FP {o['fp']})")
    print("offset ms:", {k: round(v, 2) for k, v in o["offset_ms"].items()})
    worst = sorted(report["per_record"], key=lambda r: min(r["sensitivity"], r["positive_predictivity"]))[:4]
    for r in worst:
        print(f"  record {r['record']}: Se {r['sensitivity']:.4f} +P {r['positive_predictivity']:.4f} off {r['median_offset_ms']:.1f} ms")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
