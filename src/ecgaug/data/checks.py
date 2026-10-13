"""Checks of the PTB-XL processing choices, measured on MIT-BIH DS1 (never DS2).

PTB-XL recordings are 10 seconds long, MIT-BIH records 30 minutes. Two questions follow, and both
can be answered with DS1 because its records can be cut into 10-second pieces:

1. Edge effect: if a 10-second piece is filtered on its own, how different are its beats from the
   same beats filtered inside the full record, and which padding keeps them closest?
2. Mains interference: how strong is it in each database? (MIT-BIH: 60 Hz. PTB-XL: 50 Hz, reported
   by ``ecgaug.data.ptbxl``.)

Usage: ``python -m ecgaug.data.checks`` writes ``results/data/ptbxl_method_checks_ds1.json``.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from ecgaug import paths
from ecgaug.data import raw
from ecgaug.data.beats import extract_beats
from ecgaug.data.ptbxl import EDGE_PAD_MODE, EDGE_PAD_S, N_SAMPLES, powerline_ratio
from ecgaug.data.records import DS1, FS
from ecgaug.data.representation import USABLE_POST_S, USABLE_PRE_S, RepresentationSpec, transform_record

SEGMENTS_PER_RECORD = 12
DISTANCE_BINS_S = ((0.4, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.01))
PAD_MODES = {"none": None, "odd reflection": ("reflect", "odd"), "mirror": ("reflect", "even"), "edge value": ("edge", None)}


def _windows(segment: np.ndarray, r: np.ndarray, spec: RepresentationSpec, mode: tuple | None) -> np.ndarray:
    if mode is None:
        return transform_record(segment, r, spec)
    pad = round(EDGE_PAD_S * FS)
    kwargs = {"reflect_type": mode[1]} if mode[0] == "reflect" else {}
    return transform_record(np.pad(segment, pad, mode=mode[0], **kwargs), r + pad, spec)


def edge_effect(spec: RepresentationSpec, seed: int = 0) -> dict:
    """Correlation between segment-filtered and record-filtered beat windows, by distance to the end."""
    rng = np.random.default_rng(seed)
    pre, post = round(USABLE_PRE_S * FS), round(USABLE_POST_S * FS)
    correlations: dict[str, list[np.ndarray]] = {name: [] for name in PAD_MODES}
    distances = []
    for record in DS1:
        signal = raw.read_signal(record)
        beats = extract_beats(record)["r_idx"].to_numpy()
        for start in rng.integers(10 * N_SAMPLES, signal.size - 10 * N_SAMPLES, size=SEGMENTS_PER_RECORD):
            r = beats[(beats >= start + pre) & (beats <= start + N_SAMPLES - post)] - start
            if r.size == 0:
                continue
            reference = transform_record(signal, r + start, spec)
            distances.append(np.minimum(r, N_SAMPLES - r) / FS)
            for name, mode in PAD_MODES.items():
                windows = _windows(signal[start : start + N_SAMPLES], r, spec, mode)
                correlations[name].append(np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(windows, reference)]))
    distance = np.concatenate(distances)
    table = {}
    for name in PAD_MODES:
        values = np.concatenate(correlations[name])
        table[name] = {
            f"{lo:.1f}-{min(hi, 5.0):.1f}s": {
                "n_beats": int(((distance >= lo) & (distance < hi)).sum()),
                "median": float(np.median(values[(distance >= lo) & (distance < hi)])),
                "p5": float(np.percentile(values[(distance >= lo) & (distance < hi)], 5)),
            }
            for lo, hi in DISTANCE_BINS_S
        }
    return {"representation": spec.to_dict(), "n_beats": int(distance.size), "correlation_by_distance_to_end": table}


def powerline_ds1() -> dict:
    """Mains-interference ratio at 60 Hz (and 50 Hz for contrast) in the first 5 minutes of each DS1 record."""
    rows = []
    for record in DS1:
        x = raw.read_signal(record)[: 300 * FS]
        rows.append({"record": record, "ratio_60hz": powerline_ratio(x, FS, 60.0), "ratio_50hz": powerline_ratio(x, FS, 50.0)})
    sixty = np.array([r["ratio_60hz"] for r in rows])
    return {
        "median_60hz": float(np.median(sixty)),
        "p90_60hz": float(np.percentile(sixty, 90)),
        "share_above_10_60hz": float((sixty > 10).mean()),
        "median_50hz": float(np.median([r["ratio_50hz"] for r in rows])),
        "per_record": rows,
    }


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="DS1-based checks of the PTB-XL processing choices.").parse_args(argv)
    band = RepresentationSpec(name="m1-B", window_pre_s=0.4, window_post_s=0.6, bandpass_hz=(0.5, 40.0))
    report = {
        "dataset": "MIT-BIH DS1 (22 records); DS2 is not read",
        "chosen_padding": {"mode": "mirror" if EDGE_PAD_MODE == "reflect" else EDGE_PAD_MODE, "seconds": EDGE_PAD_S},
        "edge_effect_bandpass": edge_effect(band),
        "powerline": powerline_ds1(),
    }
    out = paths.results_dir() / "data" / "ptbxl_method_checks_ds1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    table = report["edge_effect_bandpass"]["correlation_by_distance_to_end"]
    nearest = next(iter(table["none"]))
    for name, row in table.items():
        print(f"{name:>15}: beats {nearest} from an end: median {row[nearest]['median']:.4f}, 5th percentile {row[nearest]['p5']:.3f}")
    p = report["powerline"]
    print(f"60 Hz ratio in DS1: median {p['median_60hz']:.1f}, share above 10: {p['share_above_10_60hz']:.2f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
