"""Fake MIT-BIH-like records for tests and smoke runs.

These are random signals written in WFDB format under real record numbers so that the whole pipeline
(annotation parsing, lead selection by name, the test lock, windowing, training) can be exercised
without real data. They are never used for any result. "Synthetic" in this project means generator
output; these are called fake to keep the two apart.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import wfdb

from ecgaug.data.records import FS

# (symbol, probability, QRS width in seconds, amplitude, premature)
_BEAT_TYPES: tuple[tuple[str, float, float, float, bool], ...] = (
    ("N", 0.62, 0.012, 1.0, False),
    ("L", 0.06, 0.016, 1.0, False),
    ("A", 0.10, 0.012, 0.9, True),
    ("V", 0.10, 0.045, 1.5, True),
    ("F", 0.07, 0.028, 1.2, False),
    ("Q", 0.03, 0.020, 0.6, False),
    ("!", 0.02, 0.060, 0.8, False),
)


def write_fake_record(
    raw_dir: Path, record: int, seconds: float = 90.0, seed: int = 0, swap_leads: bool = False
) -> dict:
    """Write one fake two-lead record with annotations; return what was written (for assertions)."""
    rng = np.random.default_rng(seed + int(record))
    n = int(seconds * FS)
    symbols_pool = [b[0] for b in _BEAT_TYPES]
    probabilities = np.array([b[1] for b in _BEAT_TYPES])
    shapes = {b[0]: b[2:] for b in _BEAT_TYPES}

    samples, symbols = [], []
    t = 0.5
    while t < seconds - 0.2:
        symbol = str(rng.choice(symbols_pool, p=probabilities / probabilities.sum()))
        _, _, premature = shapes[symbol]
        gap = rng.normal(0.8, 0.03) * (0.6 if premature else 1.0)
        t += max(gap, 0.3)
        if t >= seconds - 0.2:
            break
        samples.append(int(round(t * FS)))
        symbols.append(symbol)

    time = np.arange(n) / FS
    mlii = 0.05 * np.sin(2 * np.pi * 0.3 * time) + 0.01 * rng.standard_normal(n)
    for sample, symbol in zip(samples, symbols):
        width, amplitude, _ = shapes[symbol]
        centre = sample / FS
        mlii += amplitude * np.exp(-0.5 * ((time - centre) / width) ** 2)
        mlii += 0.25 * amplitude * np.exp(-0.5 * ((time - centre - 0.25) / 0.05) ** 2)  # T wave
    other = 3.0 - 0.5 * mlii  # a second lead that is easy to tell apart from MLII

    # non-beat annotations mixed in: rhythm change at the start, a noise marker, an artifact
    ann_samples = [10, *samples, int(0.5 * n) + 3, int(0.75 * n) + 5]
    ann_symbols = ["+", *symbols, "~", "|"]
    order = np.argsort(ann_samples, kind="stable")
    ann_samples = np.asarray(ann_samples)[order]
    ann_symbols = [ann_symbols[i] for i in order]

    names = ["V5", "MLII"] if swap_leads else ["MLII", "V1"]
    channels = [other, mlii] if swap_leads else [mlii, other]
    raw_dir.mkdir(parents=True, exist_ok=True)
    wfdb.wrsamp(
        str(record),
        fs=FS,
        units=["mV", "mV"],
        sig_name=names,
        p_signal=np.stack(channels, axis=1),
        fmt=["16", "16"],
        write_dir=str(raw_dir),
    )
    wfdb.wrann(str(record), "atr", sample=ann_samples, symbol=ann_symbols, write_dir=str(raw_dir))
    return {"qrs_samples": np.asarray(samples), "qrs_symbols": symbols, "mlii": mlii, "n": n}


def write_fake_mitdb(
    raw_dir: Path, records: list[int] | tuple[int, ...], seconds: float = 90.0, seed: int = 0
) -> dict[int, dict]:
    """Write fake records; record 114 gets its leads in swapped order, as in the real database."""
    return {
        int(r): write_fake_record(raw_dir, int(r), seconds=seconds, seed=seed, swap_leads=(int(r) == 114))
        for r in records
    }
