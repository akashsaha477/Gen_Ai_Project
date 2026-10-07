"""Beat extraction: reference R-peak annotations, AAMI labels and RR intervals.

RR intervals are measured between consecutive QRS-type annotations (WFDB ``isqrs``), which include
Q-class beats and ventricular flutter waves, before any class is dropped. Dropping first would
turn the interval across a removed beat into a double-length one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ecgaug.data import raw
from ecgaug.data.records import CLASS_INDEX, FS, NON_BEAT_SYMBOLS, QRS_SYMBOLS, SYMBOL_TO_AAMI

RR_LOCAL_BEATS = 10
UNMAPPED = "-"


def rr_intervals(
    qrs_samples: np.ndarray, fs: int = FS, n_local: int = RR_LOCAL_BEATS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RR features of every QRS annotation in one record.

    Returns ``(rr_prev, rr_next, rr_ratio)``. ``rr_prev`` and ``rr_next`` are in seconds and are NaN
    for the first and last annotation. ``rr_ratio`` is ``rr_prev`` divided by the mean of the up to
    ``n_local`` RR intervals that precede it; a beat with no earlier interval gets ratio 1.
    """
    samples = np.asarray(qrs_samples, dtype=np.float64)
    n = samples.size
    rr_prev = np.full(n, np.nan)
    rr_next = np.full(n, np.nan)
    if n > 1:
        gaps = np.diff(samples) / fs
        rr_prev[1:] = gaps
        rr_next[:-1] = gaps
    # cumulative[k] = rr_prev[1] + ... + rr_prev[k]
    cumulative = np.concatenate([[0.0], np.cumsum(rr_prev[1:])])
    local_mean = np.full(n, np.nan)
    for i in range(1, n):
        last, first = i - 1, max(1, i - n_local)
        if last >= first:
            local_mean[i] = (cumulative[last] - cumulative[first - 1]) / (last - first + 1)
        else:
            local_mean[i] = rr_prev[i]
    return rr_prev, rr_next, rr_prev / local_mean


def extract_beats(record: int, access: str | None = None) -> pd.DataFrame:
    """One row per QRS-type annotation of ``record``.

    Columns: ``record``, ``r_idx`` (sample index of the annotation), ``symbol``, ``aami`` (N, S, V,
    F, Q, or '-' for QRS symbols outside the AAMI map), ``y`` (0-3, or -1 for Q and unmapped),
    ``rr_prev``, ``rr_next`` (seconds) and ``rr_ratio``.
    """
    samples, symbols = raw.read_annotations(record, access)
    unknown = set(symbols.tolist()) - QRS_SYMBOLS - NON_BEAT_SYMBOLS
    if unknown:
        raise ValueError(f"Record {record}: unknown annotation symbols {sorted(unknown)}")

    is_qrs = np.isin(symbols, sorted(QRS_SYMBOLS))
    r_idx, symbol = samples[is_qrs], symbols[is_qrs]
    if r_idx.size > 1 and not np.all(np.diff(r_idx) > 0):
        raise ValueError(f"Record {record}: QRS annotations are not strictly increasing in time")

    rr_prev, rr_next, rr_ratio = rr_intervals(r_idx)
    aami = np.array([SYMBOL_TO_AAMI.get(s, UNMAPPED) for s in symbol], dtype=str)
    y = np.array([CLASS_INDEX.get(c, -1) for c in aami], dtype=np.int8)
    return pd.DataFrame(
        {
            "record": np.full(r_idx.size, int(record), dtype=np.int16),
            "r_idx": r_idx.astype(np.int32),
            "symbol": symbol,
            "aami": aami,
            "y": y,
            "rr_prev": rr_prev,
            "rr_next": rr_next,
            "rr_ratio": rr_ratio,
        }
    )
