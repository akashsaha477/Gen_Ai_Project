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


PTBXL_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# ecg_id: (patient, fold, scp_codes, pacemaker, electrodes_problems, kind of signal)
_FAKE_PTBXL = {
    1: (10, 1, "{'NORM': 100.0, 'SR': 0.0}", "", "", "ok"),
    2: (10, 1, "{'IMI': 50.0, 'NDT': 0.0}", "", "", "ok"),
    3: (11, 9, "{'CLBBB': 100.0, 'LVH': 80.0}", "", "", "swapped"),
    4: (12, 10, "{'NORM': 80.0}", "", "", "ok"),
    5: (13, 2, "{'SR': 0.0}", "", "", "ok"),  # no diagnostic statement
    6: (14, 2, "{'NORM': 100.0}", "ja, pacemaker", "", "ok"),
    7: (15, 3, "{'NORM': 100.0}", "", "V6", "ok"),
    8: (16, 3, "{'NORM': 100.0}", "", "", "flat"),
    9: (17, 4, "{'NORM': 100.0}", "", "", "noise"),  # no detectable beats
}
_FAKE_SCP = [  # code, diagnostic, diagnostic_class
    ("NORM", 1.0, "NORM"), ("IMI", 1.0, "MI"), ("NDT", 1.0, "STTC"), ("CLBBB", 1.0, "CD"), ("LVH", 1.0, "HYP"),
    ("SR", "", ""),
]  # fmt: skip


_FAKE_EXTRA_CODES = ("NORM", "IMI", "NDT", "CLBBB", "LVH")  # one per superclass, with a QRS width each
_FAKE_EXTRA_WIDTHS = (0.010, 0.016, 0.022, 0.030, 0.040)


def write_fake_ptbxl(root: Path, seed: int = 0, n_extra: int = 0) -> dict[int, np.ndarray]:
    """Write a tiny PTB-XL-like folder (two CSV files and nine 500 Hz recordings).

    Returns the true R-peak sample indices at 500 Hz for the recordings with beats. Recording 3 has
    its leads in a different order, to check that lead II is selected by name. ``n_extra`` adds
    clean recordings (ids from 101, two per patient, all ten folds, one superclass each with its own
    QRS width) so that there is enough data to train on in a smoke run.
    """
    root = Path(root)
    rng = np.random.default_rng(seed)
    fs, n = 500, 5000
    time = np.arange(n) / fs
    rows, truth = [], {}
    records = {k: (*v, 0.012, 0.75 + 0.02 * k) for k, v in _FAKE_PTBXL.items()}
    for i in range(n_extra):
        patient = 100 + i // 2
        label = i % len(_FAKE_EXTRA_CODES)
        codes = f"{{'{_FAKE_EXTRA_CODES[label]}': 100.0}}"
        records[101 + i] = (patient, patient % 10 + 1, codes, "", "", "ok", _FAKE_EXTRA_WIDTHS[label], 0.7 + 0.01 * (i % 7))
    for ecg_id, (patient, fold, codes, pacemaker, electrodes, kind, width, period) in records.items():
        filename = f"records500/00000/{ecg_id:05d}_hr"
        lead2 = 0.01 * rng.standard_normal(n)
        if kind in ("ok", "swapped"):
            beats = np.arange(0.5 + 0.05 * (ecg_id % 3), 9.6, period)
            for b in beats:
                lead2 += np.exp(-0.5 * ((time - b) / width) ** 2) + 0.2 * np.exp(-0.5 * ((time - b - 0.25) / 0.05) ** 2)
            truth[ecg_id] = np.rint(beats * fs).astype(int)
        elif kind == "flat":
            lead2 = np.zeros(n)
        signals = np.stack([lead2 if name == "II" else 2.0 + 0.3 * i - 0.2 * lead2 for i, name in enumerate(PTBXL_LEADS)], axis=1)
        names = list(PTBXL_LEADS)
        if kind == "swapped":
            names = names[::-1]
            signals = signals[:, ::-1]
        (root / "records500" / "00000").mkdir(parents=True, exist_ok=True)
        wfdb.wrsamp(
            f"{ecg_id:05d}_hr", fs=fs, units=["mV"] * 12, sig_name=names, p_signal=np.ascontiguousarray(signals),
            fmt=["16"] * 12, write_dir=str(root / "records500" / "00000"),
        )  # fmt: skip
        rows.append(f'{ecg_id},{patient}.0,"{codes}",{fold},{filename},"{pacemaker}",{electrodes}')
    header = "ecg_id,patient_id,scp_codes,strat_fold,filename_hr,pacemaker,electrodes_problems"
    (root / "ptbxl_database.csv").write_text(header + "\n" + "\n".join(rows) + "\n")
    scp = ["," + "diagnostic,diagnostic_class"] + [f"{code},{diag},{cls}" for code, diag, cls in _FAKE_SCP]
    (root / "scp_statements.csv").write_text("\n".join(scp) + "\n")
    return truth
