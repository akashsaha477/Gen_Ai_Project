"""phase6_icentia: the P13 cnn trained on many more patients, from Icentia11k (PREREGISTRATION §10 v13, P14; post hoc).

    python -m ecgaug.experiments.phase6_icentia --config configs/phase6_icentia/icentia.yaml --part check       # training-side annotations only
    python -m ecgaug.experiments.phase6_icentia --config ... --part grid [--device cuda]   # downloads, builds, trains
    python -m ecgaug.experiments.phase6_icentia --config ... --part select
    ALLOW_TEST_POSTHOC=1 python -m ecgaug.experiments.phase6_icentia --config ... --part test        # one posthoc access

P11 to P13 and P13's design check point to patient variation: the training patients hold almost none of the rhythm patterns
that make DS2 hard (one S-dominant recording among 175 training-side recordings, and no model with a local RR reference
finds it when it is held out). The conclusion asks for more training patients. Icentia11k (PhysioNet; 11,000 patients;
technician-labelled N, S, V and Q beats; one chest lead at 250 Hz; licence CC BY-NC-SA 4.0) is the largest open source of
labelled S beats. P14 adds the first segment (about 70 minutes) of ``patients`` patients drawn at random to P13's data,
processed as P11 processed INCART and SVDB and in P13's form, and retrains the P7 cnn in four of P13's conditions at the
same strengths. Nothing is chosen on DS2.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import yaml

from ecgaug import paths, runs, workflow
from ecgaug.data import raw
from ecgaug.data.access import LockedTestDataError
from ecgaug.data.beats import UNMAPPED, rr_intervals
from ecgaug.data.contract import content_sha256, sha256_file
from ecgaug.data.records import CLASS_INDEX, CLASSES, DS1, DS2, FS, NON_BEAT_SYMBOLS, QRS_SYMBOLS, SYMBOL_TO_AAMI
from ecgaug.data.representation import RepresentationSpec, transform_record, usable_mask
from ecgaug.experiments import phase6_more_patients as mp
from ecgaug.experiments import phase6_patient_template as pt
from ecgaug.experiments import phase6_record_rr as rr
from ecgaug.experiments import phase6_relative_generators as rg
from ecgaug.experiments import phase6_rr_baseline as v5
from ecgaug.experiments import phase6_rr_reference as ref
from ecgaug.experiments.common import PreconditionError, check_preconditions
from ecgaug.experiments.data_source import DataSource, load_data
from ecgaug.generators.base import training_beats
from ecgaug.grid import runner
from ecgaug.names import PHASE6_ICENTIA
from ecgaug.train.loop import resolve_device

EXPERIMENT = PHASE6_ICENTIA
DB = "icentia11k"
URL = "https://physionet.org/files/icentia11k-continuous-ecg/1.0"
LEAD = "ecg"
N_PATIENTS = 11_000
RECORD_OFFSET = 10_000  # patient p -> record 10000 + p; MIT-BIH records are 100-234, SVDB 800-894, INCART 1-75
CONDITIONS = ("none", "class_weighted", "random_oversampling", "cvae")  # WGAN-GP's bank is too small at 0.25 of the larger N count
CONFIG_KEYS = {"experiment", "record_rr_config", "patients", "patient_seed", "n_cap_per_record", "cap_seed", "validation_every", "rule_threshold"}
GRID_NAME = "full_grid"


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if set(cfg) != CONFIG_KEYS or cfg["experiment"] != EXPERIMENT:
        raise ValueError(f"{path}: expected experiment {EXPERIMENT} and exactly the keys {sorted(CONFIG_KEYS)}")
    if not 0 < int(cfg["patients"]) <= N_PATIENTS:
        raise ValueError(f"{path}: patients must lie in 1..{N_PATIENTS}")
    return {**cfg, "file": str(path), "rr": rr.load_config(paths.repo_root() / cfg["record_rr_config"])}


# --------------------------------------------------------------------------------------------------
# Icentia11k: the sample, the download and one record in P13's form
# --------------------------------------------------------------------------------------------------


def sample_patients(n: int, seed: int) -> list[int]:
    """``n`` distinct patients of the 11,000, drawn at random with ``seed`` (no label is looked at), in sorted order."""
    return sorted(int(p) for p in np.random.default_rng(seed).choice(N_PATIENTS, size=n, replace=False))


def segment_name(patient: int) -> str:
    return f"p{patient:05d}_s00"


def segment_url(patient: int, extension: str) -> str:
    """PhysioNet's layout: ``p09/p09000/p09000_s00.hea`` for patient 9000."""
    return f"{URL}/p{patient // 1000:02d}/p{patient:05d}/{segment_name(patient)}.{extension}"


def record_number(name: str) -> int:
    """``p00002_s00`` -> 10002."""
    return RECORD_OFFSET + int(name[1:6])


def _fetch(urls: list[str], target: Path, retries: int) -> None:
    """The first of ``urls`` that answers (PhysioNet's open-data mirror on AWS, then PhysioNet itself), to ``target``."""
    part = target.with_name(target.name + ".part")
    for attempt in range(retries):
        for url in urls:
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "ecgaug-research"})
                with urllib.request.urlopen(request, timeout=120) as response, open(part, "wb") as out:
                    out.write(response.read())
                part.rename(target)
                return
            except OSError:
                continue
        time.sleep(2**attempt)
    raise OSError(f"Could not download {target.name} from {urls}")


def download(patients: list[int], dest: Path, retries: int = 4, workers: int = 8) -> list[str]:
    """Each patient's first segment (``.hea``, ``.atr``, ``.dat``) into ``dest``, skipping files already there."""
    from concurrent.futures import ThreadPoolExecutor

    dest.mkdir(parents=True, exist_ok=True)
    jobs = []
    for patient in patients:
        for extension in ("hea", "atr", "dat"):
            target = dest / f"{segment_name(patient)}.{extension}"
            if not (target.is_file() and target.stat().st_size > 0):
                url = segment_url(patient, extension)
                jobs.append(([url.replace("https://physionet.org/files/", "https://physionet-open.s3.amazonaws.com/"), url], target))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda job: _fetch(job[0], job[1], retries), jobs))
    return [segment_name(p) for p in patients]


def build_record(name: str, spec: RepresentationSpec, cap: int, cap_seed: int, half_window_s: float) -> tuple[dict | None, dict]:
    """One segment's beats in P13's form: the frozen representation after resampling to 360 Hz, minus the beat's local
    template (all QRS annotations within the half window); v2's ratio; ``rr_prev`` and ``rr_next`` over the segment's mean
    RR interval. At most ``cap`` N beats are kept (P11's rule), chosen before the templates are computed; Q beats are dropped."""
    samples, symbols, lead_signal, fs, sig_names = raw.read_external(DB, name, LEAD)
    report: dict = {"record": name, "number": record_number(name)}
    unknown = sorted(set(symbols.tolist()) - QRS_SYMBOLS - NON_BEAT_SYMBOLS)
    if unknown:
        return None, {**report, "excluded": f"unknown annotation symbols {unknown}"}
    if lead_signal is None:
        return None, {**report, "excluded": f"lead {LEAD} not in {sig_names}"}
    is_qrs = np.isin(symbols, sorted(QRS_SYMBOLS))
    r_native, symbol = samples[is_qrs], symbols[is_qrs]
    if r_native.size < 3:
        return None, {**report, "excluded": "fewer than 3 QRS annotations"}
    rr_prev, rr_next, _ = rr_intervals(r_native, fs=fs)
    v2 = v5.baseline_ratio(r_native, fs, half_window_s)
    mean = rr.record_mean_rr(r_native, fs)
    signal = np.nan_to_num(lead_signal)
    if fs != FS:
        from ecgaug.experiments import phase6_external_validation as ext

        signal = ext.resample_to_fs(signal, fs)
    r = np.rint(r_native * FS / fs).astype(np.int64)
    aami = np.array([SYMBOL_TO_AAMI.get(s, UNMAPPED) for s in symbol], dtype=str)
    y = np.array([CLASS_INDEX.get(c, -1) for c in aami], dtype=np.int8)
    usable = np.flatnonzero(usable_mask(r, signal.size, rr_prev, rr_next) & (y >= 0))
    keep = usable[mp.cap_normal(y[usable], np.zeros(usable.size, dtype=np.int64), cap, cap_seed)]
    n = int(keep.size)
    if not n:
        return None, {**report, "excluded": "no usable labelled beat"}
    x = transform_record(signal, r[keep], spec) - pt.signal_templates(signal, r, r[keep], spec, half_window_s)
    arrays = {
        "X": x.astype(np.float32), "y": y[keep], "record": np.full(n, record_number(name), dtype=np.int16), "split": np.full(n, "train", dtype="<U5"),
        "r_idx": r[keep].astype(np.int32), "rr_prev": (rr_prev[keep] / mean).astype(np.float32), "rr_next": (rr_next[keep] / mean).astype(np.float32),
        "rr_ratio": v2[keep].astype(np.float32), "is_synthetic": np.zeros(n, dtype=bool), "source": np.full(n, "real", dtype="<U4"),
    }  # fmt: skip
    report.update(fs=int(fs), mean_rr_s=mean, kept={c: int((arrays["y"] == i).sum()) for i, c in enumerate(CLASSES)},
                  usable={c: int((y[usable] == i).sum()) for i, c in enumerate(CLASSES)})  # fmt: skip
    return arrays, report


def build_icentia(cfg: dict, spec: RepresentationSpec) -> tuple[dict, dict]:
    """The sampled patients' beats, every ``validation_every``-th patient (sorted) in validation."""
    patients = sample_patients(int(cfg["patients"]), int(cfg["patient_seed"]))
    names = download(patients, raw.external_dir(DB))
    half = float(cfg["rr"]["rg"]["mp"]["pt"]["half_window_s"])
    parts, reports = [], []
    for name in names:
        arrays, report = build_record(name, spec, int(cfg["n_cap_per_record"]), int(cfg["cap_seed"]), half)
        reports.append(report)
        if arrays is not None:
            parts.append(arrays)
    if not parts:
        raise PreconditionError("No Icentia11k segment was usable")
    folder = raw.external_dir(DB)
    files = {f"{n}.{e}": sha256_file(folder / f"{n}.{e}") for n in names for e in ("hea", "atr", "dat")}
    arrays = {k: np.concatenate([p[k] for p in parts]) for k in mp.FIELDS}
    units = np.asarray(arrays["record"]).astype(np.int64)
    val = mp.validation_units(units, int(cfg["validation_every"]))
    arrays["split"] = np.where(np.isin(units, sorted(val)), "val", "train").astype("<U5")
    return arrays, {"patients": patients, "validation_units": sorted(int(u) for u in val), "records": reports, "files_sha256": files}


def concat(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """``first`` then ``second`` in ``first``'s dtype (strings as str, as P11 joined its parts)."""
    if first.dtype.kind in "US":
        return np.concatenate([first.astype(str), second.astype(str)])
    return np.concatenate([first, second.astype(first.dtype)])


def data_record() -> dict | None:
    path = paths.results_for(EXPERIMENT) / "data.json"
    return json.loads(path.read_text()) if path.is_file() else None


def build(cfg: dict) -> tuple[DataSource, DataSource, DataSource]:
    """P13's data (hash-checked) plus the Icentia11k patients; returns (P14's data, P13's, P11's)."""
    from ecgaug.experiments import phase6_external_validation as ext

    p13, p11, _ = rr.build(cfg["rr"])
    icentia, report = build_icentia(cfg, ext.frozen_spec())
    combined = {k: concat(np.asarray(p13.arrays[k]), np.asarray(icentia[k])) for k in mp.FIELDS}
    sha = content_sha256(combined)
    record = data_record()
    if record is not None and record["trainval_sha256"] != sha:
        raise PreconditionError(f"P14 data {sha[:12]}... does not match data.json ({record['trainval_sha256'][:12]}...)")
    if record is None:
        counts = {s: {c: int(((icentia["split"] == s) & (icentia["y"] == i)).sum()) for i, c in enumerate(CLASSES)} for s in ("train", "val")}
        record = {"experiment": EXPERIMENT, "rule": "PREREGISTRATION §10 v13", "trainval_sha256": sha, "p13_sha256": p13.info["content_sha256"],
                  "icentia_sha256": content_sha256(icentia), "patients": report["patients"], "validation_units": report["validation_units"],
                  "icentia_beats": counts, "records": report["records"], "files_sha256": report["files_sha256"], "git_sha": runs.git_sha()}  # fmt: skip
        folder = paths.results_for(EXPERIMENT)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "data.json").write_text(json.dumps(record, indent=2, default=float) + "\n")
    info = {"source": "P13's data plus Icentia11k patients (PREREGISTRATION v13)", "representation": p13.spec.name, "content_sha256": sha,
            "frozen": False, "version": "v4+icentia"}  # fmt: skip
    return DataSource(combined, p13.spec, info), p13, p11


# --------------------------------------------------------------------------------------------------
# design check: training-side annotations only, SVDB 865 held out as an unseen S-dominant patient
# --------------------------------------------------------------------------------------------------

FEATURE_SETS = {"recording mean only": ("pre_m", "post_m"), "recording mean and local averages": ("pre_m", "post_m", "loc10_m", "mean5_m"),
                "recording mean, v2 ratio and post/pre (P13's)": ("pre_m", "post_m", "v2", "post_pre"),
                "de Chazal's raw intervals": ("pre", "post", "loc10", "mean5"), "all": ("pre_m", "post_m", "loc10_m", "mean5_m", "v2", "post_pre")}  # fmt: skip
HELD_OUT = ("svdb", 865)


def interval_features(qrs_samples: np.ndarray, symbols: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    """For each N or S beat with both RR intervals: intervals over the recording's mean (pre_m, post_m), the mean of the 10
    preceding intervals and of the intervals within +/-150 s (loc10, mean5, also over the recording's mean), v2's ratio,
    post/pre, a robust local irregularity (median absolute successive difference of the 20 surrounding intervals over their
    median) and the class."""
    t = np.asarray(qrs_samples, dtype=np.float64) / float(fs)
    rr_s = np.diff(t)
    n = t.size
    pre, post = np.full(n, np.nan), np.full(n, np.nan)
    pre[1:], post[:-1] = rr_s, rr_s
    mean = rr_s.mean()
    mid = (t[1:] + t[:-1]) / 2
    lo, hi = np.searchsorted(mid, t - 150), np.searchsorted(mid, t + 150, side="right")
    cs = np.concatenate([[0.0], np.cumsum(rr_s)])
    mean5 = (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)
    loc10, irr = np.full(n, np.nan), np.full(n, np.nan)
    d = np.abs(np.diff(rr_s))
    for i in range(2, n):
        loc10[i] = rr_s[max(0, i - 11) : i - 1].mean()
    for i in range(n):
        a, b = max(0, i - 10), min(rr_s.size, i + 10)
        if b - a >= 4:
            irr[i] = np.median(d[a : b - 1]) / np.median(rr_s[a:b])
    f = {"pre_m": pre / mean, "post_m": post / mean, "loc10_m": loc10 / mean, "mean5_m": mean5 / mean, "v2": v5.baseline_ratio(np.asarray(qrs_samples), fs, 150.0),
         "post_pre": post / pre, "pre": pre, "post": post, "loc10": loc10, "mean5": mean5, "irr": irr}  # fmt: skip
    aami = np.array([SYMBOL_TO_AAMI.get(s, UNMAPPED) for s in np.asarray(symbols)])
    keep = np.isin(aami, ["N", "S"]) & np.all([np.isfinite(v) for v in f.values()], axis=0)
    return {k: v[keep] for k, v in f.items()} | {"is_s": aami[keep] == "S"}


def _score(pred: np.ndarray, y: np.ndarray) -> dict:
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    se, ppv = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
    return {"se": se, "ppv": ppv, "f1": 2 * se * ppv / (se + ppv) if se + ppv else 0.0}


def holdout_check(cfg: dict) -> dict:
    """Training-side annotations only (DS1, INCART, SVDB; P11's split): rules and interval models of S vs N beats, scored on
    P11's validation patients and on SVDB 865's S beats with 865 left out of training (an unseen S-dominant patient)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    from ecgaug.data.splits import load_approved_split
    from ecgaug.experiments import phase6_external_validation as ext

    p11 = mp.data_record()
    ds1_val = set(load_approved_split().val)
    inc_val, svdb_val = set(p11["external"]["incartdb"]["validation_units"]), set(p11["external"]["svdb"]["validation_units"])
    patient = ext.patient_of_record("incartdb")
    tables = []
    for record in DS1:
        samples, symbols = raw.read_annotations(record)
        qrs = np.isin(symbols, sorted(QRS_SYMBOLS))
        tables.append(("mitdb", record, "val" if record in ds1_val else "train", interval_features(samples[qrs], symbols[qrs], FS)))
    for db in mp.DATABASES:
        for name in ext.record_names(db):
            samples, symbols, fs, sig_names = raw.read_external_annotations(db, name)
            if set(symbols.tolist()) - QRS_SYMBOLS - NON_BEAT_SYMBOLS or ext.DATABASES[db]["lead"] not in sig_names:
                continue
            if ext.DATABASES[db]["exclude_paced"] and np.isin(symbols, sorted(ext.PACED)).any():
                continue
            number = ext.record_number(name)
            unit = patient[number] if db == "incartdb" else number
            split = "val" if unit in (inc_val if db == "incartdb" else svdb_val) else "train"
            qrs = np.isin(symbols, sorted(QRS_SYMBOLS))
            tables.append((db, number, split, interval_features(samples[qrs], symbols[qrs], fs)))
    train = [t for t in tables if t[2] == "train"]
    val = [t for t in tables if t[2] == "val"]
    held = [t for t in train if (t[0], t[1]) == HELD_OUT]
    without = [t for t in train if (t[0], t[1]) != HELD_OUT]

    def stack(rows: list, cols: tuple) -> tuple[np.ndarray, np.ndarray]:
        return np.column_stack([np.log(np.concatenate([r[3][c] for r in rows])) for c in cols]), np.concatenate([r[3]["is_s"] for r in rows])

    def pooled(rows: list, key: str) -> np.ndarray:
        return np.concatenate([r[3][key] for r in rows])

    rules = {}
    for ref_key, label in (("pre_m", "recording mean"), ("mean5_m", "+/-150 s mean")):
        for veto in (None, 0.1):
            name = f"{label} < 0.8" + ("" if veto is None else f", irregularity < {veto}")
            def rule(rows: list, ref_key: str = ref_key, veto: float | None = veto) -> np.ndarray:
                ratio = pooled(rows, "pre_m") / (pooled(rows, "mean5_m") if ref_key == "mean5_m" else 1.0)
                return (ratio < 0.8) & ((pooled(rows, "irr") < veto) if veto is not None else True)
            rules[name] = {"validation": _score(rule(val), pooled(val, "is_s")), "held_out_s_sensitivity": _score(rule(held), pooled(held, "is_s"))["se"]}
    models = {}
    for set_name, cols in FEATURE_SETS.items():
        for model_name, make in (("logistic regression", lambda: LogisticRegression(class_weight="balanced", max_iter=2000)),
                                 ("gradient boosting", lambda: HistGradientBoostingClassifier(class_weight="balanced", max_iter=200, random_state=0))):  # fmt: skip
            entry = {}
            for label, rows in (("without_held_out", without), ("with_held_out", train)):
                model = make().fit(*stack(rows, cols))
                xv, yv = stack(val, cols)
                xh, yh = stack(held, cols)
                entry[label] = {"validation": _score(model.predict(xv).astype(bool), yv), "held_out_s_sensitivity": _score(model.predict(xh).astype(bool), yh)["se"]}
            models[f"{model_name}: {set_name}"] = entry
    out = {"experiment": EXPERIMENT, "rule": "PREREGISTRATION §10 v13 (design check, training-side annotations only)", "held_out": list(HELD_OUT),
           "n_records": {"train": len(train), "val": len(val)}, "val_s_beats": int(pooled(val, "is_s").sum()), "held_out_s_beats": int(pooled(held, "is_s").sum()),
           "rules": rules, "models": models, "git_sha": runs.git_sha()}  # fmt: skip
    folder = paths.results_for(EXPERIMENT)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "design_check.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    return out


# --------------------------------------------------------------------------------------------------
# grid (P13's settings and strengths) and the trivial selection
# --------------------------------------------------------------------------------------------------


def icentia_grid(cfg: dict) -> dict:
    base = rr.record_grid(cfg["rr"])
    return {**base, "experiment": EXPERIMENT, "name": GRID_NAME, "conditions": {c: base["conditions"][c] for c in CONDITIONS}, "file": cfg["file"]}


def run_grid(cfg: dict, data: DataSource, p13: DataSource, p11: DataSource, device: torch.device) -> dict:
    split = check_preconditions()
    grid = icentia_grid(cfg)
    ledger = runner.read_ledger(grid)
    train_before, train_after = training_beats(p11.arrays), training_beats(p13.arrays)  # P12's bank draws refer to P11's training beats
    report = {"done_before": 0, "ran": 0}
    for cell in runner.cells(grid):
        classifier = runner.classifier_settings(grid, cell.architecture, preview=False)
        bank, bank_sha, bank_id = None, None, None
        if cell.condition in rg.FAMILIES:
            p12_bank, bank_sha = rg.load_relative_bank(cfg["rr"]["rg"], cell.condition, cell.seed)
            bank = rr.v4_bank(p12_bank, train_before, train_after, rg.generators_record(cell.condition)["bank_counts"], cell.seed)
            bank_id = grid["conditions"][cell.condition]["bank"]
        sha = runner.cell_sha(cell, grid, classifier, data, bank_sha)
        if runner.is_done(ledger.get(sha)):
            report["done_before"] += 1
            continue
        ledger[sha] = runner.run_cell(cell, grid, split, data, classifier, device, sha, bank, bank_id, [])
        runner.write_cell(grid, ledger[sha])
        report["ran"] += 1
    return report


def select(cfg: dict) -> dict:
    grid = icentia_grid(cfg)
    expected = runner.cells(grid)
    done = {(r["condition"], r["strength"], r["architecture"], r["seed"]) for r in runner.read_ledger(grid).values() if r["status"] == "done"}
    missing = [c for c in expected if (c.condition, c.strength, c.architecture, c.seed) not in done]
    if missing:
        raise PreconditionError(f"{len(missing)} of {len(expected)} cells are not done, e.g. {missing[:2]}")
    chosen, _ = workflow.choose_strengths(grid, runner.summarize(grid))
    out = {"experiment": EXPERIMENT, "rule": "PREREGISTRATION §10 v13 (strengths carried from P13)", "chosen": chosen, "git_sha": runs.git_sha()}
    (paths.results_for(EXPERIMENT) / "selection.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    return out


def load_selection() -> dict:
    path = paths.results_for(EXPERIMENT) / "selection.json"
    if not path.is_file():
        raise PreconditionError("Collect the runs first: --part select")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------------------------------
# DS2
# --------------------------------------------------------------------------------------------------


def test_report(cfg: dict) -> dict:
    """DS2 in P13's form, one posthoc access. Confirmatory: each condition trained with the Icentia11k patients vs the same
    condition in P13. Secondary: the cVAE vs none, random oversampling and class weighting. Descriptive: Huang et al.'s SVEB
    stage on top. P13's scores are recomputed and checked first."""
    from ecgaug import final_eval

    new_sel, old_sel = load_selection()["chosen"]["cnn"], rr.load_selection()["chosen"]["cnn"]
    half = cfg["rr"]["rg"]["mp"]["pt"]["half_window_s"]
    spec = load_data(check_preconditions(), preview=False).spec
    v1_test = pt.test_arrays(spec, experiment=EXPERIMENT)
    ratios = {}
    for record in DS2:
        ratios.update({(record, r): v for r, v in v5.record_ratios(record, half, access="posthoc").items()})
    v3 = pt.relative(v5.substitute(v1_test, ratios), pt.mitbih_templates(v1_test, spec, half, access="posthoc"))
    test = rr.normalize(v3, rr.mean_rr_by_record(np.asarray(v3["record"]), access="posthoc"))
    y, records = test["y"], test["record"]
    new, old, rule, unique = {}, {}, {}, None
    for c in CONDITIONS:
        probs = pt._predict(new_sel[c]["run_ids"], test)
        unique, new[c] = final_eval.per_record_cms(probs, y, records)
        _, rule[c] = final_eval.per_record_cms([rr.sveb_rule(p, test["rr_prev"], cfg["rule_threshold"]) for p in probs], y, records)
        _, old[c] = final_eval.per_record_cms(pt._predict(old_sel[c]["run_ids"], test), y, records)
    reproduced = {c: rr._reproduced(old[c], paths.results_for(rr.EXPERIMENT) / "summary.json", f"cnn/{c}") for c in CONDITIONS}
    off = {c: d for c, d in reproduced.items() if abs(d) > 1e-9}
    if off:
        raise PreconditionError(f"P13's scores do not reproduce: {off}")
    summary = {"experiment": EXPERIMENT, "post_hoc": True, "rule": "PREREGISTRATION §10 v13", "records": [int(r) for r in unique], "n_boot": ref.N_BOOT,
               "boot_seed": ref.BOOT_SEED, "rule_threshold": float(cfg["rule_threshold"]),
               "conditions": {f"cnn/{c}": ref.condition_summary(cm, unique) | {"strength": new_sel[c]["strength"]} for c, cm in new.items()},
               "before": {f"cnn/{c}": ref.condition_summary(cm, unique) for c, cm in old.items()},
               "sveb_rule": {f"cnn/{c}": ref.condition_summary(cm, unique) for c, cm in rule.items()},
               "families": {"confirmatory": mp._family({f"cnn: {c}, with Icentia11k patients vs P13": (new[c], old[c]) for c in CONDITIONS}),
                            "secondary": mp._family({f"cnn: cvae vs {c} (with Icentia11k patients)": (new["cvae"], new[c])
                                                     for c in ("none", "random_oversampling", "class_weighted")})},
               "reproduction_difference": reproduced, "run_ids": [r for c in CONDITIONS for r in new_sel[c]["run_ids"]], "git_sha": runs.git_sha()}  # fmt: skip
    (paths.results_for(EXPERIMENT) / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--part", choices=["check", "grid", "select", "test"], required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config))
    try:
        if args.part == "check":
            c = holdout_check(cfg)
            for name, e in c["rules"].items():
                print(f"  rule  {name:40s} validation S F1 {e['validation']['f1']:.3f}  held-out S sensitivity {e['held_out_s_sensitivity']:.2f}")
            for name, e in c["models"].items():
                w = e["without_held_out"]
                print(f"  model {name:62s} validation S F1 {w['validation']['f1']:.3f}  held-out S sensitivity {w['held_out_s_sensitivity']:.2f}")
        elif args.part == "grid":
            data, p13, p11 = build(cfg)
            print(json.dumps(data_record()["icentia_beats"]))
            print(json.dumps(run_grid(cfg, data, p13, p11, resolve_device(args.device))))
        elif args.part == "select":
            s = select(cfg)
            print({c: (v["strength"], round(v["mean"], 3)) for c, v in s["chosen"]["cnn"].items()})
        else:
            s = test_report(cfg)
            for family, items in s["families"].items():
                for name, c in items.items():
                    print(f"  {family:12s} {name:52s} {c['point']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] Holm p {c['p_holm']:.3f}")
    except (PreconditionError, LockedTestDataError) as err:
        print(f"Refusing to run: {err}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
