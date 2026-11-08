"""phase6_af_rule: Huang's SVEB stage with atrial-fibrillation beats vetoed (PREREGISTRATION §10 v15, P16; post hoc).

    python -m ecgaug.experiments.phase6_af_rule --config configs/phase6_af_rule/af_rule.yaml --part fit     # MIT-BIH AF database annotations only
    python -m ecgaug.experiments.phase6_af_rule --config ... --part check   # training-side annotations: the threshold, by the registered rule
    ALLOW_TEST_POSTHOC=1 python -m ecgaug.experiments.phase6_af_rule --config ... --part test        # one posthoc access

P13's SVEB stage (a beat the cnn calls N becomes S when its RR interval is below 0.8 of the recording's mean) finds most S
beats but many of its S calls are wrong. On the MIT-BIH training records 89% of those false calls lie in atrial
fibrillation or flutter, where only 16 of 933 real S beats lie: without organized atrial activity no beat is labelled S,
and the irregular rhythm makes normal beats look premature. P16 adds an AF detector from the RR intervals alone (the
irregularity of the 64 intervals around a beat after the intervals of premature beats are removed, as HRV and AF detectors
do) and vetoes the SVEB stage where it fires. The detector is fitted on the MIT-BIH Atrial Fibrillation Database (25 other
subjects; its beat detections and rhythm labels) and nothing else; its threshold is chosen on the training-side databases
(DS1, INCART, SVDB) by the registered rule, and the veto's form (the stage's own S calls, or every S call in AF) on P13's
validation patients; on DS2 it sees only annotation times, never labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from ecgaug import paths, runs
from ecgaug.data import raw
from ecgaug.data.access import LockedTestDataError
from ecgaug.data.records import CLASS_INDEX, DS1, DS2, FS, NON_BEAT_SYMBOLS, QRS_SYMBOLS, SYMBOL_TO_AAMI
from ecgaug.experiments import phase6_more_patients as mp
from ecgaug.experiments import phase6_patient_template as pt
from ecgaug.experiments import phase6_record_rr as rr
from ecgaug.experiments import phase6_rr_baseline as v5
from ecgaug.experiments import phase6_rr_reference as ref
from ecgaug.experiments.common import PreconditionError, check_preconditions
from ecgaug.experiments.data_source import load_data
from ecgaug.names import PHASE6_AF_RULE

EXPERIMENT = PHASE6_AF_RULE
AFDB = "afdb"
AF_RHYTHMS = ("(AFIB", "(AFL")
FEATURES = ("nrmssd", "cv", "nrmssd_trimmed", "jump_share", "entropy")
CONFIG_KEYS = {"experiment", "record_rr_config", "diffusion_config", "half_window_beats", "premature_ratio", "local_median_beats", "fit_every",
               "thresholds", "rule_threshold"}  # fmt: skip
DATABASES = ("mitdb_ds1", "incartdb", "svdb")


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if set(cfg) != CONFIG_KEYS or cfg["experiment"] != EXPERIMENT:
        raise ValueError(f"{path}: expected experiment {EXPERIMENT} and exactly the keys {sorted(CONFIG_KEYS)}")
    if not all(0 < float(t) < 1 for t in cfg["thresholds"]):
        raise ValueError(f"{path}: thresholds must lie in (0, 1)")
    return {**cfg, "file": str(path), "rr": rr.load_config(paths.repo_root() / cfg["record_rr_config"])}


# --------------------------------------------------------------------------------------------------
# the AF detector: RR intervals only
# --------------------------------------------------------------------------------------------------


def irregularity_features(qrs_samples: np.ndarray, fs: float, half: int = 32, premature: float = 0.85, local: int = 10) -> np.ndarray:
    """Per QRS annotation: five irregularity measures of the up to ``2 * half`` RR intervals around it, after removing every
    interval below ``premature`` of the median of its ``2 * local + 1`` neighbours and the interval right after it (the
    premature beat and its pause). NaN where fewer than 16 intervals remain."""
    t = np.sort(np.asarray(qrs_samples, dtype=np.float64)) / float(fs)
    rr_s = np.diff(t)
    n = t.size
    out = np.full((n, len(FEATURES)), np.nan)
    if rr_s.size < 16:
        return out
    med = np.array([np.median(rr_s[max(0, j - local) : j + local + 1]) for j in range(rr_s.size)])
    early = rr_s < premature * med
    drop = early.copy()
    drop[1:] |= early[:-1]
    for i in range(n):
        a, b = max(0, i - half), min(rr_s.size, i + half)
        w = rr_s[a:b][~drop[a:b]]
        if w.size < 16:
            continue
        d = np.abs(np.diff(w))
        m = w.mean()
        trimmed = np.sort(d)[:-4] if d.size > 8 else d
        hist = np.histogram(w / m, bins=16, range=(0.4, 1.6))[0] / w.size
        nz = hist[hist > 0]
        out[i] = [np.sqrt(np.mean(d**2)) / m, w.std() / m, np.sqrt(np.mean(trimmed**2)) / m, float(np.mean(d > 0.1 * m)), float(-(nz * np.log(nz)).sum() / np.log(16))]
    return out


def rhythm_at(samples: np.ndarray, symbols: np.ndarray, aux: list[str], at: np.ndarray) -> np.ndarray:
    """The rhythm label in force at each sample of ``at`` (from the '+' rhythm annotations; '(N' before the first)."""
    times, labels = [], []
    for s, sym, a in zip(np.asarray(samples), np.asarray(symbols), aux):
        if sym == "+" and a and a.strip():
            times.append(int(s))
            labels.append(a.strip().rstrip("\x00").split()[0])
    if not times:
        return np.full(np.asarray(at).size, "(N", dtype=object)
    idx = np.searchsorted(np.asarray(times), np.asarray(at), side="right") - 1
    lab = np.asarray(labels, dtype=object)
    return np.where(idx >= 0, lab[np.clip(idx, 0, len(labels) - 1)], "(N")


def _log(f: np.ndarray) -> np.ndarray:
    return np.log(np.asarray(f, dtype=np.float64) + 1e-6)


def detector_path() -> Path:
    return paths.results_for(EXPERIMENT) / "af_detector.json"


def fit_detector(cfg: dict) -> dict:
    """Logistic regression (balanced classes) of 'in AF or flutter' on the log features, from every ``fit_every``-th beat of
    the MIT-BIH AF database (beat detections ``qrs``, rhythm labels ``atr``); written to ``af_detector.json``."""
    import wfdb
    from sklearn.linear_model import LogisticRegression

    folder = raw.external_dir(AFDB)
    names = sorted(p.stem for p in folder.glob("*.qrs"))
    if not names:
        raise PreconditionError(f"{folder} holds no afdb annotations: download its .qrs, .atr and .hea files from PhysioNet")
    xs, ys, used = [], [], []
    for name in names:
        qrs, atr = wfdb.rdann(str(folder / name), "qrs"), wfdb.rdann(str(folder / name), "atr")
        f = irregularity_features(np.asarray(qrs.sample), qrs.fs, cfg["half_window_beats"], cfg["premature_ratio"], cfg["local_median_beats"])
        af = np.isin(rhythm_at(np.asarray(atr.sample), np.asarray(atr.symbol), atr.aux_note, np.asarray(qrs.sample)), AF_RHYTHMS)
        keep = np.flatnonzero(np.isfinite(f).all(axis=1))[:: int(cfg["fit_every"])]
        xs.append(_log(f[keep]))
        ys.append(af[keep])
        used.append(name)
    x, y = np.concatenate(xs), np.concatenate(ys)
    model = LogisticRegression(class_weight="balanced", max_iter=5000).fit(x, y)
    pred = model.predict(x).astype(bool)
    out = {"experiment": EXPERIMENT, "rule": "PREREGISTRATION §10 v15", "database": "afdb (MIT-BIH Atrial Fibrillation Database)", "records": used,
           "features": list(FEATURES), "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]), "beats": int(y.size),
           "af_share": float(y.mean()), "training_sensitivity": float((pred & y).sum() / y.sum()), "training_specificity": float((~pred & ~y).sum() / (~y).sum()),
           "settings": {k: cfg[k] for k in ("half_window_beats", "premature_ratio", "local_median_beats", "fit_every")}, "git_sha": runs.git_sha()}  # fmt: skip
    folder_out = paths.results_for(EXPERIMENT)
    folder_out.mkdir(parents=True, exist_ok=True)
    detector_path().write_text(json.dumps(out, indent=2) + "\n")
    return out


def load_detector() -> dict:
    if not detector_path().is_file():
        raise PreconditionError("Fit the AF detector first: --part fit")
    return json.loads(detector_path().read_text())


def af_probability(detector: dict, qrs_samples: np.ndarray, fs: float) -> np.ndarray:
    """Per QRS annotation (in sorted order): the detector's probability of AF or flutter (0 where no window is available)."""
    s = detector["settings"]
    f = irregularity_features(qrs_samples, fs, s["half_window_beats"], s["premature_ratio"], s["local_median_beats"])
    ok = np.isfinite(f).all(axis=1)
    z = _log(np.where(ok[:, None], f, 1.0)) @ np.asarray(detector["coef"]) + float(detector["intercept"])
    return np.where(ok, 1.0 / (1.0 + np.exp(-z)), 0.0)


FORMS = ("rule", "all")


def veto_rule(probs: np.ndarray, rr_prev_over_mean: np.ndarray, p_af: np.ndarray, rule_threshold: float, af_threshold: float,
              form: str = "rule") -> np.ndarray:  # fmt: skip
    """Huang's SVEB stage (as ``rr.sveb_rule``) with S calls undone where the AF probability exceeds ``af_threshold``:
    form "rule" undoes only the stage's own N-to-S conversions, form "all" every S call there (by the annotation convention
    no beat in atrial fibrillation or flutter is labelled S). One-hot rows."""
    if form not in FORMS:
        raise ValueError(f"form must be one of {FORMS}")
    base = rr.sveb_rule(probs, rr_prev_over_mean, rule_threshold).argmax(axis=1)
    called = np.asarray(probs).argmax(axis=1)
    undo = (base == CLASS_INDEX["S"]) & (np.asarray(p_af) > af_threshold)
    if form == "rule":
        undo &= called == CLASS_INDEX["N"]
    pred = np.where(undo, CLASS_INDEX["N"], base)
    return np.eye(np.asarray(probs).shape[1], dtype=np.float32)[pred]


# --------------------------------------------------------------------------------------------------
# the threshold: training-side annotations only
# --------------------------------------------------------------------------------------------------


def _check_tables(cfg: dict, detector: dict):
    """(database, record, table) for DS1 and every usable INCART and SVDB record: per N or S beat its class, its ratio to the
    recording's mean RR and its AF probability."""
    from ecgaug.experiments import phase6_external_validation as ext

    def table(samples, symbols, fs, aux=None):
        qrs = np.isin(symbols, sorted(QRS_SYMBOLS))
        s, sym = np.asarray(samples)[qrs], np.asarray(symbols)[qrs]
        t = s / float(fs)
        rr_s = np.diff(t)
        pre = np.full(t.size, np.nan)
        pre[1:] = rr_s
        p = af_probability(detector, s, fs)
        aami = np.array([SYMBOL_TO_AAMI.get(x, "-") for x in sym])
        keep = np.isin(aami, ["N", "S"]) & np.isfinite(pre)
        out = {"is_s": aami[keep] == "S", "ratio": (pre / rr_s.mean())[keep], "p_af": p[keep]}
        if aux is not None:  # MIT-BIH only: the annotated rhythm of each beat
            out["in_af"] = np.isin(rhythm_at(samples, symbols, aux, s), AF_RHYTHMS)[keep]
        return out

    for record in DS1:
        samples, symbols, aux = raw.read_annotation_notes(record)
        yield "mitdb_ds1", record, table(samples, symbols, FS, aux)
    for db in mp.DATABASES:
        for name in ext.record_names(db):
            samples, symbols, fs, sig_names = raw.read_external_annotations(db, name)
            if set(symbols.tolist()) - QRS_SYMBOLS - NON_BEAT_SYMBOLS or ext.DATABASES[db]["lead"] not in sig_names:
                continue
            if ext.DATABASES[db]["exclude_paced"] and np.isin(symbols, sorted(ext.PACED)).any():
                continue
            yield db, ext.record_number(name), table(samples, symbols, fs)


def _score(pred: np.ndarray, y: np.ndarray) -> dict:
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    se, ppv = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
    return {"se": se, "ppv": ppv, "f1": 2 * se * ppv / (se + ppv) if se + ppv else 0.0}


def _record_beats(record: int, detector: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For a training-side record, from its annotation times only: the QRS positions at 360 Hz (as ``r_idx``), each beat's
    RR interval over the recording's mean RR (P13's form) and its AF probability."""
    from ecgaug.experiments import phase6_external_validation as ext

    if record in DS2:
        raise PreconditionError(f"Record {record} is a test record")
    if record < 100 or 800 <= record < 900:
        db = "incartdb" if record < 100 else "svdb"
        name = {ext.record_number(n): n for n in ext.record_names(db)}[record]
        samples, symbols, fs, _ = raw.read_external_annotations(db, name)
    else:
        (samples, symbols), fs = raw.read_annotations(record), FS
    qrs = np.sort(np.asarray(samples)[np.isin(symbols, sorted(QRS_SYMBOLS))])
    ratio = np.full(qrs.size, np.nan)
    ratio[1:] = np.diff(qrs) / float(fs) / rr.record_mean_rr(qrs, fs)
    return np.rint(qrs * FS / float(fs)).astype(np.int64), ratio, af_probability(detector, qrs, fs)


def validation_check(cfg: dict, detector: dict, af_t: float) -> dict:
    """The veto form, on P13's validation patients with the training runs' own validation predictions (no test data): per
    compared condition and seed, S F1 of the cnn alone, with Huang's rule, and with the rule and each veto form at
    ``af_t``. The registered choice is the form with the higher mean S F1 over conditions and seeds (ties: "rule")."""
    t_rule = float(cfg["rule_threshold"])
    keys = ("cnn", "rule", *(f"veto_{f}" for f in FORMS))
    feats: dict[int, tuple] = {}
    table: dict[str, dict] = {}
    for c, run_ids in _conditions().items():
        table[c] = {k: [] for k in keys}
        for run_id in run_ids:
            folder = runs.find_run_dir(run_id)
            if folder is None or not (folder / "preds_val.npz").is_file():
                raise PreconditionError(f"No validation predictions for {run_id}")
            z = np.load(folder / "preds_val.npz")
            probs, y, rec, r_idx = z["probs"], z["y"], z["record"].astype(np.int64), z["r_idx"].astype(np.int64)
            ratio, p_af = np.full(y.size, np.nan), np.zeros(y.size)
            for r in np.unique(rec).tolist():
                if r not in feats:
                    feats[r] = _record_beats(int(r), detector)
                at, rat, prob = feats[r]
                m = rec == r
                pos = np.clip(np.searchsorted(at, r_idx[m]), 0, at.size - 1)
                if not np.array_equal(at[pos], r_idx[m]):
                    raise PreconditionError(f"Record {r}: validation beats do not match its QRS annotations")
                ratio[m], p_af[m] = rat[pos], prob[pos]
            preds = {"cnn": probs.argmax(axis=1), "rule": rr.sveb_rule(probs, ratio, t_rule).argmax(axis=1),
                     **{f"veto_{f}": veto_rule(probs, ratio, p_af, t_rule, af_t, f).argmax(axis=1) for f in FORMS}}  # fmt: skip
            for k, pred in preds.items():
                table[c][k].append(_score(pred == CLASS_INDEX["S"], y == CLASS_INDEX["S"]))
    means = {k: float(np.mean([s["f1"] for c in table for s in table[c][k]])) for k in keys}
    best = max(means[f"veto_{f}"] for f in FORMS)
    return {"conditions": {c: {k: {"f1": [s["f1"] for s in v], "se": [s["se"] for s in v], "ppv": [s["ppv"] for s in v]} for k, v in e.items()}
                           for c, e in table.items()},
            "mean_s_f1": means, "chosen_form": "rule" if means["veto_rule"] == best else "all", "records": sorted(feats)}  # fmt: skip


def design_check(cfg: dict) -> dict:
    """The rule alone and with the veto at each candidate threshold, per training-side database; the registered choice is
    the threshold with the highest mean S F1 over the three databases (ties: the higher threshold, the fewer vetoes). Then
    the veto form, on validation (``validation_check``). Also counts, on DS1's annotated rhythms, where the rule's false S
    calls and the real S beats lie."""
    detector = load_detector()
    tables: dict[str, list[dict]] = {}
    heavy = None
    for db, record, tab in _check_tables(cfg, detector):
        tables.setdefault(db, []).append(tab)
        if (db, record) == ("svdb", 865):
            heavy = tab
    pooled = {db: {k: np.concatenate([t[k] for t in ts]) for k in ts[0]} for db, ts in tables.items()}
    rule_t = float(cfg["rule_threshold"])
    out = {"experiment": EXPERIMENT, "rule": "PREREGISTRATION §10 v15 (threshold on training-side annotations)", "rule_threshold": rule_t, "databases": {}, "svdb_865": {}}
    for db in DATABASES:
        p = pooled[db]
        flag = p["ratio"] < rule_t
        out["databases"][db] = {"rule": _score(flag, p["is_s"]),
                                **{f"veto_{t}": _score(flag & ~(p["p_af"] > float(t)), p["is_s"]) for t in cfg["thresholds"]}}  # fmt: skip
    if heavy is not None:
        flag = heavy["ratio"] < rule_t
        out["svdb_865"] = {"rule": _score(flag, heavy["is_s"]), **{f"veto_{t}": _score(flag & ~(heavy["p_af"] > float(t)), heavy["is_s"]) for t in cfg["thresholds"]}}
    means = {float(t): float(np.mean([out["databases"][db][f"veto_{t}"]["f1"] for db in DATABASES])) for t in cfg["thresholds"]}
    best = max(means.values())
    out["mean_f1_by_threshold"] = {str(t): m for t, m in means.items()}
    out["mean_f1_rule"] = float(np.mean([out["databases"][db]["rule"]["f1"] for db in DATABASES]))
    out["chosen_threshold"] = max(t for t, m in means.items() if m == best)
    ds1 = pooled["mitdb_ds1"]
    false_s = (ds1["ratio"] < rule_t) & ~ds1["is_s"]
    out["ds1_rhythm"] = {"rule_false_s": int(false_s.sum()), "rule_false_s_in_af": int((false_s & ds1["in_af"]).sum()),
                         "s_beats": int(ds1["is_s"].sum()), "s_beats_in_af": int((ds1["is_s"] & ds1["in_af"]).sum()),
                         "n_beats_in_af": int((~ds1["is_s"] & ds1["in_af"]).sum())}  # fmt: skip
    out["validation"] = validation_check(cfg, detector, out["chosen_threshold"])
    out["chosen_form"] = out["validation"]["chosen_form"]
    out["git_sha"] = runs.git_sha()
    (paths.results_for(EXPERIMENT) / "design_check.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    return out


def chosen_design() -> tuple[float, str]:
    """The threshold and the veto form chosen by ``--part check``."""
    path = paths.results_for(EXPERIMENT) / "design_check.json"
    if not path.is_file():
        raise PreconditionError("Choose the threshold and the form first: --part check")
    d = json.loads(path.read_text())
    if d.get("chosen_form") not in FORMS:
        raise PreconditionError("design_check.json holds no veto form: run --part check again")
    return float(d["chosen_threshold"]), str(d["chosen_form"])


# --------------------------------------------------------------------------------------------------
# DS2
# --------------------------------------------------------------------------------------------------


def _conditions() -> dict[str, list[str]]:
    """The run IDs per compared condition: P13's five and P15's diffusion condition when its selection exists."""
    out = {c: v["run_ids"] for c, v in rr.load_selection()["chosen"]["cnn"].items()}
    from ecgaug.experiments import phase6_diffusion as dif

    path = paths.results_for(dif.EXPERIMENT) / "selection.json"
    if path.is_file():
        out["diffusion"] = json.loads(path.read_text())["chosen"]["cnn"]["diffusion"]["run_ids"]
    return out


def test_report(cfg: dict) -> dict:
    """DS2 in P13's form, one posthoc access. For every compared condition: Huang's rule alone and with the AF veto.
    Confirmatory (Holm): with the veto vs without, per condition. The rule-alone scores of P13's conditions must reproduce
    P13's summary. The AF detector sees only the annotation times of each DS2 record."""
    from ecgaug import final_eval

    detector, (af_t, form) = load_detector(), chosen_design()
    half = cfg["rr"]["rg"]["mp"]["pt"]["half_window_s"]
    spec = load_data(check_preconditions(), preview=False).spec
    v1_test = pt.test_arrays(spec, experiment=EXPERIMENT)
    ratios = {}
    for record in DS2:
        ratios.update({(record, r): v for r, v in v5.record_ratios(record, half, access="posthoc").items()})
    v3 = pt.relative(v5.substitute(v1_test, ratios), pt.mitbih_templates(v1_test, spec, half, access="posthoc"))
    test = rr.normalize(v3, rr.mean_rr_by_record(np.asarray(v3["record"]), access="posthoc"))
    y, records, r_idx = test["y"], np.asarray(test["record"]), np.asarray(test["r_idx"])
    p_af = np.zeros(y.size)
    for record in sorted(set(records.tolist())):
        samples, symbols = raw.read_annotations(int(record), access="posthoc")
        qrs = np.sort(np.asarray(samples)[np.isin(symbols, sorted(QRS_SYMBOLS))])
        prob = af_probability(detector, qrs, FS)
        mask = records == record
        pos = np.searchsorted(qrs, r_idx[mask])
        if not np.array_equal(qrs[np.clip(pos, 0, qrs.size - 1)], r_idx[mask]):
            raise PreconditionError(f"Record {record}: test beats do not match its QRS annotations")
        p_af[mask] = prob[pos]
    plain, vetoed, unique = {}, {}, None
    for c, run_ids in _conditions().items():
        probs = pt._predict(run_ids, test)
        unique, plain[c] = final_eval.per_record_cms([rr.sveb_rule(p, test["rr_prev"], cfg["rule_threshold"]) for p in probs], y, records)
        _, vetoed[c] = final_eval.per_record_cms([veto_rule(p, test["rr_prev"], p_af, cfg["rule_threshold"], af_t, form) for p in probs], y, records)
    earlier = dict(json.loads((paths.results_for(rr.EXPERIMENT) / "summary.json").read_text())["sveb_rule"])
    if "diffusion" in plain:  # P15's own test reports the diffusion condition with the stage; P16 must reproduce it
        from ecgaug.experiments import phase6_diffusion as dif

        p15 = paths.results_for(dif.EXPERIMENT) / "summary.json"
        if not p15.is_file():
            raise PreconditionError("Run P15's test first: the diffusion condition's rule scores are checked against its summary")
        earlier["cnn/diffusion"] = json.loads(p15.read_text())["sveb_rule"]["cnn/diffusion"]
    reproduced = {c: float(np.mean(M_rare(plain[c]))) - earlier[f"cnn/{c}"]["rare_macro_f1"]["point"] for c in plain}
    off = {c: d for c, d in reproduced.items() if abs(d) > 1e-9}
    if off:
        raise PreconditionError(f"The rule scores of P13 (and P15) do not reproduce: {off}")
    summary = {"experiment": EXPERIMENT, "post_hoc": True, "rule": "PREREGISTRATION §10 v15", "records": [int(r) for r in unique], "n_boot": ref.N_BOOT,
               "boot_seed": ref.BOOT_SEED, "rule_threshold": float(cfg["rule_threshold"]), "af_threshold": af_t, "veto_form": form,
               "af_share_flagged": {str(int(r)): float((p_af[records == r] > af_t).mean()) for r in unique},
               "rule_alone": {f"cnn/{c}": ref.condition_summary(cm, unique) for c, cm in plain.items()},
               "with_af_veto": {f"cnn/{c}": ref.condition_summary(cm, unique) for c, cm in vetoed.items()},
               "without_202": {name: {f"cnn/{c}": ref.condition_summary(cm[:, unique != 202], unique[unique != 202]) for c, cm in table.items()}
                               for name, table in (("rule_alone", plain), ("with_af_veto", vetoed))},
               "families": {"confirmatory": mp._family({f"cnn: {c}, rule with AF veto vs rule alone": (vetoed[c], plain[c]) for c in vetoed})},
               "reproduction_difference": reproduced, "git_sha": runs.git_sha()}  # fmt: skip
    (paths.results_for(EXPERIMENT) / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def M_rare(cms: np.ndarray) -> np.ndarray:
    from ecgaug.eval import metrics as M

    return M.rare_macro_f1_from_cm(np.asarray(cms).sum(axis=1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--part", choices=["fit", "check", "test"], required=True)
    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config))
    try:
        if args.part == "fit":
            d = fit_detector(cfg)
            print({k: d[k] for k in ("beats", "af_share", "training_sensitivity", "training_specificity")})
        elif args.part == "check":
            c = design_check(cfg)
            print("chosen threshold", c["chosen_threshold"], "mean F1 rule", round(c["mean_f1_rule"], 3), {t: round(m, 3) for t, m in c["mean_f1_by_threshold"].items()})
            for db, e in c["databases"].items():
                print(db, {k: round(v["f1"], 3) for k, v in e.items()})
            print("DS1 rhythm", c["ds1_rhythm"])
            print("validation S F1", {k: round(v, 3) for k, v in c["validation"]["mean_s_f1"].items()}, "chosen form", c["chosen_form"])
        else:
            s = test_report(cfg)
            for name, c in s["families"]["confirmatory"].items():
                print(f"  {name:52s} {c['point']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] Holm p {c['p_holm']:.3f}")
    except (PreconditionError, LockedTestDataError) as err:
        print(f"Refusing to run: {err}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
