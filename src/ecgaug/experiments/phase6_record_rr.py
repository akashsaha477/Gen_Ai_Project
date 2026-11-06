"""phase6_record_rr: RR intervals over the mean RR interval of their recording (PREREGISTRATION §10 v12, P13; post hoc).

    python -m ecgaug.experiments.phase6_record_rr --config configs/phase6_record_rr/record_rr.yaml --part check     # annotations of DS1, INCART, SVDB
    python -m ecgaug.experiments.phase6_record_rr --config ... --part grid [--device cuda]
    python -m ecgaug.experiments.phase6_record_rr --config ... --part select
    ALLOW_TEST_POSTHOC=1 python -m ecgaug.experiments.phase6_record_rr --config ... --part test      # one posthoc access

Published analyses of DS2 find that one record holds three quarters of its S beats, that S beats outnumber N beats there
and come without a compensatory pause, and that RR intervals normalized by local averages miss nearly all of them. Huang
et al. (2014) detect S beats by the RR interval over the mean RR interval of the whole recording; Lin and Yang (2014) and
Mondejar-Guerra et al. (2019) normalize RR intervals by the patient's average. P11's training data hold the same pattern
(``--part check``, annotations of training-side databases only). P13 divides ``rr_prev`` and ``rr_next`` by the mean RR
interval of their recording (all consecutive QRS annotations, pauses included, no label used; v2's local ``rr_ratio``
stays) and retrains the P7 cnn on P11's data in P11's and P12's conditions at the strengths they chose. Nothing else
changes: beats, split, caps, banks (P12's, each synthetic beat keeping the real beat its RR features came from), settings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

from ecgaug import paths, runs, workflow
from ecgaug.data import raw
from ecgaug.data.access import LockedTestDataError
from ecgaug.data.beats import rr_intervals
from ecgaug.data.contract import content_sha256
from ecgaug.data.records import CLASS_INDEX, DS1, DS2, FS, NON_BEAT_SYMBOLS, QRS_SYMBOLS, SYMBOL_TO_AAMI
from ecgaug.eval import metrics as M
from ecgaug.experiments import phase6_more_patients as mp
from ecgaug.experiments import phase6_patient_template as pt
from ecgaug.experiments import phase6_relative_generators as rg
from ecgaug.experiments import phase6_rr_baseline as v5
from ecgaug.experiments import phase6_rr_reference as ref
from ecgaug.experiments.common import PreconditionError, check_preconditions
from ecgaug.experiments.data_source import DataSource, load_data
from ecgaug.generators import GENERATED_CLASSES
from ecgaug.generators import bank as banks
from ecgaug.generators.base import training_beats
from ecgaug.grid import runner
from ecgaug.names import PHASE6_RECORD_RR
from ecgaug.train.loop import resolve_device

EXPERIMENT = PHASE6_RECORD_RR
CONFIG_KEYS = {"experiment", "relative_generators_config", "rule_threshold", "check"}
GRID_NAME = "full_grid"
GENERATIVE = rg.FAMILIES
CONDITIONS = ("none", "class_weighted", "random_oversampling", *GENERATIVE)
NORMALIZED = ("rr_prev", "rr_next")  # divided by the recording's mean RR; rr_ratio (v2, local) is kept
RATIOS = ("v1", "v2", "record")  # the design check: v1's local mean, v2's local median, the recording's mean


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if set(cfg) != CONFIG_KEYS or cfg["experiment"] != EXPERIMENT:
        raise ValueError(f"{path}: expected experiment {EXPERIMENT} and exactly the keys {sorted(CONFIG_KEYS)}")
    if not 0 < float(cfg["rule_threshold"]) < 1:
        raise ValueError(f"{path}: rule_threshold must lie in (0, 1)")
    return {**cfg, "file": str(path), "rg": rg.load_config(paths.repo_root() / cfg["relative_generators_config"])}


# --------------------------------------------------------------------------------------------------
# the feature: RR intervals over the recording's mean RR interval
# --------------------------------------------------------------------------------------------------


def record_mean_rr(qrs_samples: np.ndarray, fs: float) -> float:
    """The mean of all RR intervals of a recording (between consecutive QRS annotations, pauses included), in seconds."""
    t = np.sort(np.asarray(qrs_samples, dtype=np.float64))
    if t.size < 2:
        raise ValueError("A recording needs two QRS annotations for a mean RR interval")
    return float(np.mean(np.diff(t)) / float(fs))


def _qrs(samples: np.ndarray, symbols: np.ndarray) -> np.ndarray:
    return np.asarray(samples)[np.isin(symbols, sorted(QRS_SYMBOLS))]


def mean_rr_by_record(records: np.ndarray, access: str | None = None) -> dict[int, float]:
    """The mean RR interval of each recording, from its annotation times only: INCART (records 1-75), SVDB (800-894),
    MIT-BIH otherwise (a DS2 record is read under ``access``)."""
    from ecgaug.experiments import phase6_external_validation as ext

    wanted = sorted({int(r) for r in np.asarray(records).tolist()})
    names = {db: {ext.record_number(n): n for n in ext.record_names(db)} for db in mp.DATABASES} if any(r < 100 or 800 <= r < 900 for r in wanted) else {}
    out = {}
    for r in wanted:
        if r < 100 or 800 <= r < 900:
            db = "incartdb" if r < 100 else "svdb"
            samples, symbols, fs, _ = raw.read_external_annotations(db, names[db][r])
        else:
            (samples, symbols), fs = raw.read_annotations(r, access if r in DS2 else None), FS
        out[r] = record_mean_rr(_qrs(samples, symbols), fs)
    return out


def normalize(arrays: dict, means: dict[int, float]) -> dict:
    """The arrays with ``rr_prev`` and ``rr_next`` divided by the mean RR interval of their recording; nothing else changes."""
    rec = np.asarray(arrays["record"]).astype(np.int64)
    missing = sorted(set(rec.tolist()) - set(means))
    if missing:
        raise PreconditionError(f"No mean RR interval for records {missing[:5]}")
    keys = np.array(sorted(means), dtype=np.int64)
    m = np.array([means[int(k)] for k in keys], dtype=np.float64)[np.searchsorted(keys, rec)]
    return {**arrays, **{f: (np.asarray(arrays[f], dtype=np.float64) / m).astype(np.float32) for f in NORMALIZED}}


def sveb_rule(probs: np.ndarray, rr_prev_over_mean: np.ndarray, threshold: float) -> np.ndarray:
    """Huang et al. (2014)'s SVEB stage on top of a classifier, as one-hot rows: a beat the classifier calls N becomes S
    when its RR interval over the recording's mean is below ``threshold``; every other decision stays."""
    pred = np.asarray(probs).argmax(axis=1)
    pred = np.where((pred == CLASS_INDEX["N"]) & (np.asarray(rr_prev_over_mean) < float(threshold)), CLASS_INDEX["S"], pred)
    return np.eye(np.asarray(probs).shape[1], dtype=np.float32)[pred]


# --------------------------------------------------------------------------------------------------
# design check: the training-side databases' annotations only (DS1, INCART, SVDB; never DS2)
# --------------------------------------------------------------------------------------------------


def ratio_table(qrs_samples: np.ndarray, symbols: np.ndarray, fs: float, half_window_s: float) -> dict[str, np.ndarray]:
    """For each N or S beat with both RR intervals: its class, the three ratios of ``rr_prev`` and its post/pre ratio."""
    t = np.asarray(qrs_samples, dtype=np.int64)
    rr_prev, rr_next, v1 = rr_intervals(t, fs=int(round(fs)))
    v2 = v5.baseline_ratio(t, fs, half_window_s)
    record = rr_prev / record_mean_rr(t, fs)
    aami = np.array([SYMBOL_TO_AAMI.get(s, "-") for s in np.asarray(symbols)])
    keep = np.isfinite(rr_prev) & np.isfinite(rr_next) & np.isin(aami, ["N", "S"])
    return {"is_s": aami[keep] == "S", "v1": v1[keep], "v2": v2[keep], "record": record[keep], "post_pre": (rr_next / rr_prev)[keep]}


def _rule(table: dict, ratio: str, threshold: float) -> dict:
    pred, y = table[ratio] < threshold, table["is_s"]
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    se, ppv = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
    return {"se": se, "ppv": ppv, "f1": 2 * se * ppv / (se + ppv) if se + ppv else 0.0}


def _check_records(half_window_s: float):
    """``(database, record, table)`` for DS1 and every INCART and SVDB record P11 could use."""
    from ecgaug.experiments import phase6_external_validation as ext

    for record in DS1:
        samples, symbols = raw.read_annotations(record)
        qrs = np.isin(symbols, sorted(QRS_SYMBOLS))
        yield "mitdb_ds1", record, ratio_table(samples[qrs], symbols[qrs], FS, half_window_s)
    for db in mp.DATABASES:
        for name in ext.record_names(db):
            samples, symbols, fs, sig_names = raw.read_external_annotations(db, name)
            if set(symbols.tolist()) - QRS_SYMBOLS - NON_BEAT_SYMBOLS or ext.DATABASES[db]["lead"] not in sig_names:
                continue
            if ext.DATABASES[db]["exclude_paced"] and np.isin(symbols, sorted(ext.PACED)).any():
                continue
            qrs = np.isin(symbols, sorted(QRS_SYMBOLS))
            yield db, ext.record_number(name), ratio_table(samples[qrs], symbols[qrs], fs, half_window_s)


def design_check(cfg: dict) -> dict:
    """How well each ratio separates S from N beats (AUC; Huang's rule at the threshold) per database, and the records
    where S beats are a large share of the beats: training-side annotations only."""
    c = cfg["check"]
    threshold = float(c["threshold"])
    by_db: dict[str, list[dict]] = {}
    heavy = []
    for db, record, table in _check_records(float(cfg["rg"]["mp"]["pt"]["half_window_s"])):
        by_db.setdefault(db, []).append(table)
        n_s = int(table["is_s"].sum())
        if n_s >= int(c["s_heavy_min_beats"]) and n_s >= float(c["s_heavy_min_share"]) * table["is_s"].size:
            heavy.append({"database": db, "record": int(record), "s_beats": n_s, "n_and_s_beats": int(table["is_s"].size),
                          "median_post_pre_of_s": float(np.median(table["post_pre"][table["is_s"]])),
                          "rule_s_sensitivity": {r: _rule(table, r, threshold)["se"] for r in RATIOS}})  # fmt: skip
    databases = {}
    for db, tables in by_db.items():
        pooled = {k: np.concatenate([t[k] for t in tables]) for k in tables[0]}
        databases[db] = {"records": len(tables), "n_beats": int((~pooled["is_s"]).sum()), "s_beats": int(pooled["is_s"].sum()),
                         "auc": {r: float(roc_auc_score(pooled["is_s"], -pooled[r])) for r in RATIOS},
                         "rule": {r: _rule(pooled, r, threshold) for r in RATIOS}}  # fmt: skip
    out = {"experiment": EXPERIMENT, "rule": "PREREGISTRATION §10 v12 (design check, training-side annotations only)", "threshold": threshold,
           "ratios": {"v1": "rr_prev over the mean of up to 10 preceding RR intervals", "v2": "rr_prev over the median RR within +/-150 s",
                      "record": "rr_prev over the mean RR interval of the whole recording"},
           "databases": databases, "s_heavy_records": heavy, "git_sha": runs.git_sha()}  # fmt: skip
    folder = paths.results_for(EXPERIMENT)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "design_check.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    return out


# --------------------------------------------------------------------------------------------------
# data and banks (P11's beats, P12's banks)
# --------------------------------------------------------------------------------------------------


def data_record() -> dict | None:
    path = paths.results_for(EXPERIMENT) / "data.json"
    return json.loads(path.read_text()) if path.is_file() else None


def build(cfg: dict) -> tuple[DataSource, DataSource, dict]:
    """P11's data (hash-checked) and the same beats with ``rr_prev`` and ``rr_next`` over their recording's mean RR."""
    p11, _ = mp.build(cfg["rg"]["mp"])
    means = mean_rr_by_record(np.asarray(p11.arrays["record"]))
    arrays = normalize(p11.arrays, means)
    sha = content_sha256({k: arrays[k] for k in mp.FIELDS})
    record = data_record()
    if record is not None and record["trainval_sha256"] != sha:
        raise PreconditionError(f"P13 data {sha[:12]}... does not match data.json ({record['trainval_sha256'][:12]}...)")
    if record is None:
        record = {"experiment": EXPERIMENT, "rule": "PREREGISTRATION §10 v12", "trainval_sha256": sha, "p11_sha256": p11.info["content_sha256"],
                  "mean_rr_s": {str(r): m for r, m in means.items()}, "git_sha": runs.git_sha()}  # fmt: skip
        folder = paths.results_for(EXPERIMENT)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "data.json").write_text(json.dumps(record, indent=2) + "\n")
    info = {"source": "P11's data with rr_prev and rr_next over the recording's mean RR interval (PREREGISTRATION v12)",
            "representation": p11.spec.name, "content_sha256": sha, "frozen": False, "version": "v4"}  # fmt: skip
    return DataSource(arrays, p11.spec, info), p11, record


def p12_picks(train: dict, counts: dict[str, int], seed: int) -> dict[str, np.ndarray]:
    """The real training beats whose RR features P12's bank of ``seed`` copied, class by class: the draws of
    ``rg.sample_relative_bank`` (one generator seeded with the bank seed, used for nothing else)."""
    rng = np.random.default_rng(seed)
    out = {}
    for cls in GENERATED_CLASSES:
        n = int(counts.get(cls, 0))
        if n <= 0:
            continue
        pool = np.flatnonzero(np.asarray(train["y"]) == CLASS_INDEX[cls])
        out[cls] = rng.choice(pool, size=n, replace=True)
    return out


def v4_bank(bank: dict, train_before: dict, train_after: dict, counts: dict[str, int], seed: int) -> dict:
    """P12's bank with each synthetic beat's RR features taken from the same real training beat in P13's form. The draws
    are recomputed and must reproduce the RR features the bank stores."""
    y = np.asarray(bank["y"])
    out = {k: np.array(v, copy=True) for k, v in bank.items()}
    for cls, pick in p12_picks(train_before, counts, seed).items():
        idx = np.flatnonzero(y == CLASS_INDEX[cls])
        if idx.size != pick.size:
            raise PreconditionError(f"Seed {seed} bank holds {idx.size} {cls} beats; P12's draws give {pick.size}")
        for f in banks.RR_FIELDS:
            if not np.array_equal(np.asarray(bank[f])[idx], np.asarray(train_before[f])[pick].astype(np.float32)):
                raise PreconditionError(f"Seed {seed} bank: {cls} {f} values are not those of P12's draws")
            out[f][idx] = np.asarray(train_after[f])[pick].astype(np.float32)
    return out


# --------------------------------------------------------------------------------------------------
# the grid (strengths carried) and the trivial selection
# --------------------------------------------------------------------------------------------------


def strengths() -> dict[str, float]:
    """P11's choices for class weighting and random oversampling, P12's for the two generators."""
    p11, p12 = mp.load_selection()["chosen"]["cnn"], rg.load_selection()["chosen"]["cnn"]
    return {"class_weighted": p11["class_weighted"]["strength"], "random_oversampling": p11["random_oversampling"]["strength"],
            **{f: p12[f]["strength"] for f in GENERATIVE}}  # fmt: skip


def record_grid(cfg: dict) -> dict:
    """P11's grid (the P7 cnn, its settings and seeds) with the five conditions at the carried strengths, P12's bank policy
    (unscreened; random oversampling adds the same number of beats under either policy)."""
    base = pt.v3_grid(cfg["rg"]["mp"]["pt"])
    s = strengths()
    conditions = {"none": {}, "class_weighted": {"gammas": [s["class_weighted"]]}, "random_oversampling": {"strengths": [s["random_oversampling"]]},
                  **{f: {"bank": rg.generator_id(cfg["rg"], f), "strengths": [s[f]]} for f in GENERATIVE}}  # fmt: skip
    return {**base, "experiment": EXPERIMENT, "name": GRID_NAME, "architectures": {"cnn": base["architectures"]["cnn"]},
            "policy": {"kind": "fraction_of_n"}, "conditions": conditions, "file": cfg["file"]}  # fmt: skip


def run_grid(cfg: dict, data: DataSource, before: DataSource, device: torch.device) -> dict:
    split = check_preconditions()
    grid = record_grid(cfg)
    ledger = runner.read_ledger(grid)
    train_before, train_after = training_beats(before.arrays), training_beats(data.arrays)
    report = {"done_before": 0, "ran": 0}
    for cell in runner.cells(grid):
        classifier = runner.classifier_settings(grid, cell.architecture, preview=False)
        bank, bank_sha, bank_id = None, None, None
        if cell.condition in GENERATIVE:
            p12_bank, bank_sha = rg.load_relative_bank(cfg["rg"], cell.condition, cell.seed)
            bank = v4_bank(p12_bank, train_before, train_after, rg.generators_record(cell.condition)["bank_counts"], cell.seed)
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
    grid = record_grid(cfg)
    expected = runner.cells(grid)
    done = {(r["condition"], r["strength"], r["architecture"], r["seed"]) for r in runner.read_ledger(grid).values() if r["status"] == "done"}
    missing = [c for c in expected if (c.condition, c.strength, c.architecture, c.seed) not in done]
    if missing:
        raise PreconditionError(f"{len(missing)} of {len(expected)} cells are not done, e.g. {missing[:2]}")
    chosen, _ = workflow.choose_strengths(grid, runner.summarize(grid))
    out = {"experiment": EXPERIMENT, "rule": "PREREGISTRATION §10 v12 (strengths carried from P11 and P12)", "chosen": chosen, "git_sha": runs.git_sha()}
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


def _reproduced(cms: np.ndarray, summary_path: Path, key: str) -> float:
    expected = json.loads(summary_path.read_text())["conditions"][key]["rare_macro_f1"]["point"]
    return float(np.mean(M.rare_macro_f1_from_cm(cms.sum(axis=1)))) - expected


def test_report(cfg: dict) -> dict:
    """DS2 in v3 form plus the recording-mean RR features, one posthoc access. Confirmatory: each condition with the new
    features vs the same condition in P11 (none, class weighting, random oversampling) or P12 (the two generators).
    Secondary: with the new features, each generator vs none, random oversampling and class weighting. Descriptive:
    Huang et al.'s SVEB rule on top of each condition. P11's and P12's scores are recomputed and checked first."""
    from ecgaug import final_eval

    new_sel = load_selection()["chosen"]["cnn"]
    old_sel = {**mp.load_selection()["chosen"]["cnn"], **{f: rg.load_selection()["chosen"]["cnn"][f] for f in GENERATIVE}}
    half = cfg["rg"]["mp"]["pt"]["half_window_s"]
    spec = load_data(check_preconditions(), preview=False).spec
    v1_test = pt.test_arrays(spec, experiment=EXPERIMENT)
    ratios = {}
    for record in DS2:
        ratios.update({(record, r): v for r, v in v5.record_ratios(record, half, access="posthoc").items()})
    before = pt.relative(v5.substitute(v1_test, ratios), pt.mitbih_templates(v1_test, spec, half, access="posthoc"))
    after = normalize(before, mean_rr_by_record(np.asarray(before["record"]), access="posthoc"))
    y, records = before["y"], before["record"]
    new, old, rule, unique = {}, {}, {}, None
    for c in CONDITIONS:
        probs = pt._predict(new_sel[c]["run_ids"], after)
        unique, new[c] = final_eval.per_record_cms(probs, y, records)
        _, rule[c] = final_eval.per_record_cms([sveb_rule(p, after["rr_prev"], cfg["rule_threshold"]) for p in probs], y, records)
        _, old[c] = final_eval.per_record_cms(pt._predict(old_sel[c]["run_ids"], before), y, records)
    sources = {c: (paths.results_for(rg.EXPERIMENT if c in GENERATIVE else mp.EXPERIMENT) / "summary.json") for c in CONDITIONS}
    reproduced = {c: _reproduced(old[c], sources[c], f"cnn/{c}") for c in CONDITIONS}
    off = {c: d for c, d in reproduced.items() if abs(d) > 1e-9}
    if off:
        raise PreconditionError(f"P11's or P12's scores do not reproduce: {off}")
    summary = {"experiment": EXPERIMENT, "post_hoc": True, "rule": "PREREGISTRATION §10 v12", "records": [int(r) for r in unique], "n_boot": ref.N_BOOT,
               "boot_seed": ref.BOOT_SEED, "rule_threshold": float(cfg["rule_threshold"]),
               "conditions": {f"cnn/{c}": ref.condition_summary(cm, unique) | {"strength": new_sel[c]["strength"]} for c, cm in new.items()},
               "before": {f"cnn/{c}": ref.condition_summary(cm, unique) for c, cm in old.items()},
               "sveb_rule": {f"cnn/{c}": ref.condition_summary(cm, unique) for c, cm in rule.items()},
               "families": {"confirmatory": mp._family({f"cnn: {c}, recording-mean RR vs before": (new[c], old[c]) for c in CONDITIONS}),
                            "secondary": mp._family({f"cnn: {g} vs {c} (recording-mean RR)": (new[g], new[c])
                                                     for g in GENERATIVE for c in ("none", "random_oversampling", "class_weighted")})},
               "reproduction_difference": reproduced, "run_ids": [r for c in CONDITIONS for r in new_sel[c]["run_ids"]],
               "git_sha": runs.git_sha()}  # fmt: skip
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
            c = design_check(cfg)
            for db, e in c["databases"].items():
                print(db, {r: round(v, 3) for r, v in e["auc"].items()}, {r: round(v["f1"], 3) for r, v in e["rule"].items()})
            for h in c["s_heavy_records"]:
                print(h)
        elif args.part == "grid":
            data, before, _ = build(cfg)
            print(json.dumps(run_grid(cfg, data, before, resolve_device(args.device))))
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
