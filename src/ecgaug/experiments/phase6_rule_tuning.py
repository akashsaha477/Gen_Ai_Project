"""phase6_rule_tuning: Huang et al.'s SVEB stage and P16's AF veto chosen on P14's 400 Icentia11k patients, scored on
P19's patients and once on DS2 (PREREGISTRATION §10 v19, P20; post hoc).

    python -m ecgaug.experiments.phase6_rule_tuning --config configs/phase6_rule_tuning/rule_tuning.yaml --part build [--beats-dir DIR]   (CPU)
    python -m ecgaug.experiments.phase6_rule_tuning --config ... --part run --beats-dir DIR [--device cuda]                             (GPU)
    python -m ecgaug.experiments.phase6_rule_tuning --config ... --part score --beats-dir DIR                                          (CPU)
    ALLOW_TEST_POSTHOC=1 python -m ecgaug.experiments.phase6_rule_tuning --config ... --part test                                      (CPU)

On P19's 977 unused patients Huang's stage lowered S F1 against the ten-run ensembles alone in five of six conditions.
Its threshold 0.8 is Huang et al.'s MIT-BIH value and P16's veto threshold 0.95 came from MIT-BIH, INCART and SVDB
annotations. P20 chooses, per condition, among the ensemble alone and the stage at seven thresholds with no veto or the
veto at six thresholds (50 settings), by S F1 over P14's 400 patients, which none of P18's runs has seen. The choice is
written to ``choice.json`` before any of P19's patients is scored; P19's patients (rebuilt, hash-checked against P19)
then compare the chosen setting with P19's and with the ensemble alone, and one DS2 read compares it with P18's.

``build`` writes the two patient sets' beats to ``--beats-dir``; ``run`` predicts both with P18's sixty runs, saves each
condition's ten-run mean probabilities (``probs.npz``, beside the beats unless that folder is read-only), chooses and
scores; ``score`` does the choice and the scoring again from saved probabilities; ``test`` is the DS2 read.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from ecgaug import paths, runs
from ecgaug.data.access import LockedTestDataError
from ecgaug.data.contract import content_sha256
from ecgaug.data.records import CLASS_INDEX, CLASSES
from ecgaug.eval import metrics as M
from ecgaug.eval.bootstrap import paired_cluster_bootstrap_diff_confusion
from ecgaug.experiments import phase6_af_rule as af
from ecgaug.experiments import phase6_fresh_patients as fp
from ecgaug.experiments import phase6_more_patients as mp
from ecgaug.experiments import phase6_patient_template as pt
from ecgaug.experiments import phase6_record_rr as rr
from ecgaug.experiments import phase6_rr_reference as ref
from ecgaug.experiments import phase6_seed_ensemble as se
from ecgaug.experiments.common import PreconditionError, check_preconditions
from ecgaug.experiments.data_source import load_data
from ecgaug.names import PHASE6_RULE_TUNING
from ecgaug.train.loop import resolve_device

EXPERIMENT = PHASE6_RULE_TUNING
RULE = "PREREGISTRATION §10 v19"
CONFIG_KEYS = {"experiment", "fresh_patients_config", "rule_thresholds", "veto_thresholds"}
RULE_THRESHOLDS = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)  # fixed by PREREGISTRATION v19
VETO_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
FIXED = (0.80, 0.95)  # P19's and P18's setting: Huang's threshold, P16's veto threshold
REPRODUCTION_TOLERANCE = 1e-3  # P19's S F1 for its own setting, predicted again on a GPU
SETS = ("tuning", "evaluation")
Setting = tuple  # (rule threshold or None, veto threshold or None); (None, None) is the ensemble alone


def load_config(path: Path) -> dict:
    """P20's config: the pre-registered grid; P19's config brings P18's, P16's and P13's."""
    cfg = yaml.safe_load(Path(path).read_text())
    if set(cfg) != CONFIG_KEYS or cfg["experiment"] != EXPERIMENT:
        raise ValueError(f"{path}: expected experiment {EXPERIMENT} and exactly the keys {sorted(CONFIG_KEYS)}")
    if tuple(float(t) for t in cfg["rule_thresholds"]) != RULE_THRESHOLDS or tuple(float(a) for a in cfg["veto_thresholds"]) != VETO_THRESHOLDS:
        raise ValueError(f"{path}: the thresholds must be {RULE_THRESHOLDS} and {VETO_THRESHOLDS} (PREREGISTRATION v19)")
    fresh = fp.load_config(paths.repo_root() / cfg["fresh_patients_config"])
    if float(fresh["af"]["rule_threshold"]) != FIXED[0]:
        raise ValueError("P19's rule threshold is not 0.8")
    return {**cfg, "file": str(path), "fresh": fresh, "ensemble": fresh["ensemble"], "af": fresh["af"], "rr": fresh["rr"]}


# --------------------------------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------------------------------


def settings() -> list[Setting]:
    """The 50 pre-registered settings: the ensemble alone, the stage alone at seven thresholds, the stage with the veto."""
    return [(None, None), *((t, None) for t in RULE_THRESHOLDS), *((t, a) for t in RULE_THRESHOLDS for a in VETO_THRESHOLDS)]


def label(s: Setting) -> str:
    t, a = s
    return "ensemble alone" if t is None else f"stage {t:.2f}" + ("" if a is None else f", veto {a:.2f}")


def preference(s: Setting) -> tuple:
    """Tie order: the ensemble alone, then no veto, then the threshold nearest 0.80 (the lower), then the higher veto threshold."""
    t, a = s
    if t is None:
        return (0,)
    return (1, a is not None, round(abs(t - FIXED[0]), 9), t, -(a or 0.0))


def decide(probs: np.ndarray, rr_prev: np.ndarray, p_af: np.ndarray, s: Setting, form: str) -> np.ndarray:
    """Class index per beat for one setting (argmax; Huang's stage; the stage with P16's veto)."""
    t, a = s
    if t is None:
        return np.asarray(probs).argmax(axis=1)
    if a is None:
        return rr.sveb_rule(probs, rr_prev, t).argmax(axis=1)
    return af.veto_rule(probs, rr_prev, p_af, t, a, form).argmax(axis=1)


def pooled_f1_s(pred: np.ndarray, y: np.ndarray) -> float:
    """S F1 over all beats pooled (the bootstrap's point estimate)."""
    cm = M.confusion_by_record(y, pred, np.zeros(np.asarray(y).size, dtype=np.int64))[1][0]
    return float(M.f1_from_cm(cm)[CLASS_INDEX["S"]])


def surface(probs: dict[str, np.ndarray], arrays: dict, form: str) -> dict[str, dict[str, float]]:
    """S F1 over all beats pooled, per condition and setting."""
    y, rr_prev, p_af = np.asarray(arrays["y"]), np.asarray(arrays["rr_prev"]), np.asarray(arrays["p_af"])
    return {c: {label(s): pooled_f1_s(decide(p, rr_prev, p_af, s, form), y) for s in settings()} for c, p in probs.items()}


def choose(probs: dict[str, np.ndarray], arrays: dict) -> dict:
    """Per condition, the setting with the highest pooled S F1 on the tuning patients (ties by ``preference``)."""
    af_t, form = af.chosen_design()
    if af_t != FIXED[1]:
        raise PreconditionError(f"P16's veto threshold is {af_t}, not {FIXED[1]}")
    table = surface(probs, arrays, form)
    by_label = {label(s): s for s in settings()}
    chosen = {}
    for c, scores in table.items():
        best = max(scores.values())
        s = min((by_label[k] for k, v in scores.items() if v == best), key=preference)
        chosen[c] = {"rule_threshold": s[0], "veto_threshold": s[1], "label": label(s), "s_f1": best}
    return {"experiment": EXPERIMENT, "rule": RULE, "veto_form": form, "settings": [label(s) for s in settings()], "chosen": chosen,
            "tuning_surface": table, "patients": int(np.unique(np.asarray(arrays["record"])).size),
            "beats": {k: int((np.asarray(arrays["y"]) == i).sum()) for i, k in enumerate(CLASSES)}, "git_sha": runs.git_sha()}  # fmt: skip


def chosen_settings(choice: dict) -> dict[str, Setting]:
    return {c: (v["rule_threshold"], v["veto_threshold"]) for c, v in choice["chosen"].items()}


# --------------------------------------------------------------------------------------------------
# the beats: P14's patients (tuning) and P19's (evaluation)
# --------------------------------------------------------------------------------------------------


def default_beats_dir() -> Path:
    return paths.processed_dir() / "rule_tuning"


def locate(root: Path, name: str) -> Path:
    """``root/name``, or the one file of that name below ``root`` (a Kaggle session's attached outputs)."""
    root = Path(root)
    if (root / name).is_file():
        return root / name
    found = sorted(root.rglob(name)) if root.is_dir() else []
    if len(found) != 1:
        raise PreconditionError(f"Expected one {name} under {root}, found {len(found)}: run --part build first")
    return found[0]


def build(cfg: dict, spec, beats_dir: Path, workers: int | None = None) -> dict:
    """Both patient sets' beats, saved to ``beats_dir``; the evaluation set must reproduce P19's beats exactly."""
    p19 = json.loads((paths.results_for(fp.EXPERIMENT) / "data.json").read_text())
    half = float(cfg["rr"]["rg"]["mp"]["pt"]["half_window_s"])
    patients = {"tuning": fp.used_patients(), "evaluation": fp.sample_patients(fp.N_FRESH, fp.FRESH_SEED, fp.used_patients())}
    if patients["evaluation"] != p19["patients"]:
        raise PreconditionError("P19's patients do not come out of its seed")
    beats_dir.mkdir(parents=True, exist_ok=True)
    record = {"experiment": EXPERIMENT, "rule": RULE}
    for name in SETS:
        arrays, info = fp.build_set(patients[name], spec, half, workers)
        if name == "evaluation" and info["content_sha256"] != p19["content_sha256"]:
            raise PreconditionError("The rebuilt evaluation beats differ from P19's (content_sha256)")
        np.savez(beats_dir / f"{name}.npz", **arrays)
        record[name] = {"patients": patients[name], **info}
        del arrays
    record["evaluation_matches_p19"] = True
    record["git_sha"] = runs.git_sha()
    out = paths.results_for(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.json").write_text(json.dumps(record, indent=2, default=float) + "\n")
    return record


def load_set(beats_dir: Path, name: str) -> dict:
    """One patient set's beats as ``build`` saved them, checked against data.json."""
    record_path = paths.results_for(EXPERIMENT) / "data.json"
    if not record_path.is_file():
        raise PreconditionError("data.json is missing: run --part build and ingest its results first")
    with np.load(locate(beats_dir, f"{name}.npz")) as z:
        arrays = {k: z[k] for k in fp.FIELDS}
    if content_sha256(arrays) != json.loads(record_path.read_text())[name]["content_sha256"]:
        raise PreconditionError(f"The saved {name} beats do not match data.json")
    return arrays


# --------------------------------------------------------------------------------------------------
# prediction and scoring
# --------------------------------------------------------------------------------------------------


def ensemble_probs(arrays: dict, predict_fn) -> dict[str, np.ndarray]:
    """Each condition's ten-run mean probability (float64, summed in P18's run order as P19 did)."""
    out = {}
    for c, ids in fp.frozen_members().items():
        total = None
        for run_id in ids:
            p = np.asarray(predict_fn(run_id, arrays), dtype=np.float64)
            total = p if total is None else total + p
        out[c] = total / len(ids)
    return out


def score(choice: dict, probs: dict[str, np.ndarray], arrays: dict) -> dict:
    """On P19's patients: per condition the chosen setting, P19's setting and the ensemble alone (patient-level bootstrap);
    confirmatory chosen vs P19's, secondary chosen vs alone, both on S F1 with Holm; descriptive the evaluation surface
    and the cVAE vs each other condition with the chosen settings. P19's own S F1 for its setting must reproduce."""
    form = choice["veto_form"]
    chosen = chosen_settings(choice)
    y, records = np.asarray(arrays["y"]), np.asarray(arrays["record"])
    rr_prev, p_af = np.asarray(arrays["rr_prev"]), np.asarray(arrays["p_af"])
    kinds = {"chosen": None, "p19": FIXED, "alone": (None, None)}
    tables: dict[str, dict[str, np.ndarray]] = {k: {} for k in kinds}
    unique = None
    for c, p in probs.items():
        for k, s in kinds.items():
            pred = decide(p, rr_prev, p_af, chosen[c] if s is None else s, form)
            unique, cm = M.confusion_by_record(y, pred, records)
            tables[k][c] = cm[None]
    p19 = json.loads((paths.results_for(fp.EXPERIMENT) / "summary.json").read_text())
    reproduction = {f"cnn/{c}": float(fp._stat("f1_S")(tables["p19"][c].sum(axis=1))[0]) - p19["ensemble"]["with_af_veto"][f"cnn/{c}"]["f1_S"]["point"] for c in probs}
    off = {k: d for k, d in reproduction.items() if abs(d) > REPRODUCTION_TOLERANCE}
    if off:
        raise PreconditionError(f"P19's S F1 for its own setting does not reproduce: {off}")
    pairs = {f"cnn: {se.REFERENCE} vs {c}, chosen settings": (tables["chosen"][se.REFERENCE], tables["chosen"][c]) for c in probs if c != se.REFERENCE}
    return {
        "experiment": EXPERIMENT, "post_hoc": True, "rule": RULE, "n_boot": ref.N_BOOT, "boot_seed": ref.BOOT_SEED, "veto_form": form,
        "chosen": choice["chosen"], "patients": int(unique.size), "beats": {k: int((y == i).sum()) for i, k in enumerate(CLASSES)},
        "conditions": {"chosen": fp._block(tables["chosen"], unique), "p19_setting": fp._block(tables["p19"], unique),
                       "ensemble_alone": fp._block(tables["alone"], unique)},
        "families": {"confirmatory": fp._family({f"cnn: {c}, chosen vs P19's setting": (tables["chosen"][c], tables["p19"][c]) for c in probs}),
                     "secondary": fp._family({f"cnn: {c}, chosen vs ensemble alone": (tables["chosen"][c], tables["alone"][c]) for c in probs})},
        "versus_cvae": {k: paired_cluster_bootstrap_diff_confusion(fp._stat("f1_S"), a, b, n=ref.N_BOOT, seed=ref.BOOT_SEED) for k, (a, b) in pairs.items()},
        "evaluation_surface": surface(probs, arrays, form),
        "reproduction_difference": reproduction,
        "note": "S F1 is the primary measure (Icentia11k has no F labels); versus_cvae and evaluation_surface are descriptive",
        "git_sha": runs.git_sha(),
    }  # fmt: skip


def _save_probs(probs: dict[str, dict[str, np.ndarray]], beats_dir: Path) -> Path:
    target = beats_dir if os.access(beats_dir, os.W_OK) else default_beats_dir()
    target.mkdir(parents=True, exist_ok=True)
    np.savez(target / "probs.npz", **{f"{name}/{c}": p for name, per in probs.items() for c, p in per.items()})
    return target / "probs.npz"


def _load_probs(beats_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    with np.load(locate(beats_dir, "probs.npz")) as z:
        out: dict[str, dict[str, np.ndarray]] = {name: {} for name in SETS}
        for key in z.files:
            name, c = key.split("/")
            out[name][c] = z[key]
    if any(tuple(v) != se.CONDITIONS for v in out.values()):
        raise PreconditionError("probs.npz must hold every condition for both patient sets")
    return out


def choose_and_score(probs: dict[str, dict[str, np.ndarray]], beats_dir: Path) -> tuple[dict, dict]:
    """The choice on the tuning patients, written before the evaluation patients are loaded; then their scores."""
    out = paths.results_for(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    choice = choose(probs["tuning"], load_set(beats_dir, "tuning"))
    (out / "choice.json").write_text(json.dumps(choice, indent=2, default=float) + "\n")
    summary = score(choice, probs["evaluation"], load_set(beats_dir, "evaluation"))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return choice, summary


def run(beats_dir: Path, device: torch.device) -> tuple[dict, dict]:
    probs = {}
    for name in SETS:
        arrays = load_set(beats_dir, name)
        probs[name] = ensemble_probs(arrays, lambda run_id, a: fp.predict(run_id, a, device))
        del arrays
    print(f"probabilities saved to {_save_probs(probs, beats_dir)}")
    return choose_and_score(probs, beats_dir)


# --------------------------------------------------------------------------------------------------
# DS2: one posthoc read
# --------------------------------------------------------------------------------------------------


def load_choice() -> dict:
    path = paths.results_for(EXPERIMENT) / "choice.json"
    if not path.is_file():
        raise PreconditionError("choice.json is missing: run --part run (or score) and ingest it first")
    return json.loads(path.read_text())


def test_report(cfg: dict) -> dict:
    """DS2 in P13's form, one posthoc access. Each condition's ten-run ensemble with its chosen setting vs with P18's
    (rare-class macro-F1, Holm 6); P18's values must reproduce exactly. Descriptive: per-class F1, the ensembles alone,
    without record 202."""
    from ecgaug import final_eval

    choice = load_choice()
    chosen, form = chosen_settings(choice), choice["veto_form"]
    members = fp.frozen_members()
    p18 = se._summary(se.EXPERIMENT)
    test, p_af = se.ds2_inputs(cfg, EXPERIMENT)
    y, records, rr_prev = test["y"], np.asarray(test["record"]), test["rr_prev"]
    kinds = {"chosen": None, "p18": FIXED, "alone": (None, None)}
    tables: dict[str, dict[str, np.ndarray]] = {k: {} for k in kinds}
    unique = None
    for c, ids in members.items():
        p = se.ensemble(pt._predict(ids, test))
        for k, s in kinds.items():
            pred = decide(p, rr_prev, p_af, chosen[c] if s is None else s, form)
            unique, tables[k][c] = final_eval.per_record_cms([np.eye(p.shape[1], dtype=np.float32)[pred]], y, records)
    checks = {f"cnn/{c}": se._rare(tables["p18"][c]) - p18["ensemble"]["with_af_veto"][f"cnn/{c}"]["rare_macro_f1"]["point"] for c in members}
    off = {k: d for k, d in checks.items() if abs(d) > 1e-9}
    if off:
        raise PreconditionError(f"P18's DS2 scores do not reproduce: {off}")
    keep = unique != 202

    def block(table: dict, kept: np.ndarray | None = None) -> dict:
        if kept is None:
            return {f"cnn/{c}": ref.condition_summary(cm, unique) for c, cm in table.items()}
        return {f"cnn/{c}": ref.condition_summary(cm[:, kept], unique[kept]) for c, cm in table.items()}

    summary = {"experiment": EXPERIMENT, "post_hoc": True, "rule": RULE, "records": [int(r) for r in unique], "n_boot": ref.N_BOOT,
               "boot_seed": ref.BOOT_SEED, "veto_form": form, "chosen": choice["chosen"],
               "conditions": {"chosen": block(tables["chosen"]), "p18_setting": block(tables["p18"]), "ensemble_alone": block(tables["alone"])},
               "without_202": {"chosen": block(tables["chosen"], keep), "p18_setting": block(tables["p18"], keep)},
               "families": {"ds2": mp._family({f"cnn: {c}, chosen vs P18's setting": (tables["chosen"][c], tables["p18"][c]) for c in members})},
               "note": "DS2 family (Holm, 6) on rare-class macro-F1; the rest is descriptive",
               "reproduction_difference": checks, "run_ids": members, "git_sha": runs.git_sha()}  # fmt: skip
    (paths.results_for(EXPERIMENT) / "test_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--part", choices=["build", "run", "score", "test"], required=True)
    parser.add_argument("--beats-dir", default=None, help="where build writes the beats and run and score find them (searched below)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config))
    beats_dir = Path(args.beats_dir) if args.beats_dir else default_beats_dir()
    try:
        if args.part == "build":
            spec = load_data(check_preconditions(), preview=False).spec
            record = build(cfg, spec, beats_dir)
            for name in SETS:
                print(name, json.dumps({k: record[name][k] for k in ("segments_used", "beats", "patients_with", "s_dominant_patients")}))
            return 0
        if args.part == "test":
            s = test_report(cfg)
            for name, c in s["families"]["ds2"].items():
                print(f"  ds2 {name:44s} {c['point']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] Holm p {c['p_holm']:.3f}")
            return 0
        if args.part == "run":
            choice, s = run(beats_dir, resolve_device(args.device))
        else:
            choice, s = choose_and_score(_load_probs(beats_dir), beats_dir)
        for c, v in choice["chosen"].items():
            print(f"  chosen for {c:20s} {v['label']:24s} tuning S F1 {v['s_f1']:.3f}")
        for family, items in s["families"].items():
            for name, c in items.items():
                print(f"  {family:12s} {name:44s} {c['point']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] Holm p {c['p_holm']:.3f}")
    except (PreconditionError, LockedTestDataError) as err:
        print(f"Refusing to run: {err}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
