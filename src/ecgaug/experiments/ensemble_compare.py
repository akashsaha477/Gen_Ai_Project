"""New ten-run ensembles against P18's, each with the stage and the veto P20's rule chooses (P22 and P23, PREREGISTRATION
§10 v21; post hoc).

Conditions are named ``<architecture>/<condition>``: ``cnn/cvae`` is P18's cVAE ensemble, ``resnet/none`` P22's ResNet
without augmentation, ``cnn/cvae_fitted`` P23's. ``pairs`` maps each new condition to the P18 condition it is compared
with. On P14's patients P20's rule chooses each new condition's setting (``choose``); P18's conditions keep P20's
choices. ``score_patients`` compares the pairs on P19's patients (S F1) and ``ds2_report`` on DS2 (rare-class macro-F1),
each one Holm family; P20's values for the baselines must reproduce (within 0.001 on P19's patients, exactly on DS2).
``within`` pairs are descriptive (paired intervals, no correction).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ecgaug import paths, runs
from ecgaug.data.records import CLASSES
from ecgaug.eval import metrics as M
from ecgaug.experiments import phase6_fresh_patients as fp
from ecgaug.experiments import phase6_more_patients as mp
from ecgaug.experiments import phase6_patient_template as pt
from ecgaug.experiments import phase6_rr_reference as ref
from ecgaug.experiments import phase6_rule_tuning as rt
from ecgaug.experiments import phase6_seed_ensemble as se
from ecgaug.experiments.common import PreconditionError

BASE = "cnn/"  # P18's conditions, as P20 names them in its summaries
PATIENT_TOLERANCE = rt.REPRODUCTION_TOLERANCE
DS2_TOLERANCE = 1e-9


def base_name(name: str) -> str:
    """``cnn/cvae`` -> ``cvae`` (P20's choice is keyed by P18's condition)."""
    if not name.startswith(BASE):
        raise ValueError(f"{name} is not one of P18's conditions")
    return name[len(BASE) :]


def baseline_members(pairs: dict[str, str]) -> dict[str, list[str]]:
    """P18's ten run IDs for every baseline in ``pairs`` (checked as P18's test read checked them)."""
    frozen = fp.frozen_members()
    return {b: frozen[base_name(b)] for b in sorted(set(pairs.values()))}


def baseline_settings(pairs: dict[str, str]) -> dict[str, tuple]:
    chosen = rt.chosen_settings(rt.load_choice())
    return {b: chosen[base_name(b)] for b in sorted(set(pairs.values()))}


def ensemble_probs(members: dict[str, list[str]], arrays: dict, predict_fn) -> dict[str, np.ndarray]:
    """Each condition's mean probability (float64, summed in run order)."""
    out = {}
    for c, ids in members.items():
        total = None
        for run_id in ids:
            p = np.asarray(predict_fn(run_id, arrays), dtype=np.float64)
            total = p if total is None else total + p
        out[c] = total / len(ids)
    return out


def choose(experiment: str, rule: str, probs: dict[str, np.ndarray], arrays: dict, new: list[str]) -> dict:
    """P20's rule for the new conditions on P14's patients."""
    choice = rt.choose({c: probs[c] for c in new}, arrays)
    return {**choice, "experiment": experiment, "rule": f"{rule} (P20's rule, PREREGISTRATION v19)"}


def _block(tables: dict[str, np.ndarray], records: np.ndarray) -> dict:
    return {c: {s: fp.cluster_bootstrap_confusion(fp._stat(s), cm, records, n=ref.N_BOOT, seed=ref.BOOT_SEED) for s in fp.STATS} for c, cm in tables.items()}


def score_patients(experiment: str, rule: str, choice: dict, probs: dict[str, np.ndarray], arrays: dict, pairs: dict[str, str],
                   within: dict[str, tuple[str, str]] | None = None) -> dict:  # fmt: skip
    """On P19's patients: each new condition with its chosen setting vs its baseline with P20's (S F1, one Holm family);
    the ensembles alone and ``within`` pairs beside. P20's S F1 for the baselines' settings must reproduce within 0.001."""
    form = choice["veto_form"]
    settings = {**{c: (v["rule_threshold"], v["veto_threshold"]) for c, v in choice["chosen"].items()}, **baseline_settings(pairs)}
    y, records = np.asarray(arrays["y"]), np.asarray(arrays["record"])
    rr_prev, p_af = np.asarray(arrays["rr_prev"]), np.asarray(arrays["p_af"])
    tables: dict[str, dict[str, np.ndarray]] = {"chosen": {}, "alone": {}}
    unique = None
    for c in (*pairs, *sorted(set(pairs.values()))):
        for k, s in (("chosen", settings[c]), ("alone", (None, None))):
            unique, cm = M.confusion_by_record(y, rt.decide(probs[c], rr_prev, p_af, s, form), records)
            tables[k][c] = cm[None]
    p20 = json.loads((paths.results_for(rt.EXPERIMENT) / "summary.json").read_text())["conditions"]["chosen"]
    reproduction = {b: float(fp._stat("f1_S")(tables["chosen"][b].sum(axis=1))[0]) - p20[b]["f1_S"]["point"] for b in sorted(set(pairs.values()))}
    off = {k: d for k, d in reproduction.items() if abs(d) > PATIENT_TOLERANCE}
    if off:
        raise PreconditionError(f"P20's S F1 for the baselines does not reproduce: {off}")
    out = {"experiment": experiment, "post_hoc": True, "rule": rule, "n_boot": ref.N_BOOT, "boot_seed": ref.BOOT_SEED, "veto_form": form,
           "chosen": choice["chosen"], "baseline_settings": {b: rt.label(s) for b, s in baseline_settings(pairs).items()},
           "patients": int(unique.size), "beats": {k: int((y == i).sum()) for i, k in enumerate(CLASSES)},
           "conditions": {"chosen": _block(tables["chosen"], unique), "ensemble_alone": _block(tables["alone"], unique)},
           "families": {"secondary": fp._family({f"{c} vs {b}, chosen settings": (tables["chosen"][c], tables["chosen"][b]) for c, b in pairs.items()})},
           "versus_baseline_alone": fp._family({f"{c} vs {b}, ensembles alone": (tables["alone"][c], tables["alone"][b]) for c, b in pairs.items()}),
           "evaluation_surface": rt.surface({c: probs[c] for c in pairs}, arrays, form), "reproduction_difference": reproduction,
           "note": f"secondary family (Holm, {len(pairs)}) on S F1; the rest is descriptive", "git_sha": runs.git_sha()}  # fmt: skip
    if within:
        out["within"] = fp._family({f"{a} vs {b}, chosen settings": (tables["chosen"][a], tables["chosen"][b]) for a, b in within.values()})
    return out


def run_patients(experiment: str, rule: str, members: dict[str, list[str]], pairs: dict[str, str], beats_dir: Path, device: torch.device,
                 within: dict[str, tuple[str, str]] | None = None) -> tuple[dict, dict]:  # fmt: skip
    """Predict both patient sets with every member, choose on P14's, score P19's; returns (choice, summary)."""
    everything = {**members, **baseline_members(pairs)}
    probs = {}
    for name in rt.SETS:
        arrays = rt.load_set(beats_dir, name)
        probs[name] = ensemble_probs(everything, arrays, lambda run_id, a: fp.predict(run_id, a, device))
        del arrays
    choice = choose(experiment, rule, probs["tuning"], rt.load_set(beats_dir, "tuning"), list(pairs))
    summary = score_patients(experiment, rule, choice, probs["evaluation"], rt.load_set(beats_dir, "evaluation"), pairs, within)
    return choice, summary


def ds2_report(experiment: str, rule: str, se_cfg: dict, choice: dict, members: dict[str, list[str]], pairs: dict[str, str],
               within: dict[str, tuple[str, str]] | None = None) -> dict:  # fmt: skip
    """DS2 in P13's form, one posthoc access logged under ``experiment``: each new condition with its chosen setting vs its
    baseline with P20's (rare-class macro-F1, one Holm family); P20's DS2 values for the baselines must reproduce exactly.
    Descriptive: the ensembles alone, per-class F1, without record 202, ``within`` pairs."""
    from ecgaug import final_eval

    form = choice["veto_form"]
    p20_test = json.loads((paths.results_for(rt.EXPERIMENT) / "test_summary.json").read_text())["conditions"]["chosen"]
    settings = {**{c: (v["rule_threshold"], v["veto_threshold"]) for c, v in choice["chosen"].items()}, **baseline_settings(pairs)}
    everything = {**members, **baseline_members(pairs)}
    test, p_af = se.ds2_inputs(se_cfg, experiment)
    y, records, rr_prev = test["y"], np.asarray(test["record"]), test["rr_prev"]
    tables: dict[str, dict[str, np.ndarray]] = {"chosen": {}, "alone": {}}
    unique = None
    for c, ids in everything.items():
        p = se.ensemble(pt._predict(ids, test))
        for k, s in (("chosen", settings[c]), ("alone", (None, None))):
            pred = rt.decide(p, rr_prev, p_af, s, form)
            unique, tables[k][c] = final_eval.per_record_cms([np.eye(p.shape[1], dtype=np.float32)[pred]], y, records)
    checks = {b: se._rare(tables["chosen"][b]) - p20_test[b]["rare_macro_f1"]["point"] for b in sorted(set(pairs.values()))}
    off = {k: d for k, d in checks.items() if abs(d) > DS2_TOLERANCE}
    if off:
        raise PreconditionError(f"P20's DS2 scores do not reproduce: {off}")
    keep = unique != 202

    def block(table: dict, kept: np.ndarray | None = None) -> dict:
        if kept is None:
            return {c: ref.condition_summary(cm, unique) for c, cm in table.items()}
        return {c: ref.condition_summary(cm[:, kept], unique[kept]) for c, cm in table.items()}

    out = {"experiment": experiment, "post_hoc": True, "rule": rule, "records": [int(r) for r in unique], "n_boot": ref.N_BOOT,
           "boot_seed": ref.BOOT_SEED, "veto_form": form, "settings": {c: rt.label(s) for c, s in settings.items()},
           "conditions": {"chosen": block(tables["chosen"]), "ensemble_alone": block(tables["alone"])},
           "without_202": {"chosen": block(tables["chosen"], keep)},
           "families": {"confirmatory": mp._family({f"{c} vs {b}, chosen settings": (tables["chosen"][c], tables["chosen"][b]) for c, b in pairs.items()})},
           "versus_baseline_alone": mp._family({f"{c} vs {b}, ensembles alone": (tables["alone"][c], tables["alone"][b]) for c, b in pairs.items()}),
           "note": f"confirmatory family (Holm, {len(pairs)}) on rare-class macro-F1; the rest is descriptive",
           "reproduction_difference": checks, "run_ids": everything, "git_sha": runs.git_sha()}  # fmt: skip
    if within:
        out["within"] = mp._family({f"{a} vs {b}, chosen settings": (tables["chosen"][a], tables["chosen"][b]) for a, b in within.values()})
    return out
