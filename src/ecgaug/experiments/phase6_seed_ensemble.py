"""phase6_seed_ensemble: the cnn's class probabilities averaged over ten runs (PREREGISTRATION §10 v17, P18; post hoc).

    python -m ecgaug.experiments.phase6_seed_ensemble --config configs/phase6_seed_ensemble/seed_ensemble.yaml --part grid [--device cuda]
    python -m ecgaug.experiments.phase6_seed_ensemble --config ... --part check     # validation patients, the runs' stored predictions
    ALLOW_TEST_POSTHOC=1 python -m ecgaug.experiments.phase6_seed_ensemble --config ... --part test      # one posthoc access

Each run of the cnn keeps the epoch with the highest validation S F1, and within one run that score moves by 0.09 to 0.34
between consecutive epochs (P13's and P15's selected runs), so the model a run keeps is a noisy draw. Averaging the class
probabilities of runs trained independently (a deep ensemble; Lakshminarayanan et al., 2017) is the standard way to reduce
that noise, and such ensembles held up best under dataset shift in Ovadia et al. (2019). P18 trains seeds 3-9 in P16's six
conditions (P13's five and P15's diffusion condition) with every setting unchanged; a run of seed s uses bank seed s mod 3,
the three frozen banks of each generator. A condition's ensemble is the mean of its ten runs' probabilities (seeds 0-2 are
the existing runs), then argmax; Huang et al.'s SVEB stage and P16's AF veto go on top, as in P16. Nothing is chosen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from ecgaug import paths, runs
from ecgaug.data.access import LockedTestDataError
from ecgaug.data.records import CLASS_INDEX, DS2
from ecgaug.eval import metrics as M
from ecgaug.experiments import phase6_af_rule as af
from ecgaug.experiments import phase6_diffusion as dif
from ecgaug.experiments import phase6_more_patients as mp
from ecgaug.experiments import phase6_patient_template as pt
from ecgaug.experiments import phase6_record_rr as rr
from ecgaug.experiments import phase6_relative_generators as rg
from ecgaug.experiments import phase6_rr_baseline as v5
from ecgaug.experiments import phase6_rr_reference as ref
from ecgaug.experiments.common import PreconditionError, check_preconditions
from ecgaug.experiments.data_source import load_data
from ecgaug.generators.base import training_beats
from ecgaug.grid import runner
from ecgaug.names import PHASE6_SEED_ENSEMBLE
from ecgaug.train.loop import resolve_device

EXPERIMENT = PHASE6_SEED_ENSEMBLE
RULE = "PREREGISTRATION §10 v17"
CONFIG_KEYS = {"experiment", "af_rule_config", "new_seeds"}
EXISTING_SEEDS = (0, 1, 2)  # P13's and P15's selected runs
NEW_SEEDS = tuple(range(3, 10))
N_BANKS = len(EXISTING_SEEDS)  # every generator has three frozen banks, seeds 0-2
GRID_NAME = "full_grid"
CONDITIONS = (*rr.CONDITIONS, dif.FAMILY)  # P16's six: none, class weighting, random oversampling, cVAE, WGAN-GP, diffusion
GENERATIVE = (*rg.FAMILIES, dif.FAMILY)
REFERENCE = "cvae"  # the secondary family: the cVAE's ensemble against every other condition's
KINDS = ("cnn", "rule", "veto")  # the cnn alone, with Huang's stage, with the stage and P16's AF veto


def load_config(path: Path) -> dict:
    """P18's config. The seeds to train must be 3-9 (PREREGISTRATION v17); P16's config brings P13's."""
    cfg = yaml.safe_load(Path(path).read_text())
    if set(cfg) != CONFIG_KEYS or cfg["experiment"] != EXPERIMENT:
        raise ValueError(f"{path}: expected experiment {EXPERIMENT} and exactly the keys {sorted(CONFIG_KEYS)}")
    if tuple(int(s) for s in cfg["new_seeds"]) != NEW_SEEDS:
        raise ValueError(f"{path}: new_seeds must be {list(NEW_SEEDS)} (PREREGISTRATION v17)")
    af_cfg = af.load_config(paths.repo_root() / cfg["af_rule_config"])
    return {**cfg, "file": str(path), "af": af_cfg, "rr": af_cfg["rr"]}


def bank_seed(seed: int) -> int:
    """The bank a run of ``seed`` uses: seed mod 3, so the new seeds cycle over each generator's three frozen banks."""
    if int(seed) < 0:
        raise ValueError("Seeds are non-negative")
    return int(seed) % N_BANKS


def ensemble(probs: list[np.ndarray]) -> np.ndarray:
    """The mean of the runs' class probabilities, beat by beat (one ``[beats, classes]`` array per run)."""
    stacked = np.stack([np.asarray(p, dtype=np.float64) for p in probs]) if len(probs) else np.empty((0,))
    if stacked.ndim != 3 or stacked.shape[0] == 0:
        raise ValueError("Expected one [beats, classes] array per run, at least one run")
    return stacked.mean(axis=0)


def decisions(probs: np.ndarray, rr_prev: np.ndarray, p_af: np.ndarray, rule_threshold: float, af_threshold: float, form: str) -> dict[str, np.ndarray]:
    """One-hot decisions from class probabilities: the cnn alone (argmax), with Huang's SVEB stage, and with the stage and
    P16's AF veto."""
    p = np.asarray(probs)
    return {"cnn": np.eye(p.shape[1], dtype=np.float32)[p.argmax(axis=1)], "rule": rr.sveb_rule(p, rr_prev, rule_threshold),
            "veto": af.veto_rule(p, rr_prev, p_af, rule_threshold, af_threshold, form)}  # fmt: skip


# --------------------------------------------------------------------------------------------------
# the new runs: P13's grid (and P15's diffusion condition) for seeds 3-9
# --------------------------------------------------------------------------------------------------


def existing_runs() -> dict[str, list[str]]:
    """P16's compared runs, seeds 0-2 in seed order: P13's five conditions and P15's diffusion condition."""
    chosen = {**rr.load_selection()["chosen"]["cnn"], dif.FAMILY: dif.load_selection()["chosen"]["cnn"][dif.FAMILY]}
    wrong = [c for c in CONDITIONS if c not in chosen or tuple(chosen[c]["seeds"]) != EXISTING_SEEDS]
    if wrong:
        raise PreconditionError(f"P13's or P15's selection lacks seeds 0-2 for {wrong}")
    return {c: list(chosen[c]["run_ids"]) for c in CONDITIONS}


def strengths() -> dict[str, float | None]:
    """The strength of every condition's selected runs (none has no strength)."""
    chosen = {**rr.load_selection()["chosen"]["cnn"], dif.FAMILY: dif.load_selection()["chosen"]["cnn"][dif.FAMILY]}
    return {c: chosen[c]["strength"] for c in CONDITIONS}


def ensemble_grid(cfg: dict) -> dict:
    """P13's grid (the P7 cnn, its settings, the strengths carried from P11 and P12) plus P15's diffusion condition at its
    chosen strength, for seeds 3-9."""
    base = rr.record_grid(cfg["rr"])
    conditions = {**base["conditions"], dif.FAMILY: {"bank": dif.generator_id(), "strengths": [strengths()[dif.FAMILY]]}}
    if tuple(conditions) != CONDITIONS:
        raise PreconditionError(f"Expected the conditions {CONDITIONS}, got {tuple(conditions)}")
    return {**base, "experiment": EXPERIMENT, "name": GRID_NAME, "seeds": list(NEW_SEEDS), "conditions": conditions, "file": cfg["file"]}


def load_bank(cfg: dict, condition: str, seed: int, train_before: dict, train_after: dict) -> tuple[dict, str]:
    """The frozen bank a run of ``seed`` uses, and its SHA-256: P12's with its RR features in P13's form (as P13), or P15's."""
    b = bank_seed(seed)
    if condition == dif.FAMILY:
        return dif.load_diffusion_bank(b)
    bank, sha = rg.load_relative_bank(cfg["rr"]["rg"], condition, b)
    return rr.v4_bank(bank, train_before, train_after, rg.generators_record(condition)["bank_counts"], b), sha


def run_grid(cfg: dict, device: torch.device) -> dict:
    """The 42 new runs (six conditions, seeds 3-9) on P13's data, hash-checked; cells already done are skipped."""
    data, before, _ = rr.build(cfg["rr"])
    split = check_preconditions()
    grid = ensemble_grid(cfg)
    ledger = runner.read_ledger(grid)
    train_before, train_after = training_beats(before.arrays), training_beats(data.arrays)
    loaded: dict[tuple[str, int], tuple[dict, str]] = {}
    report = {"done_before": 0, "ran": 0}
    for cell in runner.cells(grid):
        classifier = runner.classifier_settings(grid, cell.architecture, preview=False)
        bank, bank_sha, bank_id = None, None, None
        if cell.condition in GENERATIVE:
            key = (cell.condition, bank_seed(cell.seed))
            if key not in loaded:
                loaded = {k: v for k, v in loaded.items() if k[0] == cell.condition}  # one condition's banks in memory at a time
                loaded[key] = load_bank(cfg, cell.condition, cell.seed, train_before, train_after)
            bank, bank_sha = loaded[key]
            bank_id = grid["conditions"][cell.condition]["bank"]
        sha = runner.cell_sha(cell, grid, classifier, data, bank_sha)
        if runner.is_done(ledger.get(sha)):
            report["done_before"] += 1
            continue
        ledger[sha] = runner.run_cell(cell, grid, split, data, classifier, device, sha, bank, bank_id, [])
        runner.write_cell(grid, ledger[sha])
        report["ran"] += 1
    return report


def members(cfg: dict) -> dict[str, list[str]]:
    """Per condition, the run IDs of seeds 0-9 in seed order: P13's and P15's selected runs, then P18's new runs (one done
    ledger row per condition and seed, at the carried strength)."""
    existing, carried = existing_runs(), strengths()
    grid = ensemble_grid(cfg)
    rows: dict[tuple[str, int], list[dict]] = {}
    for r in runner.read_ledger(grid).values():
        if r["status"] == "done":
            rows.setdefault((r["condition"], int(r["seed"])), []).append(r)
    out = {}
    for c in CONDITIONS:
        new = []
        for s in NEW_SEEDS:
            found = rows.get((c, s), [])
            if len(found) != 1 or found[0]["strength"] != carried[c]:
                raise PreconditionError(f"{c} seed {s}: expected one done run at strength {carried[c]}, found {len(found)}")
            new.append(found[0]["run_id"])
        out[c] = [*existing[c], *new]
    return out


# --------------------------------------------------------------------------------------------------
# validation (P13's validation patients, the runs' stored predictions; descriptive)
# --------------------------------------------------------------------------------------------------


def _f1(pred: np.ndarray, y: np.ndarray, cls: str) -> float:
    k = CLASS_INDEX[cls]
    tp, fp, fn = int(((pred == k) & (y == k)).sum()), int(((pred == k) & (y != k)).sum()), int(((pred != k) & (y == k)).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


def _validation_features(rec: np.ndarray, r_idx: np.ndarray, detector: dict, cache: dict) -> tuple[np.ndarray, np.ndarray]:
    """Each validation beat's RR interval over its recording's mean and its AF probability, from annotation times only."""
    ratio, p_af = np.full(rec.size, np.nan), np.zeros(rec.size)
    for r in np.unique(rec).tolist():
        if r not in cache:
            cache[r] = af._record_beats(int(r), detector)
        at, rat, prob = cache[r]
        m = rec == r
        pos = np.clip(np.searchsorted(at, r_idx[m]), 0, at.size - 1)
        if not np.array_equal(at[pos], r_idx[m]):
            raise PreconditionError(f"Record {r}: validation beats do not match its QRS annotations")
        ratio[m], p_af[m] = rat[pos], prob[pos]
    return ratio, p_af


def _validation_predictions(run_ids: list[str]) -> tuple[list[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    probs, beats = [], None
    for run_id in run_ids:
        folder = runs.find_run_dir(run_id)
        if folder is None or not (folder / "preds_val.npz").is_file():
            raise PreconditionError(f"No validation predictions for {run_id}")
        z = np.load(folder / "preds_val.npz")
        these = (np.asarray(z["y"]), np.asarray(z["record"]).astype(np.int64), np.asarray(z["r_idx"]).astype(np.int64))
        if beats is None:
            beats = these
        elif not all(np.array_equal(a, b) for a, b in zip(these, beats)):
            raise PreconditionError(f"{run_id}: its validation beats differ from the other runs'")
        probs.append(np.asarray(z["probs"]))
    return probs, beats


def validation_check(cfg: dict) -> dict:
    """On P13's validation patients, from the runs' stored validation predictions (no test data): per condition, S, V and F
    F1 of the cnn alone, with Huang's stage, and with the stage and P16's veto, for each of the ten single runs, the
    three-run ensemble of seeds 0-2 and the ten-run ensemble. Descriptive; nothing is chosen. Writes ``members.json``,
    the run IDs the test read uses."""
    m = members(cfg)
    detector, (af_t, form) = af.load_detector(), af.chosen_design()
    t_rule = float(cfg["af"]["rule_threshold"])
    cache: dict[int, tuple] = {}
    table = {}
    for c, ids in m.items():
        probs, (y, rec, r_idx) = _validation_predictions(ids)
        ratio, p_af = _validation_features(rec, r_idx, detector, cache)
        single = [decisions(p, ratio, p_af, t_rule, af_t, form) for p in probs]
        three, ten = (decisions(ensemble(group), ratio, p_af, t_rule, af_t, form) for group in (probs[: len(EXISTING_SEEDS)], probs))
        table[c] = {k: {cls: {"single": [_f1(d[k].argmax(axis=1), y, cls) for d in single], "three": _f1(three[k].argmax(axis=1), y, cls),
                              "ten": _f1(ten[k].argmax(axis=1), y, cls)} for cls in ("S", "V", "F")} for k in KINDS}  # fmt: skip
    mean_s = {k: {"single": float(np.mean([np.mean(table[c][k]["S"]["single"]) for c in table])),
                  "three": float(np.mean([table[c][k]["S"]["three"] for c in table])),
                  "ten": float(np.mean([table[c][k]["S"]["ten"] for c in table]))} for k in KINDS}  # fmt: skip
    out = {"experiment": EXPERIMENT, "rule": f"{RULE} (validation patients, the runs' stored predictions; descriptive)",
           "rule_threshold": t_rule, "af_threshold": af_t, "veto_form": form, "members": m, "validation": table,
           "mean_s_f1": mean_s, "records": sorted(int(r) for r in cache), "git_sha": runs.git_sha()}  # fmt: skip
    folder = paths.results_for(EXPERIMENT)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "members.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    return out


def load_members() -> dict:
    path = paths.results_for(EXPERIMENT) / "members.json"
    if not path.is_file():
        raise PreconditionError("Run the validation check first: --part check")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------------------------------
# DS2
# --------------------------------------------------------------------------------------------------


def _summary(experiment: str) -> dict:
    path = paths.results_for(experiment) / "summary.json"
    if not path.is_file():
        raise PreconditionError(f"{experiment} has no test summary yet; P18 checks its scores against it")
    return json.loads(path.read_text())


def _rare(cms: np.ndarray) -> float:
    return float(np.mean(M.rare_macro_f1_from_cm(np.asarray(cms).sum(axis=1))))


def test_report(cfg: dict) -> dict:
    """DS2 in P13's form, one posthoc access. Each condition's ten runs predict; its ensemble is their mean probability.
    Confirmatory (Holm, 6 tests): per condition, the ten-run ensemble vs P16's single runs (seeds 0-2), both with Huang's
    stage and P16's AF veto. Secondary (Holm, 5 tests): the cVAE's ensemble vs every other condition's, with the stage and
    the veto. Descriptive: the ensembles alone and with the stage alone, the ten single runs, the three-run ensemble, and
    without record 202. P13's, P15's and P16's scores are recomputed from seeds 0-2 and must match their summaries."""
    from ecgaug import final_eval

    m = load_members()["members"]
    if tuple(m) != CONDITIONS or any(len(ids) != len(EXISTING_SEEDS) + len(NEW_SEEDS) for ids in m.values()):
        raise PreconditionError("members.json must list ten runs for each of P16's six conditions")
    if {c: ids[: len(EXISTING_SEEDS)] for c, ids in m.items()} != existing_runs():
        raise PreconditionError("members.json's seeds 0-2 are not P13's and P15's selected runs")
    detector, (af_t, form) = af.load_detector(), af.chosen_design()
    t_rule = float(cfg["af"]["rule_threshold"])
    p13_s, p15_s, p16_s = _summary(rr.EXPERIMENT), _summary(dif.EXPERIMENT), _summary(af.EXPERIMENT)
    half = cfg["rr"]["rg"]["mp"]["pt"]["half_window_s"]
    spec = load_data(check_preconditions(), preview=False).spec
    v1_test = pt.test_arrays(spec, experiment=EXPERIMENT)
    ratios = {}
    for record in DS2:
        ratios.update({(record, r): v for r, v in v5.record_ratios(record, half, access="posthoc").items()})
    v3 = pt.relative(v5.substitute(v1_test, ratios), pt.mitbih_templates(v1_test, spec, half, access="posthoc"))
    test = rr.normalize(v3, rr.mean_rr_by_record(np.asarray(v3["record"]), access="posthoc"))
    y, records = test["y"], np.asarray(test["record"])
    p_af = af.ds2_af_probability(detector, records, np.asarray(test["r_idx"]))
    single: dict[str, dict] = {k: {} for k in KINDS}  # [10 runs, records, C, C]
    ten: dict[str, dict] = {k: {} for k in KINDS}  # [1, records, C, C]
    three: dict[str, dict] = {k: {} for k in KINDS}
    unique = None
    for c, ids in m.items():
        probs = pt._predict(ids, test)
        per_run = [decisions(p, test["rr_prev"], p_af, t_rule, af_t, form) for p in probs]
        d10 = decisions(ensemble(probs), test["rr_prev"], p_af, t_rule, af_t, form)
        d3 = decisions(ensemble(probs[: len(EXISTING_SEEDS)]), test["rr_prev"], p_af, t_rule, af_t, form)
        for k in KINDS:
            unique, single[k][c] = final_eval.per_record_cms([d[k] for d in per_run], y, records)
            _, ten[k][c] = final_eval.per_record_cms([d10[k]], y, records)
            _, three[k][c] = final_eval.per_record_cms([d3[k]], y, records)
    first = len(EXISTING_SEEDS)
    earlier = {c: (p15_s if c == dif.FAMILY else p13_s)["conditions"][f"cnn/{c}"] for c in CONDITIONS}
    checks = {f"plain {c}": _rare(single["cnn"][c][:first]) - earlier[c]["rare_macro_f1"]["point"] for c in CONDITIONS}
    checks |= {f"rule {c}": _rare(single["rule"][c][:first]) - p16_s["rule_alone"][f"cnn/{c}"]["rare_macro_f1"]["point"] for c in CONDITIONS}
    checks |= {f"veto {c}": _rare(single["veto"][c][:first]) - p16_s["with_af_veto"][f"cnn/{c}"]["rare_macro_f1"]["point"] for c in CONDITIONS}
    off = {k: d for k, d in checks.items() if abs(d) > 1e-9}
    if off:
        raise PreconditionError(f"Earlier scores do not reproduce: {off}")
    keep = unique != 202

    def block(table: dict, records_kept: np.ndarray | None = None) -> dict:
        if records_kept is None:
            return {f"cnn/{c}": ref.condition_summary(cm, unique) for c, cm in table.items()}
        return {f"cnn/{c}": ref.condition_summary(cm[:, records_kept], unique[records_kept]) for c, cm in table.items()}

    p16_veto = {c: cm[:first] for c, cm in single["veto"].items()}
    summary = {"experiment": EXPERIMENT, "post_hoc": True, "rule": RULE, "records": [int(r) for r in unique], "n_boot": ref.N_BOOT,
               "boot_seed": ref.BOOT_SEED, "rule_threshold": t_rule, "af_threshold": af_t, "veto_form": form,
               "seeds": [*EXISTING_SEEDS, *NEW_SEEDS], "strengths": strengths(),
               "ensemble": {"with_af_veto": block(ten["veto"]), "rule_alone": block(ten["rule"]), "cnn_alone": block(ten["cnn"])},
               "p16": {"with_af_veto": block(p16_veto)},
               "single_runs": {"with_af_veto": block(single["veto"])},
               "per_run_points": {f"cnn/{c}": [float(v) for v in M.rare_macro_f1_from_cm(cm.sum(axis=1))] for c, cm in single["veto"].items()},
               "three_run_ensemble": {"with_af_veto": block(three["veto"])},
               "without_202": {"ensemble": block(ten["veto"], keep), "p16": block(p16_veto, keep)},
               "families": {"confirmatory": mp._family({f"cnn: {c}, ten-run ensemble vs single runs (P16)": (ten["veto"][c], p16_veto[c]) for c in CONDITIONS}),
                            "secondary": mp._family({f"cnn: {REFERENCE} vs {c}, ten-run ensembles": (ten["veto"][REFERENCE], ten["veto"][c])
                                                     for c in CONDITIONS if c != REFERENCE})},
               "note": "every row has Huang et al.'s SVEB stage and P16's AF veto on top unless its name says otherwise",
               "reproduction_difference": checks, "run_ids": m, "git_sha": runs.git_sha()}  # fmt: skip
    (paths.results_for(EXPERIMENT) / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--part", choices=["grid", "check", "test"], required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config))
    try:
        if args.part == "grid":
            print(json.dumps(run_grid(cfg, resolve_device(args.device))))
        elif args.part == "check":
            c = validation_check(cfg)
            for kind, means in c["mean_s_f1"].items():
                print(f"  validation S F1, {kind:4s}: single runs {means['single']:.3f}, three-run ensemble {means['three']:.3f}, ten-run ensemble {means['ten']:.3f}")
        else:
            s = test_report(cfg)
            for family, items in s["families"].items():
                for name, c in items.items():
                    print(f"  {family:12s} {name:58s} {c['point']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] Holm p {c['p_holm']:.3f}")
    except (PreconditionError, LockedTestDataError) as err:
        print(f"Refusing to run: {err}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
