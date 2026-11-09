"""phase6_diffusion_long: P15's diffusion model trained five times longer (PREREGISTRATION §10 v16, P17; post hoc).

    python -m ecgaug.experiments.phase6_diffusion_long --config configs/phase6_diffusion_long/diffusion_long.yaml --part train [--device cuda]
    python -m ecgaug.experiments.phase6_diffusion_long --config ... --part shape    # local: bank class medians vs real training beats
    python -m ecgaug.experiments.phase6_diffusion_long --config ... --part select
    ALLOW_TEST_POSTHOC=1 python -m ecgaug.experiments.phase6_diffusion_long --config ... --part test        # one posthoc access

At the phase 3 budget (20,000 steps) P15's training loss was still falling, and the class medians of its samples matched
the real training beats less closely than the cVAE's; sampling with the raw weights instead of their moving average, or
without guidance, changed little (training-side checks only). P17 trains the same model on the same beats, with the same
batch, seeds, sampler and banks, for 100,000 steps; the strength is chosen on validation by the phase 4 gate rule; one DS2
read. Huang et al.'s SVEB stage and P16's AF veto on top are descriptive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from ecgaug import paths, runs
from ecgaug.data.access import LockedTestDataError
from ecgaug.data.records import DS2
from ecgaug.eval import metrics as M
from ecgaug.experiments import phase6_af_rule as af
from ecgaug.experiments import phase6_diffusion as dif
from ecgaug.experiments import phase6_more_patients as mp
from ecgaug.experiments import phase6_patient_template as pt
from ecgaug.experiments import phase6_record_rr as rr
from ecgaug.experiments import phase6_rr_baseline as v5
from ecgaug.experiments import phase6_rr_reference as ref
from ecgaug.experiments.common import PreconditionError, check_preconditions
from ecgaug.experiments.data_source import load_data
from ecgaug.names import PHASE6_DIFFUSION_LONG
from ecgaug.train.loop import resolve_device

EXPERIMENT = PHASE6_DIFFUSION_LONG
RULE = "PREREGISTRATION §10 v16"
CONFIG_KEYS = dif.CONFIG_KEYS | {"diffusion_config", "af_rule_config"}
SAME_AS_P15 = ("record_rr_config", "classes", "generator", "seeds", "strengths", "mechanism_beats", "rule_threshold")
BEFORE = ("none", "class_weighted", "random_oversampling", "cvae")  # P13's conditions
LONG = "diffusion_long"


def load_config(path: Path) -> dict:
    """P17's config; every setting but the step budget must equal P15's, and the budget must be larger."""
    cfg = yaml.safe_load(Path(path).read_text())
    if set(cfg) != CONFIG_KEYS or cfg["experiment"] != EXPERIMENT:
        raise ValueError(f"{path}: expected experiment {EXPERIMENT} and exactly the keys {sorted(CONFIG_KEYS)}")
    p15 = dif.load_config(paths.repo_root() / cfg["diffusion_config"])
    differ = [k for k in SAME_AS_P15 if cfg[k] != p15[k]]
    differ += [f"training.{k}" for k in sorted(set(cfg["training"]) | set(p15["training"])) if k != "final_steps" and cfg["training"].get(k) != p15["training"].get(k)]
    if differ:
        raise ValueError(f"{path}: only the step budget may differ from P15, but {differ} do")
    if int(cfg["training"]["final_steps"]) <= int(p15["training"]["final_steps"]):
        raise ValueError(f"{path}: P17 must train longer than P15's {p15['training']['final_steps']} steps")
    return {**cfg, "file": str(path), "rr": rr.load_config(paths.repo_root() / cfg["record_rr_config"]), "p15": p15,
            "af": af.load_config(paths.repo_root() / cfg["af_rule_config"])}  # fmt: skip


def _summary(experiment: str) -> dict:
    path = paths.results_for(experiment) / "summary.json"
    if not path.is_file():
        raise PreconditionError(f"{experiment} has no test summary yet; P17 checks its scores against it")
    return json.loads(path.read_text())


def _rare(cms: np.ndarray) -> float:
    return float(np.mean(M.rare_macro_f1_from_cm(np.asarray(cms).sum(axis=1))))


def real_training_beats(per_class: int = 2000, seed: int = 0) -> dict[int, np.ndarray]:
    """Real DS1 training beats of S, V and F in the form the cnn sees them (beat minus the patient's +/-150 s template), up
    to ``per_class`` per class drawn with ``seed``; training records only (the local v1 data and raw MIT-BIH files)."""
    from ecgaug.data.loaders import load_trainval
    from ecgaug.experiments.phase6_external_validation import frozen_spec

    if not (paths.processed_dir() / "v1" / "trainval.npz").is_file():
        raise PreconditionError("The shape check needs the v1 arrays (data/processed/v1) and the raw MIT-BIH files")
    v1 = load_trainval("v1")
    y, record, split, r_idx = (np.asarray(v1[k]) for k in ("y", "record", "split", "r_idx"))
    rng = np.random.default_rng(seed)
    chosen = {}
    for c in (1, 2, 3):
        idx = np.flatnonzero((y == c) & (split == "train"))
        chosen[c] = np.sort(rng.choice(idx, min(per_class, idx.size), replace=False))
    wanted = np.sort(np.concatenate(list(chosen.values())))
    rel = np.empty((wanted.size, v1["X"].shape[1]), dtype=np.float32)
    spec = frozen_spec()
    for r in np.unique(record[wanted]).tolist():
        at = wanted[record[wanted] == r]
        order = at[np.argsort(r_idx[at])]
        rel[np.searchsorted(wanted, order)] = v1["X"][order] - pt.record_templates(int(r), r_idx[order], spec, 150.0)
    return {c: rel[np.searchsorted(wanted, idx)] for c, idx in chosen.items()}


SHAPE_BANKS = {"cvae_p12": "phase6_relative_generators/phase3_cvae-D", "diffusion_p15": f"{dif.EXPERIMENT}/diffusion", "diffusion_p17": f"{EXPERIMENT}/diffusion"}


def shape_check(per_class: int = 2000) -> dict:
    """Descriptive, training data only: per rare class, the correlation and RMSE between the median beat of each bank seed
    and the median of real DS1 training beats in the cnn's form (P12's cVAE, P15's and P17's diffusion banks, those on disk)."""
    from ecgaug.data.records import CLASSES
    from ecgaug.generators import bank as banks

    real = real_training_beats(per_class)
    med = {c: np.median(beats, axis=0) for c, beats in real.items()}
    out = {"experiment": EXPERIMENT, "rule": f"{RULE} (descriptive)", "real_beats": {CLASSES[c]: int(len(b)) for c, b in real.items()}, "banks": {}}
    for name, gid in SHAPE_BANKS.items():
        for seed in (0, 1, 2):
            path = banks.bank_dir(gid) / f"seed{seed}.npz"
            if not path.is_file():
                continue
            with np.load(path) as z:
                X, y = np.asarray(z["X"], dtype=np.float64), np.asarray(z["y"])
            entry = {"bank_sha256": banks.sha256_file(path)}
            for c, m in med.items():
                b = np.median(X[y == c], axis=0)
                entry[CLASSES[c]] = {"corr": float(np.corrcoef(b, m)[0, 1]), "rmse": float(np.sqrt(np.mean((b - m) ** 2)))}
            out["banks"].setdefault(name, {})[str(seed)] = entry
    folder = paths.results_for(EXPERIMENT)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "shape_check.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def test_report(cfg: dict) -> dict:
    """DS2 in P13's form, one posthoc access. Confirmatory (Holm): the longer-trained model vs P15's and vs P13's cnn without
    augmentation. Secondary (Holm): vs P13's cVAE, class weighting and random oversampling. Descriptive: Huang et al.'s SVEB
    stage, and the stage with P16's AF veto, on top of every compared condition. P13's, P15's and P16's scores are
    recomputed and must match their summaries."""
    from ecgaug import final_eval

    new_sel = dif.load_selection(EXPERIMENT)["chosen"]["cnn"]["diffusion"]
    p15_sel = dif.load_selection()["chosen"]["cnn"]["diffusion"]
    old_sel = rr.load_selection()["chosen"]["cnn"]
    detector, (af_t, form) = af.load_detector(), af.chosen_design()
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
    run_ids = {LONG: new_sel["run_ids"], "diffusion": p15_sel["run_ids"], **{c: old_sel[c]["run_ids"] for c in BEFORE}}
    plain, rule, veto, unique = {}, {}, {}, None
    for c, ids in run_ids.items():
        probs = pt._predict(ids, test)
        unique, plain[c] = final_eval.per_record_cms(probs, y, records)
        _, rule[c] = final_eval.per_record_cms([rr.sveb_rule(p, test["rr_prev"], cfg["rule_threshold"]) for p in probs], y, records)
        _, veto[c] = final_eval.per_record_cms([af.veto_rule(p, test["rr_prev"], p_af, cfg["rule_threshold"], af_t, form) for p in probs], y, records)
    earlier = {**{c: p13_s["conditions"][f"cnn/{c}"] for c in BEFORE}, "diffusion": p15_s["conditions"]["cnn/diffusion"]}
    checks = {f"plain {c}": _rare(plain[c]) - e["rare_macro_f1"]["point"] for c, e in earlier.items()}
    checks |= {f"rule {c}": _rare(rule[c]) - p16_s["rule_alone"][f"cnn/{c}"]["rare_macro_f1"]["point"] for c in earlier}
    checks |= {f"veto {c}": _rare(veto[c]) - p16_s["with_af_veto"][f"cnn/{c}"]["rare_macro_f1"]["point"] for c in earlier}
    off = {k: d for k, d in checks.items() if abs(d) > 1e-9}
    if off:
        raise PreconditionError(f"Earlier scores do not reproduce: {off}")
    summary = {"experiment": EXPERIMENT, "post_hoc": True, "rule": RULE, "records": [int(r) for r in unique], "n_boot": ref.N_BOOT,
               "boot_seed": ref.BOOT_SEED, "rule_threshold": float(cfg["rule_threshold"]), "af_threshold": af_t, "veto_form": form,
               "conditions": {f"cnn/{LONG}": ref.condition_summary(plain[LONG], unique) | {"strength": new_sel["strength"]}},
               "before": {f"cnn/{c}": ref.condition_summary(plain[c], unique) for c in ("diffusion", *BEFORE)},
               "sveb_rule": {f"cnn/{c}": ref.condition_summary(rule[c], unique) for c in run_ids},
               "with_af_veto": {f"cnn/{c}": ref.condition_summary(veto[c], unique) for c in run_ids},
               "families": {"confirmatory": mp._family({"cnn: diffusion_long vs diffusion (P15)": (plain[LONG], plain["diffusion"]),
                                                        "cnn: diffusion_long vs none (P13)": (plain[LONG], plain["none"])}),
                            "secondary": mp._family({f"cnn: diffusion_long vs {c} (P13)": (plain[LONG], plain[c]) for c in ("cvae", "class_weighted", "random_oversampling")})},
               "note": "the SVEB-rule and AF-veto rows are descriptive (no test)", "reproduction_difference": checks,
               "run_ids": list(new_sel["run_ids"]), "git_sha": runs.git_sha()}  # fmt: skip
    (paths.results_for(EXPERIMENT) / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--part", choices=["train", "shape", "select", "test"], required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config))
    try:
        if args.part == "train":
            data, _, _ = rr.build(cfg["rr"])  # P13's data, hash-checked against its data.json
            device = resolve_device(args.device)
            record = dif.train_generators(cfg, data, device, experiment=EXPERIMENT, rule=RULE)
            print(json.dumps({s: {k: e[k] for k in ("steps_done", "diverged", "generator_seconds")} for s, e in record["seeds"].items()}, default=float))
            print(json.dumps(dif.run_grid(cfg, data, device, experiment=EXPERIMENT)))
            m = dif.mechanism(cfg, data, experiment=EXPERIMENT)
            for cls, e in m["classes"].items():
                print(cls, {k: (round(v["effective_records"], 1), round(v["top_share"], 2)) for k, v in e.items()})
        elif args.part == "shape":
            s = shape_check()
            for name, seeds in s["banks"].items():
                for seed, e in seeds.items():
                    print(name, seed, {c: (round(v["corr"], 3), round(v["rmse"], 3)) for c, v in e.items() if c != "bank_sha256"})
        elif args.part == "select":
            s = dif.select(cfg, experiment=EXPERIMENT, rule=f"{RULE} (the phase 4 gate rule)")
            print({c: (v["strength"], round(v["mean"], 3)) for c, v in s["chosen"]["cnn"].items()})
        else:
            s = test_report(cfg)
            for family, items in s["families"].items():
                for name, c in items.items():
                    print(f"  {family:12s} {name:44s} {c['point']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] Holm p {c['p_holm']:.3f}")
    except (PreconditionError, LockedTestDataError) as err:
        print(f"Refusing to run: {err}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
