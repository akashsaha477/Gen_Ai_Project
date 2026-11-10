"""phase6_physics_fitted: P21 with the simulator fitted to each generated beat (PREREGISTRATION §10 v21, P23; post hoc).

    python -m ecgaug.experiments.phase6_physics_fitted --config configs/phase6_physics_fitted/physics_fitted.yaml --part check     # Kaggle CPU
    python -m ecgaug.experiments.phase6_physics_fitted --config ... --part train --family cvae|wgan_gp [--device cuda]
    python -m ecgaug.experiments.phase6_physics_fitted --config ... --part score --beats-dir DIR [--device cuda]
    ALLOW_TEST_POSTHOC=1 python -m ecgaug.experiments.phase6_physics_fitted --config ... --part test      # one posthoc access

P21's Euler loss drew the simulator's parameters at random from the beat's class, so its value was set by the draw and it
barely moved the generators. P23 keeps everything else of P21 and changes the loss: each generated beat (plus a real
training patient's template, as in P21) is compared with the simulator's beat that fits it best for the class's mean wave
positions and widths (heights and baseline solved exactly), and the loss is what the simulator cannot explain, scaled so
real beats score 1 (``ecgaug.generators.physics.FittedPrior``). ``--part check`` computes that scale and the stop rule;
``--part train`` chooses the weight on validation and trains the rest; ``--part score`` and ``--part test`` compare the
two new ensembles with P18's (``ensemble_compare``).
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
from ecgaug.data.records import CLASS_INDEX
from ecgaug.experiments import ensemble_compare as ec
from ecgaug.experiments import generator_runs as gr
from ecgaug.experiments import phase6_augmentation_mechanism as mech
from ecgaug.experiments import phase6_physics_prior as pp
from ecgaug.experiments import phase6_record_rr as rr
from ecgaug.experiments import phase6_relative_generators as rg
from ecgaug.experiments import phase6_rule_tuning as rt
from ecgaug.experiments import phase6_seed_ensemble as se
from ecgaug.experiments.common import PreconditionError
from ecgaug.generators import bank as banks
from ecgaug.generators import build_generator
from ecgaug.generators import physics as ph
from ecgaug.generators.base import training_beats
from ecgaug.names import PHASE6_PHYSICS_FITTED
from ecgaug.train.loop import resolve_device

EXPERIMENT = PHASE6_PHYSICS_FITTED
RULE = "PREREGISTRATION §10 v21"
CONFIG_KEYS = {"experiment", "physics_prior_config", "lambdas"}
LAMBDAS = pp.LAMBDAS  # (0.1, 1, 10), as P21
R2_STOP = 0.5
PHYSICS_CLASSES = pp.PHYSICS_CLASSES
FAMILIES = pp.FAMILIES
NEW = {"cvae_fitted": "cvae", "wgan_gp_fitted": "wgan_gp"}
PAIRS = {f"{ec.BASE}{c}": f"{ec.BASE}{b}" for c, b in NEW.items()}
SELECTION_SEEDS, SELECTION_BANK, GENERATOR_SEEDS, SEEDS = pp.SELECTION_SEEDS, pp.SELECTION_BANK, pp.GENERATOR_SEEDS, pp.SEEDS
DESCRIBE_BEATS = pp.DESCRIBE_BEATS
P21_CHOSEN = {"cvae": 1.0, "wgan_gp": 1.0}  # P21's chosen weights (results/phase6_physics_prior/<family>.json), checked on use


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if set(cfg) != CONFIG_KEYS or cfg["experiment"] != EXPERIMENT:
        raise ValueError(f"{path}: expected experiment {EXPERIMENT} and exactly the keys {sorted(CONFIG_KEYS)}")
    if tuple(float(v) for v in cfg["lambdas"]) != LAMBDAS:
        raise ValueError(f"{path}: lambdas must be those of PREREGISTRATION v21")
    p21 = pp.load_config(paths.repo_root() / cfg["physics_prior_config"])
    return {**cfg, "file": str(path), "p21": p21, "se": p21["se"], "rr": p21["rr"], "rg": p21["rg"]}


def _out() -> Path:
    folder = paths.results_for(EXPERIMENT)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _write(name: str, payload: dict) -> Path:
    path = _out() / name
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    return path


def _read(name: str, hint: str) -> dict:
    path = paths.results_for(EXPERIMENT) / name
    if not path.is_file():
        raise PreconditionError(f"{name} is missing: {hint}")
    return json.loads(path.read_text())


def p21_physics() -> dict:
    """P21's fits (``physics.json``): the class means of the fitted wave positions and widths come from it."""
    path = paths.results_for(pp.EXPERIMENT) / "physics.json"
    if not path.is_file():
        raise PreconditionError("P21's physics.json is missing")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------------------------------
# --part check: kappa' and the stop rule on P21's fitted real beats; descriptive scores of existing banks
# --------------------------------------------------------------------------------------------------


def fit_samples(train: dict, whole: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """P21's fitted beats: per class S, V, F (in that order, one generator), up to 5,000 training beats drawn with seed 0."""
    rng = np.random.default_rng(pp.FIT["seed"])
    y = np.asarray(train["y"])
    out = {}
    for cls in PHYSICS_CLASSES:
        pool = np.flatnonzero(y == CLASS_INDEX[cls])
        pick = np.sort(rng.choice(pool, size=min(pp.FIT["per_class"], pool.size), replace=False))
        out[cls] = (np.asarray(whole)[pick], ph.omega_of(np.asarray(train["rr_prev"])[pick], np.asarray(train["rr_ratio"])[pick]))
    return out


def score_class(whole: np.ndarray, omega: np.ndarray, distribution: dict, chunk: int = 4096) -> dict[str, np.ndarray]:
    """Per beat, what the simulator with the class's mean positions and widths cannot explain (distance) and R^2."""
    positions, widths = ph.class_shape(distribution)
    d, r2 = [], []
    for start in range(0, len(whole), chunk):
        with torch.no_grad():
            r = ph.simulation_fit(torch.as_tensor(np.asarray(whole[start : start + chunk], dtype=np.float64)),
                                  torch.as_tensor(np.asarray(omega[start : start + chunk], dtype=np.float64)), positions, widths)  # fmt: skip
        d.append(r["distance"].numpy())
        r2.append(r["r2"].numpy())
    return {"distance": np.concatenate(d), "r2": np.concatenate(r2)}


def make_prior(train: dict, whole: np.ndarray, record: dict, scale: float) -> ph.FittedPrior:
    p21 = p21_physics()
    shapes = {CLASS_INDEX[c]: {"mean": p21["classes"][c]["mean"], "kappa": record["classes"][c]["kappa"]} for c in PHYSICS_CLASSES}
    return ph.FittedPrior.build(pp.class_pools(train, whole), shapes, scale)


def roughness(beats: np.ndarray) -> float:
    """Mean squared first difference."""
    x = np.asarray(beats, dtype=np.float64)
    return float(np.mean(np.diff(x, axis=1) ** 2))


def describe(named: dict[str, dict], train: dict, prior: ph.FittedPrior, scale: float) -> dict:
    """Per bank and class S, V, F: the mean physics loss (as in training; real relative beats scored the same way first,
    then the templates alone), roughness over the real beats', the class median's correlation with the real one, and for
    S and F the effective number of training patients of the nearest real beat (P12's check)."""
    prior.to("cpu")
    y, x = np.asarray(train["y"]), np.asarray(train["X"], dtype=np.float32)
    units = rg.patient_units(np.asarray(train["record"]))
    real = {c: x[y == CLASS_INDEX[c]] for c in PHYSICS_CLASSES}
    zeros = {"X": np.zeros((DESCRIBE_BEATS * len(PHYSICS_CLASSES), x.shape[1]), dtype=np.float32),
             "y": np.repeat([CLASS_INDEX[c] for c in PHYSICS_CLASSES], DESCRIBE_BEATS)}  # fmt: skip
    out: dict[str, dict] = {}
    for name, bank in {"real": {"X": x, "y": y}, "templates_alone": zeros, **named}.items():
        bx, by = np.asarray(bank["X"], dtype=np.float32), np.asarray(bank["y"])
        entry = {}
        for c in PHYSICS_CLASSES:
            idx = np.flatnonzero(by == CLASS_INDEX[c])[:DESCRIBE_BEATS]
            torch.manual_seed(0)
            with torch.no_grad():
                term = prior(torch.as_tensor(bx[idx] / scale), torch.full((idx.size,), CLASS_INDEX[c], dtype=torch.long))
            e = {"physics": float(term) if term is not None else float("nan")}
            if name != "templates_alone":
                e["roughness"] = roughness(bx[idx]) / roughness(real[c])
                e["median_corr"] = float(np.corrcoef(np.median(bx[idx], axis=0), np.median(real[c], axis=0))[0, 1])
            if c in ("S", "F") and name not in ("real", "templates_alone"):
                m = y == CLASS_INDEX[c]
                e["nearest_patients"] = mech._entry(units[m][mech.nearest(x[m], bx[idx])])["effective_records"]
            entry[c] = e
        out[name] = entry
    return out


def p21_bank(family: str) -> tuple[dict, str]:
    """P21's seed-0 bank at its chosen weight (checked against P21's record)."""
    chosen = float(json.loads((paths.results_for(pp.EXPERIMENT) / f"{family}.json").read_text())["chosen_lambda"])
    if chosen != P21_CHOSEN[family]:
        raise PreconditionError(f"P21's chosen weight for {family} is {chosen}, not {P21_CHOSEN[family]}")
    return pp.load_bank(family, chosen, 0)


def check(cfg: dict) -> dict:
    """``physics.json``: kappa'_c and the R^2 of P21's fitted real beats with the class's positions and fitted heights,
    the stop rule, and the descriptive scores of P12's and P21's seed-0 banks and of the templates alone."""
    _, p11 = pp.build_data(cfg["p21"])
    train = training_beats(p11.arrays)
    whole = pp.whole_training_beats(train)
    share = pp.zscore_share(whole)
    if share < 0.999:
        raise PreconditionError(f"Only {share:.4f} of the whole training beats are z-scored windows: the match is wrong")
    p21 = p21_physics()
    classes = {}
    for cls, (beats, omega) in fit_samples(train, whole).items():
        s = score_class(beats, omega, p21["classes"][cls])
        if len(beats) != p21["classes"][cls]["beats"]:
            raise PreconditionError(f"{cls}: {len(beats)} beats, P21 fitted {p21['classes'][cls]['beats']}")
        classes[cls] = {"beats": int(len(beats)), "kappa": float(s["distance"].mean()), "median_r2": float(np.nanmedian(s["r2"])),
                        "r2_quantiles": pp._quantiles(s["r2"]), "p21_median_r2": p21["classes"][cls]["median_r2"]}  # fmt: skip
        print(f"{cls}: median R^2 {classes[cls]['median_r2']:.3f} (P21's own-position fit {classes[cls]['p21_median_r2']:.3f})", flush=True)
    record = {"classes": classes}
    scale = rg.scale_of(train["X"])
    prior = make_prior(train, whole, record, scale)
    named = {f"p12_{f}_seed0": rg.load_relative_bank(cfg["rg"], f, 0)[0] for f in FAMILIES}
    named.update({f"p21_{f}_seed0": p21_bank(f)[0] for f in FAMILIES})
    out = {"experiment": EXPERIMENT, "post_hoc": True, "rule": RULE, "r2_stop": R2_STOP, "zscore_share": share, "p11_sha256": p11.info["content_sha256"],
           "classes": classes, "physics_passed": all(c["median_r2"] >= R2_STOP for c in classes.values()),
           "describe": describe(named, train, prior, scale), "git_sha": runs.git_sha()}  # fmt: skip
    _write("physics.json", out)
    return out


def load_physics() -> dict:
    record = _read("physics.json", "run --part check on Kaggle and ingest it first")
    if not record["physics_passed"]:
        raise PreconditionError("The fitted simulator failed its stop rule (physics.json): both conditions stop")
    return record


# --------------------------------------------------------------------------------------------------
# --part train: the generators with the fitted physics loss, the lambda choice, the ten cnn runs (as P21)
# --------------------------------------------------------------------------------------------------


def generator_id(family: str, value: float) -> str:
    return f"{EXPERIMENT}/{family}-lambda{pp.lambda_tag(value)}"


def train_generator(cfg: dict, family: str, value: float, seed: int, train: dict, val_records: list[int], prior: ph.FittedPrior,
                    device: torch.device) -> dict:  # fmt: skip
    """One generator at lambda ``value`` and ``seed`` with P12's settings and steps, and its bank (P12's counts and RR
    draws). Already-trained ones (bank on disk with the recorded hash) are kept."""
    name = pp.generators_name(family)
    path = paths.results_for(EXPERIMENT) / name
    record = json.loads(path.read_text()) if path.is_file() else {"experiment": EXPERIMENT, "rule": RULE, "family": family, "fits": {}}
    key = f"{pp.lambda_tag(value)}/{seed}"
    gid = generator_id(family, value)
    entry = record["fits"].get(key)
    if entry and "bank_sha256" in entry and (banks.bank_dir(gid) / f"seed{seed}.npz").is_file():
        return entry
    g = cfg["rg"]["gen"][family]
    p12 = rg.generators_record(family)
    scale = rg.scale_of(train["X"])
    if abs(scale - float(p12["scale"])) > 1e-9 * max(1.0, scale):
        raise PreconditionError(f"The training scale {scale} is not P12's {p12['scale']}")
    part = {**train, "X": (np.asarray(train["X"], dtype=np.float64) / scale).astype(np.float32)}
    settings = gr.generator_settings(g, g["generator"])
    steps = int(g["training"]["final_steps"])
    run_id = runs.make_run_id(EXPERIMENT, "generator", int(seed), tag=f"{family}.lambda{pp.lambda_tag(value)}")
    run_dir = paths.runs_dir() / EXPERIMENT / "generators" / family / run_id
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"experiment": EXPERIMENT, "family": family, "settings": settings, "steps": steps, "seed": int(seed),
                                                         "scale": scale, "lambda": float(value), "prior": "fitted physics loss (physics.json)"}))  # fmt: skip
    generator = build_generator(g["family"], settings, int(part["X"].shape[1]), int(seed))
    generator.prior, generator.prior_weight = prior, float(value)
    fit = generator.fit(part, steps, int(seed), device, val_records=val_records, fractions=(1.0,), log_every=int(g["training"]["log_every"]))
    rg._write_log(run_dir / "train_log.csv", fit.history or [{"step": 0}])
    (run_dir / "env.json").write_text(json.dumps({**runs.capture_env(str(device)), "wall_clock_s": fit.wall_clock_s}, indent=2) + "\n")
    entry = {"run_id": run_id, "lambda": float(value), "seed": int(seed), "steps_done": fit.steps_done, "diverged": fit.diverged,
             "divergence": fit.divergence, "generator_seconds": fit.generator_seconds, "final_losses": fit.history[-1] if fit.history else {}}  # fmt: skip
    if not fit.diverged:
        checkpoint = run_dir / "checkpoints" / "final.pt"
        torch.save(generator.state_dict(), checkpoint)
        bank = rg.sample_relative_bank(generator, train, p12["bank_counts"], int(seed), scale, source=gid)
        exported = banks.export_bank(bank, gid, int(seed), {"run_id": run_id, "scale": scale, "lambda": float(value), "step": fit.steps_done,
                                                            "checkpoint_sha256": banks.sha256_file(checkpoint)})  # fmt: skip
        entry.update(checkpoint_sha256=exported["checkpoint_sha256"], bank_sha256=exported["sha256"])
    (run_dir / "metrics_val.json").write_text(json.dumps(entry, indent=2, default=float) + "\n")
    record["fits"][key] = entry
    _write(name, record)
    return entry


def load_bank(family: str, value: float, seed: int) -> tuple[dict, str]:
    entry = _read(pp.generators_name(family), f"run --part train --family {family}")["fits"].get(f"{pp.lambda_tag(value)}/{seed}")
    if entry is None or "bank_sha256" not in entry:
        raise PreconditionError(f"No {family} bank at lambda {value} for seed {seed} (did its generator diverge?)")
    return banks.load_bank(generator_id(family, value), seed, expected_sha256=entry["bank_sha256"]), entry["bank_sha256"]


def cnn_grid(cfg: dict, name: str, conditions: dict[str, tuple[str, str]], seeds: tuple[int, ...]) -> dict:
    """P13's grid (the P7 cnn and its settings) with ``conditions`` = {name: (bank id, base family)} at the base family's
    carried strength, for ``seeds``."""
    base = rr.record_grid(cfg["rr"])
    strength = rr.strengths()
    conds = {c: {"bank": bank_id, "strengths": [strength[family]]} for c, (bank_id, family) in conditions.items()}
    return {**base, "experiment": EXPERIMENT, "name": name, "seeds": list(seeds), "conditions": conds, "file": cfg["file"]}


def train_family(cfg: dict, family: str, device: torch.device) -> dict:
    """The lambda choice (seed 0 at each lambda, three cnn runs each, lambda 0 beside), seeds 1-2 at the chosen lambda,
    the ten cnn runs of the new condition and the descriptive bank checks; writes ``<family>.json``."""
    record = load_physics()
    p13, p11 = pp.build_data(cfg["p21"])
    train = training_beats(p11.arrays)
    val_records = sorted(set(np.asarray(p11.arrays["record"])[np.asarray(p11.arrays["split"]) == "val"].astype(int).tolist()))
    whole = pp.whole_training_beats(train)
    scale = rg.scale_of(train["X"])
    prior = make_prior(train, whole, record, scale)
    train_before, train_after = train, training_beats(p13.arrays)
    counts = rg.generators_record(family)["bank_counts"]
    for value in LAMBDAS:
        train_generator(cfg, family, value, SELECTION_BANK, train, val_records, prior, device)
    cache: dict[tuple[str, float, int], tuple[dict, str]] = {}

    def p13_bank(value: float, b: int) -> tuple[dict, str]:
        key = (family, value, b)
        if key not in cache:
            for k in [k for k in cache if k[1] != value]:
                del cache[k]  # one lambda's banks in memory at a time
            relative, sha = rg.load_relative_bank(cfg["rg"], family, b) if value == 0 else load_bank(family, value, b)
            cache[key] = (rr.v4_bank(relative, train_before, train_after, counts, b), sha)
        return cache[key]

    selection = {pp.selection_condition(family, v): (generator_id(family, v) if v else rg.generator_id(cfg["rg"], family), family) for v in (0.0, *LAMBDAS)}
    sel_grid = cnn_grid(cfg, f"selection_{family}", selection, SELECTION_SEEDS)
    by_name = {pp.selection_condition(family, v): v for v in (0.0, *LAMBDAS)}
    report = {"selection": pp.run_cells(sel_grid, p13, lambda c, s: p13_bank(by_name[c], SELECTION_BANK), device)}
    choice = pp.choose_lambda(pp.done_rows(sel_grid), family)
    chosen = float(choice["chosen"])
    for seed in GENERATOR_SEEDS:
        if seed != SELECTION_BANK:
            train_generator(cfg, family, chosen, seed, train, val_records, prior, device)
    condition = f"{family}_fitted"
    final_grid = cnn_grid(cfg, f"final_{family}", {condition: (generator_id(family, chosen), family)}, SEEDS)
    report["final"] = pp.run_cells(final_grid, p13, lambda c, s: p13_bank(chosen, se.bank_seed(s)), device)
    members = pp.members_of(pp.done_rows(final_grid), condition)
    named = {"p12_seed0": rg.load_relative_bank(cfg["rg"], family, 0)[0], "p21_seed0": p21_bank(family)[0],
             **{f"lambda{pp.lambda_tag(v)}_seed0": load_bank(family, v, 0)[0] for v in LAMBDAS}}  # fmt: skip
    fits = _read(pp.generators_name(family), "")["fits"]
    out = {"experiment": EXPERIMENT, "post_hoc": True, "rule": RULE, "family": family, "condition": condition, "scale": scale,
           "strength": rr.strengths()[family], "lambda_choice": choice, "chosen_lambda": chosen,
           "banks": {str(s): fits[f"{pp.lambda_tag(chosen)}/{s}"]["bank_sha256"] for s in GENERATOR_SEEDS},
           "members": members, "describe": describe(named, train, prior, scale), "report": report, "git_sha": runs.git_sha()}  # fmt: skip
    _write(f"{family}.json", out)
    return out


# --------------------------------------------------------------------------------------------------
# --part score and --part test: the two ensembles against P18's
# --------------------------------------------------------------------------------------------------


def new_members() -> dict[str, list[str]]:
    load_physics()
    out = {}
    for c, family in NEW.items():
        ids = list(_read(f"{family}.json", f"run --part train --family {family} and ingest it first")["members"])
        if len(ids) != len(SEEDS):
            raise PreconditionError(f"{c}: expected {len(SEEDS)} runs")
        out[f"{ec.BASE}{c}"] = ids
    return out


def run_score(beats_dir: Path, device: torch.device) -> tuple[dict, dict]:
    choice, summary = ec.run_patients(EXPERIMENT, RULE, new_members(), PAIRS, beats_dir, device)
    _write("choice.json", choice)
    _write("summary.json", summary)
    return choice, summary


def test_report(cfg: dict) -> dict:
    choice = _read("choice.json", "run --part score and ingest it first")
    out = ec.ds2_report(EXPERIMENT, RULE, cfg["se"], choice, new_members(), PAIRS)
    _write("test_summary.json", out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--part", choices=("check", "train", "score", "test"), required=True)
    parser.add_argument("--family", choices=FAMILIES, default=None)
    parser.add_argument("--beats-dir", type=Path, default=rt.default_beats_dir(), help="P20's built beats (score)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    try:
        if args.part == "check":
            out = check(cfg)
            print(json.dumps({"physics_passed": out["physics_passed"], "classes": out["classes"]}, indent=2, default=float))
        elif args.part == "train":
            if args.family is None:
                raise PreconditionError("--part train needs --family")
            out = train_family(cfg, args.family, resolve_device(args.device))
            print(json.dumps({"chosen_lambda": out["chosen_lambda"], "lambda_choice": out["lambda_choice"]["table"], "report": out["report"]}, indent=2, default=float))
        elif args.part == "score":
            choice, summary = run_score(args.beats_dir, resolve_device(args.device))
            print(json.dumps({"chosen": choice["chosen"], "secondary": {k: {"point": v["point"], "p_holm": v["p_holm"]} for k, v in summary["families"]["secondary"].items()}}, indent=2, default=float))
        else:
            out = test_report(cfg)
            print(json.dumps({k: {"point": v["point"], "p_holm": v["p_holm"]} for k, v in out["families"]["confirmatory"].items()}, indent=2, default=float))
    except (PreconditionError, LockedTestDataError) as exc:
        print(f"refused: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
