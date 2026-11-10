"""phase6_physics_prior: a physics prior for the generators and physiological fusion beats (PREREGISTRATION §10 v20, P21; post hoc).

    python -m ecgaug.experiments.phase6_physics_prior --config configs/phase6_physics_prior/physics_prior.yaml --part check     # Kaggle CPU
    python -m ecgaug.experiments.phase6_physics_prior --config ... --part train --family cvae|wgan_gp [--device cuda]
    python -m ecgaug.experiments.phase6_physics_prior --config ... --part fusion [--device cuda]
    python -m ecgaug.experiments.phase6_physics_prior --config ... --part score --beats-dir DIR [--device cuda]
    ALLOW_TEST_POSTHOC=1 python -m ecgaug.experiments.phase6_physics_prior --config ... --part test      # one posthoc access

SimGANs (Golany, Radinsky and Freedman, ICML 2020) add McSharry et al.'s (2003) ECG simulator to a GAN's generator
objective: the Euler loss, how far a generated beat is from satisfying the simulator's z equation for parameters drawn
from its class (``ecgaug.generators.physics``). P21 adds it to P12's cVAE and WGAN-GP. The simulator describes whole
beats, while P12's generators learn patient-relative beats, so a generated beat gets a real training patient's template
before it is scored. ``--part check`` fits the simulator to real training beats (and the fusion model to real F beats)
and applies the pre-registered stop rules; ``--part train`` chooses the loss weight on validation and trains the rest;
``--part fusion`` replaces the F beats of P12's cVAE banks with fractions of real V deviations. Each condition runs
through P18's ten-run pipeline; ``--part score`` chooses Huang's stage and P16's veto on P14's patients by P20's rule
and scores P19's patients; ``--part test`` reads DS2 once.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import yaml

from ecgaug import paths, runs
from ecgaug.data.access import LockedTestDataError
from ecgaug.data.contract import content_sha256
from ecgaug.data.records import CLASS_INDEX, CLASSES
from ecgaug.eval import metrics as M
from ecgaug.experiments import generator_runs as gr
from ecgaug.experiments import phase6_augmentation_mechanism as mech
from ecgaug.experiments import phase6_fresh_patients as fp
from ecgaug.experiments import phase6_more_patients as mp
from ecgaug.experiments import phase6_patient_template as pt
from ecgaug.experiments import phase6_record_rr as rr
from ecgaug.experiments import phase6_relative_generators as rg
from ecgaug.experiments import phase6_rr_reference as ref
from ecgaug.experiments import phase6_rule_tuning as rt
from ecgaug.experiments import phase6_seed_ensemble as se
from ecgaug.experiments.common import PreconditionError, check_preconditions
from ecgaug.experiments.data_source import DataSource, load_data
from ecgaug.generators import bank as banks
from ecgaug.generators import build_generator
from ecgaug.generators import physics as ph
from ecgaug.generators.base import training_beats
from ecgaug.grid import runner
from ecgaug.names import PHASE6_PHYSICS_PRIOR
from ecgaug.train.loop import resolve_device

EXPERIMENT = PHASE6_PHYSICS_PRIOR
RULE = "PREREGISTRATION §10 v20"
CONFIG_KEYS = {"experiment", "seed_ensemble_config", "lambdas", "fit", "fusion"}
LAMBDAS = (0.1, 1.0, 10.0)  # fixed by PREREGISTRATION v20
FIT = {"per_class": 5000, "seed": 0, "iterations": 500, "lr": 0.01}
FUSION = {"min_v_beats": 5, "min_f_beats": 50, "min_recordings": 3, "low_quantile": 0.1, "high_quantile": 0.9}
R2_STOP = 0.5  # a class (or the fusion model) whose median R^2 is below this stops its conditions
PHYSICS_CLASSES = ("S", "V", "F")
FAMILIES = rg.FAMILIES  # ("cvae", "wgan_gp")
NEW = {"cvae_physics": "cvae", "wgan_gp_physics": "wgan_gp", "cvae_fusion": "cvae"}  # new condition -> the P18 condition it is compared with
FUSION_CONDITION = "cvae_fusion"
SELECTION_SEEDS = (0, 1, 2)  # cnn seeds of the lambda choice, all with generator (bank) seed 0
SELECTION_BANK = 0
GENERATOR_SEEDS = (0, 1, 2)
SEEDS = tuple(range(10))  # the ten-run ensemble; a run of seed s uses bank seed s mod 3 (se.bank_seed)
DESCRIBE_BEATS = 2000  # beats per class for the descriptive bank checks
DS2_TOLERANCE = 1e-9
ICENTIA_TOLERANCE = rt.REPRODUCTION_TOLERANCE


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if set(cfg) != CONFIG_KEYS or cfg["experiment"] != EXPERIMENT:
        raise ValueError(f"{path}: expected experiment {EXPERIMENT} and exactly the keys {sorted(CONFIG_KEYS)}")
    if tuple(float(v) for v in cfg["lambdas"]) != LAMBDAS or dict(cfg["fit"]) != FIT or dict(cfg["fusion"]) != FUSION:
        raise ValueError(f"{path}: lambdas, fit and fusion must be those of PREREGISTRATION v20")
    se_cfg = se.load_config(paths.repo_root() / cfg["seed_ensemble_config"])
    return {**cfg, "file": str(path), "se": se_cfg, "rr": se_cfg["rr"], "rg": se_cfg["rr"]["rg"]}


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


# --------------------------------------------------------------------------------------------------
# data: P11's training beats (the generators'), P13's (the cnn's), whole beats for the simulator
# --------------------------------------------------------------------------------------------------


def build_data(cfg: dict) -> tuple[DataSource, DataSource]:
    """P13's data and P11's (hash-checked by P13's builder)."""
    p13, p11, _ = rr.build(cfg["rr"])
    return p13, p11


def match_whole(train: dict, sources: list[dict]) -> np.ndarray:
    """The whole beat (frozen per-beat z-scored window) of every training beat, found by (record, r_idx) in ``sources``.
    MIT-BIH (100-234), SVDB (800-894) and INCART (1-75) record numbers do not overlap."""
    lookup: dict[tuple[int, int], tuple[int, int]] = {}
    for s, arrays in enumerate(sources):
        for i, key in enumerate(zip(np.asarray(arrays["record"]).astype(int).tolist(), np.asarray(arrays["r_idx"]).astype(int).tolist())):
            if key in lookup:
                raise PreconditionError(f"Beat {key} appears twice in the whole-beat sources")
            lookup[key] = (s, i)
    keys = list(zip(np.asarray(train["record"]).astype(int).tolist(), np.asarray(train["r_idx"]).astype(int).tolist()))
    missing = [k for k in keys if k not in lookup]
    if missing:
        raise PreconditionError(f"{len(missing)} training beats have no whole beat, e.g. {missing[:3]}")
    out = np.empty_like(np.asarray(train["X"], dtype=np.float32))
    for j, k in enumerate(keys):
        s, i = lookup[k]
        out[j] = sources[s]["X"][i]
    return out


def whole_training_beats(train: dict) -> np.ndarray:
    """P11's training beats as whole beats: DS1's from the frozen v1 contract, INCART's and SVDB's from their builds."""
    from ecgaug.experiments import phase6_external_validation as ext

    v1 = load_data(check_preconditions(), preview=False).arrays
    sources = [{k: np.asarray(v1[k]) for k in ("X", "record", "r_idx")}]
    for db in mp.DATABASES:
        arrays, _ = ext.load_database(db)
        sources.append({k: np.asarray(arrays[k]) for k in ("X", "record", "r_idx")})
    return match_whole(train, sources)


def zscore_share(whole: np.ndarray, tolerance: float = 1e-2) -> float:
    """Share of whole beats whose mean is 0 and SD 1 within ``tolerance`` (the frozen representation z-scores each beat)."""
    x = np.asarray(whole, dtype=np.float64)
    return float(np.mean((np.abs(x.mean(axis=1)) < tolerance) & (np.abs(x.std(axis=1) - 1) < tolerance)))


def class_pools(train: dict, whole: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per class S, V, F: every training beat's template (whole minus relative) and omega."""
    y = np.asarray(train["y"])
    pools = {}
    for cls in PHYSICS_CLASSES:
        m = y == CLASS_INDEX[cls]
        omega = ph.omega_of(np.asarray(train["rr_prev"])[m], np.asarray(train["rr_ratio"])[m])
        if not np.isfinite(omega).all() or (omega <= 0).any():
            raise PreconditionError(f"{cls}: some training beats have no finite local RR interval")
        pools[CLASS_INDEX[cls]] = ((np.asarray(whole)[m] - np.asarray(train["X"])[m]).astype(np.float32), omega.astype(np.float32))
    return pools


# --------------------------------------------------------------------------------------------------
# --part check: the simulator and the fusion model fitted to real training beats; the stop rules
# --------------------------------------------------------------------------------------------------


def _quantiles(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return {f"q{int(q * 100):02d}": float(np.quantile(v, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}


def fit_class(whole: np.ndarray, omega: np.ndarray, seed: int = FIT["seed"]) -> tuple[dict, dict]:
    """The simulator fitted to one class's beats: summary (R^2, distribution, kappa) and three example beats."""
    f = ph.fit(whole, omega, FIT["iterations"], FIT["lr"])
    dist = ph.class_distribution(f["eta"])
    kappa = ph.kappa(whole, omega, dist, seed=seed)
    own = ph.euler_distance(torch.as_tensor(np.asarray(whole, dtype=np.float64)), torch.as_tensor(f["eta"]), torch.as_tensor(omega))
    r2 = f["r2"]
    order = np.argsort(np.where(np.isfinite(r2), r2, -np.inf))
    examples = {}
    for q in (0.1, 0.5, 0.9):
        i = int(order[min(len(order) - 1, int(q * (len(order) - 1)))])
        examples[f"q{int(q * 100):02d}"] = {"real": np.asarray(whole[i], dtype=np.float64).tolist(), "fitted": f["fitted"][i].tolist(), "r2": float(r2[i])}
    hist, edges = np.histogram(np.clip(r2[np.isfinite(r2)], 0, 1), bins=20, range=(0.0, 1.0))
    summary = {"beats": int(len(r2)), "median_r2": float(np.nanmedian(r2)), "r2_quantiles": _quantiles(r2),
               "r2_histogram": {"counts": hist.tolist(), "edges": edges.tolist()}, "mean": dist["mean"], "sd": dist["sd"],
               "kappa": kappa, "own_parameters_euler": float(own.mean()) / kappa}  # fmt: skip
    return summary, examples


def fusion_fit(train: dict) -> tuple[dict, dict]:
    """alpha-hat of every training F beat in a recording with enough training V beats, against that recording's mean
    relative V beat; the fusion model's R^2; the uniform range of alpha; the stop rule."""
    y, records, x = np.asarray(train["y"]), np.asarray(train["record"]), np.asarray(train["X"], dtype=np.float64)
    v, f = CLASS_INDEX["V"], CLASS_INDEX["F"]
    alphas, r2s, used, example = [], [], [], None
    for r in sorted(set(records[y == f].tolist())):
        vm = (records == r) & (y == v)
        if int(vm.sum()) < FUSION["min_v_beats"]:
            continue
        vbar = x[vm].mean(axis=0)
        beats = x[(records == r) & (y == f)]
        a = beats @ vbar / float(vbar @ vbar)
        pred = a[:, None] * vbar
        sst = ((beats - beats.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
        alphas.append(a)
        r2s.append(1 - ((beats - pred) ** 2).sum(axis=1) / np.where(sst > 0, sst, np.nan))
        used.append(int(r))
        if example is None or beats.shape[0] > example["n"]:
            i = int(np.argsort(a)[len(a) // 2])
            example = {"record": int(r), "n": int(beats.shape[0]), "real": beats[i].tolist(), "fitted": pred[i].tolist(), "vbar": vbar.tolist(), "alpha": float(a[i])}
    alpha = np.concatenate(alphas) if alphas else np.zeros(0)
    r2 = np.concatenate(r2s) if r2s else np.zeros(0)
    low = float(np.clip(np.quantile(alpha, FUSION["low_quantile"]), 0, 1)) if alpha.size else float("nan")
    high = float(np.clip(np.quantile(alpha, FUSION["high_quantile"]), 0, 1)) if alpha.size else float("nan")
    median_r2 = float(np.nanmedian(r2)) if r2.size else float("nan")
    passed = bool(alpha.size >= FUSION["min_f_beats"] and len(used) >= FUSION["min_recordings"] and median_r2 >= R2_STOP and low < high)
    summary = {"f_beats": int(alpha.size), "recordings": used, "alpha_quantiles": _quantiles(alpha) if alpha.size else {},
               "alpha_low": low, "alpha_high": high, "median_r2": median_r2, "r2_quantiles": _quantiles(r2) if r2.size else {}, "passed": passed}  # fmt: skip
    return summary, {"fusion": example}


def check(cfg: dict) -> dict:
    """``physics.json``: the simulator fitted per class to P11's training beats, its class distributions and kappas, the
    fusion model's fit, and both stop rules. Training beats only."""
    _, p11 = build_data(cfg)
    train = training_beats(p11.arrays)
    whole = whole_training_beats(train)
    share = zscore_share(whole)
    if share < 0.999:
        raise PreconditionError(f"Only {share:.4f} of the whole training beats are z-scored windows: the match is wrong")
    rng = np.random.default_rng(FIT["seed"])
    y = np.asarray(train["y"])
    classes, examples = {}, {}
    for cls in PHYSICS_CLASSES:
        pool = np.flatnonzero(y == CLASS_INDEX[cls])
        pick = np.sort(rng.choice(pool, size=min(FIT["per_class"], pool.size), replace=False))
        omega = ph.omega_of(np.asarray(train["rr_prev"])[pick], np.asarray(train["rr_ratio"])[pick])
        classes[cls], examples[cls] = fit_class(whole[pick], omega)
        classes[cls]["training_beats"] = int(pool.size)
        print(f"{cls}: {pick.size} beats, median R^2 {classes[cls]['median_r2']:.3f}", flush=True)
    fusion, fusion_example = fusion_fit(train)
    out = {"experiment": EXPERIMENT, "post_hoc": True, "rule": RULE, "fit": FIT, "fusion_rule": FUSION, "r2_stop": R2_STOP,
           "waves": list(ph.WAVES), "eta_layout": "positions (5), heights (5), widths (5), baseline", "zscore_share": share,
           "p11_sha256": p11.info["content_sha256"], "classes": classes, "physics_passed": all(c["median_r2"] >= R2_STOP for c in classes.values()),
           "fusion": fusion, "examples": {**examples, **fusion_example}, "git_sha": runs.git_sha()}  # fmt: skip
    _write("physics.json", out)
    return out


def load_physics(need: str) -> dict:
    record = _read("physics.json", "run --part check on Kaggle and ingest it first")
    if need == "physics" and not record["physics_passed"]:
        raise PreconditionError("The simulator failed its stop rule (physics.json): the generator conditions stop")
    if need == "fusion" and not record["fusion"]["passed"]:
        raise PreconditionError("The fusion model failed its stop rule (physics.json): the fusion condition stops")
    return record


def make_prior(train: dict, whole: np.ndarray, record: dict, scale: float) -> ph.EulerPrior:
    pools = class_pools(train, whole)
    physics = {CLASS_INDEX[c]: record["classes"][c] for c in PHYSICS_CLASSES}
    return ph.EulerPrior.build(pools, physics, scale)


# --------------------------------------------------------------------------------------------------
# --part train: the generators with the Euler loss, the lambda choice, the ten cnn runs
# --------------------------------------------------------------------------------------------------


def lambda_tag(value: float) -> str:
    return f"{value:g}"


def generator_id(family: str, value: float) -> str:
    """E.g. ``phase6_physics_prior/cvae-lambda0.1`` (a folder under data/synthetic/)."""
    return f"{EXPERIMENT}/{family}-lambda{lambda_tag(value)}"


def selection_condition(family: str, value: float) -> str:
    return f"{family}_l{lambda_tag(value)}"


def generators_name(family: str) -> str:
    return f"{family}_generators.json"


def train_generator(cfg: dict, family: str, value: float, seed: int, train: dict, val_records: list[int], prior: ph.EulerPrior,
                    device: torch.device) -> dict:  # fmt: skip
    """One generator of ``family`` at lambda ``value`` and ``seed``, P12's settings and steps, and its bank (P12's counts
    and RR draws). Already-trained ones (bank on disk with the recorded hash) are kept."""
    name = generators_name(family)
    path = paths.results_for(EXPERIMENT) / name
    record = json.loads(path.read_text()) if path.is_file() else {"experiment": EXPERIMENT, "rule": RULE, "family": family, "fits": {}}
    key = f"{lambda_tag(value)}/{seed}"
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
    run_id = runs.make_run_id(EXPERIMENT, "generator", int(seed), tag=f"{family}.lambda{lambda_tag(value)}")
    run_dir = paths.runs_dir() / EXPERIMENT / "generators" / family / run_id
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"experiment": EXPERIMENT, "family": family, "settings": settings, "steps": steps, "seed": int(seed),
                                                         "scale": scale, "lambda": float(value), "prior": "Euler loss (physics.json)"}))  # fmt: skip
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
    """A P21 bank (patient-relative) and its SHA-256, checked against the generators record."""
    entry = _read(generators_name(family), f"run --part train --family {family}")["fits"].get(f"{lambda_tag(value)}/{seed}")
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


def run_cells(grid: dict, data: DataSource, bank_of: Callable[[str, int], tuple[dict, str]], device: torch.device) -> dict:
    """Every cell of ``grid`` not yet done; ``bank_of(condition, cnn seed)`` gives the bank in P13's form and its hash."""
    split = check_preconditions()
    ledger = runner.read_ledger(grid)
    report = {"done_before": 0, "ran": 0}
    for cell in runner.cells(grid):
        classifier = runner.classifier_settings(grid, cell.architecture, preview=False)
        bank, bank_sha = bank_of(cell.condition, cell.seed)
        sha = runner.cell_sha(cell, grid, classifier, data, bank_sha)
        if runner.is_done(ledger.get(sha)):
            report["done_before"] += 1
            continue
        ledger[sha] = runner.run_cell(cell, grid, split, data, classifier, device, sha, bank, grid["conditions"][cell.condition]["bank"], [])
        runner.write_cell(grid, ledger[sha])
        report["ran"] += 1
    return report


def done_rows(grid: dict) -> dict[tuple[str, int], list[dict]]:
    rows: dict[tuple[str, int], list[dict]] = {}
    for r in runner.read_ledger(grid).values():
        if r["status"] == "done":
            rows.setdefault((r["condition"], int(r["seed"])), []).append(r)
    return rows


def choose_lambda(rows: dict[tuple[str, int], list[dict]], family: str) -> dict:
    """The lambda with the highest mean validation S F1 over cnn seeds 0-2 (ties to the smaller); lambda 0 beside."""
    table = {}
    for value in (0.0, *LAMBDAS):
        c = selection_condition(family, value)
        found = [rows.get((c, s), []) for s in SELECTION_SEEDS]
        if any(len(f) != 1 for f in found):
            raise PreconditionError(f"{c}: expected one done run per seed {SELECTION_SEEDS}")
        values = [float(f[0]["selection_value"]) for f in found]
        table[lambda_tag(value)] = {"mean": float(np.mean(values)), "values": values, "run_ids": [f[0]["run_id"] for f in found]}
    chosen = LAMBDAS[0]
    for value in LAMBDAS[1:]:
        if table[lambda_tag(value)]["mean"] > table[lambda_tag(chosen)]["mean"]:
            chosen = value
    return {"rule": "highest mean validation S F1 of cnn seeds 0-2 (bank seed 0), ties to the smaller lambda; lambda 0 descriptive",
            "table": table, "chosen": chosen}  # fmt: skip


def members_of(rows: dict[tuple[str, int], list[dict]], condition: str) -> list[str]:
    out = []
    for s in SEEDS:
        found = rows.get((condition, s), [])
        if len(found) != 1:
            raise PreconditionError(f"{condition} seed {s}: expected one done run, found {len(found)}")
        out.append(found[0]["run_id"])
    return out


def describe_banks(named: dict[str, dict], train: dict, prior: ph.EulerPrior, scale: float) -> dict:
    """Descriptive, training beats only: per bank and class S, V, F, the mean Euler loss (the training loss's value; real
    relative beats scored the same way beside), the class median's correlation with the real one, and for S and F the
    effective number of training patients of the nearest real beat (P12's check)."""
    prior.to("cpu")
    y, x = np.asarray(train["y"]), np.asarray(train["X"], dtype=np.float32)
    units = rg.patient_units(np.asarray(train["record"]))
    real_median = {c: np.median(x[y == CLASS_INDEX[c]], axis=0) for c in PHYSICS_CLASSES}
    out: dict[str, dict] = {}
    for name, bank in {"real": {"X": x, "y": y}, **named}.items():
        bx, by = np.asarray(bank["X"], dtype=np.float32), np.asarray(bank["y"])
        entry = {}
        for c in PHYSICS_CLASSES:
            idx = np.flatnonzero(by == CLASS_INDEX[c])[:DESCRIBE_BEATS]
            beats = torch.as_tensor(bx[idx] / scale)
            torch.manual_seed(0)
            with torch.no_grad():
                term = prior(beats, torch.full((idx.size,), CLASS_INDEX[c], dtype=torch.long))
            e = {"euler": float(term) if term is not None else float("nan"),
                 "median_corr": float(np.corrcoef(np.median(bx[idx], axis=0), real_median[c])[0, 1])}  # fmt: skip
            if c in ("S", "F") and name != "real":
                m = y == CLASS_INDEX[c]
                e["nearest_patients"] = mech._entry(units[m][mech.nearest(x[m], bx[idx])])["effective_records"]
            entry[c] = e
        out[name] = entry
    return out


def train_family(cfg: dict, family: str, device: torch.device) -> dict:
    """lambda choice (seed 0 at each lambda, three cnn runs each, lambda 0 beside), seeds 1-2 at the chosen lambda, the
    ten cnn runs of the new condition, and the descriptive bank checks; writes ``<family>.json``."""
    record = load_physics("physics")
    p13, p11 = build_data(cfg)
    train = training_beats(p11.arrays)
    val_records = sorted(set(np.asarray(p11.arrays["record"])[np.asarray(p11.arrays["split"]) == "val"].astype(int).tolist()))
    whole = whole_training_beats(train)
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

    selection = {selection_condition(family, v): (generator_id(family, v) if v else rg.generator_id(cfg["rg"], family), family) for v in (0.0, *LAMBDAS)}
    sel_grid = cnn_grid(cfg, f"selection_{family}", selection, SELECTION_SEEDS)
    by_name = {selection_condition(family, v): v for v in (0.0, *LAMBDAS)}
    report = {"selection": run_cells(sel_grid, p13, lambda c, s: p13_bank(by_name[c], SELECTION_BANK), device)}
    choice = choose_lambda(done_rows(sel_grid), family)
    chosen = float(choice["chosen"])
    for seed in GENERATOR_SEEDS:
        if seed != SELECTION_BANK:
            train_generator(cfg, family, chosen, seed, train, val_records, prior, device)
    condition = f"{family}_physics"
    final_grid = cnn_grid(cfg, f"final_{family}", {condition: (generator_id(family, chosen), family)}, SEEDS)
    report["final"] = run_cells(final_grid, p13, lambda c, s: p13_bank(chosen, se.bank_seed(s)), device)
    members = members_of(done_rows(final_grid), condition)
    named = {"p12_seed0": rg.load_relative_bank(cfg["rg"], family, 0)[0], **{f"lambda{lambda_tag(v)}_seed0": load_bank(family, v, 0)[0] for v in LAMBDAS}}
    fits = _read(generators_name(family), "")["fits"]
    out = {"experiment": EXPERIMENT, "post_hoc": True, "rule": RULE, "family": family, "condition": condition, "scale": scale,
           "strength": rr.strengths()[family], "lambda_choice": choice, "chosen_lambda": chosen,
           "banks": {str(s): fits[f"{lambda_tag(chosen)}/{s}"]["bank_sha256"] for s in GENERATOR_SEEDS},
           "members": members, "describe": describe_banks(named, train, prior, scale), "report": report, "git_sha": runs.git_sha()}  # fmt: skip
    _write(f"{family}.json", out)
    return out


# --------------------------------------------------------------------------------------------------
# --part fusion: P12's cVAE banks with physiological F beats
# --------------------------------------------------------------------------------------------------


def fusion_bank(bank: dict, v_pool: np.ndarray, low: float, high: float, seed: int) -> dict:
    """``bank`` with each F beat's waveform replaced by alpha v (v a relative training V beat drawn at random, alpha
    uniform on [low, high]); every other field and beat unchanged."""
    out = {k: np.array(v, copy=True) for k, v in bank.items()}
    idx = np.flatnonzero(np.asarray(out["y"]) == CLASS_INDEX["F"])
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(low, high, size=idx.size)
    pick = rng.integers(0, v_pool.shape[0], size=idx.size)
    out["X"][idx] = (alpha[:, None] * np.asarray(v_pool, dtype=np.float64)[pick]).astype(out["X"].dtype)
    source = np.asarray(out["source"]).astype(object)
    source[idx] = f"{EXPERIMENT}/fusion"
    out["source"] = source.astype(str)
    return out


def run_fusion(cfg: dict, device: torch.device) -> dict:
    record = load_physics("fusion")
    low, high = record["fusion"]["alpha_low"], record["fusion"]["alpha_high"]
    p13, p11 = build_data(cfg)
    train_before, train_after = training_beats(p11.arrays), training_beats(p13.arrays)
    v_pool = np.asarray(train_after["X"])[np.asarray(train_after["y"]) == CLASS_INDEX["V"]]
    counts = rg.generators_record("cvae")["bank_counts"]
    made: dict[int, tuple[dict, str]] = {}
    for b in GENERATOR_SEEDS:
        relative, _ = rg.load_relative_bank(cfg["rg"], "cvae", b)
        bank = fusion_bank(rr.v4_bank(relative, train_before, train_after, counts, b), v_pool, low, high, b)
        made[b] = (bank, content_sha256(bank))
    grid = cnn_grid(cfg, "final_fusion", {FUSION_CONDITION: (f"{EXPERIMENT}/fusion", "cvae")}, SEEDS)
    report = run_cells(grid, p13, lambda c, s: made[se.bank_seed(s)], device)
    y = np.asarray(train_after["y"])
    units = rg.patient_units(np.asarray(train_after["record"]))
    m = y == CLASS_INDEX["F"]
    nearest = {}
    for name, bank in (("fusion_seed0", made[0][0]), ("p12_seed0", rr.v4_bank(rg.load_relative_bank(cfg["rg"], "cvae", 0)[0], train_before, train_after, counts, 0))):
        idx = np.flatnonzero(np.asarray(bank["y"]) == CLASS_INDEX["F"])[:DESCRIBE_BEATS]
        nearest[name] = mech._entry(units[m][mech.nearest(np.asarray(train_after["X"])[m], np.asarray(bank["X"])[idx])])["effective_records"]
    out = {"experiment": EXPERIMENT, "post_hoc": True, "rule": RULE, "condition": FUSION_CONDITION, "alpha": [low, high],
           "strength": rr.strengths()["cvae"], "banks": {str(b): sha for b, (_, sha) in made.items()},
           "members": members_of(done_rows(grid), FUSION_CONDITION), "nearest_patients_F": nearest, "report": report, "git_sha": runs.git_sha()}  # fmt: skip
    _write("fusion.json", out)
    return out


# --------------------------------------------------------------------------------------------------
# --part score: the setting chosen on P14's patients, scored on P19's
# --------------------------------------------------------------------------------------------------


def new_members() -> dict[str, list[str]]:
    out = {}
    for c, family in NEW.items():
        name = "fusion.json" if c == FUSION_CONDITION else f"{family}.json"
        out[c] = list(_read(name, f"run the part that makes {c} and ingest it first")["members"])
        if len(out[c]) != len(SEEDS):
            raise PreconditionError(f"{c}: expected {len(SEEDS)} runs")
    return out


def baseline_members() -> dict[str, list[str]]:
    frozen = fp.frozen_members()
    return {b: frozen[b] for b in sorted(set(NEW.values()))}


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


def choose(probs: dict[str, np.ndarray], arrays: dict) -> dict:
    """P20's rule for the new conditions on P14's patients."""
    choice = rt.choose({c: probs[c] for c in NEW}, arrays)
    return {**choice, "experiment": EXPERIMENT, "rule": f"{RULE} (P20's rule, PREREGISTRATION v19)"}


def score(choice: dict, probs: dict[str, np.ndarray], arrays: dict) -> dict:
    """On P19's patients: each new condition with its chosen setting vs its P18 condition with P20's (S F1, Holm 3); the
    ensembles alone beside. P20's S F1 for the baselines' chosen settings must reproduce within 0.001."""
    form = choice["veto_form"]
    p20 = rt.load_choice()
    settings = {**rt.chosen_settings(choice), **{b: s for b, s in rt.chosen_settings(p20).items() if b in NEW.values()}}
    y, records = np.asarray(arrays["y"]), np.asarray(arrays["record"])
    rr_prev, p_af = np.asarray(arrays["rr_prev"]), np.asarray(arrays["p_af"])
    tables: dict[str, dict[str, np.ndarray]] = {"chosen": {}, "alone": {}}
    unique = None
    for c, p in probs.items():
        for k, s in (("chosen", settings[c]), ("alone", (None, None))):
            unique, cm = M.confusion_by_record(y, rt.decide(p, rr_prev, p_af, s, form), records)
            tables[k][c] = cm[None]
    p20_summary = json.loads((paths.results_for(rt.EXPERIMENT) / "summary.json").read_text())
    reproduction = {f"cnn/{b}": float(fp._stat("f1_S")(tables["chosen"][b].sum(axis=1))[0]) - p20_summary["conditions"]["chosen"][f"cnn/{b}"]["f1_S"]["point"]
                    for b in sorted(set(NEW.values()))}  # fmt: skip
    off = {k: d for k, d in reproduction.items() if abs(d) > ICENTIA_TOLERANCE}
    if off:
        raise PreconditionError(f"P20's S F1 for the baselines does not reproduce: {off}")
    return {"experiment": EXPERIMENT, "post_hoc": True, "rule": RULE, "n_boot": ref.N_BOOT, "boot_seed": ref.BOOT_SEED, "veto_form": form,
            "chosen": choice["chosen"], "baseline_settings": {b: p20["chosen"][b] for b in sorted(set(NEW.values()))},
            "patients": int(unique.size), "beats": {k: int((y == i).sum()) for i, k in enumerate(CLASSES)},
            "conditions": {"chosen": fp._block(tables["chosen"], unique), "ensemble_alone": fp._block(tables["alone"], unique)},
            "families": {"secondary": fp._family({f"cnn: {c} vs {b}, chosen settings": (tables["chosen"][c], tables["chosen"][b]) for c, b in NEW.items()})},
            "versus_baseline_alone": fp._family({f"cnn: {c} vs {b}, ensembles alone": (tables["alone"][c], tables["alone"][b]) for c, b in NEW.items()}),
            "evaluation_surface": rt.surface({c: probs[c] for c in NEW}, arrays, form), "reproduction_difference": reproduction,
            "note": "secondary family (Holm, 3) on S F1; the ensembles alone and the surface are descriptive (Holm within that block only)",
            "git_sha": runs.git_sha()}  # fmt: skip


def run_score(beats_dir: Path, device: torch.device) -> tuple[dict, dict]:
    members = {**new_members(), **baseline_members()}
    probs = {}
    for name in rt.SETS:
        arrays = rt.load_set(beats_dir, name)
        probs[name] = ensemble_probs(members, arrays, lambda run_id, a: fp.predict(run_id, a, device))
        del arrays
    choice = choose(probs["tuning"], rt.load_set(beats_dir, "tuning"))
    _write("choice.json", choice)
    summary = score(choice, probs["evaluation"], rt.load_set(beats_dir, "evaluation"))
    _write("summary.json", summary)
    return choice, summary


# --------------------------------------------------------------------------------------------------
# --part test: one posthoc DS2 read
# --------------------------------------------------------------------------------------------------


def test_report(cfg: dict) -> dict:
    """DS2 in P13's form, one posthoc access. Confirmatory (Holm, 3): each new condition with its chosen setting vs its P18
    condition with P20's, on rare-class macro-F1; P20's DS2 values for the baselines must reproduce exactly. Descriptive:
    the ensembles alone, per-class F1, without record 202."""
    from ecgaug import final_eval

    choice = _read("choice.json", "run --part score and ingest it first")
    p20 = rt.load_choice()
    p20_test = json.loads((paths.results_for(rt.EXPERIMENT) / "test_summary.json").read_text())
    form = choice["veto_form"]
    settings = {**rt.chosen_settings(choice), **{b: s for b, s in rt.chosen_settings(p20).items() if b in NEW.values()}}
    members = {**new_members(), **baseline_members()}
    test, p_af = se.ds2_inputs(cfg["se"], EXPERIMENT)
    y, records, rr_prev = test["y"], np.asarray(test["record"]), test["rr_prev"]
    tables: dict[str, dict[str, np.ndarray]] = {"chosen": {}, "alone": {}}
    unique = None
    for c, ids in members.items():
        p = se.ensemble(pt._predict(ids, test))
        for k, s in (("chosen", settings[c]), ("alone", (None, None))):
            pred = rt.decide(p, rr_prev, p_af, s, form)
            unique, tables[k][c] = final_eval.per_record_cms([np.eye(p.shape[1], dtype=np.float32)[pred]], y, records)
    checks = {f"cnn/{b}": se._rare(tables["chosen"][b]) - p20_test["conditions"]["chosen"][f"cnn/{b}"]["rare_macro_f1"]["point"] for b in sorted(set(NEW.values()))}
    off = {k: d for k, d in checks.items() if abs(d) > DS2_TOLERANCE}
    if off:
        raise PreconditionError(f"P20's DS2 scores do not reproduce: {off}")
    keep = unique != 202

    def block(table: dict, kept: np.ndarray | None = None) -> dict:
        if kept is None:
            return {f"cnn/{c}": ref.condition_summary(cm, unique) for c, cm in table.items()}
        return {f"cnn/{c}": ref.condition_summary(cm[:, kept], unique[kept]) for c, cm in table.items()}

    summary = {"experiment": EXPERIMENT, "post_hoc": True, "rule": RULE, "records": [int(r) for r in unique], "n_boot": ref.N_BOOT,
               "boot_seed": ref.BOOT_SEED, "veto_form": form, "settings": {c: rt.label(s) for c, s in settings.items()},
               "conditions": {"chosen": block(tables["chosen"]), "ensemble_alone": block(tables["alone"])},
               "without_202": {"chosen": block(tables["chosen"], keep)},
               "families": {"confirmatory": mp._family({f"cnn: {c} vs {b}, chosen settings": (tables["chosen"][c], tables["chosen"][b]) for c, b in NEW.items()})},
               "versus_baseline_alone": mp._family({f"cnn: {c} vs {b}, ensembles alone": (tables["alone"][c], tables["alone"][b]) for c, b in NEW.items()}),
               "note": "confirmatory family (Holm, 3) on rare-class macro-F1; the rest is descriptive",
               "reproduction_difference": checks, "run_ids": members, "git_sha": runs.git_sha()}  # fmt: skip
    _write("test_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--part", choices=("check", "train", "fusion", "score", "test"), required=True)
    parser.add_argument("--family", choices=FAMILIES, default=None)
    parser.add_argument("--beats-dir", type=Path, default=rt.default_beats_dir(), help="P20's built beats (score)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    try:
        if args.part == "check":
            out = check(cfg)
            print(json.dumps({"physics_passed": out["physics_passed"], "fusion": out["fusion"],
                              "median_r2": {c: v["median_r2"] for c, v in out["classes"].items()}}, indent=2, default=float))  # fmt: skip
        elif args.part == "train":
            if args.family is None:
                raise PreconditionError("--part train needs --family")
            out = train_family(cfg, args.family, resolve_device(args.device))
            print(json.dumps({"chosen_lambda": out["chosen_lambda"], "lambda_choice": out["lambda_choice"]["table"], "report": out["report"]}, indent=2, default=float))
        elif args.part == "fusion":
            out = run_fusion(cfg, resolve_device(args.device))
            print(json.dumps({"alpha": out["alpha"], "nearest_patients_F": out["nearest_patients_F"], "report": out["report"]}, indent=2, default=float))
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
