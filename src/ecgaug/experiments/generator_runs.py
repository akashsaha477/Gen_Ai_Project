"""The runner shared by the two phase 3 experiments, phase3_cvae and phase3_wgan_gp.

    python -m ecgaug.experiments.phase3_cvae --config configs/phase3_cvae/A_conv_concat.yaml            # pilot, seeds 0 and 1
    python -m ecgaug.experiments.phase3_cvae --config configs/phase3_cvae/A_conv_concat.yaml --final    # full budget, grid seeds, banks
    python -m ecgaug.experiments.phase3_cvae --config configs/phase3_cvae/A_conv_concat.yaml --smoke    # fake data, end to end
    add --preview to run before the gates a run needs; the results are marked provisional

**Pilot** (PLAN Phase 3, items 1 and 3). Each pilot seed trains the generator for the pilot step
budget. At each quarter of the budget the pilot harness scores the generator:

1. it samples synthetic S, V and F beats up to the pilot policy (each minority class up to 50% of the
   training N count), with RR features copied from real training beats of the same class;
2. it measures the diagnostics (diversity, memorization, C2ST) and, once the phase 2 gate has fixed
   the composite, the quality metrics;
3. it trains the pilot classifier (the ``cnn`` chosen at the phase 2 gate, seed 0) on the real
   training beats plus these synthetic beats, and evaluates it on validation (100% real).

Every checkpoint is logged in the run's ``checkpoints.csv``: that is the RQ3 dataset. Only the final
checkpoint counts for selection; earlier checkpoints are never selectable.

**Final.** The variant chosen at the phase 3a gate is trained at the full budget for every grid seed
with the same harness, and its bank is exported from the final checkpoint (``ecgaug.generators.bank``).

**Preconditions.** An official pilot needs the phase 1 gate (the frozen dataset) and the phase 2 gate
(the pilot classifier and the quality composite); an official final run also needs the phase 3a gate.
``--preview`` runs before them with stand-ins and marks the results provisional: the candidate
dataset of the representation named by ``--representation`` (default: the winner in the phase 1 gate
report), the phase1_input_representation probe CNN as pilot classifier, and the diagnostics only.
Every run needs the approved validation split and a clean git tree. The test set is never read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from ecgaug import augment, figures, paths, quality, runs, workflow
from ecgaug.data.contract import TRAINVAL_FILE, build_candidate, content_sha256
from ecgaug.data.counts import count_record
from ecgaug.data.fake import write_fake_mitdb
from ecgaug.data.loaders import load_candidate, load_trainval
from ecgaug.data.records import CLASSES
from ecgaug.data.representation import RepresentationSpec
from ecgaug.data.splits import Split, check_split, selection_metric_for
from ecgaug.eval.selection import tie_break_value
from ecgaug.experiments import phase1_input_representation as phase1
from ecgaug.experiments.common import SIDE_FIELDS, PreconditionError, check_preconditions, mean_sd, split_part, temporary_root
from ecgaug.generators import GENERATED_CLASSES, build_generator
from ecgaug.generators.bank import bank_counts, export_bank, sample_bank
from ecgaug.generators.base import BeatGenerator, training_beats
from ecgaug.models import build_model
from ecgaug.names import PHASE1_GATE, PHASE2_GATE, PHASE3_CVAE, PHASE3_WGAN_GP, PHASE3A_GATE
from ecgaug.quality.diagnostics import prepare_references, score_beats
from ecgaug.train.loop import FitResult, TrainConfig, fit_classifier, resolve_device, set_seed

FAMILY_OF = {PHASE3_CVAE: "cvae", PHASE3_WGAN_GP: "wgan_gp"}
VARIANT_KEYS = {"base", "variant", "description", "generator"}
SHARED_KEYS = {
    "experiment", "family", "classes", "training", "generator_defaults", "pilot_seeds", "final_seeds",
    "checkpoint_fractions", "pilot", "diagnostics", "hard_gates", "bank",
}  # fmt: skip
DIAGNOSTICS = ("diversity_ratio", "memorization_fraction", "c2st_accuracy")
REFERENCE_SEED = 0  # held-out beats and real subsamples are the same for every run, so diagnostics compare
DEFAULT_SETTING = "default"


# --------------------------------------------------------------------------------------------------
# configs
# --------------------------------------------------------------------------------------------------


def load_config(path: Path) -> dict:
    """Merge a variant file with the shared settings it points to.

    A variant may set only its letter, description and the generator settings that define it; every
    budget (steps, seeds, batch, learning rate, pilot, diagnostics, hard gates) comes from the shared
    file, so budgets cannot drift between variants (docs/RULES.md rule 7). A generator setting the shared
    defaults do not define is an error.
    """
    path = Path(path)
    variant = yaml.safe_load(path.read_text())
    extra = set(variant) - VARIANT_KEYS
    if extra:
        raise ValueError(f"{path}: keys {sorted(extra)} belong in the shared settings file, not in a variant")
    shared_path = path.parent / variant["base"]
    shared = yaml.safe_load(shared_path.read_text())
    if set(shared) != SHARED_KEYS:
        raise ValueError(f"{shared_path}: expected exactly the keys {sorted(SHARED_KEYS)}")
    if FAMILY_OF.get(shared["experiment"]) != shared["family"]:
        raise ValueError(f"{shared_path}: experiment {shared['experiment']} does not train the family {shared['family']}")
    own = variant.get("generator") or {}
    unknown = set(own) - set(shared["generator_defaults"])
    if unknown:
        raise ValueError(f"{path}: generator settings {sorted(unknown)} are not defined in generator_defaults")
    return {
        **shared,
        "variant": str(variant["variant"]),
        "description": variant.get("description", ""),
        "generator": {**shared["generator_defaults"], **own},
        "config_name": path.stem,
    }


def settings_grid(cfg: dict) -> list[tuple[str, dict]]:
    """The generator settings a variant trains: one, or one per value of a list-valued ``beta`` (cVAE B)."""
    g = cfg["generator"]
    if isinstance(g.get("beta"), list):
        return [(f"beta{float(b):g}", {**g, "beta": float(b)}) for b in g["beta"]]
    return [(DEFAULT_SETTING, g)]


def generator_settings(cfg: dict, generator: dict) -> dict:
    return {**cfg["training"], **generator, "classes": list(cfg["classes"])}


def generator_id(cfg: dict, setting: str) -> str:
    """Name of a variant's bank, e.g. ``phase3_cvae-A`` or ``phase3_cvae-B.beta0.25``."""
    base = f"{cfg['experiment']}-{cfg['variant']}"
    return base if setting == DEFAULT_SETTING else f"{base}.{setting}"


# --------------------------------------------------------------------------------------------------
# what a run needs: data, pilot classifier, quality metrics
# --------------------------------------------------------------------------------------------------


@dataclass
class DataSource:
    arrays: dict[str, np.ndarray]
    spec: RepresentationSpec
    info: dict
    provisional: str | None = None  # why this is not the frozen dataset


def reported_phase1_winner() -> str | None:
    """The winner in the written (not necessarily approved) phase 1 gate report."""
    report = paths.results_for(PHASE1_GATE) / "report.json"
    return json.loads(report.read_text())["winner"] if report.is_file() else None


def load_data(split: Split, preview: bool, representation: str | None = None) -> DataSource:
    """The training and validation beats a run uses.

    After the phase 1 gate: the frozen dataset, from ``data/processed/v1`` or, where that is not on
    disk (Kaggle), rebuilt from the raw records and checked against the frozen hash. In a preview:
    the candidate dataset of ``representation`` or of the phase 1 gate report's winner.
    """
    winner = workflow.phase1_winner()
    letter = representation or winner or reported_phase1_winner()
    if letter is None:
        raise PreconditionError(
            "No input representation to use: the phase 1 gate is not approved, its report is not written, "
            "and --representation was not given."
        )
    configs = workflow.variant_configs()
    if letter not in configs:
        raise PreconditionError(f"No phase1_input_representation variant {letter!r}")
    spec = RepresentationSpec(**phase1.load_config(configs[letter])["representation"])
    if winner is not None and letter == winner:
        expected = workflow.manifest()[PHASE1_GATE]["files"][TRAINVAL_FILE]["content_sha256"]
        if (paths.processed_dir() / "v1" / TRAINVAL_FILE).is_file():
            arrays, source = load_trainval("v1"), "data/processed/v1 (frozen at the phase 1 gate)"
        else:
            build_candidate(spec, split)
            arrays, source = load_candidate(spec.name), f"rebuilt from the raw records ({spec.name}) and checked against the frozen hash"
        actual = content_sha256(arrays)
        if actual != expected:
            raise PreconditionError(
                f"The training data ({actual[:12]}...) does not match the frozen dataset ({expected[:12]}..., "
                "results/manifest.json). Use the frozen files (data/processed/v1) instead of rebuilding."
            )
        return DataSource(arrays, spec, {"source": source, "representation": spec.name, "content_sha256": actual, "frozen": True})
    if not preview:
        raise PreconditionError(
            "Phase 1 is not finished: the phase 1 gate has not frozen the dataset. Approve the phase 1 gate first "
            "(python -m ecgaug.workflow status), or add --preview."
        )
    build_candidate(spec, split)
    arrays = load_candidate(spec.name)
    reason = "the phase 1 gate has not been approved" if winner is None else f"the phase 1 gate chose {winner}, not {letter}"
    info = {"source": f"candidate dataset {spec.name}", "representation": spec.name, "content_sha256": content_sha256(arrays), "frozen": False}
    return DataSource(arrays, spec, info, f"{reason}: the beats come from the candidate dataset {spec.name}")


def pilot_classifier(cfg: dict, preview: bool) -> tuple[dict, str | None]:
    """The pilot classifier's model and training settings, and why it is a stand-in (or None)."""
    name = cfg["pilot"]["classifier"]
    chosen = (workflow.manifest().get(PHASE2_GATE) or {}).get(name)
    if chosen:
        return {"source": f"{name} chosen at the phase 2 gate", **chosen}, None
    if not preview:
        raise PreconditionError(
            f"Phase 2 is not finished: the phase 2 gate has not chosen the pilot classifier ({name}). "
            "Finish phase 2 first (python -m ecgaug.workflow status), or add --preview."
        )
    stand_in = {"source": "preview stand-in: the phase1_input_representation probe CNN", **cfg["pilot"]["preview_classifier"]}
    return stand_in, f"the phase 2 gate has not chosen the {name}: the pilot classifier is the phase1_input_representation probe CNN"


def quality_families(preview: bool) -> tuple[list[str], str | None]:
    """The metric families of the composite quality rank, fixed at the phase 2 gate."""
    families = (workflow.manifest().get(PHASE2_GATE) or {}).get("quality_composite")
    if families:
        missing = [f for f in families if f not in quality.METRIC_FAMILIES]
        if missing:
            raise PreconditionError(f"The quality composite names metric families that are not implemented: {missing}")
        return list(families), None
    if not preview:
        raise PreconditionError(
            "Phase 2 is not finished: the phase 2 gate has not fixed the quality composite (phase2_quality_metrics). "
            "Finish phase 2 first, or add --preview."
        )
    return [], "the quality composite is not fixed yet (phase2_quality_metrics): only the diagnostics are measured"


def chosen_setting(cfg: dict, preview: bool) -> tuple[str, str | None]:
    """The setting a final run trains: the one approved at the phase 3a gate, or in a preview the pilot's choice."""
    entry = (workflow.manifest().get(PHASE3A_GATE) or {}).get(cfg["experiment"])
    if entry and entry["winner"] == cfg["variant"]:
        return entry["setting"], None
    if not preview:
        if entry:
            right = Path(entry["config"]).name
            raise PreconditionError(f"The phase 3a gate chose variant {entry['winner']}; run configs/{cfg['experiment']}/{right}, or add --preview.")
        raise PreconditionError("The phase 3a gate has not chosen a generator yet (python -m ecgaug.workflow status), or add --preview.")
    pilot = paths.results_for(cfg["experiment"]) / f"{cfg['variant']}.json"
    setting = json.loads(pilot.read_text())["chosen_setting"] if pilot.is_file() else settings_grid(cfg)[0][0]
    reason = "the phase 3a gate has not chosen this variant" if not entry else f"the phase 3a gate chose variant {entry['winner']}"
    return setting, f"{reason}: the final run and its bank are a preview"


# --------------------------------------------------------------------------------------------------
# the pilot harness and one generator run
# --------------------------------------------------------------------------------------------------


class PilotHarness:
    """Scores a generator at a checkpoint: diagnostics, quality metrics and the downstream pilot score."""

    def __init__(self, cfg: dict, split: Split, data: DataSource, classifier: dict, families: list[str], device: torch.device) -> None:
        self.cfg, self.split, self.spec, self.device = cfg, split, data.spec, device
        self.train = training_beats(data.arrays)
        self.val = split_part(data.arrays, "val", data.spec.rr_features)
        self.counts = augment.class_counts(self.train["y"])
        self.targets = augment.fraction_of_n_targets(self.counts, cfg["pilot"]["policy_fraction_of_n"])
        self.references = prepare_references(self.train, GENERATED_CLASSES, cfg["diagnostics"], REFERENCE_SEED)
        self.classifier, self.families = classifier, families

    def quality_values(self, bank: dict[str, np.ndarray]) -> dict[str, float]:
        """Each composite metric family, averaged over the generated classes."""
        out = {}
        for family in self.families:
            scores = [
                quality.METRIC_FAMILIES[family].score(self.train["X"][self.train["y"] == CLASSES.index(c)], bank["X"][bank["y"] == CLASSES.index(c)])
                for c in GENERATED_CLASSES
            ]
            out[family] = float(np.mean(scores))
        return out

    def score(self, generator: BeatGenerator, seed: int) -> tuple[dict, dict, FitResult, dict]:
        bank = sample_bank(generator, self.train, self.targets, seed=seed, source="pilot")
        diagnostics = score_beats(bank, self.references, self.cfg["diagnostics"], seed)
        train = split_part(augment.with_added(self.train, bank), "train", self.spec.rr_features)
        c, classifier_seed = self.classifier, int(self.cfg["pilot"]["classifier_seed"])
        set_seed(classifier_seed)
        model = build_model(c["model"], len(CLASSES), len(SIDE_FIELDS) if self.spec.rr_features else 0)
        train_cfg = TrainConfig(
            lr=float(c["lr"]), max_epochs=int(c["max_epochs"]), patience=int(c["patience"]),
            batch_size=int(c["batch_size"]), weight_decay=float(c["weight_decay"]),
        )  # fmt: skip
        fit = fit_classifier(model, train, self.val, train_cfg, self.split.selection_metric, seed=classifier_seed, device=self.device)
        return diagnostics, self.quality_values(bank), fit, bank


def write_csv(path: Path, rows: list[dict]) -> None:
    """Rows with possibly different keys; the header is the union of keys in order of appearance."""
    fields: list[str] = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields or ["step"])
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_row(k: int, step: int, steps: int, diagnostics: dict, quality_values: dict, fit: FitResult, selection: str) -> dict:
    """One line of ``checkpoints.csv``: everything RQ3 needs about one checkpoint."""
    m = fit.best_metrics
    return {
        "checkpoint": k, "step": step, "fraction": step / steps,
        "pilot_selection_metric": selection, "pilot_selection_value": fit.best_value,
        "pilot_tie_break_value": tie_break_value(m, selection), "pilot_rare_macro_f1": m["rare_macro_f1"],
        **{f"pilot_f1_{c}": m["per_class"][c]["f1"] for c in CLASSES},
        **{f"pilot_pr_auc_{c}": m["pr_auc"][c] for c in CLASSES},
        "pilot_n_recall": m["n_recall"], "pilot_best_epoch": fit.best_epoch, "pilot_epochs_run": fit.epochs_run,
        **{f"{name}_{c}": diagnostics[c][name] for c in GENERATED_CLASSES for name in DIAGNOSTICS},
        **{f"quality_{f}": v for f, v in quality_values.items()},
    }  # fmt: skip


def new_run_dir(run_id: str, group: tuple[str, ...]) -> Path:
    if runs.find_run_dir(run_id) is not None:
        raise FileExistsError(f"Run {run_id} already exists")
    run_dir = paths.runs_dir().joinpath(*group, run_id)
    (run_dir / "checkpoints").mkdir(parents=True)
    return run_dir


def train_one(
    cfg: dict, setting: str, generator_cfg: dict, seed: int, steps: int, mode: str, harness: PilotHarness,
    data: DataSource, reasons: list[str], export: bool = False,
) -> dict:  # fmt: skip
    """Train one generator (one variant, setting and seed), score its checkpoints, log the run.

    With ``export``, the bank is sampled from the final checkpoint and written to ``data/synthetic/``.
    Returns the run's summary for the result file.
    """
    experiment, letter = cfg["experiment"], cfg["variant"]
    tags = [t for t in ("final" if mode == "final" else None, None if setting == DEFAULT_SETTING else setting) if t]
    run_id = runs.make_run_id(experiment, letter, seed, tag=".".join(tags) or None)
    group = (experiment, cfg["config_name"], mode)
    run_dir = new_run_dir(run_id, group)
    settings = generator_settings(cfg, generator_cfg)
    generator = build_generator(cfg["family"], settings, data.spec.n_samples, seed)
    selection = harness.split.selection_metric
    rows: list[dict] = []
    scored: dict[int, tuple] = {}
    hashes: dict[str, str] = {}

    def on_checkpoint(k: int, step: int, gen: BeatGenerator) -> dict:
        path = run_dir / "checkpoints" / f"q{k}.pt"
        torch.save(gen.state_dict(), path)
        hashes[f"q{k}"] = _sha256(path)
        diagnostics, quality_values, fit, bank = harness.score(gen, seed)
        write_csv(run_dir / "pilot_classifier" / f"q{k}_train_log.csv", fit.history)
        rows.append({**checkpoint_row(k, step, steps, diagnostics, quality_values, fit, selection), "checkpoint_sha256": hashes[f"q{k}"]})
        scored[k] = (diagnostics, quality_values, fit, bank)
        return {"pilot_selection_value": fit.best_value}

    fit = generator.fit(
        harness.train, steps, seed, harness.device, val_records=harness.split.val, on_checkpoint=on_checkpoint,
        fractions=cfg["checkpoint_fractions"], log_every=int(cfg["training"]["log_every"]),
    )  # fmt: skip
    last = len(cfg["checkpoint_fractions"])
    done = not fit.diverged and last in scored
    summary = {
        "seed": seed, "run_id": run_id, "setting": setting, "steps": steps, "steps_done": fit.steps_done,
        "diverged": fit.diverged, "divergence": fit.divergence, "n_parameters": generator.n_parameters(),
        "generator_seconds": fit.generator_seconds, "wall_clock_s": fit.wall_clock_s,
        "checkpoints": [
            {"checkpoint": r["checkpoint"], "step": r["step"], "selection_value": r["pilot_selection_value"],
             "min_diversity_ratio": min(r[f"diversity_ratio_{c}"] for c in GENERATED_CLASSES),
             "max_memorization_fraction": max(r[f"memorization_fraction_{c}"] for c in GENERATED_CLASSES)}
            for r in rows
        ],
    }  # fmt: skip
    metrics: dict = {"selection_metric": selection, "diverged": fit.diverged, "divergence": fit.divergence, "steps_done": fit.steps_done}
    preds = None
    if done:
        diagnostics, quality_values, final_fit, pilot_bank = scored[last]
        m = final_fit.best_metrics
        metrics.update({**m, "selection_value": final_fit.best_value, "best_epoch": final_fit.best_epoch,
                        "epochs_run": final_fit.epochs_run, "diagnostics": diagnostics, "quality": quality_values})  # fmt: skip
        summary.update({
            "selection_value": final_fit.best_value, "tie_break_value": tie_break_value(m, selection),
            "rare_macro_f1": m["rare_macro_f1"], "f1": {c: m["per_class"][c]["f1"] for c in CLASSES},
            "pr_auc": m["pr_auc"], "n_recall": m["n_recall"], "diagnostics": diagnostics, "quality": quality_values,
            "classifier_best_epoch": final_fit.best_epoch, "classifier_epochs_run": final_fit.epochs_run,
        })  # fmt: skip
        preds = {"probs": final_fit.val_probs, "y": harness.val["y"], "record": harness.val["record"], "r_idx": harness.val["r_idx"]}
    else:
        metrics.update({"selection_value": float("nan"), "rare_macro_f1": float("nan")})
        summary.update({"selection_value": float("nan"), "tie_break_value": None, "diagnostics": None, "quality": {}})

    bank_entry = None
    if export and done:
        gid = generator_id(cfg, setting)
        bank = sample_bank(generator, harness.train, bank_counts(harness.train["y"], cfg["bank"]["size"]), seed=seed, source=gid)
        bank_diagnostics = score_beats(bank, harness.references, cfg["diagnostics"], seed)
        info = {
            "run_id": run_id, "config": f"configs/{experiment}/{cfg['config_name']}.yaml", "setting": setting,
            "step": fit.steps_done, "checkpoint_sha256": hashes[f"q{last}"], "train_only_check": "passed",
            "dataset": data.info, "diagnostics": bank_diagnostics, "provisional": bool(reasons), "provisional_reasons": reasons,
        }  # fmt: skip
        bank_entry = {"generator_id": gid, "seed": seed, **export_bank(bank, gid, seed, info)}
        out = paths.results_for(experiment) / "banks"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{gid}-seed{seed}.json").write_text(json.dumps(bank_entry, indent=2) + "\n")
        summary["bank"] = {k: bank_entry[k] for k in ("generator_id", "file", "sha256", "content_sha256", "counts")}
        figures.safe_call(figures.phase3_sample_figures, paths.figures_for(experiment) / "final" / gid / f"seed{seed}",
                          f"{gid}, seed {seed}: bank", harness.train, bank, data.spec)  # fmt: skip
    elif done and seed == cfg["pilot_seeds"][0] and mode == "pilot":
        out = paths.figures_for(experiment) / letter / ("" if setting == DEFAULT_SETTING else setting)
        figures.safe_call(figures.phase3_sample_figures, out, f"{experiment}, variant {letter}"
                          + ("" if setting == DEFAULT_SETTING else f" ({setting})") + f", seed {seed}: pilot bank at the final checkpoint",
                          harness.train, scored[last][3], data.spec)  # fmt: skip

    run_config = {
        **{k: v for k, v in cfg.items() if k != "generator_defaults"}, "generator": settings, "setting": setting,
        "mode": mode, "seed": seed, "steps": steps, "run_id": run_id,
        "split": {"name": harness.split.name, "val": list(harness.split.val), "selection_metric": selection},
        "dataset": data.info, "pilot_classifier": harness.classifier, "quality_families": harness.families,
        "provisional": bool(reasons), "provisional_reasons": reasons,
    }  # fmt: skip
    (run_dir / "config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False))
    write_csv(run_dir / "train_log.csv", fit.history)
    write_csv(run_dir / "checkpoints.csv", rows)
    (run_dir / "metrics_val.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if preds is not None:
        np.savez_compressed(run_dir / "preds_val.npz", **preds)
    env = {
        **runs.capture_env(str(harness.device)), "wall_clock_s": fit.wall_clock_s, "generator_seconds": fit.generator_seconds,
        "checkpoints": {k: {"path": f"checkpoints/{k}.pt", "sha256": v} for k, v in hashes.items()},
    }  # fmt: skip
    (run_dir / "env.json").write_text(json.dumps(env, indent=2) + "\n")
    runs.upsert_runs_index(runs.index_row(run_id, metrics, fit.wall_clock_s, status="done" if done else "diverged", group=group))
    return summary


# --------------------------------------------------------------------------------------------------
# a variant's result: hard gates and the summary the phase 3a gate reads
# --------------------------------------------------------------------------------------------------


def hard_gates(per_seed: list[dict], gates: dict) -> dict:
    """PLAN Phase 3 hard gates for one setting: no divergence on any pilot seed, diversity of every
    generated class on every seed at least the floor, memorization of every class on every seed at
    most the ceiling."""
    floor, ceiling = float(gates["diversity_min_ratio"]), float(gates["memorization_max_fraction"])
    diverged = [s["seed"] for s in per_seed if s["diverged"]]
    scored = [s for s in per_seed if s.get("diagnostics")]
    diversity = [s["diagnostics"][c]["diversity_ratio"] for s in scored for c in GENERATED_CLASSES]
    memorization = [s["diagnostics"][c]["memorization_fraction"] for s in scored for c in GENERATED_CLASSES]
    complete = not diverged and len(scored) == len(per_seed)
    lowest = float(min(diversity)) if diversity and np.isfinite(diversity).all() else float("nan")
    highest = float(max(memorization)) if memorization and np.isfinite(memorization).all() else float("nan")
    checks = [
        {"check": "every pilot seed trains without divergence", "passed": not diverged, "diverged_seeds": diverged},
        {"check": f"diversity ratio >= {floor:g} for every class and seed", "passed": complete and lowest >= floor, "value": lowest},
        {"check": f"memorization fraction <= {ceiling:g} for every class and seed", "passed": complete and highest <= ceiling, "value": highest},
    ]
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def summarise_setting(per_seed: list[dict], gates: dict, generator_cfg: dict, steps: int) -> dict:
    values = [s["selection_value"] for s in per_seed]
    mean, sd = mean_sd(values)
    entry = {"generator": generator_cfg, "steps": steps, "per_seed": per_seed, "mean": mean, "sd": sd,
             "run_ids": [s["run_id"] for s in per_seed], "hard_gates": hard_gates(per_seed, gates),
             "n_parameters": per_seed[0]["n_parameters"], "wall_clock_s": float(sum(s["wall_clock_s"] for s in per_seed))}  # fmt: skip
    ties = [s.get("tie_break_value") for s in per_seed]
    entry["tie_break_mean"], entry["tie_break_sd"] = mean_sd(ties) if all(t is not None for t in ties) else (None, None)
    quality_keys = sorted({k for s in per_seed for k in (s.get("quality") or {})})
    entry["quality_mean"] = {k: float(np.mean([s["quality"][k] for s in per_seed])) for k in quality_keys} if all(s.get("quality") for s in per_seed) else {}
    return entry


def choose_setting(settings: dict[str, dict]) -> str:
    """For a variant with several settings (cVAE B): the passing setting with the best mean; if none
    passes, the one with the best mean; otherwise the first."""
    passing = [k for k, v in settings.items() if v["hard_gates"]["passed"] and np.isfinite(v["mean"])]
    finite = [k for k, v in settings.items() if np.isfinite(v["mean"])]
    pool = passing or finite
    return max(pool, key=lambda k: settings[k]["mean"]) if pool else next(iter(settings))


def run_pilot(cfg: dict, split: Split, data: DataSource, classifier: dict, families: list[str], device: torch.device, reasons: list[str]) -> dict:
    """Pilot every setting of a variant on the pilot seeds and write ``results/<experiment>/<letter>.json``."""
    harness = PilotHarness(cfg, split, data, classifier, families, device)
    grid = settings_grid(cfg)
    steps = int(cfg["training"]["pilot_steps"]) // len(grid)  # several settings share the variant's pilot budget
    settings = {}
    for setting, generator_cfg in grid:
        per_seed = [train_one(cfg, setting, generator_cfg, seed, steps, "pilot", harness, data, reasons) for seed in cfg["pilot_seeds"]]
        settings[setting] = summarise_setting(per_seed, cfg["hard_gates"], generator_cfg, steps)
    chosen = choose_setting(settings)
    best = settings[chosen]
    notes = []
    if len(grid) > 1:
        notes.append(f"{len(grid)} settings share the pilot budget: {steps} steps each instead of {cfg['training']['pilot_steps']}")
    if cfg["generator"].get("conditioning") == "per_class":
        notes.append("one GAN per class; each gets a third of the steps (step t trains class t mod 3)")
    result = {
        "experiment": cfg["experiment"], "variant": cfg["variant"], "description": cfg["description"],
        "family": cfg["family"], "mode": "pilot", "provisional": bool(reasons), "provisional_reasons": reasons,
        "dataset": data.info, "split": {"name": split.name, "val": list(split.val)}, "selection_metric": split.selection_metric,
        "pilot": {"steps": int(cfg["training"]["pilot_steps"]), "seeds": list(cfg["pilot_seeds"]),
                  "policy": f"each of S, V, F up to {cfg['pilot']['policy_fraction_of_n']:g} x the training N count",
                  "targets": harness.targets, "classifier": classifier, "quality_families": families},
        "hard_gate_thresholds": cfg["hard_gates"], "settings": settings, "chosen_setting": chosen,
        "mean": best["mean"], "sd": best["sd"], "tie_break_mean": best["tie_break_mean"], "tie_break_sd": best["tie_break_sd"],
        "per_seed": best["per_seed"], "run_ids": best["run_ids"], "hard_gates": best["hard_gates"],
        "quality_mean": best["quality_mean"], "n_parameters": best["n_parameters"],
        "wall_clock_s": float(sum(v["wall_clock_s"] for v in settings.values())),
        "budget": {"training": cfg["training"], "pilot_seeds": cfg["pilot_seeds"], "checkpoint_fractions": cfg["checkpoint_fractions"],
                   "pilot": {k: v for k, v in cfg["pilot"].items() if k != "preview_classifier"}},
        "budget_deviations": [], "budget_notes": notes, "device": str(device), "git_sha": runs.git_sha(),
    }  # fmt: skip
    out = paths.results_for(cfg["experiment"])
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{cfg['variant']}.json").write_text(json.dumps(result, indent=2) + "\n")
    notes_file = out / f"{cfg['variant']}_notes.md"
    if not notes_file.exists():
        notes_file.write_text(
            f"# {cfg['experiment']}, variant {cfg['variant']}: notes\n\n"
            "What worked:\n\nWhat failed (divergence, collapse, artifacts):\n\nSurprises:\n\nSanity checks (losses, "
            "real vs synthetic figure, diagnostics per checkpoint, pilot classifier curves):\n\nFigure paths:\n"
        )
    figures.safe_for_result(out / f"{cfg['variant']}.json")
    return result


def run_final(cfg: dict, split: Split, data: DataSource, classifier: dict, families: list[str], device: torch.device,
              reasons: list[str], setting: str) -> dict:  # fmt: skip
    """Train the chosen setting at the full budget for every grid seed and export the banks."""
    harness = PilotHarness(cfg, split, data, classifier, families, device)
    grid = dict(settings_grid(cfg))
    if setting not in grid:
        raise ValueError(f"Variant {cfg['variant']} has no setting {setting!r}; it has {sorted(grid)}")
    steps = int(cfg["training"]["final_steps"])
    per_seed = [train_one(cfg, setting, grid[setting], seed, steps, "final", harness, data, reasons, export=True) for seed in cfg["final_seeds"]]
    result = {
        "experiment": cfg["experiment"], "variant": cfg["variant"], "family": cfg["family"], "mode": "final",
        "generator_id": generator_id(cfg, setting), "setting": setting, "provisional": bool(reasons),
        "provisional_reasons": reasons, "dataset": data.info, "steps": steps, "seeds": list(cfg["final_seeds"]),
        "per_seed": per_seed, "run_ids": [s["run_id"] for s in per_seed], "banks": [s.get("bank") for s in per_seed],
        "diverged_seeds": [s["seed"] for s in per_seed if s["diverged"]], "device": str(device), "git_sha": runs.git_sha(),
    }  # fmt: skip
    out = paths.results_for(cfg["experiment"])
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{cfg['variant']}_final.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


# --------------------------------------------------------------------------------------------------
# smoke run and command line
# --------------------------------------------------------------------------------------------------


def smoke(cfg: dict, final: bool = False) -> dict:
    """The pilot (and with ``final`` the final run and bank) on fake records in a temporary root."""
    train_records, val_records = (101, 106, 108, 114), (118, 201)
    small = {
        **cfg, "pilot_seeds": [0], "final_seeds": [0],
        "training": {**cfg["training"], "pilot_steps": 8, "final_steps": 8, "batch_size": 16, "log_every": 2},
        "diagnostics": {**cfg["diagnostics"], "max_beats": 60, "c2st_folds": 2},
    }  # fmt: skip
    classifier = {**cfg["pilot"]["preview_classifier"], "max_epochs": 1, "source": "smoke run"}
    with temporary_root():
        write_fake_mitdb(paths.raw_dir(), train_records + val_records, seconds=120.0)
        counts = pd.DataFrame([count_record(r) for r in train_records + val_records])
        split = Split(name="smoke", train=train_records, val=val_records)
        check_split(split, require_all_ds1=False)
        split = Split(name="smoke", train=train_records, val=val_records, selection_metric=selection_metric_for(counts, split))
        spec = RepresentationSpec(name="smoke_short_window_rr", window_pre_s=0.25, window_post_s=0.25, rr_features=True)
        build_candidate(spec, split, require_all_ds1=False)
        arrays = load_candidate(spec.name)
        data = DataSource(arrays, spec, {"source": "fake records (smoke run)", "representation": spec.name}, "smoke run on fake data")
        reasons = ["smoke run on fake data"]
        result = run_pilot(small, split, data, classifier, [], torch.device("cpu"), reasons)
        run_dirs = [runs.find_run_dir(r) for s in result["settings"].values() for r in s["run_ids"]]
        expected = {"config.yaml", "env.json", "train_log.csv", "checkpoints.csv", "metrics_val.json", "checkpoints", "pilot_classifier"}
        missing = sorted({name for d in run_dirs for name in expected - {p.name for p in d.iterdir()}})
        if missing:
            raise RuntimeError(f"Smoke run did not produce {missing}")
        summary = {
            "variant": result["variant"], "family": result["family"], "settings": sorted(result["settings"]),
            "chosen_setting": result["chosen_setting"], "checkpoints_logged": len(result["per_seed"][0]["checkpoints"]),
            "hard_gates_evaluated": [c["check"] for c in result["hard_gates"]["checks"]],
            "run_files": sorted(p.name for p in run_dirs[0].iterdir()),
            "result_file": f"results/{cfg['experiment']}/{cfg['variant']}.json",
        }  # fmt: skip
        if final:
            final_result = run_final(small, split, data, classifier, [], torch.device("cpu"), reasons, result["chosen_setting"])
            bank = final_result["banks"][0]
            if bank is None:
                raise RuntimeError("Smoke final run exported no bank")
            summary["bank"] = {"generator_id": bank["generator_id"], "counts": bank["counts"]}
        summary["note"] = "fake data; the numbers of a smoke run mean nothing and are discarded"
        return summary


def main(experiment: str, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"{experiment}: generator pilots, final banks and the pilot harness.")
    parser.add_argument("--config", required=True, help=f"configs/{experiment}/<variant>.yaml")
    parser.add_argument("--final", action="store_true", help="full budget on the grid seeds and export the banks (the phase 3a gate's choice)")
    parser.add_argument("--device", default=None, help="cpu, cuda, mps or auto (default: auto)")
    parser.add_argument("--smoke", action="store_true", help="end-to-end run on fake data in a temp directory")
    parser.add_argument("--preview", action="store_true", help="run before the gates this run needs; results are marked provisional")
    parser.add_argument("--representation", default=None, help="preview only: the phase1_input_representation variant letter to use")
    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config))
    if cfg["experiment"] != experiment:
        print(f"{args.config} belongs to {cfg['experiment']}, not {experiment}")
        return 2
    if args.smoke:
        print(json.dumps(smoke(cfg, final=args.final), indent=2))
        return 0
    if args.representation and not args.preview:
        print("--representation is for previews only; official runs use the dataset frozen at the phase 1 gate")
        return 2
    try:
        reasons: list[str] = []
        if not args.preview and workflow.phase1_winner() is None:
            raise PreconditionError("Phase 1 is not finished: approve the phase 1 gate first (python -m ecgaug.workflow status), or add --preview.")
        classifier, why_classifier = pilot_classifier(cfg, args.preview)
        families, why_quality = quality_families(args.preview)
        setting, why_setting = chosen_setting(cfg, args.preview) if args.final else (None, None)
        split = check_preconditions()
        data = load_data(split, args.preview, args.representation)
        reasons = [r for r in (data.provisional, why_classifier, why_quality, why_setting) if r]
    except PreconditionError as err:
        print(f"Refusing to run {experiment} variant {cfg['variant']}: {err}")
        return 2
    device = resolve_device(args.device or "auto")
    if args.final:
        result = run_final(cfg, split, data, classifier, families, device, reasons, setting)
        for s in result["per_seed"]:
            bank = s.get("bank")
            print(f"seed {s['seed']}: {s['run_id']}" + (f"; bank {bank['generator_id']} {bank['counts']}" if bank else "; DIVERGED, no bank"))
    else:
        result = run_pilot(cfg, split, data, classifier, families, device, reasons)
        for name, s in result["settings"].items():
            gates = "pass" if s["hard_gates"]["passed"] else "FAIL: " + "; ".join(c["check"] for c in s["hard_gates"]["checks"] if not c["passed"])
            print(f"{cfg['variant']} {name}: pilot {result['selection_metric']} = {s['mean']:.4f} ± {s['sd']:.4f}; hard gates {gates}; runs: {', '.join(s['run_ids'])}")
    if reasons:
        print("PROVISIONAL: " + "; ".join(reasons))
    return 0
