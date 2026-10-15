"""Phase 5 experiment phase5_explainability: one saliency method per architecture (PLAN Phase 5).

Usage:
    python -m ecgaug.experiments.phase5_explainability --config configs/phase5_explainability/B_integrated_gradients.yaml
    python -m ecgaug.experiments.phase5_explainability --config ... --smoke

The models are fixed at the phase 4 gate: for each architecture (``cnn``, ``rnn``) and each of none,
cVAE and WGAN-GP, the condition's runs at its chosen strength; the best checkpoint is the seed with the
highest validation selection metric, and the other seeds serve the stability score. Their checkpoints
must be on disk (ingest the Kaggle zips of the grid). Everything is computed on validation beats,
stratified by class; the test set is never read.

For each model the variant's method explains the predicted class of every sampled beat, and the maps
are scored for faithfulness (deletion against random order), stability across seeds and physiological
alignment (``ecgaug.explain.evaluate``). For augmented models the method is also run on synthetic bank
beats, to see whether the model looks at different places in them (RQ4: possible artifact reliance).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from ecgaug import figures, paths, runs, workflow
from ecgaug.data.records import CLASS_INDEX, CLASSES
from ecgaug.experiments.common import SIDE_FIELDS, PreconditionError, check_preconditions, load_run_classifier, split_part
from ecgaug.experiments.data_source import DataSource, load_data, smoke_data
from ecgaug.explain.evaluate import faithfulness, segment_masses, stability
from ecgaug.explain.methods import applies, saliency
from ecgaug.generators.bank import load_bank
from ecgaug.names import PHASE4_GATE, PHASE5_EXPLAINABILITY

EXPERIMENT = PHASE5_EXPLAINABILITY
VARIANT_KEYS = {"base", "variant", "description", "method", "cost"}
SHARED_KEYS = {"experiment", "conditions", "beats_per_class", "sample_seed", "deletion_fractions", "random_orders",
               "synthetic_beats_per_class", "method_settings"}  # fmt: skip


def load_config(path: Path) -> dict:
    path = Path(path)
    variant = yaml.safe_load(path.read_text())
    extra = set(variant) - VARIANT_KEYS
    if extra:
        raise ValueError(f"{path}: keys {sorted(extra)} belong in the shared settings file, not in a variant")
    shared = yaml.safe_load((path.parent / variant["base"]).read_text())
    if set(shared) != SHARED_KEYS:
        raise ValueError(f"{path.parent / variant['base']}: expected exactly the keys {sorted(SHARED_KEYS)}")
    return {**shared, **{k: variant[k] for k in VARIANT_KEYS - {"base"}}, "variant": str(variant["variant"]), "config_name": path.stem}


def model_set(conditions: list[str]) -> dict:
    """Per architecture and condition: the best run (highest validation selection metric) and every seed's run."""
    gate = workflow.manifest().get(PHASE4_GATE)
    if not gate:
        raise PreconditionError("The phase 4 gate has not fixed the models yet (python -m ecgaug.workflow status)")
    out: dict = {}
    for arch, chosen in gate["chosen"].items():
        for condition in conditions:
            if condition not in chosen:
                continue
            run_ids = chosen[condition]["run_ids"]
            values = {r: json.loads((runs.find_run_dir(r) / "metrics_val.json").read_text())["selection_value"] for r in run_ids}
            out.setdefault(arch, {})[condition] = {"best": max(values, key=values.get), "seeds": list(run_ids), "bank": chosen[condition].get("bank")}
    return out


def sample_beats(val: dict, per_class: int, seed: int) -> np.ndarray:
    """Indices of up to ``per_class`` validation beats of each class (stratified, fixed seed)."""
    rng = np.random.default_rng(seed)
    picks = [rng.permutation(np.flatnonzero(val["y"] == i))[:per_class] for i in range(len(CLASSES))]
    return np.sort(np.concatenate(picks))


def _by_class(values: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {c: float(values[y == i].mean()) for c, i in CLASS_INDEX.items() if (y == i).any()}


def run_variant(cfg: dict, data: DataSource, models: dict, reasons: list[str]) -> dict:
    """Explain every model with the variant's method and score the maps; write ``results/<experiment>/<letter>.json``."""
    started = time.perf_counter()
    rr = data.spec.rr_features
    val = split_part(data.arrays, "val", rr)
    idx = sample_beats(val, int(cfg["beats_per_class"]), int(cfg["sample_seed"]))
    x, side, y = val["X"][idx], (val["side"][idx] if rr else None), val["y"][idx]
    settings = {**cfg["method_settings"], "fs": data.spec.effective_fs}
    pre = data.spec.window_samples[0]
    per_arch: dict = {}
    class_average: dict = {}
    for arch, conditions in models.items():
        entry: dict = {"applicable": True, "conditions": {}}
        for condition, m in conditions.items():
            best = load_run_classifier(m["best"], rr)
            if not applies(cfg["method"], best):
                entry = {"applicable": False, "conditions": {}}
                break
            maps = saliency(cfg["method"], best, x, side, settings)
            faith = faithfulness(best, x, side, maps, list(cfg["deletion_fractions"]), int(cfg["random_orders"]), int(cfg["sample_seed"]))
            seed_maps = [maps] + [saliency(cfg["method"], load_run_classifier(r, rr), x, side, settings) for r in m["seeds"] if r != m["best"]]
            masses = segment_masses(maps, pre, data.spec.effective_fs)
            result = {
                "best_run": m["best"], "seed_runs": m["seeds"], "n_beats": int(len(idx)), "faithfulness_gain": float(faith["gain"].mean()),
                "faithfulness_gain_by_class": _by_class(faith["gain"], y), "auc_saliency": float(faith["auc_saliency"].mean()),
                "auc_random": float(faith["auc_random"].mean()), "stability": stability(seed_maps),
                "alignment": {seg: _by_class(v, y) for seg, v in masses.items()},
            }  # fmt: skip
            if condition != "none" and m.get("bank"):
                try:
                    bank = load_bank(m["bank"], int(runs.parse_run_id(m["best"])["seed"]))
                    rng = np.random.default_rng(int(cfg["sample_seed"]))
                    pick = np.sort(np.concatenate([rng.permutation(np.flatnonzero(bank["y"] == CLASS_INDEX[c]))[: int(cfg["synthetic_beats_per_class"])] for c in ("S", "V", "F")]))
                    bx, by = bank["X"][pick], bank["y"][pick]
                    bside = np.stack([bank[f][pick] for f in SIDE_FIELDS], axis=1).astype(np.float32) if rr else None
                    synth = segment_masses(saliency(cfg["method"], best, bx, bside, settings), pre, data.spec.effective_fs)
                    result["alignment_on_synthetic"] = {seg: _by_class(v, by) for seg, v in synth.items()}
                except Exception as err:  # noqa: BLE001 (the bank is optional here; report why it was skipped)
                    result["alignment_on_synthetic"] = {"skipped": str(err)}
            entry["conditions"][condition] = result
            class_average.setdefault(arch, {})[condition] = {c: np.abs(maps[y == i]).mean(axis=0).tolist() for c, i in CLASS_INDEX.items() if (y == i).any()}
        if entry["applicable"]:
            gains = [c["faithfulness_gain"] for c in entry["conditions"].values()]
            stabilities = [c["stability"] for c in entry["conditions"].values()]
            entry.update(mean_gain=float(np.mean(gains)), sd_gain=float(np.std(gains, ddof=1)) if len(gains) > 1 else 0.0,
                         mean_stability=float(np.nanmean(stabilities)))  # fmt: skip
        per_arch[arch] = entry
    result = {
        "experiment": EXPERIMENT, "variant": cfg["variant"], "description": cfg["description"], "method": cfg["method"], "cost": cfg["cost"],
        "per_architecture": per_arch, "settings": settings, "deletion_fractions": cfg["deletion_fractions"],
        "provisional": bool(reasons), "provisional_reasons": reasons, "dataset": data.info, "wall_clock_s": time.perf_counter() - started,
        "git_sha": runs.git_sha(),
    }  # fmt: skip
    out = paths.results_for(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{cfg['variant']}.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / f"{cfg['variant']}_class_average_saliency.json").write_text(json.dumps({"pre_samples": pre, "fs": data.spec.effective_fs, "maps": class_average}) + "\n")
    figures.safe_for_result(out / f"{cfg['variant']}.json")
    return result


def smoke(cfg: dict) -> dict:
    """Tiny cnn and rnn models (two seeds, one epoch) on fake records stand in for the phase 4 models."""
    from ecgaug.experiments.common import train_grid

    small = {**cfg, "beats_per_class": 6, "random_orders": 1, "method_settings": {**cfg["method_settings"], "ig_steps": 4, "smoothgrad_samples": 3}}
    with smoke_data() as (split, data):
        rr = data.spec.rr_features
        train, val = split_part(data.arrays, "train", rr), split_part(data.arrays, "val", rr)
        models: dict = {}
        for arch, block in (("cnn", {"name": "shallow_cnn", "channels": [8, 16, 16], "kernel": 7, "side_hidden": 4}),
                            ("rnn", {"name": "bilstm", "hidden": 8, "layers": 1, "side_hidden": 4})):  # fmt: skip
            budget = {"seeds": [0, 1], "lr_grid": [1e-3], "max_epochs": 1, "patience": 5, "batch_size": 256, "weight_decay": 0.01, "model": block}
            grid, _ = train_grid(budget, split, data.spec, train, val, experiment=EXPERIMENT, label=f"smoke{arch}", device=__import__("torch").device("cpu"),
                                 dataset=data.info, group=(EXPERIMENT, "smoke", arch))  # fmt: skip
            run_ids = [r["run_id"] for r in next(iter(grid.values()))["per_seed"]]
            models[arch] = {"none": {"best": run_ids[0], "seeds": run_ids, "bank": None}}
        result = run_variant(small, data, models, [data.provisional])
        return {"variant": result["variant"], "method": result["method"],
                "applicable": {a: e["applicable"] for a, e in result["per_architecture"].items()}, "note": "fake data; the numbers mean nothing"}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="phase5_explainability: score a saliency method on the phase 4 models.")
    parser.add_argument("--config", required=True, help="configs/phase5_explainability/<variant>.yaml")
    parser.add_argument("--smoke", action="store_true", help="end-to-end run on fake data with tiny models")
    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config))
    if args.smoke:
        print(json.dumps(smoke(cfg), indent=2))
        return 0
    try:
        if workflow.phase1_winner() is None:
            raise PreconditionError("Phase 1 is not finished: approve the phase 1 gate first.")
        split = check_preconditions()
        data = load_data(split, preview=False)
        models = model_set(list(cfg["conditions"]))
    except (PreconditionError, FileNotFoundError) as err:
        print(f"Refusing to run {EXPERIMENT} variant {cfg['variant']}: {err}")
        return 2
    result = run_variant(cfg, data, models, [])
    for arch, entry in result["per_architecture"].items():
        if entry["applicable"]:
            print(f"{arch}: faithfulness gain {entry['mean_gain']:+.4f} (mean over {len(entry['conditions'])} models), stability {entry['mean_stability']:.3f}")
        else:
            print(f"{arch}: {result['method']} does not apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
