"""phase6_rr_reference: fixed RR-feature reference classifiers (PREREGISTRATION §10 v4, P1; post hoc).

    python -m ecgaug.experiments.phase6_rr_reference --part fit                                   # DS1 only
    ALLOW_TEST_POSTHOC=1 python -m ecgaug.experiments.phase6_rr_reference --part test             # one logged posthoc access

Two references with nothing to tune, in the spirit of the linear discriminants of de Chazal et al. (2004):
``rr_lda`` (rr_prev, rr_next, rr_ratio) and ``rr_morph_lda`` (the same plus every 10th sample of the frozen
representation's z-scored window). Uniform class priors, features standardised on the DS1 training beats, one
deterministic fit each, trained on the frozen v1 training records. ``--part fit`` writes each model as a run with
its validation metrics; ``--part test`` scores DS2 with the record-level bootstrap and compares each reference with
the cnn without augmentation and with cVAE (the final evaluation's saved predictions), as paired differences with
95% intervals and no test. Nothing here changes or selects any model of the main study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import yaml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ecgaug import paths, runs, workflow
from ecgaug.data.access import LockedTestDataError
from ecgaug.data.loaders import load_test_posthoc, load_trainval
from ecgaug.data.records import CLASSES
from ecgaug.eval import metrics as M
from ecgaug.eval.bootstrap import cluster_bootstrap_confusion, paired_cluster_bootstrap_diff_confusion
from ecgaug.experiments.common import PreconditionError
from ecgaug.names import PHASE4_GATE, PHASE6_RR_REFERENCE

EXPERIMENT = PHASE6_RR_REFERENCE
REFERENCES = {"rr_lda": "LDA on rr_prev, rr_next and rr_ratio", "rr_morph_lda": "LDA on the RR features plus every 10th window sample"}
MORPH_STEP = 10
N_BOOT, BOOT_SEED = 10_000, 0
COMPARED = ("none", "cvae")  # the cnn conditions each reference is set beside


def features(arrays: dict, reference: str) -> np.ndarray:
    """The fixed feature matrix of one reference."""
    rr = np.stack([arrays["rr_prev"], arrays["rr_next"], arrays["rr_ratio"]], axis=1).astype(np.float64)
    if reference == "rr_lda":
        return rr
    if reference == "rr_morph_lda":
        return np.concatenate([rr, np.asarray(arrays["X"], dtype=np.float64)[:, ::MORPH_STEP]], axis=1)
    raise ValueError(f"Unknown reference {reference!r}")


def fit_reference(train: dict, reference: str):
    """Standardise on the training beats, then LDA with uniform priors over the classes."""
    model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="svd", priors=np.full(len(CLASSES), 1 / len(CLASSES))))
    return model.fit(features(train, reference), np.asarray(train["y"], dtype=np.int64))


def predict(model, arrays: dict, reference: str) -> np.ndarray:
    """Probabilities [n, 4] in class order (every class is present in training)."""
    return model.predict_proba(features(arrays, reference))


def _part(arrays: dict, name: str) -> dict:
    mask = np.asarray(arrays["split"]) == name
    return {k: np.asarray(v)[mask] for k, v in arrays.items()}


def model_path(run_id: str) -> Path:
    return runs.find_run_dir(run_id) / "checkpoints" / "model.pkl"


def fit_all() -> dict:
    """Fit both references on the frozen training beats; one run each, with validation metrics."""
    arrays = load_trainval("v1")
    train, val = _part(arrays, "train"), _part(arrays, "val")
    out = {}
    for reference, description in REFERENCES.items():
        model = fit_reference(train, reference)
        probs = predict(model, val, reference)
        m = M.evaluate(probs, val["y"])
        run_id = runs.make_run_id(EXPERIMENT, reference.replace("_", ""), 0)
        run_dir = paths.runs_dir() / EXPERIMENT / run_id
        (run_dir / "checkpoints").mkdir(parents=True)
        blob = pickle.dumps(model)
        (run_dir / "checkpoints" / "model.pkl").write_bytes(blob)
        sha = hashlib.sha256(blob).hexdigest()
        config = {"experiment": EXPERIMENT, "reference": reference, "description": description, "features": reference,
                  "morph_step": MORPH_STEP, "priors": "uniform", "dataset": "data/processed/v1", "train_records": "v1 split 'train'", "run_id": run_id}  # fmt: skip
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        (run_dir / "env.json").write_text(json.dumps({**runs.capture_env("cpu"), "wall_clock_s": 0.0, "checkpoint": {"path": "checkpoints/model.pkl", "sha256": sha}}, indent=2) + "\n")
        (run_dir / "train_log.csv").write_text("step\n0\n")
        metrics = {**m, "selection_metric": "", "selection_value": float("nan")}
        (run_dir / "metrics_val.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n")
        np.savez_compressed(run_dir / "preds_val.npz", probs=probs, y=val["y"], record=val["record"], r_idx=val["r_idx"])
        runs.upsert_runs_index(runs.index_row(run_id, metrics, 0.0, group=(EXPERIMENT,)))
        out[reference] = {"run_id": run_id, "sha256": sha, "validation": {"rare_macro_f1": m["rare_macro_f1"], "f1": {c: m["per_class"][c]["f1"] for c in CLASSES}}}
    result_dir = paths.results_for(EXPERIMENT)
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "fit.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    return out


def load_fitted() -> dict[str, object]:
    """The fitted references, checked against the hashes written when they were fitted."""
    fitted = json.loads((paths.results_for(EXPERIMENT) / "fit.json").read_text())
    models = {}
    for reference, entry in fitted.items():
        blob = model_path(entry["run_id"]).read_bytes()
        if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
            raise PreconditionError(f"{entry['run_id']}: the fitted model changed since it was written")
        models[reference] = pickle.loads(blob)  # noqa: S301 (our own file, hash checked)
    return models


def condition_summary(cms: np.ndarray, records: np.ndarray) -> dict:
    """Rare-class macro-F1, per-class F1 and S Se/+P with record-level intervals, from ``[n_seeds, n_records, C, C]``."""
    from ecgaug.final_eval import STATISTICS

    keep = ("rare_macro_f1", "f1_S", "f1_V", "f1_F", "se_S", "ppv_S", "n_recall")
    return {name: cluster_bootstrap_confusion(STATISTICS[name], cms, records, n=N_BOOT, seed=BOOT_SEED) for name in keep}


def test_report() -> dict:
    """DS2: each reference, and its paired difference from the cnn without augmentation and with cVAE."""
    from ecgaug.final_eval import per_record_cms

    models = load_fitted()
    gate = workflow.manifest()[PHASE4_GATE]
    fitted = json.loads((paths.results_for(EXPERIMENT) / "fit.json").read_text())
    test = load_test_posthoc("v1", f"{EXPERIMENT}: DS2 predictions of the references", run_ids=", ".join(e["run_id"] for e in fitted.values()))
    y, records = test["y"], test["record"]
    cnn = {}
    for condition in COMPARED:
        probs = []
        for run_id in gate["chosen"]["cnn"][condition]["run_ids"]:
            with np.load(runs.find_run_dir(run_id) / "preds_test.npz") as saved:
                if not np.array_equal(saved["r_idx"], test["r_idx"]):
                    raise PreconditionError(f"{run_id}: saved test predictions do not match the frozen test beats")
                probs.append(saved["probs"])
        unique, cnn[condition] = per_record_cms(probs, y, records)
    summary: dict = {"experiment": EXPERIMENT, "post_hoc": True, "rule": "PREREGISTRATION §10 v4, P1", "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                     "references": {}, "cnn": {c: condition_summary(cnn[c], unique) for c in COMPARED}, "git_sha": runs.git_sha()}  # fmt: skip
    for reference, model in models.items():
        probs = predict(model, test, reference)
        np.savez_compressed(runs.find_run_dir(fitted[reference]["run_id"]) / "preds_test.npz", probs=probs, y=y, record=records, r_idx=test["r_idx"])
        unique, cms = per_record_cms([probs], y, records)
        entry = {"run_id": fitted[reference]["run_id"], "description": REFERENCES[reference], **condition_summary(cms, unique), "vs_cnn": {}}
        for condition in COMPARED:
            diff = paired_cluster_bootstrap_diff_confusion(M.rare_macro_f1_from_cm, cms, cnn[condition], n=N_BOOT, seed=BOOT_SEED)
            diff["cnn_weaker"] = bool(condition == "none" and diff["ci_low"] > 0)  # the pre-registered wording rule
            entry["vs_cnn"][condition] = diff
        summary["references"][reference] = entry
    out = paths.results_for(EXPERIMENT)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=["fit", "test"], required=True)
    args = parser.parse_args(argv)
    try:
        if args.part == "fit":
            for reference, entry in fit_all().items():
                print(f"{reference}: validation rare-class macro-F1 {entry['validation']['rare_macro_f1']:.3f} ({entry['run_id']})")
        else:
            summary = test_report()
            for reference, e in summary["references"].items():
                r = e["rare_macro_f1"]
                print(f"{reference}: DS2 rare-class macro-F1 {r['point']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]; S Se {e['se_S']['point']:.3f}, S +P {e['ppv_S']['point']:.3f}")
                for condition, d in e["vs_cnn"].items():
                    print(f"    minus cnn {condition}: {d['point']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]")
            print(f"written: results/{EXPERIMENT}/summary.json")
    except (PreconditionError, LockedTestDataError) as err:
        print(f"Refusing to run: {err}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
