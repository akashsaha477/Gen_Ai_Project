"""Error analysis from saved predictions (PLAN Phase 5): no new test access, ever.

    python -m ecgaug.report.error_analysis            # validation predictions of the phase 4 gate's models
    python -m ecgaug.report.error_analysis --test     # after the final evaluation: its saved test predictions

For each architecture and condition fixed at the phase 4 gate: the confusion matrix (mean over seeds),
errors per record for S and F, and the N/S confusions with their RR context (``rr_ratio``: how early the
beat came relative to the 10 before it). Validation RR values come from the dataset; test RR values
come from the prediction files written by the final evaluation, so the test set is never reopened.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from ecgaug import paths, runs, workflow
from ecgaug.data.records import CLASS_INDEX
from ecgaug.eval.metrics import confusion_matrix
from ecgaug.names import FINAL_EVAL, PHASE4_GATE

OUT = "error_analysis"


def _predictions(run_id: str, split: str) -> dict[str, np.ndarray]:
    path = runs.find_run_dir(run_id) / f"preds_{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing" + (" (run the final evaluation first)" if split == "test" else " (ingest the Kaggle zip)"))
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def _rr_lookup(arrays: dict) -> dict[tuple[int, int], float]:
    return {(int(r), int(i)): float(v) for r, i, v in zip(arrays["record"], arrays["r_idx"], arrays["rr_ratio"])}


def analyse(chosen: dict, split: str, rr_lookup: dict | None = None) -> dict:
    """Confusions, per-record S/F errors and N/S confusions with their RR ratios, per architecture and condition."""
    out: dict = {}
    for arch, conditions in chosen.items():
        for condition, c in conditions.items():
            preds = [_predictions(r, split) for r in c["run_ids"]]
            y, records = preds[0]["y"], preds[0]["record"]
            predicted = [p["probs"].argmax(axis=1) for p in preds]
            cm = np.mean([confusion_matrix(y, yp) for yp in predicted], axis=0)
            per_record = {}
            for record in np.unique(records):
                mask = records == record
                entry = {}
                for cls in ("S", "F"):
                    true = mask & (y == CLASS_INDEX[cls])
                    if true.any():
                        missed = np.mean([(yp[true] != CLASS_INDEX[cls]).mean() for yp in predicted])
                        entry[cls] = {"support": int(true.sum()), "missed_fraction": float(missed)}
                if entry:
                    per_record[str(int(record))] = entry
            rr = preds[0]["rr_ratio"] if "rr_ratio" in preds[0] else np.array([rr_lookup.get((int(r), int(i)), np.nan) for r, i in zip(records, preds[0]["r_idx"])])
            yp = predicted[0]  # the RR context of the first seed's decisions
            groups = {
                "N correct": (y == CLASS_INDEX["N"]) & (yp == CLASS_INDEX["N"]), "N called S": (y == CLASS_INDEX["N"]) & (yp == CLASS_INDEX["S"]),
                "S correct": (y == CLASS_INDEX["S"]) & (yp == CLASS_INDEX["S"]), "S called N": (y == CLASS_INDEX["S"]) & (yp == CLASS_INDEX["N"]),
                "S called V": (y == CLASS_INDEX["S"]) & (yp == CLASS_INDEX["V"]),
            }  # fmt: skip
            rr_context = {name: {"n": int(m.sum()), "rr_ratio_median": float(np.nanmedian(rr[m])) if m.any() else None,
                                 "rr_ratio_values": [float(v) for v in rr[m][:2000]]} for name, m in groups.items()}  # fmt: skip
            out[f"{arch}/{condition}"] = {"confusion_matrix": cm.tolist(), "per_record": per_record, "rr_context": rr_context, "split": split}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Error analysis from saved predictions (validation, or test after the final evaluation).")
    parser.add_argument("--test", action="store_true", help="use the final evaluation's saved test predictions")
    args = parser.parse_args(argv)
    gate = workflow.manifest().get(PHASE4_GATE)
    if not gate:
        print("The phase 4 gate has not fixed the models yet")
        return 2
    if args.test and not (paths.results_for(FINAL_EVAL) / "summary.json").is_file():
        print("The final evaluation has not run; there are no saved test predictions")
        return 2
    rr_lookup = None
    if not args.test:
        from ecgaug.experiments.common import check_preconditions
        from ecgaug.experiments.data_source import load_data

        rr_lookup = _rr_lookup(load_data(check_preconditions(allow_dirty=True), preview=False).arrays)
    split = "test" if args.test else "val"
    result = analyse(gate["chosen"], split, rr_lookup)
    out = paths.results_for(OUT)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{split}.json").write_text(json.dumps(result, indent=2) + "\n")
    from ecgaug import figures

    figures.safe_call(figures.error_analysis_figures, result, split)
    print(f"written: results/{OUT}/{split}.json and figures/{OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
