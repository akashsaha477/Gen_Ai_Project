"""RQ3: do synthetic-quality metrics predict the downstream benefit? (PAPER_TRACK RQ3, H3; PREREGISTRATION §9)

    python -m ecgaug.eval.rq3 [--n-boot 2000] [--preview]

Data: every checkpoint of every official generator run behind the phase3_cvae and phase3_wgan_gp result
files (pilot runs of every setting, and final runs), read from each run's ``checkpoints.csv``. For each
quality metric: Spearman's rho between the metric and the downstream gain across checkpoints, with a 95%
percentile interval from resampling generator training runs (variant x setting x seed) with
replacement, because the checkpoints of one run are dependent. TSTR is reported apart from the
fidelity and diversity metrics: it and the gain are both validation scores of a classifier trained on
the same synthetic beats, so some correlation is built in. Exploratory: no decision uses it.

Gain = the pilot score at a checkpoint minus the score of the same classifier (the phase 2 gate's cnn,
its seed-0 run) trained without synthetic beats. Subtracting one constant does not change rho, so rho
is the same whether or not that reference exists; it only sets the scatter's axis.

Outputs: ``results/phase4_quality_vs_gain/`` (summary.json, checkpoints.csv) and
``figures/phase4_quality_vs_gain/quality_vs_gain.png``. ``python -m ecgaug.paper`` builds the table.
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np
from scipy.stats import spearmanr

from ecgaug import paths, runs, workflow
from ecgaug.names import PHASE2_GATE, PHASE4_QUALITY_VS_GAIN

N_BOOT = 2000
BOOT_SEED = 0
CLASSIFIER_SEED = 0  # the pilot classifier's seed (configs/phase3_*/generator_settings.yaml, pilot.classifier_seed)
DIAGNOSTICS = ("diversity_ratio", "memorization_fraction", "c2st_accuracy")
GENERATED = ("S", "V", "F")
SEPARATE = ("quality_tstr",)  # analysed apart (PREREGISTRATION §9)


def generator_runs(preview: bool = False) -> tuple[list[dict], list[str]]:
    """Every generator run behind the phase 3 result files, and the provisional result files left out."""
    found, skipped = [], []
    for experiment in workflow.GENERATOR_EXPERIMENTS:
        folder = paths.results_for(experiment)
        for path in sorted(folder.glob("[A-Z].json")) + sorted(folder.glob("[A-Z]_final.json")):
            result = json.loads(path.read_text())
            if result.get("provisional") and not preview:
                skipped.append(str(path.relative_to(paths.repo_root())))
                continue
            if result["mode"] == "pilot":
                pairs = [(setting, run_id) for setting, s in result["settings"].items() for run_id in s["run_ids"]]
            else:
                pairs = [(result["setting"], run_id) for run_id in result["run_ids"]]
            for setting, run_id in pairs:
                found.append({"run_id": run_id, "experiment": experiment, "family": result["family"], "variant": result["variant"],
                              "setting": setting, "mode": result["mode"], "provisional": bool(result.get("provisional"))})  # fmt: skip
    return found, skipped


def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def checkpoint_rows(run: dict) -> list[dict]:
    """The scored checkpoints of one run, with the class-mean diagnostics added."""
    run_dir = runs.find_run_dir(run["run_id"])
    path = run_dir / "checkpoints.csv" if run_dir else None
    if path is None or not path.is_file():
        return []
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for row in rows:
        values = {k: _number(v) for k, v in row.items() if k != "pilot_selection_metric"}
        values["pilot_selection_metric"] = row.get("pilot_selection_metric")
        for name in DIAGNOSTICS:
            values[f"{name}_mean"] = float(np.nanmean([values.get(f"{name}_{c}", np.nan) for c in GENERATED]))
        out.append({**run, **values})
    return out


def metric_columns(rows: list[dict]) -> list[str]:
    """The metrics to correlate: the composite's fidelity families, the class-mean diagnostics, then TSTR."""
    families = sorted({k for r in rows for k in r if k.startswith("quality_")})
    main = [f for f in families if f not in SEPARATE] + [f"{name}_mean" for name in DIAGNOSTICS]
    return main + [f for f in families if f in SEPARATE]


def reference_score() -> dict | None:
    """The phase 2 gate cnn's seed-0 run without synthetic beats: the zero of the gain axis."""
    cnn = (workflow.manifest().get(PHASE2_GATE) or {}).get("cnn")
    for run_id in (cnn or {}).get("run_ids", []):
        if runs.parse_run_id(run_id)["seed"] != CLASSIFIER_SEED:
            continue
        run_dir = runs.find_run_dir(run_id)
        if run_dir and (run_dir / "metrics_val.json").is_file():
            value = json.loads((run_dir / "metrics_val.json").read_text())["selection_value"]
            return {"run_id": run_id, "selection_value": float(value)}
    return None


def _rho(x: np.ndarray, y: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.ptp(x[keep]) == 0 or np.ptp(y[keep]) == 0:
        return float("nan")
    return float(spearmanr(x[keep], y[keep]).statistic)


def correlate(rows: list[dict], metric: str, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Spearman's rho of ``metric`` with the pilot score, and a run-level percentile bootstrap interval."""
    by_run: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        by_run.setdefault(r["run_id"], []).append((r.get(metric, np.nan), r.get("pilot_selection_value", np.nan)))
    groups = [np.asarray(v, dtype=np.float64) for v in by_run.values()]
    pooled = np.concatenate(groups) if groups else np.zeros((0, 2))
    rho = _rho(pooled[:, 0], pooled[:, 1])
    n_points = int((np.isfinite(pooled[:, 0]) & np.isfinite(pooled[:, 1])).sum())
    out = {"rho": rho, "n_points": n_points, "n_runs": len(groups), "ci_low": None, "ci_high": None, "excludes_zero": None, "frac_undefined": None}
    if len(groups) < 2 or not np.isfinite(rho):
        return out
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        sample = np.concatenate([groups[i] for i in rng.integers(0, len(groups), size=len(groups))])
        reps[b] = _rho(sample[:, 0], sample[:, 1])
    finite = reps[np.isfinite(reps)]
    if finite.size:
        low, high = np.percentile(finite, [2.5, 97.5])
        out.update(ci_low=float(low), ci_high=float(high), excludes_zero=bool(low > 0 or high < 0))
    out["frac_undefined"] = float(1 - finite.size / n_boot)
    return out


def analyse(preview: bool = False, n_boot: int = N_BOOT) -> dict:
    """The RQ3 summary; raises FileNotFoundError when no generator run has scored checkpoints yet."""
    found, skipped = generator_runs(preview)
    rows = [row for run in found for row in checkpoint_rows(run)]
    if not rows:
        raise FileNotFoundError("No scored generator checkpoints yet: run and ingest the phase 3 pilots first")
    reference = None if preview else reference_score()
    for row in rows:
        row["gain"] = row["pilot_selection_value"] - reference["selection_value"] if reference else float("nan")
    metrics = {m: {"group": "tstr" if m in SEPARATE else "fidelity and diversity", **correlate(rows, m, n_boot)} for m in metric_columns(rows)}
    with_rows = {r["run_id"] for r in rows}
    return {
        "question": "RQ3 / H3 (exploratory): Spearman rho between each quality metric and the downstream gain across generator checkpoints",
        "selection_metric": rows[0].get("pilot_selection_metric"), "n_boot": n_boot, "boot_seed": BOOT_SEED,
        "resampling_unit": "generator training run (variant x setting x seed)", "reference": reference,
        "provisional": preview or any(r["provisional"] for r in found), "skipped_provisional_results": skipped,
        "n_runs": len(with_rows), "n_checkpoints": len(rows), "runs": [r for r in found if r["run_id"] in with_rows],
        "runs_without_checkpoints": [r["run_id"] for r in found if r["run_id"] not in with_rows],
        "metrics": metrics, "rows": rows, "git_sha": runs.git_sha(),
    }  # fmt: skip


def write_outputs(summary: dict) -> None:
    out = paths.results_for(PHASE4_QUALITY_VS_GAIN)
    out.mkdir(parents=True, exist_ok=True)
    rows = summary.pop("rows")
    fields = ["run_id", "experiment", "family", "variant", "setting", "mode", "checkpoint", "step", "pilot_selection_value", "gain",
              *summary["metrics"]]  # fmt: skip
    with (out / "checkpoints.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    from ecgaug import figures

    figures.safe_call(figures.quality_vs_gain_figure, summary, rows)
    figures.safe_call(figures.quality_rho_figure, summary)
    summary["rows"] = rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--preview", action="store_true", help="include provisional generator results (marked provisional)")
    args = parser.parse_args(argv)
    try:
        summary = analyse(args.preview, args.n_boot)
    except FileNotFoundError as err:
        print(err)
        return 2
    write_outputs(summary)
    print(f"{summary['n_checkpoints']} checkpoints from {summary['n_runs']} runs" + (" (provisional)" if summary["provisional"] else ""))
    for name, m in summary["metrics"].items():
        ci = f"[{m['ci_low']:+.2f}, {m['ci_high']:+.2f}]" if m["ci_low"] is not None else "no interval"
        print(f"  {name:28s} rho {m['rho']:+.2f} {ci}  ({m['group']})")
    print(f"written: results/{PHASE4_QUALITY_VS_GAIN}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
