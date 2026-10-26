"""Phase 2 experiment phase2_quality_metrics: four families of synthetic-quality metrics, sanity-tested.

Usage:
    python -m ecgaug.experiments.phase2_quality_metrics --config configs/phase2_quality_metrics/A_dtw.yaml
    python -m ecgaug.experiments.phase2_quality_metrics --config configs/phase2_quality_metrics/A_dtw.yaml --smoke
    add --preview [--representation C] to run before the phase 1 gate (results are marked provisional)

No generator is involved: each family is scored on real held-out beats and on degraded copies of them
(``ecgaug.quality.sanity``), and passes if it ranks the real beats best, gets worse with every
degradation's severity and is stable across resamples. The passing families form the composite quality
rank, fixed at the phase 2 gate before any generator is evaluated (PLAN Phase 2). Variant C also trains
and saves the frozen feature network of the FID-analog metric (``ecgaug.quality.feature_net``).

Official runs need the phase 1 gate (the frozen dataset), the approved validation split and a clean
git tree. The test set is never read.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from ecgaug import figures, paths, quality, runs, workflow
from ecgaug.experiments.common import PreconditionError, check_preconditions, split_part
from ecgaug.experiments.data_source import DataSource, load_data, smoke_data
from ecgaug.generators.base import training_beats
from ecgaug.names import PHASE2_QUALITY_METRICS
from ecgaug.quality.feature_net import ensure_feature_net
from ecgaug.quality.sanity import score_family
from ecgaug.train.loop import resolve_device

EXPERIMENT = PHASE2_QUALITY_METRICS
VARIANT_KEYS = {"base", "variant", "description", "family"}
SHARED_KEYS = {"experiment", "resamples", "degradations", "pass_rule", "metrics", "feature_net"}


def load_config(path: Path) -> dict:
    path = Path(path)
    variant = yaml.safe_load(path.read_text())
    extra = set(variant) - VARIANT_KEYS
    if extra:
        raise ValueError(f"{path}: keys {sorted(extra)} belong in the shared settings file, not in a variant")
    shared = yaml.safe_load((path.parent / variant["base"]).read_text())
    if set(shared) != SHARED_KEYS:
        raise ValueError(f"{path.parent / variant['base']}: expected exactly the keys {sorted(SHARED_KEYS)}")
    if variant["family"] not in quality.METRIC_FAMILIES:
        raise ValueError(f"{path}: unknown metric family {variant['family']!r}; choose from {sorted(quality.METRIC_FAMILIES)}")
    return {**shared, "variant": str(variant["variant"]), "description": variant.get("description", ""),
            "family": variant["family"], "config_name": path.stem}  # fmt: skip


def run_variant(cfg: dict, split, data: DataSource, device_name: str | None = None, reasons: list[str] | None = None) -> dict:
    """Score the variant's family on every sanity set and write ``results/<experiment>/<letter>.json``."""
    reasons = list(reasons or [])
    device = str(resolve_device(device_name or "auto"))
    train = training_beats(data.arrays)
    val = split_part(data.arrays, "val", data.spec.rr_features)
    started = time.perf_counter()
    feature_net, net_info = None, None
    if cfg["family"] == "fid_analog":
        feature_net, net_info = ensure_feature_net(
            split_part(data.arrays, "train", False), split_part(data.arrays, "val", False), cfg["feature_net"],
            split.selection_metric, data.info.get("content_sha256", "unknown"), device,
        )  # fmt: skip
    report = score_family(
        cfg["family"], train, cfg["metrics"], cfg["degradations"], int(cfg["resamples"]), cfg["pass_rule"],
        val=val, spec=data.spec, selection=split.selection_metric, feature_net=feature_net, device=device,
    )  # fmt: skip
    result = {
        "experiment": EXPERIMENT, "variant": cfg["variant"], "description": cfg["description"], **report,
        "provisional": bool(reasons), "provisional_reasons": reasons, "dataset": data.info,
        "split": {"name": split.name, "val": list(split.val)}, "selection_metric": split.selection_metric,
        "settings": {"resamples": cfg["resamples"], "degradations": cfg["degradations"], "pass_rule": cfg["pass_rule"], "metrics": cfg["metrics"]},
        "feature_net": net_info, "wall_clock_s": time.perf_counter() - started, "device": device, "git_sha": runs.git_sha(),
    }  # fmt: skip
    out_dir = paths.results_for(EXPERIMENT)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{cfg['variant']}.json").write_text(json.dumps(result, indent=2) + "\n")
    notes = out_dir / f"{cfg['variant']}_notes.md"
    if not notes.exists():
        notes.write_text(f"# {EXPERIMENT}, variant {cfg['variant']} ({cfg['family']}): notes\n\nWhat worked:\n\nWhat failed:\n\nSurprises:\n\nFigure paths:\n")
    figures.safe_for_result(out_dir / f"{cfg['variant']}.json")
    return result


def smoke(cfg: dict) -> dict:
    """The variant on fake records in a temporary root with one resample and small settings."""
    small = {**cfg, "resamples": 2, "metrics": {**cfg["metrics"], "dtw_max_beats": 10, "fid_max_beats": 60,
             "tstr_classifier": {**cfg["metrics"]["tstr_classifier"], "max_epochs": 1}},
             "feature_net": {**cfg["feature_net"], "max_epochs": 1}}  # fmt: skip
    with smoke_data() as (split, data):
        result = run_variant(small, split, data, "cpu", [data.provisional])
        return {"variant": result["variant"], "family": result["family"], "sets_scored": len(result["scores"]),
                "checks": [c["check"].split(" (")[0] for c in result["checks"]], "passed_on_fake_data": result["passed"],
                "note": "fake data; the numbers of a smoke run mean nothing and are discarded"}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="phase2_quality_metrics: sanity-test a synthetic-quality metric family.")
    parser.add_argument("--config", required=True, help="configs/phase2_quality_metrics/<variant>.yaml")
    parser.add_argument("--device", default=None, help="cpu, cuda, mps or auto (default: auto)")
    parser.add_argument("--smoke", action="store_true", help="end-to-end run on fake data in a temp directory")
    parser.add_argument("--preview", action="store_true", help="run before the phase 1 gate; results are marked provisional")
    parser.add_argument("--representation", default=None, help="preview only: the phase1_input_representation variant letter")
    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config))
    if args.smoke:
        print(json.dumps(smoke(cfg), indent=2))
        return 0
    if args.representation and not args.preview:
        print("--representation is for previews only; official runs use the dataset frozen at the phase 1 gate")
        return 2
    try:
        if not args.preview and workflow.phase1_winner() is None:
            raise PreconditionError("Phase 1 is not finished: approve the phase 1 gate first (python -m ecgaug.workflow status), or add --preview.")
        split = check_preconditions()
        data = load_data(split, args.preview, args.representation)
    except PreconditionError as err:
        print(f"Refusing to run {EXPERIMENT} variant {cfg['variant']}: {err}")
        return 2
    result = run_variant(cfg, split, data, args.device, [data.provisional] if data.provisional else [])
    print(f"variant {result['variant']} ({result['family']}): {'PASSES' if result['passed'] else 'fails'} the sanity harness")
    for check in result["checks"]:
        print(f"  {'PASS' if check['passed'] else 'FAIL'}: {check['check']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
