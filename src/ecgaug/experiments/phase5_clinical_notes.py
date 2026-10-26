"""Phase 5 experiment phase5_clinical_notes: grounded notes for classified beats (PLAN Phase 5).

Usage:
    python -m ecgaug.experiments.phase5_clinical_notes --config configs/phase5_clinical_notes/A_template.yaml
    python -m ecgaug.experiments.phase5_clinical_notes --config configs/phase5_clinical_notes/D_llm_fact_check_loop.yaml
    python -m ecgaug.experiments.phase5_clinical_notes --rating-sheet     # blinded notes of every variant for the team's ratings
    python -m ecgaug.experiments.phase5_clinical_notes --config ... --smoke

Every variant writes a note for the same 50 validation cases (stratified by true class, at least 10 of
low confidence), built once from the phase 4 gate's model (``model`` in the shared settings) and its
saliency, and saved in ``results/phase5_clinical_notes/cases.json``. Each note is fact-checked
(``ecgaug.report.notes.check_note``). The hard gate is zero unsupported claims over the 50 notes; then
coverage of the required parts, the team's blinded readability ratings, latency and cost.

The LLM variants need ``LLM_API_KEY`` (and optionally ``ECGAUG_LLM_MODEL``); without it they are
recorded as not run. The test set is never read.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ecgaug import paths, runs, workflow
from ecgaug.data.records import CLASSES
from ecgaug.experiments.common import PreconditionError, check_preconditions, load_run_classifier, split_part
from ecgaug.experiments.data_source import load_data
from ecgaug.explain.evaluate import segment_masses
from ecgaug.explain.methods import saliency
from ecgaug.names import PHASE4_GATE, PHASE5_CLINICAL_NOTES, PHASE5_GATE
from ecgaug.report.notes import check_note, client_from_env, generate, note_input
from ecgaug.train.loop import predict_proba

EXPERIMENT = PHASE5_CLINICAL_NOTES
VARIANT_KEYS = {"base", "variant", "description", "generator"}
SHARED_KEYS = {"experiment", "n_cases", "min_low_confidence", "low_confidence_threshold", "case_seed", "model", "saliency_method",
               "method_settings", "max_retries"}  # fmt: skip


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


def choose_cases(y: np.ndarray, top_probability: np.ndarray, n_cases: int, min_low: int, threshold: float, seed: int) -> np.ndarray:
    """Indices stratified by true class (equal shares, as available), with at least ``min_low`` low-confidence ones."""
    rng = np.random.default_rng(seed)
    low = top_probability < threshold
    chosen: list[int] = []
    low_pool = rng.permutation(np.flatnonzero(low))
    chosen += list(low_pool[:min_low])  # the low-confidence quota first, from any class
    classes = [i for i in range(len(CLASSES)) if (y == i).any()]
    share = int(np.ceil((n_cases - len(chosen)) / max(1, len(classes))))
    for i in classes:
        pool = [j for j in rng.permutation(np.flatnonzero(y == i)) if j not in set(chosen)]
        chosen += pool[:share]
    return np.sort(np.asarray(chosen[:n_cases], dtype=np.int64))


def build_cases(cfg: dict, data, run_id: str, method: str) -> list[dict]:
    """The note inputs of the evaluation set: the model's validation outputs and saliency."""
    rr = data.spec.rr_features
    model = load_run_classifier(run_id, rr)
    val = split_part(data.arrays, "val", rr)
    side = torch.from_numpy(val["side"]) if rr else None
    probs = predict_proba(model, torch.from_numpy(val["X"]), side, torch.device("cpu"))
    idx = choose_cases(val["y"], probs.max(axis=1), int(cfg["n_cases"]), int(cfg["min_low_confidence"]), float(cfg["low_confidence_threshold"]), int(cfg["case_seed"]))
    maps = saliency(method, model, val["X"][idx], val["side"][idx] if rr else None, {**cfg["method_settings"], "fs": data.spec.effective_fs})
    masses = segment_masses(maps, data.spec.window_samples[0], data.spec.effective_fs)
    cases = []
    for k, i in enumerate(idx):
        payload = note_input(f"case-{k:02d}", {c: float(probs[i, j]) for j, c in enumerate(CLASSES)}, {s: float(m[k]) for s, m in masses.items()},
                             run_id, float(cfg["low_confidence_threshold"]))  # fmt: skip
        cases.append({"payload": payload, "true_class": CLASSES[int(val["y"][i])], "record": int(val["record"][i]), "r_idx": int(val["r_idx"][i])})
    return cases


def cases_path() -> Path:
    return paths.results_for(EXPERIMENT) / "cases.json"


def run_variant(cfg: dict, cases: list[dict], client, reasons: list[str]) -> dict:
    """Notes for every case, fact-checked; write ``results/<experiment>/<letter>.json`` and the notes."""
    variant = cfg["generator"]
    out = paths.results_for(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    if variant != "A" and client is None:
        result = {"experiment": EXPERIMENT, "variant": cfg["variant"], "generator": variant, "not_run": True,
                  "reason": "no LLM configured (set LLM_API_KEY)", "provisional": bool(reasons), "provisional_reasons": reasons}  # fmt: skip
        (out / f"{cfg['variant']}.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    notes, started = [], time.perf_counter()
    for case in cases:
        made = generate(variant, case["payload"], client, int(cfg["max_retries"]))
        report = check_note(made["note"], case["payload"])
        notes.append({"case": case["payload"]["record_anon_id"], **made, **report})
    used = {v: sum(n["variant_used"] == v for n in notes) for v in "ABCD"}
    result = {
        "experiment": EXPERIMENT, "variant": cfg["variant"], "description": cfg["description"], "generator": variant, "not_run": False,
        "n_cases": len(cases), "n_low_confidence": sum(c["payload"]["low_confidence"] for c in cases),
        "unsupported_claims": int(sum(n["unsupported_claims"] for n in notes)), "notes_with_issues": int(sum(bool(n["issues"]) for n in notes)),
        "passes_hard_gate": bool(sum(n["unsupported_claims"] for n in notes) == 0),
        "coverage": float(np.mean([n["coverage_fraction"] for n in notes])), "fell_back_to_template": int(sum(n["fell_back"] for n in notes)),
        "variant_used": used, "latency_s_mean": float(np.mean([n["latency_s"] for n in notes])),
        "input_tokens": int(sum(n["input_tokens"] for n in notes)), "output_tokens": int(sum(n["output_tokens"] for n in notes)),
        "model": getattr(client, "model", None), "wall_clock_s": time.perf_counter() - started,
        "provisional": bool(reasons), "provisional_reasons": reasons, "git_sha": runs.git_sha(),
    }  # fmt: skip
    (out / f"{cfg['variant']}.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / f"{cfg['variant']}_notes.json").write_text(json.dumps(notes, indent=2) + "\n")
    return result


def rating_sheet(seed: int = 0) -> Path:
    """Every variant's notes shuffled into one blinded CSV for the team, plus the key (kept apart)."""
    out = paths.results_for(EXPERIMENT)
    rows = []
    for path in sorted(out.glob("[A-D]_notes.json")):
        letter = path.name[0]
        rows += [{"variant": letter, "case": n["case"], "note": n["note"]} for n in json.loads(path.read_text())]
    random.Random(seed).shuffle(rows)
    sheet, key = out / "rating" / "notes_to_rate.csv", out / "rating" / "key.csv"
    sheet.parent.mkdir(parents=True, exist_ok=True)
    with sheet.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["item", "note", "clarity_1_5", "accuracy_of_emphasis_1_5", "brevity_1_5", "rater"])
        writer.writerows([[i, r["note"], "", "", "", ""] for i, r in enumerate(rows)])
    with key.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["item", "variant", "case"])
        writer.writerows([[i, r["variant"], r["case"]] for i, r in enumerate(rows)])
    return sheet


def notes_model() -> tuple[str, str]:
    """The run whose outputs the notes describe, and the saliency method (phase 4 and phase 5 gates)."""
    gate = workflow.manifest().get(PHASE4_GATE)
    if not gate:
        raise PreconditionError("The phase 4 gate has not fixed the models yet")
    settings = yaml.safe_load((paths.configs_dir() / EXPERIMENT / "shared_settings.yaml").read_text())
    arch = settings["model"]["architecture"]
    condition = settings["model"]["condition"]
    if condition == "best_generative":
        condition = gate["best_generative"][arch]
    run_ids = gate["chosen"][arch][condition]["run_ids"]
    values = {r: json.loads((runs.find_run_dir(r) / "metrics_val.json").read_text())["selection_value"] for r in run_ids}
    method = ((workflow.manifest().get(PHASE5_GATE) or {}).get("explainability") or {}).get(arch, settings["saliency_method"])
    return max(values, key=values.get), method


def smoke(cfg: dict) -> dict:
    """Template and fact checker on generated inputs (no model, no network)."""
    rng = np.random.default_rng(0)
    cases = []
    for k in range(12):
        p = rng.dirichlet(np.ones(4))
        cases.append({"payload": note_input(f"case-{k:02d}", dict(zip(CLASSES, p)), dict(zip(("P", "QRS", "ST", "T"), rng.dirichlet(np.ones(4)))),
                                            "smoke-model", float(cfg["low_confidence_threshold"]))})  # fmt: skip
    from ecgaug.experiments.common import temporary_root

    with temporary_root():
        result = run_variant(cfg, cases, None, ["smoke run"])
    return {"variant": result["variant"], "not_run": result["not_run"], "unsupported_claims": result.get("unsupported_claims"),
            "coverage": result.get("coverage"), "note": "generated inputs; the numbers mean nothing"}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="phase5_clinical_notes: fact-checked notes for 50 validation cases.")
    parser.add_argument("--config", help="configs/phase5_clinical_notes/<variant>.yaml")
    parser.add_argument("--smoke", action="store_true", help="template and fact checker on generated inputs")
    parser.add_argument("--rating-sheet", action="store_true", help="write the blinded rating sheet from every variant's notes")
    args = parser.parse_args(argv)
    if args.rating_sheet:
        print(f"written: {rating_sheet().relative_to(paths.repo_root())} (the key is next to it; keep it from the raters)")
        return 0
    if not args.config:
        parser.error("--config is required")
    cfg = load_config(Path(args.config))
    if args.smoke:
        print(json.dumps(smoke(cfg), indent=2))
        return 0
    try:
        if workflow.phase1_winner() is None:
            raise PreconditionError("Phase 1 is not finished: approve the phase 1 gate first.")
        split = check_preconditions()
        run_id, method = notes_model()
        if cases_path().is_file():
            cases = json.loads(cases_path().read_text())["cases"]
        else:
            cases = build_cases(cfg, load_data(split, preview=False), run_id, method)
            cases_path().parent.mkdir(parents=True, exist_ok=True)
            cases_path().write_text(json.dumps({"run_id": run_id, "saliency_method": method, "cases": cases}, indent=2) + "\n")
    except (PreconditionError, FileNotFoundError) as err:
        print(f"Refusing to run {EXPERIMENT} variant {cfg['variant']}: {err}")
        return 2
    result = run_variant(cfg, cases, client_from_env(), [])
    if result["not_run"]:
        print(f"variant {result['variant']}: not run ({result['reason']})")
    else:
        print(f"variant {result['variant']}: {result['unsupported_claims']} unsupported claims in {result['n_cases']} notes "
              f"({'passes' if result['passes_hard_gate'] else 'FAILS'} the hard gate); coverage {result['coverage']:.2f}; "
              f"fell back to the template {result['fell_back_to_template']} times")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
