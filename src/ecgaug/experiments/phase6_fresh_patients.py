"""phase6_fresh_patients: P18's frozen pipeline on 1,000 Icentia11k patients no analysis has used (PREREGISTRATION §10 v18,
P19; post hoc).

    python -m ecgaug.experiments.phase6_fresh_patients --config configs/phase6_fresh_patients/fresh_patients.yaml --part run [--device cuda]
    (--part build downloads and builds the beats only; --part evaluate scores beats that --part build saved)

P18's highest score (the cVAE's ten-run ensemble with Huang et al.'s SVEB stage and P16's AF veto, rare-class macro-F1 0.587)
is the best of 14 post hoc reads of the same 22 DS2 patients, and one of them (record 232) holds three quarters of DS2's S
beats. P19 asks whether the result holds on patients that played no part in any choice: the first segment (about 70
minutes) of 1,000 Icentia11k patients drawn at random from the 10,600 that P14 did not draw, every usable N, S and V beat
(Icentia11k has no F labels; Q beats are dropped as everywhere), in P13's form as P14 built them but without P14's cap on N
beats. P18's sixty runs, P16's AF detector with its threshold and form, and Huang's threshold are used as they are: nothing
is fitted, chosen or tuned. The primary measure is S F1 (there are no F labels); V F1 is reported with it. DS2 is not read.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

from ecgaug import paths, runs
from ecgaug.data import raw
from ecgaug.data.contract import content_sha256, sha256_file
from ecgaug.data.records import CLASS_INDEX, CLASSES, FS, QRS_SYMBOLS
from ecgaug.eval import metrics as M
from ecgaug.eval.bootstrap import cluster_bootstrap_confusion, holm, paired_cluster_bootstrap_diff_confusion
from ecgaug.experiments import phase6_af_rule as af
from ecgaug.experiments import phase6_icentia as ic
from ecgaug.experiments import phase6_patient_template as pt
from ecgaug.experiments import phase6_rr_reference as ref
from ecgaug.experiments import phase6_seed_ensemble as se
from ecgaug.experiments.common import PreconditionError, check_preconditions
from ecgaug.experiments.data_source import load_data
from ecgaug.names import PHASE6_FRESH_PATIENTS
from ecgaug.train.loop import predict_proba, resolve_device

EXPERIMENT = PHASE6_FRESH_PATIENTS
RULE = "PREREGISTRATION §10 v18"
CONFIG_KEYS = {"experiment", "seed_ensemble_config", "patients", "patient_seed"}
N_FRESH, FRESH_SEED = 1000, 1  # fixed by PREREGISTRATION v18
P14_PATIENTS, P14_SEED = 400, 0  # P14's sample, the only Icentia11k patients used before
ALL_BEATS = 10**9  # no cap on N beats: an evaluation keeps every usable beat
FIELDS = ("X", "y", "record", "r_idx", "rr_prev", "rr_next", "rr_ratio", "p_af")
STATS = ("f1_S", "f1_V", "se_S", "ppv_S", "n_recall")


def _stat(name: str):
    from ecgaug.final_eval import STATISTICS

    return STATISTICS[name]


def load_config(path: Path) -> dict:
    """P19's config: 1,000 patients with seed 1 (PREREGISTRATION v18); P18's config brings P16's and P13's."""
    cfg = yaml.safe_load(Path(path).read_text())
    if set(cfg) != CONFIG_KEYS or cfg["experiment"] != EXPERIMENT:
        raise ValueError(f"{path}: expected experiment {EXPERIMENT} and exactly the keys {sorted(CONFIG_KEYS)}")
    if (int(cfg["patients"]), int(cfg["patient_seed"])) != (N_FRESH, FRESH_SEED):
        raise ValueError(f"{path}: patients and patient_seed must be {N_FRESH} and {FRESH_SEED} (PREREGISTRATION v18)")
    ens = se.load_config(paths.repo_root() / cfg["seed_ensemble_config"])
    return {**cfg, "file": str(path), "ensemble": ens, "af": ens["af"], "rr": ens["rr"]}


# --------------------------------------------------------------------------------------------------
# the patients and their beats
# --------------------------------------------------------------------------------------------------


def used_patients() -> list[int]:
    """P14's patients, checked against its data record when one is present."""
    drawn = ic.sample_patients(P14_PATIENTS, P14_SEED)
    record = ic.data_record()
    if record is not None and record["patients"] != drawn:
        raise PreconditionError("P14's data record lists other patients than its seed draws")
    return drawn


def sample_patients(n: int, seed: int, exclude: list[int]) -> list[int]:
    """``n`` distinct patients drawn at random with ``seed`` from those not in ``exclude`` (no label is looked at), sorted."""
    pool = np.array(sorted(set(range(ic.N_PATIENTS)) - {int(p) for p in exclude}))
    if n > pool.size:
        raise ValueError(f"Only {pool.size} patients are left to draw from")
    return sorted(int(p) for p in np.random.default_rng(seed).choice(pool, size=n, replace=False))


def build_one(name: str, spec, half_window_s: float, detector: dict) -> tuple[dict | None, dict]:
    """One segment's usable N, S and V beats in P13's form (P14's ``build_record`` without the N cap) and each beat's AF
    probability from the segment's QRS annotation times."""
    try:
        arrays, report = ic.build_record(name, spec, ALL_BEATS, 0, half_window_s)
    except Exception as err:  # a malformed segment is left out and reported, as in P14
        return None, {"record": name, "number": ic.record_number(name), "excluded": f"{type(err).__name__}: {err}"}
    if arrays is None:
        return None, report
    samples, symbols, _, fs, _ = raw.read_external(ic.DB, name, ic.LEAD)
    qrs = np.sort(np.asarray(samples)[np.isin(symbols, sorted(QRS_SYMBOLS))])
    prob = af.af_probability(detector, qrs, fs)
    r = np.rint(qrs * FS / fs).astype(np.int64)
    pos = np.searchsorted(r, arrays["r_idx"])
    if not np.array_equal(r[np.clip(pos, 0, r.size - 1)], arrays["r_idx"]):
        raise PreconditionError(f"{name}: beats do not match its QRS annotations")
    out = {k: arrays[k] for k in FIELDS if k in arrays}
    out["p_af"] = prob[pos].astype(np.float32)
    return out, report


def _build_star(args: tuple) -> tuple[dict | None, dict]:
    return build_one(*args)


def build(cfg: dict, spec, workers: int | None = None) -> tuple[dict, dict]:
    """The sampled patients' beats and a data record (patients, exclusions, class counts, hashes); no model is involved."""
    patients = sample_patients(N_FRESH, FRESH_SEED, used_patients())
    arrays, info = build_set(patients, spec, float(cfg["rr"]["rg"]["mp"]["pt"]["half_window_s"]), workers)
    record = {"experiment": EXPERIMENT, "rule": RULE, "patients": patients, "used_before": P14_PATIENTS, **info, "git_sha": runs.git_sha()}
    out = paths.results_for(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.json").write_text(json.dumps(record, indent=2, default=float) + "\n")
    return arrays, record


def build_set(patients: list[int], spec, half: float, workers: int | None = None) -> tuple[dict, dict]:
    """Every usable beat of these patients' first segments (``build_one``, in patient order), and what was built:
    segments used, exclusions, class counts, S concentration, the content hash and the raw files' hashes."""
    names = ic.download(patients, raw.external_dir(ic.DB))
    detector = af.load_detector()
    jobs = [(ic.segment_name(p), spec, half, detector) for p in patients if ic.segment_name(p) in names]
    reports = [{"record": ic.segment_name(p), "number": ic.RECORD_OFFSET + p, "excluded": "download failed"} for p in patients if ic.segment_name(p) not in names]
    parts = []
    workers = workers or max(1, (os.cpu_count() or 1))
    if workers == 1:
        results = [build_one(*job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_build_star, jobs, chunksize=4))
    for arrays, report in results:
        reports.append(report)
        if arrays is not None:
            parts.append(arrays)
    if not parts:
        raise PreconditionError("No Icentia11k segment was usable")
    arrays = {k: np.concatenate([p[k] for p in parts]) for k in FIELDS}
    folder = raw.external_dir(ic.DB)
    files = {f"{n}.{e}": sha256_file(folder / f"{n}.{e}") for n in names for e in ("hea", "atr", "dat")}
    y, rec = np.asarray(arrays["y"]), np.asarray(arrays["record"])
    per = {int(r): {c: int(((rec == r) & (y == i)).sum()) for i, c in enumerate(CLASSES)} for r in np.unique(rec)}
    s_total = sum(v["S"] for v in per.values())
    info = {"segments_used": len(parts), "excluded": [r for r in reports if "excluded" in r],
            "beats": {c: int((y == i).sum()) for i, c in enumerate(CLASSES)},
            "patients_with": {c: sum(v[c] > 0 for v in per.values()) for c in ("S", "V")},
            "s_dominant_patients": sum(v["S"] > v["N"] for v in per.values()),
            "largest_s_share": max(v["S"] for v in per.values()) / s_total if s_total else 0.0,
            "content_sha256": content_sha256({k: arrays[k] for k in FIELDS}), "files_sha256": files}  # fmt: skip
    return arrays, info


def beats_path() -> Path:
    return paths.processed_dir() / "fresh_icentia" / "beats.npz"


def load_beats() -> dict:
    """Beats that ``--part build`` saved, checked against data.json."""
    record_path = paths.results_for(EXPERIMENT) / "data.json"
    if not beats_path().is_file() or not record_path.is_file():
        raise PreconditionError("Run --part build first")
    with np.load(beats_path()) as z:
        arrays = {k: z[k] for k in FIELDS}
    if content_sha256(arrays) != json.loads(record_path.read_text())["content_sha256"]:
        raise PreconditionError("The saved beats do not match data.json")
    return arrays


# --------------------------------------------------------------------------------------------------
# scoring: P18's frozen runs, nothing chosen
# --------------------------------------------------------------------------------------------------


def predict(run_id: str, arrays: dict, device: torch.device) -> np.ndarray:
    """One run's class probabilities for every beat."""
    model = pt.load_run_classifier(run_id, True).to(device)
    x = torch.from_numpy(np.asarray(arrays["X"], dtype=np.float32))
    side = torch.from_numpy(np.stack([np.asarray(arrays[f], dtype=np.float32) for f in pt.SIDE_FIELDS], axis=1))
    return predict_proba(model, x, side, device, batch_size=8192)


def frozen_members() -> dict[str, list[str]]:
    """P18's ten run IDs per condition (members.json), checked as P18's test read checked them."""
    m = se.load_members()["members"]
    if tuple(m) != se.CONDITIONS or any(len(ids) != len(se.EXISTING_SEEDS) + len(se.NEW_SEEDS) for ids in m.values()):
        raise PreconditionError("P18's members.json must list ten runs for each of its six conditions")
    if {c: ids[: len(se.EXISTING_SEEDS)] for c, ids in m.items()} != se.existing_runs():
        raise PreconditionError("P18's members.json does not start with P13's and P15's selected runs")
    return m


def _cms(decision: np.ndarray, y: np.ndarray, records: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return M.confusion_by_record(y, np.asarray(decision).argmax(axis=1), records)


def _block(table: dict[str, np.ndarray], records: np.ndarray) -> dict:
    return {f"cnn/{c}": {s: cluster_bootstrap_confusion(_stat(s), cm, records, n=ref.N_BOOT, seed=ref.BOOT_SEED) for s in STATS} for c, cm in table.items()}


def _family(items: dict[str, tuple[np.ndarray, np.ndarray]], stat: str = "f1_S") -> dict:
    out = {k: paired_cluster_bootstrap_diff_confusion(_stat(stat), a, b, n=ref.N_BOOT, seed=ref.BOOT_SEED) for k, (a, b) in items.items()}
    adjusted = holm({k: v["p_two_sided"] for k, v in out.items()})
    for k, v in out.items():
        v.update(p_holm=adjusted[k]["p_adjusted"], reject_holm=adjusted[k]["reject"])
    return out


def score(cfg: dict, arrays: dict, predict_fn) -> dict:
    """Per condition: each of the ten runs, the three-run (seeds 0-2) and the ten-run ensemble, the cnn alone, with Huang's
    stage and with the stage and P16's veto; per-patient confusion matrices, record-level bootstrap. Families on S F1:
    confirmatory (Holm, 6) the ten-run ensemble vs P16's single runs, both with the stage and the veto; secondary (Holm, 5)
    the cVAE's ensemble vs each other condition's. Probabilities are streamed run by run."""
    members = frozen_members()
    af_t, form = af.chosen_design()
    t_rule = float(cfg["af"]["rule_threshold"])
    y, records = np.asarray(arrays["y"]), np.asarray(arrays["record"])
    rr_prev, p_af = np.asarray(arrays["rr_prev"]), np.asarray(arrays["p_af"])
    single = {k: {} for k in se.KINDS}
    ten = {k: {} for k in se.KINDS}
    three = {k: {} for k in se.KINDS}
    f_calls: dict[str, float] = {}
    unique = None
    for c, ids in members.items():
        total, per_run = None, {k: [] for k in se.KINDS}
        for i, run_id in enumerate(ids):
            p = np.asarray(predict_fn(run_id, arrays), dtype=np.float64)
            d = se.decisions(p, rr_prev, p_af, t_rule, af_t, form)
            for k in se.KINDS:
                unique, cm = _cms(d[k], y, records)
                per_run[k].append(cm)
            total = p if total is None else total + p
            if i + 1 == len(se.EXISTING_SEEDS):
                d3 = se.decisions(total / (i + 1), rr_prev, p_af, t_rule, af_t, form)
                for k in se.KINDS:
                    three[k][c] = _cms(d3[k], y, records)[1][None]
        d10 = se.decisions(total / len(ids), rr_prev, p_af, t_rule, af_t, form)
        f_calls[f"cnn/{c}"] = float((d10["veto"].argmax(axis=1) == CLASS_INDEX["F"]).mean())
        for k in se.KINDS:
            single[k][c] = np.stack(per_run[k])
            ten[k][c] = _cms(d10[k], y, records)[1][None]
    first = len(se.EXISTING_SEEDS)
    p16 = {k: {c: cm[:first] for c, cm in single[k].items()} for k in se.KINDS}
    stages = {f"cnn/{c}: {k} vs cnn alone": (ten[k][c], ten["cnn"][c]) for c in members for k in ("rule", "veto")}
    return {
        "experiment": EXPERIMENT, "post_hoc": True, "rule": RULE, "n_boot": ref.N_BOOT, "boot_seed": ref.BOOT_SEED,
        "rule_threshold": t_rule, "af_threshold": af_t, "veto_form": form, "patients": int(unique.size),
        "beats": {c: int((y == i).sum()) for i, c in enumerate(CLASSES)},
        "ensemble": {"with_af_veto": _block(ten["veto"], unique), "rule_alone": _block(ten["rule"], unique), "cnn_alone": _block(ten["cnn"], unique)},
        "p16": {"with_af_veto": _block(p16["veto"], unique), "cnn_alone": _block(p16["cnn"], unique)},
        "three_run_ensemble": {"with_af_veto": _block(three["veto"], unique)},
        "per_run_points": {f"cnn/{c}": [float(v) for v in _stat("f1_S")(cm.sum(axis=1))] for c, cm in single["veto"].items()},
        "families": {"confirmatory": _family({f"cnn: {c}, ensemble vs P16": (ten["veto"][c], p16["veto"][c]) for c in members}),
                     "secondary": _family({f"cnn: {se.REFERENCE} vs {c}, ensembles": (ten["veto"][se.REFERENCE], ten["veto"][c])
                                           for c in members if c != se.REFERENCE})},
        "stages": {name: paired_cluster_bootstrap_diff_confusion(_stat("f1_S"), a, b, n=ref.N_BOOT, seed=ref.BOOT_SEED) for name, (a, b) in stages.items()},
        "f_calls_share": f_calls,
        "note": "S F1 is the primary measure (Icentia11k has no F labels); every comparison family is on S F1; the stage rows are descriptive",
        "run_ids": members, "git_sha": runs.git_sha(),
    }  # fmt: skip


def evaluate(cfg: dict, arrays: dict, device: torch.device) -> dict:
    summary = score(cfg, arrays, lambda run_id, a: predict(run_id, a, device))
    record = paths.results_for(EXPERIMENT) / "data.json"
    if record.is_file():
        summary["data"] = {k: v for k, v in json.loads(record.read_text()).items() if k not in ("files_sha256", "excluded", "patients")}
    (paths.results_for(EXPERIMENT) / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--part", choices=["build", "evaluate", "run"], required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config))
    try:
        spec = load_data(check_preconditions(), preview=False).spec
        if args.part in ("build", "run"):
            arrays, record = build(cfg, spec)
            print(json.dumps({k: record[k] for k in ("segments_used", "beats", "patients_with", "s_dominant_patients", "largest_s_share")}))
            if args.part == "build":
                beats_path().parent.mkdir(parents=True, exist_ok=True)
                np.savez(beats_path(), **arrays)
                return 0
        else:
            arrays = load_beats()
        s = evaluate(cfg, arrays, resolve_device(args.device))
        for family, items in s["families"].items():
            for name, c in items.items():
                print(f"  {family:12s} {name:40s} {c['point']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] Holm p {c['p_holm']:.3f}")
        for key, e in s["ensemble"]["with_af_veto"].items():
            print(f"  {key:26s} S F1 {e['f1_S']['point']:.3f}  V F1 {e['f1_V']['point']:.3f}  (cnn alone S F1 {s['ensemble']['cnn_alone'][key]['f1_S']['point']:.3f})")
    except PreconditionError as err:
        print(f"Refusing to run: {err}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
