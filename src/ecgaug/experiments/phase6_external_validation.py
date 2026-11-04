"""phase6_external_validation: the frozen models on two databases never used before (PREREGISTRATION §10 v4, P2; post hoc).

    python -m ecgaug.experiments.phase6_external_validation --part data       # download check and beat arrays
    python -m ecgaug.experiments.phase6_external_validation --part evaluate   # the 36 checkpoints and the two references

INCART (``incartdb``, primary: lead II, 257 Hz) and SVDB (``svdb``, secondary: signal ECG1, 128 Hz), from
PhysioNet into ``data/raw/<database>/``. Each record is read by lead name, resampled to 360 Hz (polyphase) and
cut into beats exactly as the frozen dataset was (the AAMI map, Q policy, usable-beat rule and representation of
``data/processed/v1/contract.json``); RR features are in seconds from the annotations at the native rate, so
resampling cannot move them. INCART records with paced beats are left out (rule), and any record whose annotations
hold a symbol outside the WFDB table is reported and left out.

Nothing is trained, chosen or tuned: the checkpoints are the final evaluation's (hash-checked) and the references
are the phase6_rr_reference fits. Per database: every condition with record-level bootstrap intervals and the 4
confirmatory comparisons with Holm within the database, on rare-class macro-F1, or on S F1 when the database holds
fewer than 50 usable F beats (rule fixed before any model saw the data). "Replicated" = the same Holm verdict as on DS2.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import scipy.signal as sps
import torch

from ecgaug import paths, runs, workflow
from ecgaug.data.beats import UNMAPPED, rr_intervals
from ecgaug.data.contract import content_sha256, save_npz_deterministic
from ecgaug.data.records import AAMI, CLASS_INDEX, CLASSES, FS, NON_BEAT_SYMBOLS, QRS_SYMBOLS, SYMBOL_TO_AAMI
from ecgaug.data.representation import RepresentationSpec, transform_record, usable_mask
from ecgaug.eval import metrics as M
from ecgaug.eval.bootstrap import holm, paired_cluster_bootstrap_diff_confusion
from ecgaug.experiments import phase6_rr_reference as ref
from ecgaug.experiments.common import SIDE_FIELDS, PreconditionError, load_run_classifier
from ecgaug.names import FINAL_EVAL, PHASE4_GATE, PHASE6_EXTERNAL_VALIDATION
from ecgaug.train.loop import predict_proba, resolve_device

EXPERIMENT = PHASE6_EXTERNAL_VALIDATION
DATABASES = {"incartdb": {"lead": "II", "role": "primary", "exclude_paced": True}, "svdb": {"lead": "ECG1", "role": "secondary", "exclude_paced": False}}
MIN_F_BEATS = 50
PACED = set(AAMI["Q"]) - {"Q"}  # '/' paced and 'f' fusion of paced and normal


def raw_dir(db: str) -> Path:
    return paths.raw_dir().parent / db  # next to data/raw/mitdb


def record_names(db: str) -> list[str]:
    folder = raw_dir(db)
    if not folder.is_dir():
        raise PreconditionError(f"{folder} is missing: python -c \"import wfdb; wfdb.dl_database('{db}', '{folder}')\"")
    return sorted(p.stem for p in folder.glob("*.hea"))


def record_number(name: str) -> int:
    """INCART's I01 -> 1, SVDB's 800 -> 800."""
    return int("".join(ch for ch in name if ch.isdigit()))


def resample_to_fs(signal: np.ndarray, fs: int) -> np.ndarray:
    divisor = math.gcd(FS, fs)
    return sps.resample_poly(np.asarray(signal, dtype=np.float64), FS // divisor, fs // divisor)


def build_record(db: str, name: str, spec: RepresentationSpec) -> tuple[dict | None, dict]:
    """Contract-format arrays of one external record (or None when it is left out) and its report."""
    import wfdb

    stem = str(raw_dir(db) / name)
    ann = wfdb.rdann(stem, "atr")
    samples, symbols = np.asarray(ann.sample, dtype=np.int64), np.asarray(ann.symbol, dtype=str)
    report: dict = {"record": name, "number": record_number(name)}
    unknown = sorted(set(symbols.tolist()) - QRS_SYMBOLS - NON_BEAT_SYMBOLS)
    if unknown:
        return None, {**report, "excluded": f"unknown annotation symbols {unknown}"}
    if DATABASES[db]["exclude_paced"] and np.isin(symbols, sorted(PACED)).any():
        return None, {**report, "excluded": "paced beats"}
    rec = wfdb.rdrecord(stem, channel_names=[DATABASES[db]["lead"]])
    if rec.sig_name != [DATABASES[db]["lead"]]:
        return None, {**report, "excluded": f"lead {DATABASES[db]['lead']} not in {rec.sig_name}"}
    fs = int(rec.fs)
    is_qrs = np.isin(symbols, sorted(QRS_SYMBOLS))
    r_native, symbol = samples[is_qrs], symbols[is_qrs]
    rr_prev, rr_next, rr_ratio = rr_intervals(r_native, fs=fs)
    signal = resample_to_fs(np.nan_to_num(rec.p_signal[:, 0]), fs)
    r = np.rint(r_native * FS / fs).astype(np.int64)
    aami = np.array([SYMBOL_TO_AAMI.get(s, UNMAPPED) for s in symbol], dtype=str)
    y = np.array([CLASS_INDEX.get(c, -1) for c in aami], dtype=np.int8)
    keep = usable_mask(r, signal.size, rr_prev, rr_next) & (y >= 0)
    n = int(keep.sum())
    arrays = {
        "X": transform_record(signal, r[keep], spec), "y": y[keep], "record": np.full(n, record_number(name), dtype=np.int16),
        "split": np.full(n, "test", dtype="<U5"), "r_idx": r[keep].astype(np.int32), "rr_prev": rr_prev[keep].astype(np.float32),
        "rr_next": rr_next[keep].astype(np.float32), "rr_ratio": rr_ratio[keep].astype(np.float32),
        "is_synthetic": np.zeros(n, dtype=bool), "source": np.full(n, "real", dtype="<U4"),
    }  # fmt: skip
    report.update(fs=fs, lead=DATABASES[db]["lead"], kept={c: int((arrays["y"] == i).sum()) for i, c in enumerate(CLASSES)},
                  q_or_unmapped=int((y < 0).sum()), dropped_unusable=int(((y >= 0) & ~keep).sum()))  # fmt: skip
    return arrays, report


def frozen_spec() -> RepresentationSpec:
    rep = json.loads((paths.processed_dir() / "v1" / "contract.json").read_text())["representation"]
    return RepresentationSpec(**{k: rep[k] for k in RepresentationSpec.__dataclass_fields__ if k in rep})


def build_database(db: str) -> dict:
    """All usable beats of a database, saved under ``data/processed/external/``; returns the data report."""
    spec = frozen_spec()
    parts, reports = [], []
    for name in record_names(db):
        arrays, report = build_record(db, name, spec)
        reports.append(report)
        if arrays is not None and arrays["y"].size:
            parts.append(arrays)
    arrays = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if not np.isfinite(arrays["X"]).all():
        raise ValueError(f"{db}: non-finite values in X")
    out = paths.processed_dir() / "external"
    out.mkdir(parents=True, exist_ok=True)
    save_npz_deterministic(out / f"{db}.npz", arrays)
    counts = {c: int((arrays["y"] == i).sum()) for i, c in enumerate(CLASSES)}
    data = {"database": db, **DATABASES[db], "representation": spec.name, "records_used": len(parts), "records_excluded": [r for r in reports if "excluded" in r],
            "beats": counts, "primary_metric": "rare_macro_f1" if counts["F"] >= MIN_F_BEATS else "f1_S", "content_sha256": content_sha256(arrays), "records": reports}  # fmt: skip
    result_dir = paths.results_for(EXPERIMENT)
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / f"{db}_data.json").write_text(json.dumps(data, indent=2) + "\n")
    return data


def load_database(db: str) -> tuple[dict, dict]:
    data = json.loads((paths.results_for(EXPERIMENT) / f"{db}_data.json").read_text())
    with np.load(paths.processed_dir() / "external" / f"{db}.npz", allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    if content_sha256(arrays) != data["content_sha256"]:
        raise PreconditionError(f"data/processed/external/{db}.npz changed since it was built")
    return arrays, data


def evaluate_database(db: str, device: torch.device) -> dict:
    """Every frozen checkpoint and both references on one database, with the replicated confirmatory family."""
    from ecgaug import final_eval

    arrays, data = load_database(db)
    info = final_eval.plan()
    bad = [c["run_id"] for c in info["checkpoints"] if not c["hash_matches"]]
    if bad:
        raise PreconditionError(f"Checkpoints missing or changed since the phase 4 gate: {bad[:3]}")
    gate = info["gate"]
    rr = bool(json.loads((paths.processed_dir() / "v1" / "contract.json").read_text())["representation"]["rr_features"])
    x = torch.from_numpy(arrays["X"])
    side = torch.from_numpy(np.stack([arrays[f] for f in SIDE_FIELDS], axis=1)) if rr else None
    y, records = arrays["y"], arrays["record"]
    cms = {}
    for c in info["checkpoints"]:
        probs = predict_proba(load_run_classifier(c["run_id"], rr), x, side, device)
        unique, cm = final_eval.per_record_cms([probs], y, records)
        cms.setdefault((c["architecture"], c["condition"]), []).append(cm[0])
    cms = {k: np.stack(v) for k, v in cms.items()}
    primary = data["primary_metric"]
    stat = final_eval.STATISTICS[primary]
    conditions = {f"{a}/{c}": ref.condition_summary(cm, unique) | {"strength": gate["chosen"][a][c]["strength"]} for (a, c), cm in cms.items()}
    references = {}
    for reference, model in ref.load_fitted().items():
        _, cm = final_eval.per_record_cms([ref.predict(model, arrays, reference)], y, records)
        references[reference] = ref.condition_summary(cm, unique)
    family = {}
    for arch, a, b in final_eval.comparisons(gate)["confirmatory"]:
        family[f"{arch}: {a} vs {b}"] = paired_cluster_bootstrap_diff_confusion(stat, cms[(arch, a)], cms[(arch, b)], n=ref.N_BOOT, seed=ref.BOOT_SEED)
    adjusted = holm({k: v["p_two_sided"] for k, v in family.items()})
    ds2 = json.loads((paths.results_for(FINAL_EVAL) / "summary.json").read_text())["families"]["confirmatory"]
    for key, value in family.items():
        value.update(p_holm=adjusted[key]["p_adjusted"], reject_holm=adjusted[key]["reject"], ds2_reject_holm=ds2[key]["reject_holm"],
                     replicated=adjusted[key]["reject"] == ds2[key]["reject_holm"])  # fmt: skip
    summary = {"experiment": EXPERIMENT, "post_hoc": True, "rule": "PREREGISTRATION §10 v4, P2", "database": db, "role": data["role"],
               "primary_metric": primary, "beats": data["beats"], "records": len(unique), "n_boot": ref.N_BOOT, "conditions": conditions,
               "references": references, "confirmatory": family, "run_ids": [c["run_id"] for c in info["checkpoints"]], "git_sha": runs.git_sha()}  # fmt: skip
    (paths.results_for(EXPERIMENT) / f"{db}_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=["data", "evaluate"], required=True)
    parser.add_argument("--database", choices=[*DATABASES, "all"], default="all")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    dbs = list(DATABASES) if args.database == "all" else [args.database]
    try:
        for db in dbs:
            if args.part == "data":
                d = build_database(db)
                print(f"{db}: {d['records_used']} records, beats {d['beats']}, excluded {len(d['records_excluded'])}; primary metric {d['primary_metric']}")
            else:
                s = evaluate_database(db, resolve_device(args.device))
                metric = s["primary_metric"]
                for key, c in s["conditions"].items():
                    print(f"  {db} {key:28s} {metric} {c[metric]['point']:.3f} [{c[metric]['ci_low']:.3f}, {c[metric]['ci_high']:.3f}]")
                for key, c in s["references"].items():
                    print(f"  {db} {key:28s} {metric} {c[metric]['point']:.3f} [{c[metric]['ci_low']:.3f}, {c[metric]['ci_high']:.3f}]")
                for key, c in s["confirmatory"].items():
                    print(f"  {db} {key:28s} {c['point']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] Holm p {c['p_holm']:.3f}; replicated {c['replicated']}")
    except PreconditionError as err:
        print(f"Refusing to run: {err}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
