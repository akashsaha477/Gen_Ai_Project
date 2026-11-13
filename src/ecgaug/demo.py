"""Demo: one validation beat in; an offline analysis by our models, then a review by Gemini checked against it (PLAN Phase 5).

    python -m ecgaug.demo --record 118 --beat 112                   # the beat at position 112 among the record's usable beats
    python -m ecgaug.demo --record 118 --beat 112 --notes template  # the template note only, no LLM call
    python -m ecgaug.demo --smoke                                   # a fake record and a tiny model, to check a fresh clone

It prints three parts, in this order:

1. Offline analysis, with no LLM. The model the phase 4 gate fixed (by default the best seed of the best generative
   condition) classifies the beat; the saliency method the phase 5 gate chose for the architecture (integrated
   gradients until then) shows where it looked, as the share of saliency in the P, QRS, ST and T windows; the beat is
   looked up in the saved validation predictions of every model the phase 4 gate fixed; the figure is saved.
2. Review by Gemini. The note variant the phase 5 gate chose writes the review from the offline result only (the JSON
   of docs/RULES.md §5, never the signal); it needs ``GEMINI_API_KEY`` (see ``ecgaug.report.notes``).
3. The review against the offline analysis: every number, class and segment the review names, next to the offline
   value it restates. A review with any claim the offline analysis does not support is replaced by the template note,
   as is the review when no key is set or the call fails (daily quota, network); the demo says which and why.

Only validation records are accepted (the test records are locked anyway).
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ecgaug import paths, runs, workflow
from ecgaug.data import raw
from ecgaug.data.beats import extract_beats
from ecgaug.data.records import CLASSES, FS
from ecgaug.data.representation import RepresentationSpec, transform_record, usable_mask
from ecgaug.explain.evaluate import SEGMENTS_MS, segment_masses
from ecgaug.explain.methods import saliency
from ecgaug.names import PHASE4_GATE, PHASE5_GATE
from ecgaug.report.notes import (
    API_KEYS,
    CLASS_NAMES,
    LLMClient,
    LLMQuotaError,
    check_note,
    client_from_env,
    generate,
    note_claims,
    note_input,
    template_note,
)

DEFAULT_METHOD = "integrated_gradients"
METHOD_SETTINGS = {"ig_steps": 50, "smoothgrad_samples": 25, "smoothgrad_sd": 0.1, "occlusion_window_ms": 50, "occlusion_stride_ms": 25, "batch_size": 64}
LOW_CONFIDENCE_THRESHOLD = 0.6
METHOD_NAMES = {"gradcam": "Grad-CAM", "integrated_gradients": "integrated gradients", "smoothgrad": "SmoothGrad", "occlusion": "occlusion"}
CONDITION_NAMES = {"none": "no augmentation", "class_weighted": "class-weighted loss", "random_oversampling": "random oversampling", "smote": "SMOTE",
                   "cvae": "cVAE", "wgan_gp": "WGAN-GP"}  # fmt: skip
SEGMENTS = ("P", "QRS", "ST", "T")
SALIENCY_COLOR, SALIENCY_LIGHT = "#7b5cc4", "#d9d0ee"  # not a class colour: saliency is a magnitude, not a class


def write_note(variant: str, payload: dict, client: LLMClient | None) -> tuple[str, str]:
    """The note for one beat and who wrote it. An LLM variant gives way to the template note, with the reason, when no
    key is set, when the call fails (daily quota, network) or when its note fails the fact checker."""
    template = template_note(payload)
    if variant == "A":
        return template, "template (variant A)"
    if client is None:
        return template, f"template (variant A), because no LLM key is set ({', '.join(API_KEYS.values())})"
    name = f"{client.provider.capitalize()} {client.model}"
    try:
        made = generate(variant, payload, client)
    except (LLMQuotaError, OSError, ValueError) as err:  # HTTP and network errors are OSErrors; the key is never in them
        return template, f"template (variant A), because the {name} call failed: {err}"
    issues = check_note(made["note"], payload)["issues"]
    if issues or made["variant_used"] == "A":  # variant D gives way to the template itself after its retries
        why = "; ".join(issues) or "still failing after the retries"
        return template, f"template (variant A), because the {name} note (variant {variant}) failed the fact checker: {why}"
    return made["note"], f"{name} (variant {variant}), passed the fact checker"


def chosen_notes_variant(notes: str = "llm") -> str:
    """The note variant: for "llm" the phase 5 gate's choice (the template, A, until the gate), for "template" A."""
    return ((workflow.manifest().get(PHASE5_GATE) or {}).get("notes_variant") or "A") if notes == "llm" else "A"


def _seed(run_id: str) -> int | None:
    found = re.search(r"-s(\d+)-", run_id)
    return int(found.group(1)) if found else None


def describe_model(run_id: str) -> dict:
    """Architecture, condition, strength and seed of a model the phase 4 gate fixed; empty for any other model."""
    for arch, conditions in ((workflow.manifest().get(PHASE4_GATE) or {}).get("chosen") or {}).items():
        for condition, entry in conditions.items():
            if run_id in entry["run_ids"]:
                return {"arch": arch, "condition": condition, "strength": entry.get("strength"), "seed": _seed(run_id)}
    return {}


def offline_runs(record: int, r_idx: int) -> list[dict]:
    """The beat in the saved validation predictions of every model the phase 4 gate fixed (nothing is run again)."""
    rows = []
    for arch, conditions in ((workflow.manifest().get(PHASE4_GATE) or {}).get("chosen") or {}).items():
        for condition, entry in conditions.items():
            for run_id in entry["run_ids"]:
                run_dir = runs.find_run_dir(run_id)
                if run_dir is None or not (run_dir / "preds_val.npz").is_file():
                    continue
                saved = np.load(run_dir / "preds_val.npz")
                hit = np.flatnonzero((saved["record"] == record) & (saved["r_idx"] == r_idx))
                if hit.size:
                    probs = saved["probs"][hit[0]]
                    rows.append({"arch": arch, "condition": condition, "seed": _seed(run_id), "run_id": run_id, "prediction": CLASSES[int(np.argmax(probs))],
                                 "probabilities": {c: float(p) for c, p in zip(CLASSES, probs)}})  # fmt: skip
    return rows


def explain_beat(model: torch.nn.Module, spec: RepresentationSpec, record: int, index: int, method: str, notes_variant: str, client,
                 model_id: str, out_dir: Path, threshold: float = LOW_CONFIDENCE_THRESHOLD) -> dict:  # fmt: skip
    """The offline analysis of the ``index``-th usable beat of a record (prediction, saliency, saved runs, figure), then
    the note written from it and the note's claims checked against it."""
    signal = raw.read_signal(record)
    beats = extract_beats(record)
    keep = usable_mask(beats["r_idx"].to_numpy(), signal.size, beats["rr_prev"].to_numpy(), beats["rr_next"].to_numpy()) & (beats["y"] >= 0).to_numpy()
    candidates = beats[keep].reset_index(drop=True)
    if not 0 <= index < len(candidates):
        raise ValueError(f"Record {record} has {len(candidates)} usable beats; --beat must be 0..{len(candidates) - 1}")
    row = candidates.iloc[index]
    r_idx = int(row["r_idx"])
    x = transform_record(signal, np.array([r_idx]), spec)
    side = np.array([[row["rr_prev"], row["rr_next"], row["rr_ratio"]]], dtype=np.float32) if spec.rr_features else None
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(x), None if side is None else torch.from_numpy(side)), dim=1).numpy()[0]
    maps = saliency(method, model, x, side, {**METHOD_SETTINGS, "fs": spec.effective_fs})
    pre = spec.window_samples[0]
    masses = {name: float(v[0]) for name, v in segment_masses(maps, pre, spec.effective_fs).items()}
    payload = note_input(f"record-{record}-beat-{index}", dict(zip(CLASSES, map(float, probs))), masses, model_id, threshold)
    saved = offline_runs(record, r_idx)
    out = {"record": record, "beat": index, "r_idx": r_idx, "time_s": r_idx / FS, "true_class": CLASSES[int(row["y"])],
           "rr": {"before_s": float(row["rr_prev"]), "after_s": float(row["rr_next"]), "ratio": float(row["rr_ratio"])},
           "model_id": model_id, "model": describe_model(model_id), "prediction": payload["predicted_class"], "probabilities": payload["probabilities"],
           "probabilities_exact": dict(zip(CLASSES, map(float, probs))), "low_confidence": payload["low_confidence"], "threshold": threshold,
           "saliency_method": method, "segment_masses": masses, "top_segments": [s["name"] for s in payload["top_segments"]], "offline_runs": saved}  # fmt: skip
    out["figure"] = str(_plot(x[0], maps[0], probs, pre, spec.effective_fs, out, out_dir / f"record{record}_beat{index}.png"))
    note, note_by = write_note(notes_variant, payload, client)
    out.update(note=note, note_by=note_by, llm_used=not note_by.startswith("template"), claims=note_claims(note, payload), fact_check=check_note(note, payload))
    return out


def _plot(beat: np.ndarray, sal: np.ndarray, probs: np.ndarray, pre: int, fs: float, out: dict, path: Path) -> Path:
    from matplotlib.patches import Rectangle

    from ecgaug import figures

    plt = figures._plt()
    fig = plt.figure(figsize=(11, 6.6))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.3, 1], width_ratios=[1, 1, 1.2])
    ax = fig.add_subplot(grid[0, :])
    ms = (np.arange(len(beat)) - pre) / fs * 1000
    weight = np.abs(sal) / max(np.abs(sal).max(), 1e-12)
    for k, (name, (lo, hi)) in enumerate(SEGMENTS_MS.items()):
        lo, hi = max(lo, ms[0]), min(hi, ms[-1])
        if lo < hi:  # the window overlaps the beat
            ax.axvspan(lo, hi, color=figures.GRID, alpha=0.45 if k % 2 == 0 else 0.2, linewidth=0)
            ax.annotate(name, ((lo + hi) / 2, 0.98), xycoords=("data", "axes fraction"), ha="center", va="top", fontsize=9, color=figures.INK_2)
    low = beat.min() - 0.15 * np.ptp(beat)
    ax.fill_between(ms, low, low + weight * (beat.max() - low), color=SALIENCY_COLOR, alpha=0.28, linewidth=0, label="where the model looked")
    ax.plot(ms, beat, color=figures.INK, label="beat (z-scored)")
    ax.axvline(0, color=figures.INK_2, linewidth=0.8, linestyle=":")
    ax.set_xlabel("ms from the R peak")
    ax.set_xlim(ms[0], ms[-1])
    ax.set_ylim(low, beat.max() + 0.25 * np.ptp(beat))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)

    seg = fig.add_subplot(grid[1, 0])
    shares = [out["segment_masses"].get(s, 0.0) for s in SEGMENTS]
    colors = [SALIENCY_COLOR if s in out["top_segments"] else SALIENCY_LIGHT for s in SEGMENTS]
    seg.barh(range(len(SEGMENTS)), shares, color=colors, height=0.7)
    for i, v in enumerate(shares):
        seg.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=8, color=figures.INK)
    seg.set_yticks(range(len(SEGMENTS)), SEGMENTS)
    seg.invert_yaxis()
    seg.set_xlim(0, max(max(shares) * 1.3, 0.1))
    seg.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    seg.grid(axis="y", visible=False)
    seg.set_title(f"where it looked ({METHOD_NAMES.get(out['saliency_method'], out['saliency_method'])})", loc="left")

    bar = fig.add_subplot(grid[1, 1])
    bar.bar(list(CLASSES), probs, color=[figures.CLASS_COLORS[c] for c in CLASSES], edgecolor=figures.SURFACE, width=0.7)
    for i, v in enumerate(probs):
        bar.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color=figures.INK)
    bar.axhline(out["threshold"], color=figures.INK_2, linewidth=0.8, linestyle="--")
    bar.text(3.4, out["threshold"] + 0.02, f"low-confidence\nthreshold {out['threshold']:.2f}", ha="right", va="bottom", fontsize=7, color=figures.INK_2)
    bar.set_ylim(0, 1.1)
    bar.grid(axis="x", visible=False)
    bar.set_title("probabilities", loc="left")

    vote = fig.add_subplot(grid[1, 2])
    saved = out["offline_runs"]
    archs = list(dict.fromkeys(r["arch"] for r in saved))
    conditions = list(dict.fromkeys(r["condition"] for r in saved))
    seeds = sorted({r["seed"] for r in saved if r["seed"] is not None})
    columns = [(a, s) for a in archs for s in seeds]
    for r in saved:
        i, j = conditions.index(r["condition"]), columns.index((r["arch"], r["seed"]))
        own = r["run_id"] == out["model_id"]
        vote.add_patch(Rectangle((j + 0.06, i + 0.06), 0.88, 0.88, color=figures.CLASS_COLORS[r["prediction"]], linewidth=0))
        if own:
            vote.add_patch(Rectangle((j + 0.02, i + 0.02), 0.96, 0.96, fill=False, edgecolor=figures.INK, linewidth=1.8))
        vote.text(j + 0.5, i + 0.5, r["prediction"], ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    if saved:
        vote.set_xlim(0, len(columns))
        vote.set_ylim(len(conditions), 0)
        vote.set_xticks([j + 0.5 for j in range(len(columns))], [f"{a}\ns{s}" for a, s in columns])
        vote.set_yticks([i + 0.5 for i in range(len(conditions))], [CONDITION_NAMES.get(c, c) for c in conditions])
        counts = Counter(r["prediction"] for r in saved)
        vote.set_title(f"our {len(saved)} saved runs: " + ", ".join(f"{c} {n}" for c, n in counts.most_common()), loc="left")
    else:
        vote.text(0.5, 0.5, "no saved runs\n(no phase 4 gate here)", ha="center", va="center", transform=vote.transAxes, color=figures.INK_2)
        vote.set_xticks([])
        vote.set_yticks([])
        vote.set_title("our saved runs", loc="left")
    vote.grid(False)
    vote.tick_params(length=0)
    for side in vote.spines.values():
        side.set_visible(False)

    title = (f"Record {out['record']}, beat {out['beat']}: predicted {out['prediction']} ({max(probs):.2f}); "
             f"reference annotation {out['true_class']}")  # fmt: skip
    caption = ("Computed offline by our models, no LLM involved. Top: the beat and where the model looked (saliency, scaled to its peak) in the "
               "P, QRS, ST and T windows. Bottom right: the beat's predicted class in the saved validation predictions of every model "
               "the phase 4 gate fixed; outlined, the model used here.")  # fmt: skip
    return figures._save(fig, path, title, caption)


RULE = "-" * 100


def _row(label: str, text: str) -> str:
    return f"   {label:<17}{text}"


def _bar(value: float, width: int = 30) -> str:
    return "█" * int(round(max(value, 0.0) * width))


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def format_review(out: dict) -> str:
    """The three parts as text: the offline analysis, the review, and the review against the offline analysis."""
    p = out["prediction"]
    lines = [f"ECG beat review: record {out['record']}, beat {out['beat']} (at {_clock(out['time_s'])} in the recording)",
             f"Reference annotation: {out['true_class']}, {CLASS_NAMES[out['true_class']]} (from the database; the model does not see it)", "",
             "1. Offline analysis by our models (no LLM)", RULE]  # fmt: skip
    m = out["model"]
    if m:
        strength = f" at strength {m['strength']}" if m.get("strength") is not None else ""
        lines.append(_row("Model", f"{m['arch']}, {CONDITION_NAMES.get(m['condition'], m['condition'])}{strength}, seed {m['seed']}"))
    lines.append(_row("Model" if not m else "", f"run {out['model_id']}"))
    rr = out["rr"]
    lines.append(_row("RR intervals", f"{rr['before_s']:.2f} s before the beat, {rr['after_s']:.2f} s after; "
                                      f"before / mean of the previous 10: {rr['ratio']:.2f}"))  # fmt: skip
    lines.append(_row("Prediction", f"{p}, {CLASS_NAMES[p]}"))
    for i, c in enumerate(CLASSES):
        v = out["probabilities"][c]
        lines.append(_row("Probabilities" if i == 0 else "", f"{c:<4}{v:5.2f}  {_bar(v):<31}" + ("<- predicted" if c == p else "")))
    confidence = out["probabilities"][p]
    if out["low_confidence"]:
        lines.append(_row("Confidence", f"{confidence:.2f}, below the low-confidence threshold {out['threshold']:.2f}: clinician review required"))
    else:
        lines.append(_row("Confidence", f"{confidence:.2f}, at or above the low-confidence threshold {out['threshold']:.2f}"))
    masses = out["segment_masses"]
    method = METHOD_NAMES.get(out["saliency_method"], out["saliency_method"])
    lines.append(_row("Explainability", f"{method}: share of the saliency in each window around the R peak"))
    for name in SEGMENTS:
        v = masses.get(name, 0.0)
        rank = out["top_segments"].index(name) + 1 if name in out["top_segments"] else None
        lines.append(_row("", f"{name:<4}{v:5.0%}  {_bar(v):<31}" + (f"<- top {rank}" if rank else "")))
    rest = 1 - sum(masses.get(name, 0.0) for name in SEGMENTS)
    if rest > 0.005:
        lines.append(_row("", f"({rest:.0%} falls between the windows)"))
    saved = out["offline_runs"]
    own = next((r for r in saved if r["run_id"] == out["model_id"]), None)
    if own:
        diff = max(abs(own["probabilities"][c] - out["probabilities_exact"][c]) for c in CLASSES)
        lines.append(_row("Saved run", f"this model's saved validation prediction for the beat: {own['prediction']} "
                                       f"{own['probabilities'][own['prediction']]:.2f} (largest difference to now: {diff:.3f})"))  # fmt: skip
    if saved:
        archs = list(dict.fromkeys(r["arch"] for r in saved))
        seeds = sorted({r["seed"] for r in saved if r["seed"] is not None})
        lines.append(_row("All our models", f"the beat in the saved validation predictions of {len(saved)} runs "
                                            f"(seeds {', '.join(map(str, seeds))}; [ ] the model used here):"))  # fmt: skip
        lines.append(_row("", f"{'':<22}" + "".join(f"{a:<12}" for a in archs)))
        for condition in dict.fromkeys(r["condition"] for r in saved):
            cells = []
            for a in archs:
                marks = [f"[{r['prediction']}]" if r["run_id"] == out["model_id"] else f" {r['prediction']} " for r in saved
                         if r["arch"] == a and r["condition"] == condition]  # fmt: skip
                cells.append(f"{''.join(marks):<12}")
            lines.append(_row("", f"{CONDITION_NAMES.get(condition, condition):<22}" + "".join(cells)))
        counts = Counter(r["prediction"] for r in saved)
        lines.append(_row("", "; ".join(f"{c} in {n} of {len(saved)} runs" for c, n in counts.most_common())))
    lines.append(_row("Figure", out["figure"]))

    lines += [""]
    if out["llm_used"]:
        lines += [f"2. Review by {out['note_by'].removesuffix(', passed the fact checker')}, written from the offline result only", RULE]
    else:
        why = out["note_by"].split("because ", 1)[1] if "because " in out["note_by"] else "--notes template was given"
        lines += ["2. Review: the template note (no LLM)", RULE, _row("Why", why)]
    lines += textwrap.wrap(out["note"], width=97, initial_indent="   ", subsequent_indent="   ")

    lines += ["", "3. The review against the offline analysis", RULE]
    width = max([len(c["claim"]) for c in out["claims"]] + [10]) + 2
    for c in out["claims"]:
        lines.append(f"   {'✓' if c['matches'] else '✗'} {repr(c['claim']) if c['claim'] not in ('disclaimer', 'review sentence') else c['claim']:<{width}} {c['offline']}")
    wrong = sum(not c["matches"] for c in out["claims"])
    if out["llm_used"]:
        lines.append(f"   Result: {'every claim matches the offline analysis' if not wrong else f'{wrong} claims do not match'}; the review is kept.")
    else:
        lines.append(f"   Result: the template restates the offline result; {wrong} unsupported claims.")
    named = out["fact_check"]["coverage"]["predicted_class"]
    agreement = f"The review names {p}, our model's prediction" if named else f"The review does not name {p}, our model's prediction"
    if saved:
        agreement += f"; {sum(r['prediction'] == p for r in saved)} of our {len(saved)} saved runs predict {p} for this beat"
    lines.append(f"   {agreement}.")
    return "\n".join(line.rstrip() for line in lines)


def chosen_model(arch: str, condition: str) -> tuple[str, str]:
    gate = workflow.manifest().get(PHASE4_GATE)
    if not gate:
        raise ValueError("The phase 4 gate has not fixed the models yet")
    if condition == "best_generative":
        condition = gate["best_generative"][arch]
    run_ids = gate["chosen"][arch][condition]["run_ids"]
    values = {r: json.loads((runs.find_run_dir(r) / "metrics_val.json").read_text())["selection_value"] for r in run_ids}
    method = ((workflow.manifest().get(PHASE5_GATE) or {}).get("explainability") or {}).get(arch, DEFAULT_METHOD)
    return max(values, key=values.get), method


def smoke(show: bool = False) -> dict:
    """A fake validation record and a tiny untrained model: checks the whole path from a fresh clone, the note as in a real run."""
    from ecgaug.experiments.data_source import smoke_data
    from ecgaug.models import build_model

    variant = chosen_notes_variant()  # read before the smoke data's temporary root replaces the repository's
    with smoke_data() as (split, data):
        torch.manual_seed(0)
        model = build_model({"name": "shallow_cnn", "channels": [8, 16, 16], "kernel": 7, "side_hidden": 4}, 4, 3 if data.spec.rr_features else 0)
        out = explain_beat(model, data.spec, split.val[0], 5, DEFAULT_METHOD, variant, client_from_env(), "smoke-model",
                           paths.repo_root() / "figures" / "demo")  # fmt: skip
        text = format_review(out)
    if show:
        print(text)
    return {"prediction": out["prediction"], "unsupported_claims": out["fact_check"]["unsupported_claims"], "note_by": out["note_by"],
            "claims_not_matching": sum(not c["matches"] for c in out["claims"]), "note": out["note"][:80] + "..."}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline analysis of one validation beat by our models, then a review by Gemini checked against it.")
    parser.add_argument("--record", type=int, help="a validation record")
    parser.add_argument("--beat", type=int, default=0, help="position among the record's usable beats (from 0)")
    parser.add_argument("--arch", default="cnn", help="cnn or rnn")
    parser.add_argument("--condition", default="best_generative", help="none, class_weighted, ..., or best_generative")
    parser.add_argument("--notes", choices=["llm", "template"], default="llm",
                        help="llm (default): the note variant chosen at the phase 5 gate, Gemini, with the template as fallback; template: no LLM call")
    parser.add_argument("--out", default="figures/demo", help="folder for the figure")
    parser.add_argument("--json", action="store_true", help="print the result as JSON instead of the three parts")
    parser.add_argument("--smoke", action="store_true", help="fake record and tiny model")
    args = parser.parse_args(argv)
    if args.smoke:
        summary = smoke(show=not args.json)
        print(json.dumps(summary, indent=2))
        return 0
    from ecgaug.data.splits import load_approved_split
    from ecgaug.experiments.common import load_run_classifier

    if args.record is None:
        parser.error("--record is required")
    split = load_approved_split()
    if args.record not in split.val:
        print(f"The demo uses validation records only: {list(split.val)}")
        return 2
    try:
        run_id, method = chosen_model(args.arch, args.condition)
        contract = json.loads((paths.processed_dir() / "v1" / "contract.json").read_text())
        rep = {k: v for k, v in contract["representation"].items() if k in RepresentationSpec.__dataclass_fields__}
        spec = RepresentationSpec(**rep)
        model = load_run_classifier(run_id, spec.rr_features)
    except (ValueError, FileNotFoundError, KeyError) as err:
        print(f"Demo not ready: {err}")
        return 2
    client = client_from_env() if args.notes == "llm" else None
    out = explain_beat(model, spec, args.record, args.beat, method, chosen_notes_variant(args.notes), client, run_id, paths.repo_root() / args.out)
    print(json.dumps(out, indent=2) if args.json else format_review(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
