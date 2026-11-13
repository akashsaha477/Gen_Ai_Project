"""Demo: one validation beat in; an offline analysis by our models, then a review by Gemini checked against it (PLAN Phase 5).

    ./demo.sh                                                       # the tour from any folder: one N, S, V and F beat
    python -m ecgaug.demo                                           # the same tour, from the repository root
    python -m ecgaug.demo S                                         # one example beat: N, S, V or F
    python -m ecgaug.demo --list                                    # beat positions to choose from, per validation record
    python -m ecgaug.demo --record 118 --beat 112                   # the beat at position 112 among the record's usable beats
    python -m ecgaug.demo --record 118 --beat 112 --notes template  # the same review filled in offline, no LLM call
    python -m ecgaug.demo --record 118 --beat 112 --notes gate      # the note variant the phase 5 gate chose, instead of the review
    python -m ecgaug.demo --smoke                                   # a fake record and a tiny model, to check a fresh clone

It prints three parts, in this order:

1. Offline analysis, with no LLM. The model the phase 4 gate fixed (by default the best seed of the best generative
   condition) classifies the beat; the saliency method the phase 5 gate chose for the architecture (integrated
   gradients until then) shows where it looked, as the share of saliency in the P, QRS, ST and T windows; the beat is
   looked up in the saved validation predictions of every model the phase 4 gate fixed; the figure is saved.
2. Review by Gemini, written as a cardiologist's short report (findings, impression, confidence, next step) from the
   offline result only, never the signal (``ecgaug.report.review``, added after the phase 5 gate for the demo, A44); it
   needs ``GEMINI_API_KEY``. ``--notes gate`` writes the note variant the phase 5 gate chose instead (``ecgaug.report.notes``).
3. The review against the offline analysis: every number, class and segment the review names, next to the offline
   value it restates. A review with any claim the offline analysis does not support is sent back (at most twice) and
   then replaced by the same report filled in offline, as is the review when no key is set or the call fails (daily
   quota, network); the demo says which and why.

Only validation records are accepted (the test records are locked anyway). In a terminal the figure opens by itself
(``--no-open`` to stop it) and the tour waits for Enter between beats.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
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
from ecgaug.report.review import check_review, review_facts, template_review, write_review
from ecgaug.report.review import claims as review_claims

DEFAULT_METHOD = "integrated_gradients"
METHOD_SETTINGS = {"ig_steps": 50, "smoothgrad_samples": 25, "smoothgrad_sd": 0.1, "occlusion_window_ms": 50, "occlusion_stride_ms": 25, "batch_size": 64}
LOW_CONFIDENCE_THRESHOLD = 0.6
METHOD_NAMES = {"gradcam": "Grad-CAM", "integrated_gradients": "integrated gradients", "smoothgrad": "SmoothGrad", "occlusion": "occlusion"}
CONDITION_NAMES = {"none": "no augmentation", "class_weighted": "class-weighted loss", "random_oversampling": "random oversampling", "smote": "SMOTE",
                   "cvae": "cVAE", "wgan_gp": "WGAN-GP"}  # fmt: skip
SEGMENTS = ("P", "QRS", "ST", "T")
SALIENCY_COLOR, SALIENCY_LIGHT = "#7b5cc4", "#d9d0ee"  # not a class colour: saliency is a magnitude, not a class
STRIP_SECONDS = 8.0
# The tour: one beat per class from the validation records, each classified correctly by the demo's model (checked 2026-11-13).
SHOWCASE = {"N": (118, 100), "S": (118, 112), "V": (118, 129), "F": (205, 1400)}
PAPER, PAPER_MINOR, PAPER_MAJOR = "#fff7f6", "#f5d9d6", "#e9aaa4"  # ECG paper: small squares 40 ms x 0.1 mV, large 200 ms x 0.5 mV


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


def clinician_review(facts: dict, client: LLMClient | None) -> tuple[str, str]:
    """The cardiologist-style review and who wrote it: Gemini when a key is set and its review passes the checker (after
    at most two rewrites), else the same report filled in offline, with the reason."""
    offline = template_review(facts)
    if client is None:
        return offline, f"offline template (no LLM), because no LLM key is set ({', '.join(API_KEYS.values())})"
    name = f"{client.provider.capitalize()} {client.model}"
    try:
        made = write_review(facts, client)
    except (LLMQuotaError, OSError, ValueError) as err:  # HTTP and network errors are OSErrors; the key is never in them
        return offline, f"offline template (no LLM), because the {name} call failed: {err}"
    if made["review"] is None:
        return offline, f"offline template (no LLM), because the {name} review failed the fact checker {made['attempts']} times: {'; '.join(made['issues'])}"
    rewrites = made["attempts"] - 1
    return made["review"], f"{name} (clinician review), passed the fact checker" + (f" after {rewrites} rewrite{'s' * (rewrites > 1)}" if rewrites else "")


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


def usable_beats(record: int):
    """The record's signal and its usable beats in order; a beat's position here is what ``--beat`` takes."""
    signal = raw.read_signal(record)
    beats = extract_beats(record)
    keep = usable_mask(beats["r_idx"].to_numpy(), signal.size, beats["rr_prev"].to_numpy(), beats["rr_next"].to_numpy()) & (beats["y"] >= 0).to_numpy()
    return signal, beats[keep].reset_index(drop=True)


def list_beats(records, shown: int = 6) -> str:
    """For each record: usable beats per reference class and the first positions of the rare classes, for ``--beat``."""
    rows = []
    for record in records:
        _, candidates = usable_beats(record)
        labels = candidates["y"].to_numpy()
        counts = [int((labels == k).sum()) for k in range(len(CLASSES))]
        firsts = [" ".join(map(str, np.flatnonzero(labels == k)[:shown])) or "-" for k in range(1, len(CLASSES))]
        rows.append((record, len(labels), counts, firsts))
    heads = [f"first {c} positions" for c in CLASSES[1:]]
    widths = [max([len(h)] + [len(r[3][k]) for r in rows]) for k, h in enumerate(heads)]
    lines = ["Validation beats to show: python -m ecgaug.demo --record <record> --beat <position>", "",
             f"{'record':<8}{'beats':>7}" + "".join(f"{c:>7}" for c in CLASSES) + "   " + "   ".join(f"{h:<{w}}" for h, w in zip(heads, widths))]  # fmt: skip
    for record, n, counts, firsts in rows:
        lines.append(f"{record:<8}{n:>7}" + "".join(f"{c:>7}" for c in counts) + "   " + "   ".join(f"{f:<{w}}" for f, w in zip(firsts, widths)))
    return "\n".join(line.rstrip() for line in lines)


def open_figure(path: str) -> None:
    """Open the figure in the system viewer (macOS ``open``, Linux ``xdg-open``); nothing happens where there is none."""
    opener = "open" if sys.platform == "darwin" else "xdg-open" if shutil.which("xdg-open") else None
    if opener:
        subprocess.Popen([opener, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def rhythm_strip(model: torch.nn.Module, spec: RepresentationSpec, signal: np.ndarray, candidates, r_idx: int) -> dict:
    """Our model's class for every usable beat within ``STRIP_SECONDS`` around the beat, with the reference annotation
    and the RR interval before each beat (offline; the signal of the strip is drawn by ``_plot``)."""
    half = int(STRIP_SECONDS / 2 * FS)
    lo, hi = max(0, r_idx - half), min(signal.size, r_idx + half)
    near = candidates[(candidates["r_idx"] >= lo) & (candidates["r_idx"] < hi)]
    peaks = near["r_idx"].to_numpy().astype(int)
    x = np.array(transform_record(signal, peaks, spec), dtype=np.float32)  # a writable copy for torch
    side = np.array(near[["rr_prev", "rr_next", "rr_ratio"]].to_numpy(dtype=np.float32)) if spec.rr_features else None
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(x), None if side is None else torch.from_numpy(side)), dim=1).numpy()
    beats = [{"time_s": round(int(peak) / FS, 3), "prediction": CLASSES[int(np.argmax(q))], "reference": CLASSES[int(y)], "rr_before_ms": round(float(rr) * 1000),
              "reviewed": int(peak) == r_idx} for peak, q, y, rr in zip(peaks, probs, near["y"], near["rr_prev"])]  # fmt: skip
    return {"start": lo, "end": hi, "beats": beats}


def explain_beat(model: torch.nn.Module, spec: RepresentationSpec, record: int, index: int, method: str, review: str, client,
                 model_id: str, out_dir: Path, threshold: float = LOW_CONFIDENCE_THRESHOLD, notes_variant: str = "A") -> dict:  # fmt: skip
    """The offline analysis of the ``index``-th usable beat of a record (prediction, saliency, saved runs, figure), then
    the review written from it and its claims checked against it. ``review``: "clinician" (Gemini as a cardiologist,
    offline template as fallback), "template" (the offline template) or "gate" (the phase 5 note ``notes_variant``)."""
    signal, candidates = usable_beats(record)
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
    strip = rhythm_strip(model, spec, signal, candidates, r_idx)
    out = {"record": record, "beat": index, "r_idx": r_idx, "time_s": r_idx / FS, "true_class": CLASSES[int(row["y"])],
           "rr": {"before_s": float(row["rr_prev"]), "after_s": float(row["rr_next"]), "ratio": float(row["rr_ratio"])},
           "model_id": model_id, "model": describe_model(model_id), "prediction": payload["predicted_class"], "probabilities": payload["probabilities"],
           "probabilities_exact": dict(zip(CLASSES, map(float, probs))), "low_confidence": payload["low_confidence"], "threshold": threshold,
           "saliency_method": method, "segment_masses": masses, "top_segments": [s["name"] for s in payload["top_segments"]], "offline_runs": saved,
           "strip": {"start_s": strip["start"] / FS, "end_s": strip["end"] / FS, "beats": strip["beats"]}}  # fmt: skip
    figure = _plot(x[0], maps[0], probs, pre, spec.effective_fs, out, signal[strip["start"] : strip["end"]], out_dir / f"record{record}_beat{index}.png")
    out["figure"] = str(figure)
    if review == "gate":
        note, note_by = write_note(notes_variant, payload, client)
        found, check = note_claims(note, payload), check_note(note, payload)
    else:
        facts = review_facts(out)
        note, note_by = clinician_review(facts, client if review == "clinician" else None)
        if review == "template":
            note_by = "offline template (no LLM), because --notes template was given"
        found, check = review_claims(note, facts), check_review(note, facts)
    out.update(review_kind=review, note=note, note_by=note_by, llm_used=not note_by.startswith(("template", "offline template")), claims=found, fact_check=check)
    return out


def _paper(ax, t0: float, t1: float, v0: float, v1: float) -> None:
    """ECG paper behind a strip: small squares 40 ms x 0.1 mV, large squares 200 ms x 0.5 mV."""
    from ecgaug import figures

    ax.set_facecolor(PAPER)
    ax.grid(False)
    for step, color, width in ((0.04, PAPER_MINOR, 0.4), (0.2, PAPER_MAJOR, 0.7)):
        for t in np.arange(np.ceil(t0 / step) * step, t1, step):
            ax.axvline(t, color=color, linewidth=width, zorder=0)
    for step, color, width in ((0.1, PAPER_MINOR, 0.4), (0.5, PAPER_MAJOR, 0.7)):
        for v in np.arange(np.ceil(v0 / step) * step, v1, step):
            ax.axhline(v, color=color, linewidth=width, zorder=0)
    ax.set_xlim(t0, t1)
    ax.set_ylim(v0, v1)
    for side in ax.spines.values():
        side.set_color(figures.GRID)


def _strip(ax, signal: np.ndarray, out: dict) -> None:
    from ecgaug import figures

    strip = out["strip"]
    t = strip["start_s"] + np.arange(signal.size) / FS
    low, high = np.floor((signal.min() - 0.6) * 2) / 2, np.ceil((signal.max() + 0.7) * 2) / 2
    _paper(ax, t[0], t[-1], low, high)
    for b in strip["beats"]:
        if b["reviewed"]:
            ax.axvspan(b["time_s"] - 0.25, b["time_s"] + 0.25, color=SALIENCY_LIGHT, alpha=0.8, linewidth=0, zorder=1)
    ax.plot(t, signal, color=figures.INK, linewidth=1.0, zorder=2)
    beats = strip["beats"]
    for k, b in enumerate(beats):
        ax.text(b["time_s"], high - 0.12, b["prediction"], ha="center", va="top", fontsize=10, fontweight="bold", color=figures.CLASS_COLORS[b["prediction"]], zorder=3)
        ax.text(b["time_s"], low + 0.08, b["reference"], ha="center", va="bottom", fontsize=8, color=figures.INK_2, zorder=3)
        if k:
            ax.text((beats[k - 1]["time_s"] + b["time_s"]) / 2, high - 0.18, f"{b['rr_before_ms']}", ha="center", va="top", fontsize=7, color=figures.INK_2, zorder=3)
    ax.set_xlabel("seconds into the recording (lead MLII)")
    ax.set_ylabel("mV")
    ax.set_title("rhythm strip: our model's class above each beat (coloured), the reference annotation below (grey), RR intervals in ms", loc="left")


def _plot(beat: np.ndarray, sal: np.ndarray, probs: np.ndarray, pre: int, fs: float, out: dict, strip_signal: np.ndarray, path: Path) -> Path:
    from matplotlib.patches import Rectangle

    from ecgaug import figures

    plt = figures._plt()
    fig = plt.figure(figsize=(11, 9.8))
    grid = fig.add_gridspec(3, 3, height_ratios=[1.15, 1.2, 1], width_ratios=[1, 1, 1.2])
    _strip(fig.add_subplot(grid[0, :]), strip_signal, out)
    ax = fig.add_subplot(grid[1, :])
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

    seg = fig.add_subplot(grid[2, 0])
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

    bar = fig.add_subplot(grid[2, 1])
    bar.bar(list(CLASSES), probs, color=[figures.CLASS_COLORS[c] for c in CLASSES], edgecolor=figures.SURFACE, width=0.7)
    for i, v in enumerate(probs):
        bar.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color=figures.INK)
    bar.axhline(out["threshold"], color=figures.INK_2, linewidth=0.8, linestyle="--")
    bar.set_ylim(0, 1.1)
    bar.grid(axis="x", visible=False)
    bar.set_title(f"probabilities (- - threshold {out['threshold']:.2f})", loc="left")

    vote = fig.add_subplot(grid[2, 2])
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
    caption = (f"Computed offline by our models, no LLM involved. Top: {STRIP_SECONDS:.0f} s of the recording on ECG paper (small squares 40 ms "
               "and 0.1 mV), the reviewed beat shaded; every usable beat classified by the same model. Middle: the reviewed beat (z-scored) and "
               "where the model looked (saliency, scaled to its peak) in the P, QRS, ST and T windows. Bottom right: the beat's class in the "
               "saved validation predictions of every model the phase 4 gate fixed; outlined, the model used here.")  # fmt: skip
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
    beats = out["strip"]["beats"]
    if beats:
        cell = [f"{'[' + b['prediction'] + ']' if b['reviewed'] else b['prediction']:^5}" for b in beats]
        ref = [f"{b['reference']:^5}" for b in beats]
        agree = sum(b["prediction"] == b["reference"] for b in beats)
        span = out["strip"]["end_s"] - out["strip"]["start_s"]
        lines.append(_row("Rhythm strip", f"{len(beats)} usable beats in the {span:.0f} s around the beat, each classified by the same model:"))
        lines.append(_row("", f"{'our model':<13}" + "".join(cell)))
        lines.append(_row("", f"{'reference':<13}" + "".join(ref)))
        lines.append(_row("", f"{'RR (ms)':<13}" + "".join(f"{b['rr_before_ms']:^5}" for b in beats)))
        lines.append(_row("", f"same class as the reference for {agree} of {len(beats)} beats"))
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
    writer = out["note_by"].split(", passed the fact checker")[0]
    if out["llm_used"]:
        kind = "as a cardiologist's report" if out["review_kind"] != "gate" else "as the phase 5 note"
        lines += [f"2. Review by {writer.replace(' (clinician review)', '')}, written {kind} from the offline result only", RULE]
        if "rewrite" in out["note_by"]:
            lines.append(_row("Checked", out["note_by"].split(", ", 1)[1]))
    else:
        title = "the template note (no LLM)" if out["review_kind"] == "gate" else "the offline template (no LLM)"
        lines += [f"2. Review: {title}", RULE, _row("Why", out["note_by"].split("because ", 1)[-1])]
    for paragraph in out["note"].split("\n"):
        label, _, rest = paragraph.partition(": ")
        if out["review_kind"] != "gate" and rest and label in ("Findings", "Impression", "Confidence", "Next step"):
            lines += textwrap.wrap(rest, width=100, initial_indent=f"   {label + ':':<14}", subsequent_indent=" " * 17)
        else:
            lines += textwrap.wrap(paragraph, width=100, initial_indent="   ", subsequent_indent="   ")

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

    with smoke_data() as (split, data):
        torch.manual_seed(0)
        model = build_model({"name": "shallow_cnn", "channels": [8, 16, 16], "kernel": 7, "side_hidden": 4}, 4, 3 if data.spec.rr_features else 0)
        out = explain_beat(model, data.spec, split.val[0], 5, DEFAULT_METHOD, "clinician", client_from_env(), "smoke-model",
                           paths.repo_root() / "figures" / "demo")  # fmt: skip
        text = format_review(out)
    if show:
        print(text)
    return {"prediction": out["prediction"], "unsupported_claims": out["fact_check"]["unsupported_claims"], "note_by": out["note_by"],
            "claims_not_matching": sum(not c["matches"] for c in out["claims"]), "note": out["note"][:80] + "..."}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline analysis of a validation beat by our models, then a review by Gemini checked against it. "
                                                 "Without arguments: a tour of a normal, an S and a V beat.")  # fmt: skip
    parser.add_argument("example", nargs="?", type=str.upper, choices=sorted(SHOWCASE), help="one example beat of this class instead of the tour")
    parser.add_argument("--record", type=int, help="a validation record (with --beat)")
    parser.add_argument("--beat", type=int, default=0, help="position among the record's usable beats (from 0; see --list)")
    parser.add_argument("--list", action="store_true", help="beat positions per validation record and class, to choose from")
    parser.add_argument("--no-open", action="store_true", help="do not open the figure")
    parser.add_argument("--arch", default="cnn", help="cnn or rnn")
    parser.add_argument("--condition", default="best_generative", help="none, class_weighted, ..., or best_generative")
    parser.add_argument("--notes", choices=["clinician", "template", "gate"], default="clinician",
                        help="clinician (default): Gemini writes a cardiologist's review, the offline template as fallback; template: the same review "
                             "filled in offline, no LLM call; gate: the note variant chosen at the phase 5 gate")
    parser.add_argument("--out", default="figures/demo", help="folder for the figure")
    parser.add_argument("--json", action="store_true", help="print the result as JSON instead of the three parts")
    parser.add_argument("--smoke", action="store_true", help="fake record and tiny model")
    args = parser.parse_args(argv)
    if args.record is not None and args.example:
        parser.error("give an example class or --record, not both")
    if args.smoke:
        summary = smoke(show=not args.json)
        print(json.dumps(summary, indent=2))
        return 0
    from ecgaug.data.splits import load_approved_split
    from ecgaug.experiments.common import load_run_classifier

    split = load_approved_split()
    if args.list:
        print(list_beats(split.val))
        return 0
    targets = [(args.record, args.beat)] if args.record is not None else [SHOWCASE[args.example]] if args.example else list(SHOWCASE.values())
    outside = sorted({r for r, _ in targets} - set(split.val))
    if outside:
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
    client = client_from_env() if args.notes != "template" else None
    interactive = sys.stdin.isatty() and sys.stdout.isatty() and not args.json
    if len(targets) > 1 and not args.json:
        stops = ", ".join(f"{CLASS_NAMES[c]} ({c}, record {r})" for c, (r, _) in SHOWCASE.items())
        print(f"Demo tour, one beat per class from the validation records: {stops}.\n")
    for k, (record, beat) in enumerate(targets):
        if k and interactive and input("\nPress Enter for the next beat, or q and Enter to stop: ").strip().lower() == "q":
            break
        if k and not args.json:
            print("\n" + "=" * 100 + "\n")
        try:
            out = explain_beat(model, spec, record, beat, method, args.notes, client, run_id, paths.repo_root() / args.out,
                               notes_variant=chosen_notes_variant("llm"))  # fmt: skip
        except ValueError as err:
            print(err)
            return 2
        print(json.dumps(out, indent=2) if args.json else format_review(out))
        if interactive and not args.no_open:
            open_figure(out["figure"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
