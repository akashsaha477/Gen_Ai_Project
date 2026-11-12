"""Demo: one validation beat in, prediction, probabilities, saliency plot and a fact-checked note out (PLAN Phase 5).

    python -m ecgaug.demo --record 118 --beat 100                   # the beat at position 100 among the record's usable beats
    python -m ecgaug.demo --record 118 --beat 100 --notes template  # the template note only, no LLM call
    python -m ecgaug.demo --smoke                                   # a fake record and a tiny model, to check a fresh clone

Only validation records are accepted (the test records are locked anyway). The model is the best seed
of the condition named by ``--condition`` (default: the best generative condition) at the strength the
phase 4 gate fixed; the saliency method is the phase 5 gate's choice for the architecture (integrated
gradients until then). The note comes from the note variant the phase 5 gate chose (Gemini, which needs
``GEMINI_API_KEY``; see ``ecgaug.report.notes``). The template note replaces it when no key is set, when the call
fails (daily quota, network) or when the note fails the fact checker; the demo prints who wrote the note and why.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ecgaug import paths, runs, workflow
from ecgaug.data import raw
from ecgaug.data.beats import extract_beats
from ecgaug.data.records import CLASSES
from ecgaug.data.representation import RepresentationSpec, transform_record, usable_mask
from ecgaug.explain.evaluate import SEGMENTS_MS, segment_masses
from ecgaug.explain.methods import saliency
from ecgaug.names import PHASE4_GATE, PHASE5_GATE
from ecgaug.report.notes import API_KEYS, LLMClient, LLMQuotaError, check_note, client_from_env, generate, note_input, template_note

DEFAULT_METHOD = "integrated_gradients"
METHOD_SETTINGS = {"ig_steps": 50, "smoothgrad_samples": 25, "smoothgrad_sd": 0.1, "occlusion_window_ms": 50, "occlusion_stride_ms": 25, "batch_size": 64}
LOW_CONFIDENCE_THRESHOLD = 0.6


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


def explain_beat(model: torch.nn.Module, spec: RepresentationSpec, record: int, index: int, method: str, notes_variant: str, client,
                 model_id: str, out_dir: Path, threshold: float = LOW_CONFIDENCE_THRESHOLD) -> dict:  # fmt: skip
    """Classify, explain and describe the ``index``-th usable beat of a record; save the figure."""
    signal = raw.read_signal(record)
    beats = extract_beats(record)
    keep = usable_mask(beats["r_idx"].to_numpy(), signal.size, beats["rr_prev"].to_numpy(), beats["rr_next"].to_numpy()) & (beats["y"] >= 0).to_numpy()
    candidates = beats[keep].reset_index(drop=True)
    if not 0 <= index < len(candidates):
        raise ValueError(f"Record {record} has {len(candidates)} usable beats; --beat must be 0..{len(candidates) - 1}")
    row = candidates.iloc[index]
    x = transform_record(signal, np.array([int(row["r_idx"])]), spec)
    side = np.array([[row["rr_prev"], row["rr_next"], row["rr_ratio"]]], dtype=np.float32) if spec.rr_features else None
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(x), None if side is None else torch.from_numpy(side)), dim=1).numpy()[0]
    maps = saliency(method, model, x, side, {**METHOD_SETTINGS, "fs": spec.effective_fs})
    pre = spec.window_samples[0]
    masses = {name: float(v[0]) for name, v in segment_masses(maps, pre, spec.effective_fs).items()}
    payload = note_input(f"record-{record}-beat-{index}", dict(zip(CLASSES, map(float, probs))), masses, model_id, threshold)
    note, note_by = write_note(notes_variant, payload, client)
    figure = _plot(x[0], maps[0], probs, pre, spec.effective_fs, payload, out_dir / f"record{record}_beat{index}.png")
    return {"record": record, "beat": index, "r_idx": int(row["r_idx"]), "true_class": CLASSES[int(row["y"])], "prediction": payload["predicted_class"],
            "probabilities": payload["probabilities"], "segment_masses": masses, "saliency_method": method, "note": note,
            "note_by": note_by, "fact_check": check_note(note, payload), "figure": str(figure)}  # fmt: skip


def _plot(beat: np.ndarray, sal: np.ndarray, probs: np.ndarray, pre: int, fs: float, payload: dict, path: Path) -> Path:
    from ecgaug import figures

    plt = figures._plt()
    ms = (np.arange(len(beat)) - pre) / fs * 1000
    fig, (ax, bar) = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [3, 1]})
    weight = np.abs(sal) / max(np.abs(sal).max(), 1e-12)
    for name, (lo, hi) in SEGMENTS_MS.items():
        lo, hi = max(lo, ms[0]), min(hi, ms[-1])
        if lo < hi:  # the window overlaps the beat
            ax.axvspan(lo, hi, color=figures.GRID, alpha=0.35, linewidth=0)
            ax.annotate(name, ((lo + hi) / 2, 1.0), xycoords=("data", "axes fraction"), ha="center", va="top", fontsize=8, color=figures.INK_2)
    ax.fill_between(ms, beat.min(), beat.min() + weight * (beat.max() - beat.min()), color=figures.CLASS_COLORS["S"], alpha=0.25, linewidth=0, label="saliency")
    ax.plot(ms, beat, color=figures.INK, label="beat (z-scored)")
    ax.set_xlabel("ms from R peak")
    ax.legend(loc="lower right")
    bar.bar(list(CLASSES), probs, color=[figures.CLASS_COLORS[c] for c in CLASSES], edgecolor=figures.SURFACE)
    bar.set_ylim(0, 1)
    bar.set_title("probabilities", loc="left")
    return figures._save(fig, path, f"{payload['record_anon_id']}: predicted {payload['predicted_class']} ({max(probs):.2f})",
                         "Shaded: R-relative segment windows. Orange: where the model looked (saliency), scaled to its peak.")


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


def smoke() -> dict:
    """A fake validation record and a tiny untrained model: checks the whole path from a fresh clone, the note as in a real run."""
    from ecgaug.experiments.data_source import smoke_data
    from ecgaug.models import build_model

    variant = chosen_notes_variant()  # read before the smoke data's temporary root replaces the repository's
    with smoke_data() as (split, data):
        torch.manual_seed(0)
        model = build_model({"name": "shallow_cnn", "channels": [8, 16, 16], "kernel": 7, "side_hidden": 4}, 4, 3 if data.spec.rr_features else 0)
        out = explain_beat(model, data.spec, split.val[0], 5, DEFAULT_METHOD, variant, client_from_env(), "smoke-model",
                           paths.repo_root() / "figures" / "demo")  # fmt: skip
        return {"prediction": out["prediction"], "unsupported_claims": out["fact_check"]["unsupported_claims"], "note_by": out["note_by"],
                "note": out["note"][:80] + "..."}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify, explain and describe one validation beat.")
    parser.add_argument("--record", type=int, help="a validation record")
    parser.add_argument("--beat", type=int, default=0, help="position among the record's usable beats (from 0)")
    parser.add_argument("--arch", default="cnn", help="cnn or rnn")
    parser.add_argument("--condition", default="best_generative", help="none, class_weighted, ..., or best_generative")
    parser.add_argument("--notes", choices=["llm", "template"], default="llm",
                        help="llm (default): the note variant chosen at the phase 5 gate, Gemini, with the template as fallback; template: no LLM call")
    parser.add_argument("--out", default="figures/demo", help="folder for the figure")
    parser.add_argument("--smoke", action="store_true", help="fake record and tiny model")
    args = parser.parse_args(argv)
    if args.smoke:
        print(json.dumps(smoke(), indent=2))
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
    print(f"record {out['record']}, beat {out['beat']} (annotated {out['true_class']}): predicted {out['prediction']} "
          f"{out['probabilities']}; saliency by {out['saliency_method']}: {out['segment_masses']}")  # fmt: skip
    print(out["note"])
    print(f"note by: {out['note_by']}")
    print(f"figure: {out['figure']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
