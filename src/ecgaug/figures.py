"""Before-and-after figures for every phase, so the team can see how the data and models change.

    python -m ecgaug.figures phase1_data          # MIT-BIH: raw signal, filtering, the four inputs, class counts
    python -m ecgaug.figures phase2_ptbxl_data    # PTB-XL: resampling and detected beats, exclusions, labels
    python -m ecgaug.figures results              # every experiment result: curves, confusion matrices, the phase 1 gate
    python -m ecgaug.figures posthoc              # the post-hoc analyses (PREREGISTRATION v4), from their summaries
    python -m ecgaug.figures all

Figures are written to ``figures/<experiment or phase folder>/``, named like ``results/`` as PNG files and are committed. The runners also call
:func:`for_result` at the end of every experiment, so a Kaggle bundle brings its figures back.
Only DS1 records are plotted (training records for anything learned from data); DS2 never is.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ecgaug import paths, runs
from ecgaug.names import (
    EXPERIMENTS,
    PHASE1_DATA,
    PHASE1_GATE,
    PHASE1_INPUT_REPRESENTATION,
    PHASE2_BASELINE_CLASSIFIERS,
    PHASE2_GATE,
    PHASE2_PTBXL_DATA,
    PHASE2_PTBXL_PRETRAINING,
    PHASE2_QUALITY_METRICS,
    PHASE3_CVAE,
    PHASE3_WGAN_GP,
    PHASE3A_GATE,
)

# Light reference palette (validated: adjacent CVD ΔE >= 9.1; aqua and yellow need direct labels).
CLASS_COLORS = {"N": "#2a78d6", "S": "#eb6834", "V": "#1baf7a", "F": "#eda100"}
ARM_COLORS = {"scratch": "#2a78d6", "finetune": "#eb6834", "pretrain": "#1baf7a"}
FAMILY_COLORS = {"cvae": "#2a78d6", "wgan_gp": "#eb6834"}  # first two hues, fixed order
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEGMENT_SECONDS = 8.0


def _plt():
    from ecgaug.eval.plotting import import_pyplot

    plt = import_pyplot()
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 10,
            "axes.titleweight": "bold", "axes.labelsize": 9, "xtick.color": INK_2, "ytick.color": INK_2,
            "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
            "font.size": 9, "lines.linewidth": 1.6, "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        }
    )  # fmt: skip
    return plt


def _save(fig, path: Path, title: str, caption: str) -> Path:
    height = fig.get_size_inches()[1]
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / height), h_pad=1.6, w_pad=1.2)
    fig.suptitle(title, x=0.01, y=1 - 0.08 / height, ha="left", va="top", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.01, -0.02, caption, ha="left", va="top", fontsize=8, color=INK_2, wrap=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------------------
# phase1_data: MIT-BIH data, before and after each preprocessing step
# --------------------------------------------------------------------------------------------------


def _phase1_specs() -> dict:
    from ecgaug.data.representation import RepresentationSpec
    from ecgaug.experiments.phase1_input_representation import load_config
    from ecgaug.workflow import variant_configs

    return {letter: (path.stem, RepresentationSpec(**load_config(path)["representation"])) for letter, path in variant_configs().items()}


def _train_records() -> tuple[int, ...]:
    from ecgaug.data.splits import load_approved_split

    return load_approved_split().train


def phase1_data_figures() -> list[Path]:
    """Raw versus filtered signal, the four inputs for the same beats, class counts, class averages."""
    import pandas as pd

    from ecgaug.data import raw
    from ecgaug.data.beats import extract_beats
    from ecgaug.data.contract import build_arrays
    from ecgaug.data.records import CLASSES, FS
    from ecgaug.data.representation import bandpass, transform_record

    plt = _plt()
    out, written = paths.figures_for(PHASE1_DATA), []
    specs = _phase1_specs()
    train = _train_records()
    counts = pd.read_csv(paths.results_for(PHASE1_DATA) / "record_counts.csv")
    counts = counts[counts["record"].isin(train)]
    record = int(counts.sort_values("S", ascending=False)["record"].iloc[0])  # the training record with most S beats

    signal = raw.read_signal(record)
    beats = extract_beats(record)
    first_s = int(beats.loc[beats["aami"] == "S", "r_idx"].iloc[0])
    start = max(0, first_s - int(3 * FS))
    stop = start + int(SEGMENT_SECONDS * FS)
    t = np.arange(start, stop) / FS
    seg = beats[(beats["r_idx"] >= start) & (beats["r_idx"] < stop)]

    # 1. raw signal and the band-passed signal of variant B
    band = specs["B"][1].bandpass_hz
    filtered = bandpass(signal, band, specs["B"][1].bandpass_order)
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True)
    for ax, y, name in ((axes[0], signal, "before: raw MLII"), (axes[1], filtered, f"after: {band[0]}–{band[1]:g} Hz band-pass (variant B)")):
        ax.plot(t, y[start:stop], color=INK_2, linewidth=0.9)
        for cls in CLASSES:
            r = seg.loc[seg["aami"] == cls, "r_idx"].to_numpy()
            if r.size:
                ax.scatter(r / FS, y[r], s=28, color=CLASS_COLORS[cls], zorder=3, label=cls, edgecolor=SURFACE, linewidth=1)
        ax.set_title(name, loc="left")
        ax.set_ylabel("mV")
    axes[0].legend(title="beat class (cardiologists' labels)", ncol=4, loc="lower right", bbox_to_anchor=(1.0, 1.0), title_fontsize=8)
    axes[1].set_xlabel(f"time in record {record} (s)")
    written.append(_save(fig, out / "01_raw_vs_filtered_signal.png", f"Record {record} (training): raw signal and band-pass filter",
                         "Coloured dots are annotated beats. The filter removes baseline wander and high-frequency noise; only variant B uses it."))  # fmt: skip

    # 2. the same two beats as each variant presents them to the model
    n_beat = int(seg.loc[seg["aami"] == "N", "r_idx"].iloc[0])
    s_row = seg.loc[seg["aami"] == "S"].iloc[0]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
    for ax, (letter, (name, spec)) in zip(axes, specs.items()):
        windows = transform_record(signal, np.array([n_beat, int(s_row["r_idx"])]), spec)
        pre, _ = spec.window_samples
        ms = (np.arange(spec.n_samples) - pre) / spec.effective_fs * 1000
        for w, cls in zip(windows, ("N", "S")):
            ax.plot(ms, w, color=CLASS_COLORS[cls], label=f"{cls} beat")
            ax.annotate(cls, (ms[-1], w[-1]), xytext=(3, 0), textcoords="offset points", color=INK_2, fontsize=8, va="center")
        ax.axvline(0, color=GRID, linewidth=1)
        ax.set_title(f"{letter}: {spec.n_inputs} inputs", loc="left")
        ax.set_xlabel("ms from R peak")
        if spec.rr_features:
            ax.text(0.02, 0.04, f"+ RR features of the S beat:\nrr_prev {s_row['rr_prev']:.2f} s, ratio {s_row['rr_ratio']:.2f}",
                    transform=ax.transAxes, fontsize=7, color=INK_2)  # fmt: skip
    axes[0].set_ylabel("z-scored amplitude")
    axes[0].legend(loc="upper left")
    written.append(_save(fig, out / "02_same_beats_in_each_variant.png", "What the model sees: one N and one S beat in variants A–D",
                         f"Record {record}. After windowing and per-beat z-scoring. The S beat looks normal; only C adds the timing (RR) that marks it as premature."))  # fmt: skip

    # 3. class counts: annotated in all of DS1 versus used for training and validation
    split_counts = {}
    from ecgaug.data.splits import load_approved_split

    split = load_approved_split()
    all_counts = pd.read_csv(paths.results_for(PHASE1_DATA) / "record_counts.csv")
    for part, records in (("train", split.train), ("validation", split.val)):
        rows = all_counts[all_counts["record"].isin(records)]
        split_counts[part] = [int(rows[c].sum()) for c in CLASSES]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(CLASSES))
    for i, (part, values) in enumerate(split_counts.items()):
        bars = ax.bar(x + (i - 0.5) * 0.38, values, width=0.36, color=["#2a78d6", "#eb6834"][i], label=part, edgecolor=SURFACE, linewidth=2)
        for bar, v in zip(bars, values):
            ax.annotate(f"{v:,}", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=8, color=INK)  # fmt: skip
    ax.set_yscale("log")
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("beats (log scale)")
    ax.legend()
    written.append(_save(fig, out / "03_class_counts_train_vs_validation.png", "Beats per class: training and validation (DS1, split V1)",
                         "Log scale. S and F are the rare classes; validation holds only 27 F beats, so selection uses S F1."))  # fmt: skip

    # 4. class-average beats per variant (training records only): what each class looks like after preprocessing
    fig, axes = plt.subplots(len(specs), len(CLASSES), figsize=(12, 2.3 * len(specs)), sharey=True)
    for row, (letter, (name, spec)) in enumerate(specs.items()):
        arrays, _ = build_arrays(list(train), spec, {r: "train" for r in train})
        pre, _ = spec.window_samples
        ms = (np.arange(spec.n_samples) - pre) / spec.effective_fs * 1000
        for col, cls in enumerate(CLASSES):
            ax = axes[row, col]
            x_cls = arrays["X"][arrays["y"] == col]
            if len(x_cls):
                q1, med, q3 = np.percentile(x_cls, [25, 50, 75], axis=0)
                ax.fill_between(ms, q1, q3, color=CLASS_COLORS[cls], alpha=0.25, linewidth=0)
                ax.plot(ms, med, color=CLASS_COLORS[cls])
            ax.set_title(f"{letter} · {cls} (n={len(x_cls):,})", loc="left", fontsize=9)
            if row == len(specs) - 1:
                ax.set_xlabel("ms from R peak")
        axes[row, 0].set_ylabel("z-scored")
    written.append(_save(fig, out / "04_class_average_beats_per_variant.png", "Median beat per class and variant, with the middle 50% shaded",
                         "Training records only. Rows: input variants A–D. Columns: classes. S looks almost like N in every variant, which is why S needs RR timing."))  # fmt: skip
    return written


# --------------------------------------------------------------------------------------------------
# phase2_ptbxl_data: PTB-XL preprocessing
# --------------------------------------------------------------------------------------------------


def phase2_ptbxl_data_figures(root: Path | None = None) -> list[Path]:
    """Raw 500 Hz versus resampled signal with detected beats; exclusions; labels; shape versus MIT-BIH."""
    from ecgaug.data import ptbxl, raw
    from ecgaug.data.representation import RepresentationSpec

    plt = _plt()
    out, written = paths.figures_for(PHASE2_PTBXL_DATA), []
    root = Path(root or paths.repo_root() / "data" / "raw" / "ptbxl")
    files = sorted(root.rglob("*_hr.hea")) if root.is_dir() else []

    if files:
        # 1. one recording before and after: raw 500 Hz lead II, then 360 Hz with detected and kept beats
        name = str(files[0].relative_to(root))[:-4]
        x, fs = raw.read_ptbxl_lead(root, name)
        processed = ptbxl.process_record(x, fs)
        fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True)
        axes[0].plot(np.arange(x.size) / fs, x, color=INK_2, linewidth=0.9)
        axes[0].set_title(f"before: raw lead II, {fs} Hz, {x.size} samples", loc="left")
        if processed["problem"] is None:
            y, peaks = processed["signal"], processed["peaks"]
            keep, _, _ = ptbxl.beat_table(peaks)
            axes[1].plot(np.arange(y.size) / 360, y, color=INK_2, linewidth=0.9)
            dropped = np.setdiff1d(np.arange(peaks.size), keep)
            axes[1].scatter(peaks[keep] / 360, y[peaks[keep]], s=30, color="#2a78d6", label="detected, kept", zorder=3)
            axes[1].scatter(peaks[dropped] / 360, y[peaks[dropped]], s=30, facecolor=SURFACE, edgecolor="#e34948",
                            linewidth=1.5, label="detected, dropped (window or RR)", zorder=3)  # fmt: skip
            axes[1].legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
            axes[1].set_title(f"after: resampled to 360 Hz ({y.size} samples), R peaks detected", loc="left")
        for ax in axes:
            ax.set_ylabel("mV")
        axes[1].set_xlabel("time (s)")
        written.append(_save(fig, out / "01_resampling_and_detected_beats.png", f"PTB-XL {files[0].stem}: before and after preprocessing",
                             "Beats too close to the ends have no room for a full window and are dropped, as in MIT-BIH."))  # fmt: skip

        # 2. average PTB-XL beat versus average MIT-BIH N beat (variant A)
        spec = RepresentationSpec(name="m1-A", window_pre_s=0.25, window_post_s=0.25)
        windows = []
        for f in files:
            xr, fsr = raw.read_ptbxl_lead(root, str(f.relative_to(root))[:-4])
            p = ptbxl.process_record(xr, fsr)
            if p["problem"] is None:
                keep, _, _ = ptbxl.beat_table(p["peaks"])
                if keep.size:
                    windows.append(ptbxl.windows_for_record(p["signal"], p["peaks"][keep], spec))
        if windows:
            from ecgaug.data.contract import build_arrays

            ptb = np.concatenate(windows)
            train = _train_records()
            mit, _ = build_arrays(list(train), spec, {r: "train" for r in train})
            mit_n = mit["X"][mit["y"] == 0]
            ms = (np.arange(spec.n_samples) - 90) / 360 * 1000
            fig, ax = plt.subplots(figsize=(7, 3.6))
            for data, label, color in ((mit_n, f"MIT-BIH N beats, training (n={len(mit_n):,})", "#2a78d6"),
                                       (ptb, f"PTB-XL beats here (n={len(ptb):,})", "#eb6834")):  # fmt: skip
                q1, med, q3 = np.percentile(data, [25, 50, 75], axis=0)
                ax.fill_between(ms, q1, q3, color=color, alpha=0.2, linewidth=0)
                ax.plot(ms, med, color=color, label=label)
            corr = float(np.corrcoef(np.median(ptb, axis=0), np.median(mit_n, axis=0))[0, 1])
            ax.set_xlabel("ms from R peak")
            ax.set_ylabel("z-scored")
            ax.legend(loc="upper right")
            written.append(_save(fig, out / "02_ptbxl_vs_mitbih_beat_shape.png", f"Do PTB-XL beats look like MIT-BIH beats? (correlation of medians {corr:.2f})",
                                 "Variant A window. Median with the middle 50% shaded. PTB-XL beats here are from the recordings on this machine."))  # fmt: skip

    report_path = paths.results_for(PHASE2_PTBXL_DATA) / "ptbxl_preprocessing.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        # 3. what was kept after each check
        stages = [("in metadata", report["in_metadata"]), ("eligible (labels, no pacemaker, electrodes ok)", report["eligible"]),
                  ("kept after signal checks", report["recordings_kept"])]  # fmt: skip
        reasons = {**report["excluded_by_metadata"], **report["excluded_by_signal"]}
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.4), gridspec_kw={"width_ratios": [1.1, 1]})
        labels, values = zip(*stages)
        bars = axes[0].barh(labels[::-1], values[::-1], color="#2a78d6", height=0.5)
        for bar, v in zip(bars, values[::-1]):
            axes[0].annotate(f"{v:,}", (v, bar.get_y() + bar.get_height() / 2), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
        axes[0].set_title("recordings kept after each check", loc="left")
        axes[0].set_xlabel("recordings")
        names, counts = zip(*sorted(reasons.items(), key=lambda kv: kv[1]))
        bars = axes[1].barh([n.replace("_", " ") for n in names], counts, color="#e34948", height=0.5)
        for bar, v in zip(bars, counts):
            axes[1].annotate(f"{v:,}", (v, bar.get_y() + bar.get_height() / 2), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
        axes[1].set_title("recordings excluded, by reason", loc="left")
        axes[1].set_xlabel("recordings")
        written.append(_save(fig, out / "03_recordings_kept_and_excluded.png",
                             f"PTB-XL preprocessing: {report['recordings_kept']:,} recordings and {report['beats_kept']:,} beats kept",
                             "From results/phase2_ptbxl_data/ptbxl_preprocessing.json (the full run on Kaggle)."))  # fmt: skip

        # 4. labels per split
        classes = list(report["settings"]["superclasses"])
        fig, ax = plt.subplots(figsize=(7, 3.4))
        x_pos = np.arange(len(classes))
        for i, part in enumerate(("train", "val")):
            per = report["per_split"][part]
            share = [per["recordings_per_class"][c] / per["recordings"] for c in classes]
            bars = ax.bar(x_pos + (i - 0.5) * 0.38, share, width=0.36, color=["#2a78d6", "#eb6834"][i], edgecolor=SURFACE, linewidth=2,
                          label=f"{'training (folds 1–8)' if part == 'train' else 'validation (fold 9)'}: {per['recordings']:,} recordings")  # fmt: skip
            for bar, v in zip(bars, share):
                ax.annotate(f"{v:.0%}", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8)
        ax.set_xticks(x_pos, classes)
        ax.set_ylabel("share of recordings")
        ax.legend(loc="upper right")
        written.append(_save(fig, out / "04_label_share_per_split.png", "PTB-XL diagnostic superclasses: training versus validation",
                             "A recording can carry several labels, so shares add up to more than 100%. Every beat inherits its recording's labels."))  # fmt: skip
    return written


# --------------------------------------------------------------------------------------------------
# Experiment results: training curves, confusion matrices, the phase 1 gate
# --------------------------------------------------------------------------------------------------


def _log(run_id: str) -> list[dict]:
    with (runs.find_run_dir(run_id) / "train_log.csv").open() as fh:
        return list(csv.DictReader(fh))


def _confusion(run_ids: list[str]) -> np.ndarray:
    cms = [np.array(json.loads((runs.find_run_dir(r) / "metrics_val.json").read_text())["confusion_matrix"]) for r in run_ids]
    return np.mean(cms, axis=0)


def _curves(ax_loss, ax_val, arm: str, run_ids: list[str], color: str, metric: str) -> None:
    for i, run_id in enumerate(run_ids):
        log = _log(run_id)
        epochs = [int(r["epoch"]) for r in log]
        ax_loss.plot(epochs, [float(r["train_loss"]) for r in log], color=color, alpha=0.85, label=arm if i == 0 else None)
        values = [float(r["val_selection"]) for r in log]
        ax_val.plot(epochs, values, color=color, alpha=0.85, label=arm if i == 0 else None)
        best = int(np.argmax(values))
        ax_val.scatter(epochs[best], values[best], s=36, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax_loss.set_title("training loss (before → after training)", loc="left")
    ax_val.set_title(f"validation {metric} (dot = epoch kept)", loc="left")
    for ax in (ax_loss, ax_val):
        ax.set_xlabel("epoch")


def _confusion_panel(ax, cm: np.ndarray, title: str) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    rows = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(rows, cmap=LinearSegmentedColormap.from_list("blues", BLUES), vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{rows[i, j]:.0%}\n{cm[i, j]:,.0f}", ha="center", va="center", fontsize=7,
                    color=SURFACE if rows[i, j] > 0.55 else INK)  # fmt: skip
    ax.set_xticks(range(4), list("NSVF"))
    ax.set_yticks(range(4), list("NSVF"))
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.grid(False)
    ax.set_title(title, loc="left")


def phase2_pretraining_figures(result_path: Path) -> list[Path]:
    """Figures for one phase2_ptbxl_pretraining result: curves, confusion matrices, per-class F1 before and after."""
    plt = _plt()
    result = json.loads(Path(result_path).read_text())
    letter = result["variant"]
    out, written = paths.figures_for(PHASE2_PTBXL_PRETRAINING) / letter, []
    metric = "S F1" if result["selection_metric"] == "s_f1" else result["selection_metric"]
    tag = " (provisional preview)" if result.get("provisional") else ""

    pre_ids = [p["run_id"] for p in result["pretraining"]["per_seed"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for i, run_id in enumerate(pre_ids):
        log = _log(run_id)
        epochs = [int(r["epoch"]) for r in log]
        axes[0].plot(epochs, [float(r["train_loss"]) for r in log], color="#1baf7a", label="seeds" if i == 0 else None)
        axes[1].plot(epochs, [float(r["val_selection"]) for r in log], color="#1baf7a")
    axes[0].set_title("training loss on PTB-XL", loc="left")
    axes[1].set_title("macro AUROC on PTB-XL fold 9", loc="left")
    for ax in axes:
        ax.set_xlabel("epoch")
    written.append(_save(fig, out / "01_pretraining_curves.png", f"phase2_ptbxl_pretraining, variant {letter}: pretraining on PTB-XL{tag}",
                         f"One line per seed. Best macro AUROC {result['pretraining']['val_macro_auroc_mean']:.3f}."))  # fmt: skip

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for arm, kind in (("scratch", "scratch"), ("pretrained", "finetune")):
        _curves(axes[0], axes[1], kind, result["arms"][arm]["run_ids"], ARM_COLORS[kind], metric)
    axes[1].legend(loc="upper right")
    written.append(_save(fig, out / "02_finetune_vs_scratch_curves.png", f"phase2_ptbxl_pretraining, variant {letter}: from scratch versus fine-tuned{tag}",
                         "One line per seed at the chosen learning rate. Blue: from scratch. Orange: fine-tuned from PTB-XL."))  # fmt: skip

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    _confusion_panel(axes[0], _confusion(result["arms"]["scratch"]["run_ids"]), "from scratch")
    _confusion_panel(axes[1], _confusion(result["arms"]["pretrained"]["run_ids"]), "fine-tuned from PTB-XL")
    written.append(_save(fig, out / "03_confusion_matrices.png", f"phase2_ptbxl_pretraining, variant {letter}: validation confusion matrices{tag}",
                         "Mean over seeds. Each row shows where the beats of one true class went; the diagonal is correct."))  # fmt: skip

    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(4)
    for i, (arm, kind) in enumerate((("scratch", "scratch"), ("pretrained", "finetune"))):
        per_class = np.array([[s["f1"][c] for c in "NSVF"] for s in result["arms"][arm]["per_seed"]])
        mean, sd = per_class.mean(axis=0), per_class.std(axis=0, ddof=1) if len(per_class) > 1 else np.zeros(4)
        bars = ax.bar(x + (i - 0.5) * 0.38, mean, width=0.36, yerr=sd, color=ARM_COLORS[kind], edgecolor=SURFACE, linewidth=2,
                      error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2}, label=kind)  # fmt: skip
        for bar, v, e in zip(bars, mean, sd):
            ax.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, v + e), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x, list("NSVF"))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 on validation")
    ax.legend(loc="upper right")
    written.append(_save(fig, out / "04_f1_per_class_before_after_pretraining.png", f"phase2_ptbxl_pretraining, variant {letter}: F1 per class, from scratch versus fine-tuned{tag}",
                         f"Mean ± SD over seeds. Winner by the selection rule: {result['winner']}."))  # fmt: skip
    return written


def phase1_variant_figures(result_path: Path) -> list[Path]:
    """Figures for one classifier variant (phase1_input_representation, phase2_baseline_classifiers):
    training curves and the validation confusion matrix."""
    plt = _plt()
    result = json.loads(Path(result_path).read_text())
    letter = result["variant"]
    experiment = result.get("experiment", PHASE1_INPUT_REPRESENTATION)
    out, written = paths.figures_for(experiment) / letter, []
    metric = "S F1" if result["selection_metric"] == "s_f1" else result["selection_metric"]
    tag = " (provisional preview)" if result.get("provisional") else ""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    _curves(axes[0], axes[1], f"variant {letter}", result["run_ids"], CLASS_COLORS["N"], metric)
    _confusion_panel(axes[2], _confusion(result["run_ids"]), "validation confusion (mean over seeds)")
    written.append(_save(fig, out / "training_and_confusion.png", f"{experiment}, variant {letter}: {metric} {result['mean']:.3f} ± {result['sd']:.3f}{tag}",
                         f"One line per seed at the chosen learning rate ({result['chosen_lr']:g})."))  # fmt: skip
    return written


def phase1_gate_figure() -> list[Path]:
    """All four phase1_input_representation variants side by side: what the phase 1 gate decides on."""
    from ecgaug.workflow import phase1_results

    results = {k: v for k, v in phase1_results().items() if v}
    if not results:
        return []
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, (letter, r) in enumerate(results.items()):
        seeds = [s["selection_value"] for s in r["per_seed"]]
        ax.scatter([i] * len(seeds), seeds, s=22, color="#2a78d6", alpha=0.6, zorder=3)
        ax.errorbar(i, r["mean"], yerr=r["sd"], fmt="o", color=INK, markersize=6, capsize=3, zorder=4)
        ax.annotate(f"{r['mean']:.3f}", (i, r["mean"]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8)
    ax.set_xticks(range(len(results)), [f"{k}\n{v['complexity_inputs_per_beat']} inputs" for k, v in results.items()])
    metric = next(iter(results.values()))["selection_metric"]
    ax.set_ylabel("S F1 on validation" if metric == "s_f1" else metric)
    out = paths.figures_for(PHASE1_GATE)
    written = [_save(fig, out / "variants.png", "Phase 1 gate: the phase1_input_representation variants on the validation split",
                     "Small dots: seeds. Black: mean ± SD. The rule picks the best mean; ties within the margin go to fewer inputs.")]  # fmt: skip
    return written + _phase1_per_class_figure(results, out) + _phase1_early_stopping_figure(results, out)


def _phase1_per_class_figure(results: dict, out: Path) -> list[Path]:
    """Per-class F1 (decisions) and PR-AUC (ranking) for each variant at its chosen learning rate."""
    plt = _plt()
    letters = list(results)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), sharey=True)
    panels = [("S", "S"), ("V", "V"), ("F", "F"), ("N", "N recall")]
    for ax, (cls, title) in zip(axes, panels):
        color = CLASS_COLORS[cls]
        for i, letter in enumerate(letters):
            seeds = results[letter]["per_seed"]
            if cls == "N":
                pairs = [("recall", [s["n_recall"] for s in seeds], "o")]
            else:
                pairs = [("F1", [s["f1"][cls] for s in seeds], "o"), ("PR-AUC", [s["pr_auc"][cls] for s in seeds], "D")]
            for j, (name, values, marker) in enumerate(pairs):
                x = i + (j - (len(pairs) - 1) / 2) * 0.28
                filled = name != "PR-AUC"
                ax.scatter([x] * len(values), values, s=26, marker=marker, zorder=3, linewidth=1.4,
                           color=color if filled else SURFACE, edgecolor=color,
                           label=name if i == 0 else None)  # fmt: skip
                ax.hlines(np.mean(values), x - 0.11, x + 0.11, color=INK, linewidth=1.6, zorder=4)
        ax.set_xticks(range(len(letters)), letters)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(title, loc="left")
        ax.legend(loc="upper left")
    axes[0].set_ylabel("validation score")
    return [_save(fig, out / "per_class.png", "Phase 1 gate: what each variant gets right, per class",
                  "Each mark is one seed at the chosen learning rate; black ticks are means. F1 uses argmax decisions; "
                  "PR-AUC measures ranking and does not depend on the decision threshold.")]  # fmt: skip


def _phase1_early_stopping_figure(results: dict, out: Path) -> list[Path]:
    """Kept epoch vs epochs run for every run of the grid; runs that kept epoch 1 stopped as early as patience allows."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 3.2))
    labels = []
    for letter, r in results.items():
        for tag, point in r["lr_grid"].items():
            k = len(labels)
            labels.append(f"{letter}\n{tag.removeprefix('lr')}")
            for s in point["per_seed"]:
                x = k + (s["seed"] - 1) * 0.18
                ax.vlines(x, 0, s["epochs_run"], color=GRID, linewidth=3, zorder=2)
                stuck = s["best_epoch"] == 1
                ax.scatter(x, s["best_epoch"], s=30, zorder=3, color=CLASS_COLORS["S"] if stuck else INK,
                           label=("kept epoch 1 (S F1 was 0 for all of its first 6 epochs)" if stuck else "kept a later epoch")
                           if s["seed"] == 0 or stuck else None)  # fmt: skip
    handles, names = ax.get_legend_handles_labels()
    unique = dict(zip(names, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper left")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("epoch")
    return [_save(fig, out / "early_stopping.png", "Phase 1 gate: where early stopping ended each run",
                  "Grey bar: epochs run (one bar per seed 0, 1, 2). Dot: epoch kept. With patience 5, a run whose "
                  "validation S F1 stays at 0 for its first epochs keeps epoch 1 and stops at epoch 6.")]  # fmt: skip


# --------------------------------------------------------------------------------------------------
# phase 3: generators
# --------------------------------------------------------------------------------------------------

GENERATED = ("S", "V", "F")


def safe_call(fn, *args, **kwargs) -> list[Path]:
    """Run a figure function; a plotting error is reported and never stops a run."""
    try:
        return fn(*args, **kwargs)
    except Exception as err:  # noqa: BLE001 (figures are a by-product; the run's results are already saved)
        print(f"WARNING: figure {fn.__name__} was not made: {err}")
        return []


def _csv(run_id: str, name: str) -> list[dict]:
    with (runs.find_run_dir(run_id) / name).open() as fh:
        return list(csv.DictReader(fh))


def _number(value) -> float:
    return float(value) if value not in (None, "") else float("nan")


def phase3_sample_figures(out: Path, title: str, train: dict, synthetic: dict, spec) -> list[Path]:
    """Before and after adding synthetic beats: real versus synthetic beats per class, class counts,
    and synthetic beats next to their nearest real training beat (the memorization check, by eye)."""
    from ecgaug.data.records import CLASS_INDEX, CLASSES
    from ecgaug.quality.diagnostics import nearest_distances

    plt = _plt()
    written = []
    pre, _ = spec.window_samples
    ms = (np.arange(spec.n_samples) - pre) / spec.effective_fs * 1000
    classes = [c for c in GENERATED if (np.asarray(synthetic["y"]) == CLASS_INDEX[c]).any()]

    fig, axes = plt.subplots(1, len(classes), figsize=(4 * len(classes), 3.3), sharey=True, squeeze=False)
    for ax, cls in zip(axes[0], classes):
        for label, arrays, color, dashes in (("real", train, CLASS_COLORS[cls], ()), ("synthetic", synthetic, INK, (4, 2))):
            x = np.asarray(arrays["X"])[np.asarray(arrays["y"]) == CLASS_INDEX[cls]]
            q1, med, q3 = np.percentile(x, [25, 50, 75], axis=0)
            ax.fill_between(ms, q1, q3, color=color, alpha=0.18, linewidth=0)
            ax.plot(ms, med, color=color, dashes=dashes, label=f"{label} (n={len(x):,})")
        ax.axvline(0, color=GRID, linewidth=1)
        ax.set_title(cls, loc="left")
        ax.set_xlabel("ms from R peak")
        ax.legend(loc="upper right")
    axes[0][0].set_ylabel("z-scored amplitude")
    written.append(_save(fig, out / "03_real_vs_synthetic.png", f"{title}: real versus synthetic beats",
                         "Median with the middle 50% shaded, aligned on the R peak. Real: training beats of the class. "
                         "Synthetic: generator output after per-beat z-scoring."))  # fmt: skip

    before = [int((np.asarray(train["y"]) == i).sum()) for i in range(len(CLASSES))]
    added = [int((np.asarray(synthetic["y"]) == i).sum()) for i in range(len(CLASSES))]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(CLASSES))
    for offset, values, alpha, label in ((-0.19, before, 0.45, "before: real training beats"), (0.19, [b + a for b, a in zip(before, added)], 1.0, "after: real + synthetic")):
        bars = ax.bar(x + offset, values, width=0.36, color=[CLASS_COLORS[c] for c in CLASSES], alpha=alpha, edgecolor=SURFACE, linewidth=2, label=label)
        for bar, v in zip(bars, values):
            ax.annotate(f"{v:,}", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_yscale("log")
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("training beats (log scale)")
    ax.legend(loc="upper right")
    written.append(_save(fig, out / "04_class_counts_before_after.png", f"{title}: training beats per class before and after",
                         "Pale bars: real training beats. Solid bars: after adding the synthetic beats. Validation and test stay 100% real."))  # fmt: skip

    fig, axes = plt.subplots(len(classes), 3, figsize=(10, 2.2 * len(classes)), sharex=True, squeeze=False)
    for row, cls in enumerate(classes):
        real = np.asarray(train["X"])[np.asarray(train["y"]) == CLASS_INDEX[cls]]
        synth = np.asarray(synthetic["X"])[np.asarray(synthetic["y"]) == CLASS_INDEX[cls]][:3]
        from sklearn.neighbors import NearestNeighbors

        index = NearestNeighbors(n_neighbors=1).fit(real).kneighbors(synth)[1][:, 0]
        distances = nearest_distances(synth, real)
        for col in range(3):
            ax = axes[row][col]
            if col >= len(synth):
                ax.axis("off")
                continue
            ax.plot(ms, real[index[col]], color=CLASS_COLORS[cls], label="nearest real")
            ax.plot(ms, synth[col], color=INK, dashes=(4, 2), label="synthetic")
            ax.set_title(f"{cls} · distance {distances[col]:.1f}", loc="left", fontsize=9)
            if row == len(classes) - 1:
                ax.set_xlabel("ms from R peak")
        axes[row][0].legend(loc="upper right")
    written.append(_save(fig, out / "05_synthetic_and_nearest_real.png", f"{title}: synthetic beats and their nearest real beat",
                         "A synthetic beat drawn on top of a training beat would be a copy; the memorization fraction counts how often that happens."))  # fmt: skip
    return written


def phase3_variant_figures(result_path: Path) -> list[Path]:
    """Figures for one generator pilot result: generator training curves and every checkpoint's scores."""
    plt = _plt()
    result = json.loads(Path(result_path).read_text())
    letter, experiment = result["variant"], result["experiment"]
    out, written = paths.figures_for(experiment) / letter, []
    tag = " (provisional preview)" if result.get("provisional") else ""
    keys = ("reconstruction", "kl") if result["family"] == "cvae" else ("wasserstein", "gradient_penalty")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for i, run_id in enumerate(result["run_ids"]):
        log = _csv(run_id, "train_log.csv")
        if not log:
            continue
        steps = [int(r["step"]) for r in log]
        for ax, key in zip(axes, keys):
            columns = [k for k in log[0] if k == key or k.startswith(f"{key}_")]
            for column in columns:
                cls = column.rsplit("_", 1)[-1] if column != key else None
                ax.plot(steps, [_number(r.get(column)) for r in log], color=CLASS_COLORS.get(cls, INK_2),
                        dashes=() if i == 0 else (4, 2), label=f"seed {i}" + (f" {cls}" if cls else ""))  # fmt: skip
            ax.set_title(key.replace("_", " "), loc="left")
            ax.set_xlabel("generator step")
    axes[1].legend(loc="upper right")
    written.append(_save(fig, out / "01_generator_training.png", f"{experiment}, variant {letter}: generator training{tag}",
                         f"Setting {result['chosen_setting']}. Solid: seed {result['per_seed'][0]['seed']}; dashed: the next seed."))  # fmt: skip

    panels = [("pilot_selection_value", "pilot S F1 on validation" if result["selection_metric"] == "s_f1" else "pilot selection metric", None),
              ("diversity_ratio", "diversity ratio (floor dashed)", result["hard_gate_thresholds"]["diversity_min_ratio"]),
              ("memorization_fraction", "memorization fraction (ceiling dashed)", result["hard_gate_thresholds"]["memorization_max_fraction"]),
              ("c2st_accuracy", "C2ST accuracy (0.5 = indistinguishable)", 0.5)]  # fmt: skip
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.3))
    for i, run_id in enumerate(result["run_ids"]):
        rows = _csv(run_id, "checkpoints.csv")
        if not rows:
            continue
        fraction = [float(r["fraction"]) for r in rows]
        for ax, (key, title, line) in zip(axes, panels):
            if key == "pilot_selection_value":
                ax.plot(fraction, [_number(r[key]) for r in rows], color=INK, marker="o", dashes=() if i == 0 else (4, 2), label=f"seed {i}")
            else:
                for cls in GENERATED:
                    ax.plot(fraction, [_number(r[f"{key}_{cls}"]) for r in rows], color=CLASS_COLORS[cls], marker="o",
                            dashes=() if i == 0 else (4, 2), label=cls if i == 0 else None)  # fmt: skip
            if line is not None and i == 0:
                ax.axhline(line, color=INK_2, linewidth=1, dashes=(2, 2))
            ax.set_title(title, loc="left")
            ax.set_xlabel("fraction of the step budget")
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    written.append(_save(fig, out / "02_checkpoints.png", f"{experiment}, variant {letter}: every checkpoint, scored{tag}",
                         "Solid: first seed, dashed: second. Only the last checkpoint counts for selection; the others are the RQ3 data."))  # fmt: skip
    return written


def phase2_quality_figures(result_path: Path) -> list[Path]:
    """One metric family's sanity test: score against severity for every degradation."""
    plt = _plt()
    result = json.loads(Path(result_path).read_text())
    letter, family = result["variant"], result["family"]
    scores = result["scores"]
    reference = scores["reference:0"]
    kinds = sorted({v["kind"] for v in scores.values() if v["kind"] != "reference"})
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.4 * len(kinds), 3.2), sharey=True, squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        rows = sorted((v for v in scores.values() if v["kind"] == kind), key=lambda v: v["severity"])
        x = [0] + [v["severity"] for v in rows]
        y = [reference["mean"]] + [v["mean"] for v in rows]
        err = [reference["sd"]] + [v["sd"] for v in rows]
        ax.errorbar(x, y, yerr=err, color=INK, marker="o", capsize=3)
        ax.set_xticks(x, ["real"] + [f"{v['level']:g}" for v in rows])
        ax.set_title(f"{kind.replace('_', ' ')} (rho {result['spearman'][kind]:.2f})", loc="left")
        ax.set_xlabel("severity")
    axes[0][0].set_ylabel(f"{family} ({'higher' if result['higher_is_better'] else 'lower'} is better)")
    verdict = "passes" if result["passed"] else "fails"
    tag = " (provisional preview)" if result.get("provisional") else ""
    return [_save(fig, paths.figures_for(PHASE2_QUALITY_METRICS) / letter / "sanity.png",
                  f"phase2_quality_metrics, variant {letter} ({family}): {verdict} the sanity harness{tag}",
                  "Mean ± SD over resamples. 'real' is the unchanged held-out beats; the score should get worse to the right.")]  # fmt: skip


def phase2_gate_figure(gate: dict) -> list[Path]:
    """The baseline classifiers side by side, grouped by role (cnn: A, B; rnn: C, D)."""
    plt = _plt()
    results = gate["baselines"]["results"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, (letter, r) in enumerate(results.items()):
        seeds = [s["selection_value"] for s in r["per_seed"]]
        ax.scatter([i] * len(seeds), seeds, s=22, color=CLASS_COLORS["N"], alpha=0.6, zorder=3)
        ax.errorbar(i, r["mean"], yerr=r["sd"], fmt="o", color=INK, markersize=6, capsize=3, zorder=4)
        ax.annotate(f"{r['mean']:.3f}", (i, r["mean"]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8)
    ax.set_xticks(range(len(results)), [f"{k} ({v['role']})\n{v['n_parameters']:,} params" for k, v in results.items()])
    ax.set_ylabel("S F1 on validation")
    picks = ", ".join(f"{role}: {gate['baselines'][role]['winner']}" for role in ("cnn", "rnn"))
    return [_save(fig, paths.figures_for(PHASE2_GATE) / "baselines.png", f"Phase 2 gate: baseline classifiers ({picks})",
                  "Small dots: seeds. Black: mean ± SD. The cnn is chosen from A and B, the rnn from C and D; ties go to fewer parameters.")]  # fmt: skip


def phase5_saliency_figure(result_path: Path) -> list[Path]:
    """Class-average absolute saliency aligned on R: none versus augmented models, per architecture (paper figure)."""
    plt = _plt()
    result = json.loads(Path(result_path).read_text())
    letter = result["variant"]
    averages = json.loads((Path(result_path).parent / f"{letter}_class_average_saliency.json").read_text())
    archs = [a for a in averages["maps"] if averages["maps"][a]]
    if not archs:
        return []
    fig, axes = plt.subplots(len(archs), 4, figsize=(13, 2.6 * len(archs)), squeeze=False, sharex=True)
    colors = {"none": INK, "cvae": CLASS_COLORS["S"], "wgan_gp": CLASS_COLORS["V"]}
    for row, arch in enumerate(archs):
        for col, cls in enumerate(("N", "S", "V", "F")):
            ax = axes[row][col]
            for condition, maps in averages["maps"][arch].items():
                if cls in maps:
                    values = np.asarray(maps[cls])
                    ms = (np.arange(len(values)) - averages["pre_samples"]) / averages["fs"] * 1000
                    ax.plot(ms, values / max(values.max(), 1e-12), color=colors.get(condition, INK_2), label=condition)
            ax.axvline(0, color=GRID, linewidth=1)
            ax.set_title(f"{arch} · {cls}", loc="left")
            if row == len(archs) - 1:
                ax.set_xlabel("ms from R peak")
        axes[row][0].legend(loc="upper left", fontsize=7)
    tag = " (provisional preview)" if result.get("provisional") else ""
    return [_save(fig, paths.figures_for("phase5_explainability") / letter / "class_average_saliency.png",
                  f"phase5_explainability, variant {letter} ({result['method']}): where the models look{tag}",
                  "Mean absolute saliency over validation beats of each class, scaled to a peak of 1; aligned on the R peak.")]  # fmt: skip


def error_analysis_figures(result: dict, split: str) -> list[Path]:
    """Confusion matrices per architecture and condition, and the RR context of N/S confusions."""
    plt = _plt()
    out, written = paths.figures_for("error_analysis"), []
    keys = list(result)
    cols = min(3, len(keys))
    rows = int(np.ceil(len(keys) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 3.9 * rows), squeeze=False)
    for ax, key in zip(axes.ravel(), keys):
        _confusion_panel(ax, np.asarray(result[key]["confusion_matrix"]), key)
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    written.append(_save(fig, out / f"confusion_{split}.png", f"Error analysis ({split}): confusion matrices, mean over seeds",
                         "Each row shows where the beats of one true class went; the diagonal is correct."))  # fmt: skip
    fig, axes = plt.subplots(1, len(keys), figsize=(4.3 * len(keys), 3.3), squeeze=False, sharey=True)
    for ax, key in zip(axes[0], keys):
        groups = [(name, g["rr_ratio_values"]) for name, g in result[key]["rr_context"].items() if g["rr_ratio_values"]]
        if groups:
            ax.boxplot([g[1] for g in groups], showfliers=False)
            ax.set_xticks(range(1, len(groups) + 1), [g[0] for g in groups], rotation=35, ha="right")
        ax.axhline(1.0, color=GRID, linewidth=1)
        ax.set_title(key, loc="left")
    axes[0][0].set_ylabel("rr_ratio (below 1 = premature)")
    written.append(_save(fig, out / f"rr_context_{split}.png", f"Error analysis ({split}): how premature the confused beats were",
                         "rr_ratio is the preceding RR interval over the mean of the 10 before it; S beats are early (below 1)."))  # fmt: skip
    return written


def final_eval_figures(stats: dict) -> list[Path]:
    """Forest plot of the pre-registered differences, and per-record S and F recall on the test set."""
    plt = _plt()
    out, written = paths.figures_for("final_eval"), []
    rows = [(family, name, r) for family, results in stats["families"].items() for name, r in results.items()]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.2))
    for i, (family, name, r) in enumerate(reversed(rows)):
        color = INK if family == "confirmatory" else INK_2
        ax.errorbar(r["point"], i, xerr=[[r["point"] - r["ci_low"]], [r["ci_high"] - r["point"]]], fmt="o", color=color, capsize=3)
        ax.annotate(f"Holm p {r['p_holm']:.3f}", (r["ci_high"], i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7, color=INK_2)
    ax.axvline(0, color=GRID, linewidth=1.2)
    ax.set_yticks(range(len(rows)), [f"{family[:4]}. {name}" for family, name, _ in reversed(rows)])
    ax.set_xlabel("difference in test rare-class macro-F1 (95% record-level bootstrap CI)")
    written.append(_save(fig, out / "forest.png", "Final evaluation: pre-registered comparisons on the test set",
                         "Black: confirmatory family; grey: secondary. Intervals are unadjusted; p-values are Holm-adjusted within each family."))  # fmt: skip
    keys = list(stats["per_record"])
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("recall", BLUES[1:]).with_extremes(bad=SURFACE)
    sizes = {cls: [r for r in stats["records"] if any(stats["per_record"][k][str(r)][cls]["support"] for k in keys)] for cls in ("S", "F")}
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.32 * len(keys) + 1.6), gridspec_kw={"width_ratios": [len(sizes["S"]), len(sizes["F"]), 0.3]})
    for ax, cls in zip(axes[:2], ("S", "F")):
        records = sizes[cls]
        grid = np.array([[stats["per_record"][k][str(r)][cls]["recall"] for r in records] for k in keys], dtype=float)
        image = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i, j in np.ndindex(grid.shape):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}"[1:] if grid[i, j] < 1 else "1", ha="center", va="center", fontsize=6,
                        color=SURFACE if grid[i, j] > 0.55 else INK)  # fmt: skip
        support = [stats["per_record"][keys[0]][str(r)][cls]["support"] for r in records]
        ax.set_xticks(range(len(records)), [f"{r}\nn={n:,}" for r, n in zip(records, support)], fontsize=7)
        ax.set_yticks(range(len(keys)), keys if cls == "S" else [""] * len(keys), fontsize=7)
        ax.grid(False)
        ax.set_title(f"{cls} recall per test record", loc="left")
    fig.colorbar(image, cax=axes[2], label="recall (seed mean)")
    written.append(_save(fig, out / "per_record_S_F.png", "Final evaluation: S and F recall per test record",
                         "Rows: architecture/condition at the phase 4 gate's strength; columns: records holding S (left) or F (right) beats, "
                         "with n = beats of that class in the record. A few records hold most of each class."))  # fmt: skip
    return written


def external_data_figure() -> list[Path]:
    """P2 data: class-average beats of INCART and SVDB beside the MIT-BIH training beats (same representation)."""
    from ecgaug.data.loaders import load_trainval

    plt = _plt()
    train = load_trainval("v1")
    sets = {"MIT-BIH DS1 training": {k: np.asarray(v)[np.asarray(train["split"]) == "train"] for k, v in train.items()}}
    for db, label in (("incartdb", "INCART (lead II)"), ("svdb", "SVDB (ECG1)")):
        path = paths.processed_dir() / "external" / f"{db}.npz"
        if path.is_file():
            with np.load(path) as f:
                sets[label] = {"X": f["X"], "y": f["y"]}
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]  # first three hues, fixed order
    fig, axes = plt.subplots(1, len(CLASS_COLORS), figsize=(13, 3.2), sharey=True)
    t = (np.arange(next(iter(sets.values()))["X"].shape[1]) - 90) / 360 * 1000
    for ax, (i, cls) in zip(axes, enumerate(CLASS_COLORS)):
        for color, (label, arrays) in zip(colors, sets.items()):
            beats = arrays["X"][arrays["y"] == i]
            if len(beats):
                ax.plot(t, beats.mean(axis=0), color=color, label=f"{label} (n={len(beats):,})")
        ax.set_title(cls, loc="left")
        ax.set_xlabel("ms from R peak")
        ax.legend(loc="lower right", fontsize=6)
    axes[0].set_ylabel("z-scored amplitude (class mean)")
    caption = ("Beats cut exactly as the frozen dataset (0.5 s at 360 Hz around the reference R peak, z-scored per beat); INCART resampled from 257 Hz, "
               "SVDB from 128 Hz. Differences in shape are what the frozen models meet on external data. Post hoc (PREREGISTRATION v4, P2).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_external_validation") / "class_mean_beats.png", "External databases against the training data", caption)]


def leakage_audit_figure(summary: dict) -> list[Path]:
    """RQ2: the cnn's rare-class macro-F1 on a random beat-level split (intra-patient) beside the final evaluation (inter-patient)."""
    plt = _plt()
    names = {"none": "none", "cvae": "cVAE", "wgan_gp": "WGAN-GP"}
    conditions = [c for c in names if c in summary["conditions"]]
    intra_color, inter_color = "#2a78d6", "#eb6834"  # first two hues, fixed order
    fig, ax = plt.subplots(figsize=(8, 0.7 * len(conditions) + 1.4))
    for i, condition in enumerate(reversed(conditions)):
        intra, inter = summary["conditions"][condition]["intra"], summary["conditions"][condition]["inter"]
        ax.scatter(intra["per_seed"], [i + 0.15] * len(intra["per_seed"]), s=22, color=intra_color, alpha=0.45, edgecolor=SURFACE, linewidth=0.6)
        ax.errorbar(intra["mean"], i + 0.15, xerr=intra["sd"], fmt="o", color=intra_color, capsize=3, label="intra-patient (random beat split)" if i == 0 else None)
        ax.errorbar(inter["point"], i - 0.15, xerr=[[inter["point"] - inter["ci_low"]], [inter["ci_high"] - inter["point"]]], fmt="o",
                    color=inter_color, capsize=3, label="inter-patient (unseen patients, final evaluation)" if i == 0 else None)  # fmt: skip
        ax.annotate(f"{intra['mean']:.3f}", (intra["mean"], i + 0.15), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7, color=INK_2)
        ax.annotate(f"{inter['point']:.3f}", (inter["point"], i - 0.15), xytext=(0, -11), textcoords="offset points", ha="center", fontsize=7, color=INK_2)
    ax.set_yticks(range(len(conditions)), [names[c] for c in reversed(conditions)])
    ax.set_ylim(-0.6, len(conditions) - 0.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel("test rare-class macro-F1 (cnn)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    caption = ("Intra-patient: mean ± SD over seeds (faint dots: seeds), generative conditions at the grid's middle strength. "
               "Inter-patient: the final evaluation with its 95% record-level interval, at the strengths chosen on validation. "
               "Run after the final evaluation; never used for any decision. Exploratory (H2).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase4_leakage_audit") / "intra_vs_inter.png", "RQ2: the same pipeline, with and without patient overlap", caption)]


def _interval(ax, value: dict, y: float, color: str, label: str | None = None, marker: str = "o") -> None:
    ax.errorbar(value["point"], y, xerr=[[value["point"] - value["ci_low"]], [value["ci_high"] - value["point"]]], fmt=marker, color=color, capsize=3, label=label)


def _summary(experiment: str, name: str = "summary.json") -> dict | None:
    path = paths.results_for(experiment) / name
    return json.loads(path.read_text()) if path.is_file() else None


def rr_reference_figure(summary: dict, final: dict) -> list[Path]:
    """P1: the RR references beside the cnn without augmentation and with cVAE, on the test set."""
    plt = _plt()
    rows = [("cnn, cVAE", final["conditions"]["cnn/cvae"]), ("cnn, no augmentation", final["conditions"]["cnn/none"]),
            *((f"reference {r}", e) for r, e in summary["references"].items())]  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.5 * len(rows) + 1.4), sharey=True)
    for ax, metric, xlabel in ((axes[0], "rare_macro_f1", "test rare-class macro-F1"), (axes[1], "f1_S", "test S F1")):
        for i, (name, entry) in enumerate(reversed(rows)):
            _interval(ax, entry[metric], i, "#2a78d6" if name.startswith("cnn") else "#eb6834")
            ax.annotate(f"{entry[metric]['point']:.3f}", (entry[metric]["ci_high"], i), xytext=(5, 0), textcoords="offset points", va="center", fontsize=7, color=INK_2)
        ax.set_xlabel(f"{xlabel} (95% record-level CI)")
        ax.set_xlim(left=0)
    axes[0].set_yticks(range(len(rows)), [name for name, _ in reversed(rows)])
    diff = summary["references"]["rr_lda"]["vs_cnn"]["none"]
    caption = (f"Blue: the cnn of the main study (3 seeds, final evaluation). Orange: fixed linear discriminants fitted once on the training records "
               f"(rr_lda: three RR features; rr_morph_lda: RR features plus every 10th window sample). rr_lda minus cnn without augmentation: "
               f"{diff['point']:+.3f} [{diff['ci_low']:+.3f}, {diff['ci_high']:+.3f}]. Post hoc (PREREGISTRATION v4, P1).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_rr_reference") / "references_vs_cnn.png", "Is the cnn weaker than a simple RR-interval classifier?", caption)]


def external_validation_figure(summaries: dict[str, dict], final: dict) -> list[Path]:
    """P2: each frozen condition's primary metric on INCART and SVDB beside the same metric in the final evaluation."""
    plt = _plt()
    keys = list(final["conditions"])
    fig, axes = plt.subplots(1, len(summaries), figsize=(6.2 * len(summaries), 0.42 * len(keys) + 1.8), sharey=True, squeeze=False)
    for ax, (db, s) in zip(axes[0], summaries.items()):
        metric = s["primary_metric"]
        for i, key in enumerate(reversed(keys)):
            _interval(ax, s["conditions"][key][metric], i + 0.15, "#2a78d6", "external database" if i == 0 else None)
            _interval(ax, final["conditions"][key][metric], i - 0.15, "#eb6834", "MIT-BIH test records (final evaluation)" if i == 0 else None)
        ax.set_yticks(range(len(keys)), list(reversed(keys)), fontsize=7)
        ax.set_xlim(left=0)
        ax.set_xlabel(f"{'rare-class macro-F1' if metric == 'rare_macro_f1' else 'S F1'} (95% patient-level CI)")
        reject = sum(v["reject_holm"] for v in s["confirmatory"].values())
        label = {"incartdb": "INCART", "svdb": "SVDB"}.get(db, db)
        units = f"{s['records']} records, {s['patients']} patients" if s.get("patients", s["records"]) != s["records"] else f"{s['records']} records"
        ax.set_title(f"{label} ({s['role']}, {units}): Holm rejects {reject} of {len(s['confirmatory'])} vs none", loc="left", fontsize=9)
    axes[0][0].legend(loc="lower left", bbox_to_anchor=(0, 1.06), ncol=2, fontsize=8)
    caption = ("Every frozen checkpoint of the phase 4 gate (3 seeds) applied unchanged to external databases; beats cut like the frozen dataset. "
               "INCART: lead II, 257 Hz resampled to 360 Hz; SVDB: ECG1, 128 Hz resampled (S F1 is its primary metric, fewer than 50 F beats). "
               "Post hoc (PREREGISTRATION v4, P2).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_external_validation") / "conditions_external_vs_ds2.png", "Do the frozen models carry over to other databases?", caption)]


def extra_seeds_figure(summary: dict, final: dict) -> list[Path]:
    """P3: each pre-registered comparison with 3 seeds (final evaluation) and with 5 seeds."""
    plt = _plt()
    rows = [(family, name) for family, items in final["families"].items() for name in items]
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(rows) + 1.4))
    for i, (family, name) in enumerate(reversed(rows)):
        _interval(ax, final["families"][family][name], i + 0.15, "#2a78d6", "3 seeds (final evaluation)" if i == 0 else None)
        five = summary["families"][family][name]
        _interval(ax, five, i - 0.15, "#eb6834", "5 seeds (post hoc)" if i == 0 else None)
        ax.annotate(f"Holm p {five['p_holm']:.3f}", (five["ci_high"], i - 0.15), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7, color=INK_2)
    ax.axvline(0, color=GRID, linewidth=1.2)
    ax.set_yticks(range(len(rows)), [f"{family[:4]}. {name}" for family, name in reversed(rows)])
    ax.set_xlabel("difference in test rare-class macro-F1 (95% record-level CI)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    changed = summary["changed"]
    caption = (f"Seeds 3 and 4 trained post hoc with the frozen configurations, strengths and generators of the main study. "
               f"Holm verdicts that change with 5 seeds: {', '.join(changed) if changed else 'none'}. Post hoc (PREREGISTRATION v4, P3).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_extra_seeds") / "three_vs_five_seeds.png", "Do two more seeds change any verdict?", caption)]


def audit_check_figure(summary: dict) -> list[Path]:
    """P4: the leakage audit's intra-patient values beside the two checks."""
    plt = _plt()
    names = {"none": "none", "cvae": "cVAE", "wgan_gp": "WGAN-GP"}
    rows = [(f"split 0, {names[c]}{'' if c == 'none' else ' (0.25, audit)'}", v) for c, v in summary["audit_split0"].items()]
    for c in summary["checks"]:
        strength = "" if c["strength"] is None else f" ({c['strength']}, check)"
        rows.append((f"split {c['split_seed']}, {names[c['condition']]}{strength}", c["intra"]))
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.4))
    for i, (name, intra) in enumerate(reversed(rows)):
        ax.scatter(intra["per_seed"], [i] * len(intra["per_seed"]), s=22, color="#2a78d6", alpha=0.45, edgecolor=SURFACE, linewidth=0.6)
        ax.errorbar(intra["mean"], i, xerr=intra["sd"], fmt="o", color="#2a78d6", capsize=3)
        ax.annotate(f"{intra['mean']:.3f}", (intra["mean"], i), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7, color=INK_2)
    inter = summary["inter_patient"]["none"]
    ax.axvline(inter["point"], color="#eb6834", linewidth=1.4, label=f"inter-patient, none (final evaluation): {inter['point']:.3f}")
    ax.set_yticks(range(len(rows)), [name for name, _ in reversed(rows)])
    ax.set_xlim(0, 1)
    ax.set_xlabel("test rare-class macro-F1 (cnn), intra-patient random beat split")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0))
    caption = ("Mean ± SD over seeds 0-2 (faint dots: seeds). The audit's own rows (split 0) beside WGAN-GP at the strength the phase 4 gate chose and "
               "no augmentation on two more random beat-level splits. Post hoc (PREREGISTRATION v4, P4); never used for any decision.")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_audit_check") / "audit_checks.png", "Does the intra-patient inflation depend on the split or the strength?", caption)]


def rr_baseline_feature_figure() -> list[Path]:
    """P5 before and after: the RR ratio of training beats per class with v1 (ten preceding intervals) and v2 (patient baseline)."""
    from ecgaug.data.loaders import load_trainval

    path = paths.processed_dir() / "v2" / "trainval.npz"
    if not path.is_file():
        return []
    plt = _plt()
    v1 = load_trainval("v1")
    with np.load(path) as f:
        v2 = {k: f[k] for k in ("rr_ratio", "y", "split", "record")}
    train = np.asarray(v1["split"]) == "train"
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    bins = np.linspace(0.2, 1.8, 65)
    for ax, (label, ratio) in zip(axes, (("v1: rr_prev / mean of the 10 preceding RR", v1["rr_ratio"]), ("v2: rr_prev / median RR within ±150 s", v2["rr_ratio"]))):
        for i, cls in enumerate(CLASS_COLORS):
            mask = train & (np.asarray(v1["y"]) == i)
            below = float(np.mean(np.asarray(ratio)[mask] < 0.9))
            ax.hist(np.clip(np.asarray(ratio)[mask], bins[0], bins[-1]), bins=bins, density=True, histtype="step", linewidth=1.6,
                    color=CLASS_COLORS[cls], label=f"{cls} (n={int(mask.sum()):,}; {below:.0%} below 0.9)")  # fmt: skip
        ax.axvline(0.9, color=GRID, linewidth=1.2)
        ax.set_title(label, loc="left")
        ax.set_xlabel("RR ratio (training beats)")
        ax.legend(loc="upper right", fontsize=7)
    axes[0].set_ylabel("density")
    caption = ("Training records only; values outside 0.2-1.8 are drawn at the edges. With v1 the S beats of record 209, which come in runs, look as regular as N beats because the ten "
               "preceding intervals are short too; against the patient's baseline they are premature. Post hoc (PREREGISTRATION v5, P5).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_rr_baseline") / "rr_ratio_v1_vs_v2.png", "Before and after: the RR ratio the models see", caption)]


def rr_baseline_results_figure(summary: dict, final: dict) -> list[Path]:
    """P5: each condition with v1 and v2 features in the final evaluation's records, and the v2 comparisons."""
    plt = _plt()
    keys = list(final["conditions"])
    rows = [(family, name) for family, items in summary["families"].items() for name in items]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.42 * max(len(keys), len(rows)) + 1.8))
    ax = axes[0]
    for i, key in enumerate(reversed(keys)):
        _interval(ax, final["conditions"][key]["rare_macro_f1"], i + 0.15, "#2a78d6", "v1 features (final evaluation)" if i == 0 else None)
        _interval(ax, summary["conditions"][key]["rare_macro_f1"], i - 0.15, "#eb6834", "v2 features (post hoc)" if i == 0 else None)
    ax.set_yticks(range(len(keys)), list(reversed(keys)), fontsize=7)
    ax.set_xlim(left=0)
    ax.set_xlabel("test rare-class macro-F1 (95% record-level CI)")
    ax.set_title("each condition, v1 and v2", loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=2, fontsize=7)
    ax = axes[1]
    for i, (family, name) in enumerate(reversed(rows)):
        r = summary["families"][family][name]
        _interval(ax, r, i, INK if family == "confirmatory" else INK_2)
        ax.annotate(f"Holm p {r['p_holm']:.3f}", (r["ci_high"], i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7, color=INK_2)
    ax.axvline(0, color=GRID, linewidth=1.2)
    ax.set_yticks(range(len(rows)), [f"{family[:4]}. {name}" for family, name in reversed(rows)], fontsize=7)
    ax.set_xlabel("difference in test rare-class macro-F1, v2 (95% CI)")
    ax.set_title("pre-registered comparisons with v2 features", loc="left")
    caption = ("The full grid rerun with one change to the input (the RR ratio against the patient's baseline), selected on validation by the "
               "phase 4 gate rule; 3 seeds. Black: confirmatory family; grey: secondary; p-values Holm-adjusted within each family. "
               "Post hoc (PREREGISTRATION v5, P5); the final evaluation (blue, left) stays the primary result.")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_rr_baseline") / "v1_vs_v2_results.png", "Before and after: every condition with v1 and v2 RR features", caption)]


def rr_baseline_external_figure(v2: dict[str, dict], v1: dict[str, dict]) -> list[Path]:
    """P5: each condition on INCART and SVDB with v1 and v2 features."""
    plt = _plt()
    dbs = [db for db in v2 if db in v1]
    keys = list(next(iter(v2.values()))["conditions"])
    fig, axes = plt.subplots(1, len(dbs), figsize=(6.2 * len(dbs), 0.42 * len(keys) + 1.8), sharey=True, squeeze=False)
    for ax, db in zip(axes[0], dbs):
        metric = v2[db]["primary_metric"]
        for i, key in enumerate(reversed(keys)):
            _interval(ax, v1[db]["conditions"][key][metric], i + 0.15, "#2a78d6", "v1 features" if i == 0 else None)
            _interval(ax, v2[db]["conditions"][key][metric], i - 0.15, "#eb6834", "v2 features" if i == 0 else None)
        ax.set_yticks(range(len(keys)), list(reversed(keys)), fontsize=7)
        ax.set_xlim(left=0)
        ax.set_xlabel(f"{'rare-class macro-F1' if metric == 'rare_macro_f1' else 'S F1'} (95% patient-level CI)")
        reject = sum(v["reject_holm"] for v in v2[db]["confirmatory"].values())
        ax.set_title(f"{ {'incartdb': 'INCART', 'svdb': 'SVDB'}.get(db, db)}: v2 Holm rejects {reject} of 4 vs none", loc="left", fontsize=9)
    axes[0][0].legend(loc="lower left", bbox_to_anchor=(0, 1.06), ncol=2, fontsize=8)
    caption = "The v2 checkpoints applied unchanged to the external databases, with the v2 ratio from their annotations. Post hoc (PREREGISTRATION v5, P5)."
    return [_save(fig, paths.figures_for("phase6_rr_baseline") / "external_v1_vs_v2.png", "Before and after: external databases with v1 and v2 RR features", caption)]


VERSION_COLORS = {"v1": "#2a78d6", "v2": "#eb6834", "v3": "#1f9e6e"}
VERSION_LABELS = {"v1": "v1 (frozen)", "v2": "v2 (baseline RR ratio)", "v3": "v3 (patient-relative beats)"}


def patient_template_input_figure() -> list[Path]:
    """P6 before and after: validation beats as the models see them in v2 (z-scored window) and v3 (minus the patient's template).

    The templates of the three shown records are computed here from their own signals (validation records only)."""
    from ecgaug.data.loaders import load_trainval
    from ecgaug.experiments import phase6_patient_template as pt
    from ecgaug.experiments.phase6_external_validation import frozen_spec

    record_json = paths.results_for("phase6_patient_template") / "data.json"
    if not record_json.is_file() or not (paths.processed_dir() / "v1" / "trainval.npz").is_file():
        return []
    plt = _plt()
    half = json.loads(record_json.read_text())["half_window_s"]
    v1 = load_trainval("v1")
    y, record, split = np.asarray(v1["y"]), np.asarray(v1["record"]), np.asarray(v1["split"])
    records = [r for r in (118, 201, 223) if ((record == r) & (y == 1) & (split == "val")).any()]
    x3 = np.array(v1["X"], copy=True)
    spec = frozen_spec()
    for r in records:
        idx = np.where(record == r)[0]
        order = idx[np.argsort(np.asarray(v1["r_idx"])[idx])]
        x3[order] = v1["X"][order] - pt.record_templates(r, np.asarray(v1["r_idx"])[order], spec, half)
    t = (np.arange(v1["X"].shape[1]) - v1["X"].shape[1] // 2) / 360 * 1000
    fig, axes = plt.subplots(2, len(records), figsize=(4.2 * len(records), 5.2), sharex=True, squeeze=False)
    for j, r in enumerate(records):
        for row, (x, label) in enumerate(((v1["X"], "v1/v2: z-scored beat"), (x3, "v3: beat minus the patient's template"))):
            ax = axes[row][j]
            for i, cls in ((0, "N"), (1, "S"), (2, "V")):
                mask = (record == r) & (y == i) & (split == "val")
                if mask.any():
                    ax.plot(t, np.median(x[mask], axis=0), color=CLASS_COLORS[cls], linewidth=1.6, label=f"{cls} (n={int(mask.sum())})")
            ax.set_title(f"record {r}: {label}", loc="left", fontsize=9)
            ax.axvline(0, color=GRID, linewidth=1)
            if row == 1:
                ax.set_xlabel("ms from R")
            ax.legend(loc="best", fontsize=7, framealpha=0.85)
    caption = ("Median validation beat per class and record (validation records only). Top: the z-scored beat of v1 and v2, where the shape of "
               "a patient's normal beat differs between patients (record 118 has right bundle branch block). Bottom: minus the patient's own template "
               "(median of the beats within ±150 s, no labels), N beats sit at zero and only the deviation from the patient's usual beat remains. "
               "Post hoc (PREREGISTRATION v6, P6).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_patient_template") / "input_v2_vs_v3.png", "Before and after: the beat the models see", caption)]


def patient_template_diagnostic_figure(diag: dict, adoption: dict | None) -> list[Path]:
    """P6 diagnostic: training versus validation PR-AUC (S, V) of the no-augmentation networks per version, and the adoption rule."""
    plt = _plt()
    archs = list(diag["networks"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, cls in zip(axes[:2], ("S", "V")):
        for a, arch in enumerate(archs):
            for v, (version, g) in enumerate(diag["networks"][arch].items()):
                x = a * 4 + v
                ax.plot([x, x], [g["train"]["pr_auc"][cls], g["val"]["pr_auc"][cls]], color=VERSION_COLORS[version], linewidth=1.4)
                ax.scatter([x], [g["train"]["pr_auc"][cls]], color=VERSION_COLORS[version], marker="^", s=40, zorder=3, label="training records" if a == v == 0 else None)
                ax.scatter([x], [g["val"]["pr_auc"][cls]], color=VERSION_COLORS[version], marker="o", s=40, zorder=3, label="validation records" if a == v == 0 else None)
        ax.set_xticks([a * 4 + 1 for a in range(len(archs))], archs)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(f"{cls} PR-AUC (mean of 3 seeds)")
        ax.set_title(f"{cls}: training vs validation patients", loc="left")
    handles = [plt.Line2D([], [], color=VERSION_COLORS[v], linewidth=3, label=VERSION_LABELS[v]) for v in next(iter(diag["networks"].values()))]
    axes[1].legend(handles=handles + axes[0].get_legend_handles_labels()[0], fontsize=7, loc="lower center")
    ax = axes[2]
    names = list(diag["linear"])
    ax.barh(range(len(names)), [diag["linear"][n]["f1"]["S"] for n in names], color=INK_2, height=0.5)
    ax.set_yticks(range(len(names)), [{"rr": "RR (v2)", "rr_raw": "RR + raw beat", "rr_relative": "RR + relative beat"}.get(n, n) for n in names])
    ax.set_xlabel("validation S F1 (LDA, uniform priors)")
    title = "linear models on fixed features"
    if adoption is not None:
        title += f"\nadoption: v3 - v2 = {adoption['mean_difference']:+.3f}, margin {adoption['margin']:.3f} -> {'adopted' if adoption['adopted'] else 'not adopted'}"
    ax.set_title(title, loc="left", fontsize=9)
    caption = ("Validation data only. Triangles: training records; circles: validation records; the gap is how much a network depends on the patients it "
               "was trained on. Right: LDAs on three fixed feature sets, and the pre-registered adoption rule (mean over cnn and rnn of the seed-mean "
               "validation S F1 without augmentation, v3 minus v2, against max(0.02, pooled seed SD)). Post hoc (PREREGISTRATION v6, P6).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_patient_template") / "diagnostic.png", "Why the networks miss S beats, and what patient-relative beats change", caption)]


def patient_template_results_figure(summary: dict, final: dict, v2: dict | None) -> list[Path]:
    """P6: each condition with v1, v2 and v3 inputs on the test records, and the v3 comparisons."""
    plt = _plt()
    keys = list(final["conditions"])
    rows = [(family, name) for family, items in summary["families"].items() for name in items]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.5 * max(len(keys), len(rows)) + 1.8))
    ax = axes[0]
    versions = [("v1", final), *([("v2", v2)] if v2 else []), ("v3", summary)]
    offsets = np.linspace(0.22, -0.22, len(versions))
    for i, key in enumerate(reversed(keys)):
        for (version, s), off in zip(versions, offsets):
            _interval(ax, s["conditions"][key]["rare_macro_f1"], i + off, VERSION_COLORS[version], VERSION_LABELS[version] if i == 0 else None)
    ax.set_yticks(range(len(keys)), list(reversed(keys)), fontsize=7)
    ax.set_xlim(left=0)
    ax.set_xlabel("test rare-class macro-F1 (95% record-level CI)")
    ax.set_title("each condition, by input version", loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=len(versions), fontsize=7)
    ax = axes[1]
    for i, (family, name) in enumerate(reversed(rows)):
        r = summary["families"][family][name]
        _interval(ax, r, i, INK if family == "confirmatory" else INK_2)
        ax.annotate(f"Holm p {r['p_holm']:.3f}", (r["ci_high"], i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7, color=INK_2)
    ax.axvline(0, color=GRID, linewidth=1.2)
    ax.set_yticks(range(len(rows)), [f"{family[:4]}. {name}" for family, name in reversed(rows)], fontsize=7)
    ax.set_xlabel("difference in test rare-class macro-F1, v3 (95% CI)")
    ax.set_title("pre-registered comparisons with v3 inputs", loc="left")
    caption = ("The full grid rerun with patient-relative beats and the v2 RR ratio, selected on validation by the phase 4 gate rule; 3 seeds. Black: "
               "confirmatory family; grey: secondary; Holm within each family. Post hoc (PREREGISTRATION v6, P6); the final evaluation (v1) stays the primary result.")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_patient_template") / "results_by_version.png", "Before and after: every condition with v1, v2 and v3 inputs", caption)]


def patient_template_external_figure(versions: dict[str, dict[str, dict]]) -> list[Path]:
    """P6: each condition on INCART and SVDB with v1, v2 and v3 inputs."""
    plt = _plt()
    v3 = versions["v3"]
    dbs = list(v3)
    keys = list(next(iter(v3.values()))["conditions"])
    fig, axes = plt.subplots(1, len(dbs), figsize=(6.2 * len(dbs), 0.5 * len(keys) + 1.8), sharey=True, squeeze=False)
    present = [v for v in ("v1", "v2", "v3") if all(db in versions.get(v, {}) for db in dbs)]
    offsets = np.linspace(0.22, -0.22, len(present))
    for ax, db in zip(axes[0], dbs):
        metric = v3[db]["primary_metric"]
        for i, key in enumerate(reversed(keys)):
            for version, off in zip(present, offsets):
                _interval(ax, versions[version][db]["conditions"][key][metric], i + off, VERSION_COLORS[version], VERSION_LABELS[version] if i == 0 else None)
        ax.set_yticks(range(len(keys)), list(reversed(keys)), fontsize=7)
        ax.set_xlim(left=0)
        ax.set_xlabel(f"{'rare-class macro-F1' if metric == 'rare_macro_f1' else 'S F1'} (95% patient-level CI)")
        reject = sum(v["reject_holm"] for v in v3[db]["confirmatory"].values())
        ax.set_title(f"{ {'incartdb': 'INCART', 'svdb': 'SVDB'}.get(db, db)}: v3 Holm rejects {reject} of 4 vs none", loc="left", fontsize=9)
    axes[0][0].legend(loc="lower left", bbox_to_anchor=(0, 1.06), ncol=len(present), fontsize=8)
    caption = "The v3 checkpoints applied unchanged to the external databases, with templates and v2 ratios from their own signals and annotations. Post hoc (PREREGISTRATION v6, P6)."
    return [_save(fig, paths.figures_for("phase6_patient_template") / "external_by_version.png", "Before and after: external databases by input version", caption)]


def relative_cnn_results_figure(summary: dict, final: dict, v2: dict | None) -> list[Path]:
    """P7: each cnn condition with v1, v2 and P7 (v3 inputs) on the test records, and the P7 comparisons."""
    plt = _plt()
    keys = list(summary["conditions"])
    rows = [(family, name) for family, items in summary["families"].items() for name in items]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.55 * max(len(keys), len(rows)) + 2.0))
    ax = axes[0]
    versions = [("v1", final, "v1 (final evaluation)"), *([("v2", v2, "v2 (P5)")] if v2 else []), ("v3", summary, "v3 cnn (P7)")]
    offsets = np.linspace(0.22, -0.22, len(versions))
    for i, key in enumerate(reversed(keys)):
        for (version, s, label), off in zip(versions, offsets):
            _interval(ax, s["conditions"][key]["rare_macro_f1"], i + off, VERSION_COLORS[version], label if i == 0 else None)
    ax.set_yticks(range(len(keys)), list(reversed(keys)), fontsize=8)
    ax.set_xlim(left=0)
    ax.set_xlabel("test rare-class macro-F1 (95% record-level CI)")
    ax.set_title("each cnn condition, by input version", loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=len(versions), fontsize=7)
    ax = axes[1]
    for i, (family, name) in enumerate(reversed(rows)):
        r = summary["families"][family][name]
        _interval(ax, r, i, INK if family == "confirmatory" else INK_2)
        ax.annotate(f"Holm p {r['p_holm']:.3f}", (r["ci_high"], i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7, color=INK_2)
    ax.axvline(0, color=GRID, linewidth=1.2)
    ax.set_yticks(range(len(rows)), [f"{family[:4]}. {name}" for family, name in reversed(rows)], fontsize=8)
    ax.set_xlabel("difference in test rare-class macro-F1 (95% CI)")
    ax.set_title("pre-registered comparisons, P7", loc="left")
    caption = ("The cnn alone with patient-relative beats and the v2 RR ratio, the full comparison selected on validation by the phase 4 gate rule; "
               "3 seeds. Black: confirmatory family; grey: secondary; Holm within each family. Post hoc (PREREGISTRATION v7, P7); the final "
               "evaluation (v1) stays the primary result.")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_relative_cnn") / "results_by_version.png", "Before and after: the cnn with v1, v2 and v3 inputs", caption)]


def relative_cnn_external_figure(versions: dict[str, dict[str, dict]]) -> list[Path]:
    """P7: each cnn condition on INCART and SVDB with v1, v2 and v3 inputs."""
    plt = _plt()
    p7 = versions["v3"]
    dbs = list(p7)
    keys = list(next(iter(p7.values()))["conditions"])
    present = [v for v in ("v1", "v2", "v3") if all(db in versions.get(v, {}) for db in dbs)]
    offsets = np.linspace(0.22, -0.22, len(present))
    fig, axes = plt.subplots(1, len(dbs), figsize=(6.2 * len(dbs), 0.55 * len(keys) + 2.0), sharey=True, squeeze=False)
    for ax, db in zip(axes[0], dbs):
        metric = p7[db]["primary_metric"]
        for i, key in enumerate(reversed(keys)):
            for version, off in zip(present, offsets):
                _interval(ax, versions[version][db]["conditions"][key][metric], i + off, VERSION_COLORS[version], VERSION_LABELS[version] if i == 0 else None)
        ax.set_yticks(range(len(keys)), list(reversed(keys)), fontsize=8)
        ax.set_xlim(left=0)
        ax.set_xlabel(f"{'rare-class macro-F1' if metric == 'rare_macro_f1' else 'S F1'} (95% patient-level CI)")
        reject = sum(v["reject_holm"] for v in p7[db]["confirmatory"].values())
        ax.set_title(f"{ {'incartdb': 'INCART', 'svdb': 'SVDB'}.get(db, db)}: P7 Holm rejects {reject} of {len(p7[db]['confirmatory'])} vs none", loc="left", fontsize=9)
    axes[0][0].legend(loc="lower left", bbox_to_anchor=(0, 1.06), ncol=len(present), fontsize=8)
    caption = "The P7 cnn checkpoints applied unchanged to the external databases, with templates and v2 ratios from their own signals and annotations. Post hoc (PREREGISTRATION v7, P7)."
    return [_save(fig, paths.figures_for("phase6_relative_cnn") / "external_by_version.png", "Before and after: the cnn on the external databases", caption)]


def mechanism_figure(attribution: dict | None, gap: dict | None) -> list[Path]:
    """P8: which training patients the synthetic S beats resemble, and the training-versus-validation gap of every condition."""
    plt = _plt()
    panels = [x for x in (attribution, gap) if x is not None]
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 4.2), squeeze=False)
    k = 0
    if attribution is not None:
        ax = axes[0][k]
        k += 1
        s = attribution["classes"]["S"]
        records = list(s["real_training"]["shares"])[:6]
        series = [("real training S beats", s["real_training"]["shares"], INK)]
        for condition, color in (("cvae", "#2a78d6"), ("wgan_gp", "#eb6834")):
            seeds = s.get(f"{condition}_bank", {})
            if seeds:
                series.append((f"synthetic ({'cVAE' if condition == 'cvae' else 'WGAN-GP'}), nearest real", {r: float(np.mean([e["shares"].get(r, 0.0) for e in seeds.values()])) for r in records}, color))
        if "validation_nearest" in s:
            series.append(("validation S beats, nearest real", s["validation_nearest"]["shares"], "#1f9e6e"))
        width = 0.8 / len(series)
        for j, (label, shares, color) in enumerate(series):
            ax.bar(np.arange(len(records)) + (j - (len(series) - 1) / 2) * width, [shares.get(r, 0.0) for r in records], width=width * 0.9, color=color, label=label)
        ax.set_xticks(range(len(records)), [f"record {r}" for r in records], fontsize=8)
        ax.set_ylabel("share of S beats")
        ax.set_title("S beats by training patient", loc="left")
        ax.legend(fontsize=7, loc="upper right")
    if gap is not None:
        ax = axes[0][k]
        conds = list(next(iter(gap["networks"].values())))
        archs = list(gap["networks"])
        width = 0.8 / len(archs)
        for j, arch in enumerate(archs):
            ax.bar(np.arange(len(conds)) + (j - (len(archs) - 1) / 2) * width, [gap["networks"][arch][c]["gap"]["S"] for c in conds], width=width * 0.9,
                   color=("#2a78d6", "#eb6834")[j % 2], label=arch)  # fmt: skip
        ax.set_xticks(range(len(conds)), [c.replace("_", " ") for c in conds], fontsize=8, rotation=20)
        ax.set_ylabel("S PR-AUC, training minus validation patients")
        ax.set_title("gap between training and new patients", loc="left")
        ax.legend(fontsize=8)
    caption = ("DS1 only, descriptive. Left: each S beat attributed to the training patient of its nearest real training S beat (Euclidean distance "
               "between z-scored windows; synthetic shares are means over the three bank seeds). Right: seed-mean S PR-AUC of the frozen networks "
               "on the real training beats minus on the validation beats, per condition. Post hoc (PREREGISTRATION v7, P8).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_augmentation_mechanism") / "mechanism.png", "Synthetic beats copy the training patients", caption)]


def patient_transfer_example_figure() -> list[Path]:
    """P9 before and after: one real training S beat of record 209, its deviation from the patient's template, and the beat moved
    onto the templates of three other training patients (training records only)."""
    from ecgaug.data.loaders import load_trainval
    from ecgaug.data.representation import zscore_per_beat
    from ecgaug.experiments import phase6_patient_template as pt
    from ecgaug.experiments.phase6_external_validation import frozen_spec

    if not (paths.results_for("phase6_patient_transfer") / "data.json").is_file() or not (paths.processed_dir() / "v1" / "trainval.npz").is_file():
        return []
    plt = _plt()
    half = json.loads((paths.results_for("phase6_patient_transfer") / "data.json").read_text())["half_window_s"]
    v1 = load_trainval("v1")
    y, record, split, r_idx = (np.asarray(v1[k]) for k in ("y", "record", "split", "r_idx"))
    spec = frozen_spec()
    source = np.flatnonzero((record == 209) & (y == 1) & (split == "train"))[len(np.flatnonzero((record == 209) & (y == 1) & (split == "train"))) // 2]
    t_p = pt.record_templates(209, r_idx[[source]], spec, half)[0]
    deviation = v1["X"][source] - t_p
    targets = []
    for q in (118, 124, 207):
        n = np.flatnonzero((record == q) & (y == 0) & (split == "train"))
        if n.size:
            i = n[n.size // 2]
            targets.append((q, pt.record_templates(q, r_idx[[i]], spec, half)[0]))
    ms = (np.arange(v1["X"].shape[1]) - v1["X"].shape[1] // 2) / 360 * 1000
    fig, axes = plt.subplots(1, 2 + len(targets), figsize=(3.6 * (2 + len(targets)), 3.2), sharex=True)
    axes[0].plot(ms, v1["X"][source], color=CLASS_COLORS["S"], linewidth=1.6, label="S beat")
    axes[0].plot(ms, t_p, color=CLASS_COLORS["N"], linewidth=1.2, label="patient's template")
    axes[0].set_title("record 209: a real S beat", loc="left", fontsize=9)
    axes[0].legend(fontsize=7)
    axes[1].plot(ms, deviation, color=INK, linewidth=1.4)
    axes[1].set_title("its deviation", loc="left", fontsize=9)
    for ax, (q, t_q) in zip(axes[2:], targets):
        ax.plot(ms, t_q, color=CLASS_COLORS["N"], linewidth=1.2, label=f"template of {q}")
        ax.plot(ms, zscore_per_beat((t_q + deviation)[None, :])[0], color=CLASS_COLORS["S"], linewidth=1.6, label="transferred S beat")
        ax.set_title(f"moved onto record {q}", loc="left", fontsize=9)
        ax.legend(fontsize=7)
    for ax in axes:
        ax.axvline(0, color=GRID, linewidth=1)
        ax.set_xlabel("ms from R")
    caption = ("Training records only. The deviation of a rare-class beat from its own patient's template is placed on the template of another training "
               "patient (a random N beat's template), so the classifier sees the same arrhythmia against many patients' normal beats; RR intervals "
               "are scaled by the ratio of the two patients' median RR. Post hoc (PREREGISTRATION v8, P9).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_patient_transfer") / "transfer_example.png", "Patient transfer: one arrhythmia, many patients", caption)]


def patient_transfer_results_figure(summary: dict, v2: dict | None) -> list[Path]:
    """P9: each condition on the test records beside v5's (v2) no augmentation and plain generative augmentation, and the P9 families."""
    plt = _plt()
    archs = sorted({k.split("/")[0] for k in summary["conditions"]})
    rows, colors = [], []
    for arch in archs:
        for c in ("none", "transfer_real", "transfer_cvae", "transfer_wgan_gp"):
            rows.append((f"{arch}/{c}", summary["conditions"][f"{arch}/{c}"]["rare_macro_f1"]))
            colors.append("#1f9e6e" if c != "none" else INK)
        if v2:
            for c in ("cvae", "wgan_gp"):
                rows.append((f"{arch}/{c} (v5, plain)", v2["conditions"][f"{arch}/{c}"]["rare_macro_f1"]))
                colors.append("#eb6834")
    fam = [(family, name) for family, items in summary["families"].items() for name in items]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.42 * max(len(rows), len(fam)) + 2.0))
    ax = axes[0]
    for i, ((label, e), color) in enumerate(zip(reversed(rows), reversed(colors))):
        _interval(ax, e, i, color)
    ax.set_yticks(range(len(rows)), [label for label, _ in reversed(rows)], fontsize=8)
    ax.set_xlim(left=0)
    ax.set_xlabel("test rare-class macro-F1 (95% record-level CI)")
    ax.set_title("each condition (green: patient transfer; orange: plain generative, v5)", loc="left", fontsize=9)
    ax = axes[1]
    for i, (family, name) in enumerate(reversed(fam)):
        r = summary["families"][family][name]
        _interval(ax, r, i, INK if family == "confirmatory" else INK_2)
        ax.annotate(f"Holm p {r['p_holm']:.3f}", (r["ci_high"], i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7, color=INK_2)
    ax.axvline(0, color=GRID, linewidth=1.2)
    ax.set_yticks(range(len(fam)), [f"{family[:4]}. {name}" for family, name in reversed(fam)], fontsize=8)
    ax.set_xlabel("difference in test rare-class macro-F1 (95% CI)")
    ax.set_title("pre-registered comparisons, P9", loc="left")
    caption = ("Dataset v2 inputs, strengths chosen on validation by the phase 4 gate rule, 3 seeds. Black: confirmatory family; grey: secondary; Holm within "
               "each family. Post hoc (PREREGISTRATION v8, P9); the final evaluation (v1) stays the primary result.")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_patient_transfer") / "results.png", "Does augmentation help when it adds patients?", caption)]


def patient_adaptation_figure(adoption: dict, summary: dict | None) -> list[Path]:
    """P10 before and after: validation S F1 without and with per-patient normalization, and (if read) the test families."""
    plt = _plt()
    keys = list(adoption["pairs"])
    panels = 2 if summary else 1
    fig, axes = plt.subplots(1, panels, figsize=(6.5 * panels, 3.6), squeeze=False)
    ax = axes[0][0]
    for i, key in enumerate(keys):
        e = adoption["pairs"][key]
        for j, (name, color) in enumerate((("plain", INK_2), ("adapted", "#1f9e6e"))):
            values = e["per_seed"][name]["f1_S"]
            ax.scatter([i + (j - 0.5) * 0.25] * len(values), values, color=color, s=18, zorder=3, label=("without adaptation" if j == 0 else "with adaptation") if i == 0 else None)
            ax.hlines(np.mean(values), i + (j - 0.5) * 0.25 - 0.1, i + (j - 0.5) * 0.25 + 0.1, color=color, linewidth=2)
        ax.annotate("adopted" if e["adopted"] else "not adopted", (i, max(max(e["per_seed"]["plain"]["f1_S"]), max(e["per_seed"]["adapted"]["f1_S"]))),
                    xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7, color=INK_2)  # fmt: skip
    ax.set_xticks(range(len(keys)), keys)
    ax.set_ylabel("validation S F1 (no augmentation; dots = seeds)")
    ax.set_title("validation: the adoption rule", loc="left")
    ax.legend(fontsize=7, loc="lower right")
    if summary:
        ax = axes[0][1]
        rows = [(family, name) for family, items in summary["families"].items() for name in items]
        for i, (family, name) in enumerate(reversed(rows)):
            r = summary["families"][family][name]
            _interval(ax, r, i, INK if family == "adaptation" else INK_2)
            ax.annotate(f"Holm p {r['p_holm']:.3f}", (r["ci_high"], i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7, color=INK_2)
        ax.axvline(0, color=GRID, linewidth=1.2)
        ax.set_yticks(range(len(rows)), [name for _, name in reversed(rows)], fontsize=7)
        ax.set_xlabel("difference in test rare-class macro-F1 (95% CI)")
        ax.set_title("test: the two families", loc="left")
    caption = ("Adaptive batch normalization: each patient classified by a copy of the frozen network whose batch-norm layers use that patient's "
               "statistics (no labels, no retraining). Left: validation S F1 per seed without and with it. Right: the pre-registered test families "
               "(black: adaptation vs none; grey: generative vs none under adaptation). Post hoc (PREREGISTRATION v9, P10).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_patient_adaptation") / "adaptation.png", "Normalizing each patient inside the network", caption)]


def more_patients_figure(summary: dict, p7: dict) -> list[Path]:
    """P11 before and after: each cnn condition trained on DS1 only (P7) and on DS1 plus the INCART and SVDB patients."""
    plt = _plt()
    keys = list(summary["conditions"])
    rows = [(family, name) for family, items in summary["families"].items() for name in items]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.6 * max(len(keys), len(rows)) + 2.0))
    ax = axes[0]
    for i, key in enumerate(reversed(keys)):
        _interval(ax, p7["conditions"][key]["rare_macro_f1"], i + 0.15, "#1f9e6e", "DS1 patients only (P7)" if i == 0 else None)
        _interval(ax, summary["conditions"][key]["rare_macro_f1"], i - 0.15, "#7a4fd6", "DS1 + INCART + SVDB patients (P11)" if i == 0 else None)
    ax.set_yticks(range(len(keys)), list(reversed(keys)), fontsize=8)
    ax.set_xlim(left=0)
    ax.set_xlabel("test rare-class macro-F1 (95% record-level CI)")
    ax.set_title("each condition, by training patients", loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=2, fontsize=7)
    ax = axes[1]
    for i, (family, name) in enumerate(reversed(rows)):
        r = summary["families"][family][name]
        _interval(ax, r, i, INK if family == "confirmatory" else INK_2)
        ax.annotate(f"Holm p {r['p_holm']:.3f}", (r["ci_high"], i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7, color=INK_2)
    ax.axvline(0, color=GRID, linewidth=1.2)
    ax.set_yticks(range(len(rows)), [name for _, name in reversed(rows)], fontsize=7)
    ax.set_xlabel("difference in test rare-class macro-F1 (95% CI)")
    ax.set_title("pre-registered comparisons, P11", loc="left")
    caption = ("The P7 cnn (patient-relative beats) trained on more patients: all S, V and F beats of INCART and SVDB and at most 600 N beats per "
               "record, strengths fixed at P7's validation choices, 3 seeds. Black: confirmatory (more patients vs DS1 only, no augmentation); "
               "grey: each condition vs none. Post hoc (PREREGISTRATION v10, P11).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_more_patients") / "results.png", "Before and after: more training patients", caption)]


def relative_generators_mechanism_figure(mechs: dict[str, dict]) -> list[Path]:
    """P12 before and after: over how many training patients the synthetic beats spread (the patient of each one's nearest real
    training beat), for the bank P11 used and the P12 bank, beside the real beats themselves."""
    plt = _plt()
    names = {"cvae": "cVAE", "wgan_gp": "WGAN-GP"}
    colors = {"cvae": "#2a78d6", "wgan_gp": "#eb6834"}
    fig, axes = plt.subplots(1, len(mechs), figsize=(6.2 * len(mechs), 4.0), squeeze=False)
    for ax, (family, m) in zip(axes[0], mechs.items()):
        classes = list(m["classes"])
        series = [("real training beats", "real", INK), ("bank used in P11 (phase 3 generator)", "p11", "#b8b6b0"), (f"P12 bank ({names[family]})", "p12", colors[family])]
        width = 0.8 / len(series)
        for j, (label, key, color) in enumerate(series):
            values = [m["classes"][c][key]["effective_records"] for c in classes]
            x = np.arange(len(classes)) + (j - (len(series) - 1) / 2) * width
            ax.bar(x, values, width=width * 0.9, color=color, label=label)
            for xi, v in zip(x, values):
                ax.annotate(f"{v:.1f}", (xi, v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7, color=INK_2)
        ax.set_xticks(range(len(classes)), [f"{c} beats" for c in classes])
        ax.set_axisbelow(True)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("effective number of training patients")
        ax.set_title(names[family], loc="left")
        ax.legend(fontsize=7, loc="upper right")
    caption = ("For each synthetic beat (5,000 per class from the seed-0 banks), the patient of its nearest real training beat (Euclidean, patient-relative "
               "form); bars give exp of the entropy of those patient shares (P8's measure, by patient). Real: the training beats' own spread. "
               "Post hoc (PREREGISTRATION v11, P12).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_relative_generators") / "mechanism.png", "Before and after: how many patients the synthetic beats resemble", caption)]


def record_rr_check_figure(check: dict) -> list[Path]:
    """P13's design check (training-side annotations only): the S beats each RR ratio finds in the most S-dominant training
    recording, and the rule's S F1 per database."""
    plt = _plt()
    labels = {"v1": "v1: mean of 10 preceding", "v2": "v2: median within ±150 s", "record": "P13: mean of the recording"}
    colors = {"v1": "#b8b6b0", "v2": INK_2, "record": "#2a78d6"}
    heavy = sorted(check["s_heavy_records"], key=lambda h: -h["s_beats"])
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.8), gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axes[0]
    if heavy:
        h = heavy[0]
        values = [100 * h["rule_s_sensitivity"][r] for r in labels]
        ax.bar(range(len(labels)), values, color=[colors[r] for r in labels], width=0.6)
        for i, v in enumerate(values):
            ax.annotate(f"{v:.1f}%", (i, v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8, color=INK_2)
        ax.set_xticks(range(len(labels)), [labels[r].split(":")[0] for r in labels])
        ax.set_ylim(0, 105)
        ax.set_ylabel("S beats found by the rule (%)")
        ax.set_title(f"SVDB {h['record']}: S {h['s_beats']:,} of {h['n_and_s_beats']:,} beats", loc="left")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    dbs = {"mitdb_ds1": "DS1", "incartdb": "INCART", "svdb": "SVDB"}
    names = list(check["databases"])
    width = 0.8 / len(labels)
    for j, r in enumerate(labels):
        x = np.arange(len(names)) + (j - (len(labels) - 1) / 2) * width
        values = [check["databases"][db]["rule"][r]["f1"] for db in names]
        ax.bar(x, values, width=width * 0.9, color=colors[r], label=labels[r])
        for xi, v in zip(x, values):
            ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7, color=INK_2)
    ax.set_xticks(range(len(names)), [f"{dbs.get(db, db)} ({check['databases'][db]['records']} records)" for db in names])
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"S F1 of the rule ratio < {check['threshold']}")
    ax.set_title("every recording of each database", loc="left")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    caption = (f"The rule 'RR interval below {check['threshold']} of the reference' (Huang et al.'s SVEB stage) with three references, on the annotations of "
               "the training-side databases only. Left: the recording where S beats are most dominant; the local references are set by the S beats "
               "themselves. Right: pooled over each database the recording mean is not better everywhere, so P13 adds it beside v2's ratio. "
               "Post hoc (PREREGISTRATION v12, P13).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_record_rr") / "check.png", "Before and after: what the RR reference sees in an S-dominant recording", caption)]


def _before_after_rule(summary: dict, labels: tuple[str, str], after_color: str, left_title: str, comparisons_title: str) -> object:
    """Three panels from a P13-style summary: rare-class macro-F1 and S F1 per condition (before, after, after with Huang's
    SVEB stage) and the pre-registered comparisons with their Holm-adjusted p."""
    plt = _plt()
    names = {"none": "no augmentation", "class_weighted": "class-weighted loss", "random_oversampling": "random oversampling", "cvae": "cVAE", "wgan_gp": "WGAN-GP"}
    keys = list(summary["conditions"])
    fig, axes = plt.subplots(1, 3, figsize=(17, 0.62 * len(keys) * 3 / 2 + 2.6), gridspec_kw={"width_ratios": [1, 0.75, 1.15]})
    rows = []
    for k in keys:
        c = k.split("/")[1]
        rows += [(f"{names[c]}: {labels[0]}", summary["before"][k], "#b8b6b0"), (f"{names[c]}: {labels[1]}", summary["conditions"][k], after_color),
                 (f"{names[c]}: + SVEB rule", summary["sveb_rule"][k], "#eb6834")]  # fmt: skip
    for ax, metric, xlabel, title in ((axes[0], "rare_macro_f1", "test rare-class macro-F1 (95% record-level CI)", left_title),
                                      (axes[1], "f1_S", "test S F1 (95% record-level CI)", "S beats only")):  # fmt: skip
        for i, (label, entry, color) in enumerate(reversed(rows)):
            _interval(ax, entry[metric], i, color)
        ax.set_yticks(range(len(rows)), [label for label, _, _ in reversed(rows)] if metric == "rare_macro_f1" else [], fontsize=7)
        ax.set_xlim(0, 1 if metric == "f1_S" else None)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
    ax = axes[2]
    comparisons = [(family, name) for family, items in summary["families"].items() for name in items]
    for i, (family, name) in enumerate(reversed(comparisons)):
        r = summary["families"][family][name]
        _interval(ax, r, i, INK if family == "confirmatory" else INK_2)
        ax.annotate(f"Holm p {r['p_holm']:.3f}", (r["ci_high"], i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7, color=INK_2)
    ax.axvline(0, color=GRID, linewidth=1.2)

    def pretty(name: str) -> str:
        for key, label in (("random_oversampling", "random oversampling"), ("class_weighted", "class weighting"), ("wgan_gp", "WGAN-GP"), ("cvae", "cVAE")):
            name = name.replace(key, label)
        return name.removeprefix("cnn: ")

    ax.set_yticks(range(len(comparisons)), [pretty(name) for _, name in reversed(comparisons)], fontsize=7)
    ax.locator_params(axis="x", nbins=6)
    ax.set_xlabel("difference in test rare-class macro-F1 (95% CI)")
    ax.set_title(comparisons_title, loc="left")
    return fig


def record_rr_results_figure(summary: dict) -> list[Path]:
    """P13 before and after: each condition with v2's local RR features (P11, P12) and with RR over the recording's mean, plus
    Huang's SVEB stage on top; the pre-registered comparisons."""
    fig = _before_after_rule(summary, ("before", "recording-mean RR"), "#2a78d6", "each condition (the P7 cnn on P11's patients)", "pre-registered comparisons, P13")
    caption = ("The same cnn, data and strengths with v2's local RR features (grey; P11 and P12) and with rr_prev and rr_next over the recording's mean "
               "RR interval (blue); orange: Huang et al.'s SVEB stage on top (descriptive). Right, black: confirmatory (each condition vs before); grey: "
               "secondary (generators vs the other conditions). Post hoc (PREREGISTRATION v12, P13).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_record_rr") / "results.png", "Before and after: RR intervals over the recording's mean", caption)]


def icentia_check_figure(check: dict) -> list[Path]:
    """P14's design check: validation S F1 against the share of an unseen S-dominant patient's S beats found, per rule or
    interval model (training-side annotations only)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    points = [(name, e["validation"]["f1"], e["held_out_s_sensitivity"]) for name, e in check["rules"].items()]
    points += [(name, e["without_held_out"]["validation"]["f1"], e["without_held_out"]["held_out_s_sensitivity"]) for name, e in check["models"].items()]
    local = []
    for name, x, y in points:
        color = "#eb6834" if "irregularity" in name else "#2a78d6" if ("recording mean" in name and "local" not in name and "v2" not in name) else INK_2
        marker = "s" if "<" in name else "o"
        ax.scatter(x, 100 * y, s=46, color=color, marker=marker, zorder=3)
        if color == INK_2:
            local.append((x, 100 * y))
            continue
        ax.annotate(name.replace("logistic regression", "logreg").replace("gradient boosting", "GBM"), (x, 100 * y), xytext=(5, 3),
                    textcoords="offset points", fontsize=7, color=INK_2)  # fmt: skip
    if local:
        x0, y0 = min(x for x, _ in local), max(y for _, y in local)
        ax.annotate(f"{len(local)} rules and models with a local reference\n(+/-150 s mean, local averages, v2 ratio, de Chazal's intervals)",
                    (x0, y0), xytext=(x0 - 0.33, y0 + 22), textcoords="data", fontsize=7, color=INK_2,
                    arrowprops={"arrowstyle": "-", "color": INK_2, "linewidth": 0.8})  # fmt: skip
    ax.set_xlim(0, 1)
    ax.set_ylim(-4, 104)
    ax.set_xlabel("validation S F1 (P11's validation patients)")
    ax.set_ylabel(f"S beats of the unseen patient found (%; SVDB {check['held_out'][1]})")
    ax.set_title("no rule or interval model is good at both", loc="left")
    caption = ("Training-side annotations only (DS1, INCART, SVDB; P11's split), SVDB 865 left out of training. Squares: rules; dots: models of the "
               "RR intervals. Blue: the recording mean alone; grey: a local reference among the inputs; orange: an irregularity veto. "
               "Post hoc (PREREGISTRATION v13, P14).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_icentia") / "check.png", "Why more patients: local references miss an unseen S-dominant patient", caption)]


def icentia_results_figure(summary: dict) -> list[Path]:
    """P14 before and after: each condition trained without (P13) and with the Icentia11k patients, plus Huang's SVEB stage."""
    fig = _before_after_rule(summary, ("P13", "with Icentia11k patients"), "#1a9a6c", "each condition (the P7 cnn)", "pre-registered comparisons, P14")
    caption = ("The same cnn, features and strengths trained on P11's 106 patients (grey; P13) and with the Icentia11k patients added (green); orange: "
               "Huang et al.'s SVEB stage on top (descriptive). Right, black: confirmatory (each condition vs P13); grey: secondary (the cVAE vs the "
               "other conditions). Post hoc (PREREGISTRATION v13, P14).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_icentia") / "results.png", "Before and after: many more training patients", caption)]


def relative_generators_results_figure(summary: dict, p11: dict) -> list[Path]:
    """P12 before and after: each generator with the phase 3 banks (P11) and with generators trained on P11's patient-relative beats (P12)."""
    plt = _plt()
    names = {"cvae": "cVAE", "wgan_gp": "WGAN-GP"}
    rows = [("P11: no augmentation", p11["conditions"]["cnn/none"], INK_2), ("P11: random oversampling", p11["conditions"]["cnn/random_oversampling"], INK_2),
            ("P11: class-weighted loss", p11["conditions"]["cnn/class_weighted"], INK_2)]  # fmt: skip
    for key in summary["conditions"]:
        family = key.split("/")[1]
        rows += [(f"{names[family]}, phase 3 generator (P11)", p11["conditions"][key], "#b8b6b0"),
                 (f"{names[family]}, P12 generator (strength {summary['conditions'][key]['strength']})", summary["conditions"][key], "#2a78d6" if family == "cvae" else "#eb6834")]  # fmt: skip
    comparisons = [(family, name) for family, items in summary["families"].items() for name in items]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.5 * max(len(rows), len(comparisons)) + 2.2))
    ax = axes[0]
    for i, (label, entry, color) in enumerate(reversed(rows)):
        _interval(ax, entry["rare_macro_f1"], i, color)
    ax.set_yticks(range(len(rows)), [label for label, _, _ in reversed(rows)], fontsize=8)
    ax.set_xlim(left=0)
    ax.set_xlabel("test rare-class macro-F1 (95% record-level CI)")
    ax.set_title("each condition (the P7 cnn on P11's patients)", loc="left")
    ax = axes[1]
    for i, (family, name) in enumerate(reversed(comparisons)):
        r = summary["families"][family][name]
        _interval(ax, r, i, INK if family == "confirmatory" else INK_2)
        ax.annotate(f"Holm p {r['p_holm']:.3f}", (r["ci_high"], i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7, color=INK_2)
    ax.axvline(0, color=GRID, linewidth=1.2)

    def pretty(name: str) -> str:
        for key, label in names.items():
            name = name.replace(key, label)
        return name.removeprefix("cnn: ")

    ax.set_yticks(range(len(comparisons)), [pretty(name) for _, name in reversed(comparisons)], fontsize=7)
    ax.locator_params(axis="x", nbins=6)
    ax.set_xlabel("difference in test rare-class macro-F1 (95% CI)")
    ax.set_title("pre-registered comparisons, P12", loc="left")
    caption = ("The cVAE and WGAN-GP retrained on P11's training beats in the form the cnn sees them (settings and step budget unchanged; coloured), "
               "against the same generators trained in phase 3 on DS1's ordinary beats (light grey); dark grey rows: P11's other conditions for "
               "reference. Right, black: confirmatory (P12 vs the same generator in P11); grey: secondary. Post hoc (PREREGISTRATION v11, P12).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_relative_generators") / "results.png", "Before and after: generators trained on what the cnn sees", caption)]


DIFFUSION_COLOR, VETO_COLOR, LONG_COLOR = "#8a5cc8", "#1a9a6c", "#c2408f"
CONDITION_NAMES = {"none": "no augmentation", "class_weighted": "class-weighted loss", "random_oversampling": "random oversampling", "cvae": "cVAE",
                   "wgan_gp": "WGAN-GP", "diffusion": "diffusion", "diffusion_long": "diffusion, 5x steps"}  # fmt: skip


def _comparisons_panel(ax, families: dict, title: str) -> None:
    """Pre-registered comparisons with their intervals and Holm-adjusted p (black: confirmatory; grey: secondary)."""
    comparisons = [(family, name) for family, items in families.items() for name in items]
    for i, (family, name) in enumerate(reversed(comparisons)):
        r = families[family][name]
        _interval(ax, r, i, INK if family == "confirmatory" else INK_2)
        ax.annotate(f"Holm p {r['p_holm']:.3f}", (r["ci_high"], i), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7, color=INK_2)
    ax.axvline(0, color=GRID, linewidth=1.2)

    def pretty(name: str) -> str:
        for key in ("diffusion_long", "random_oversampling", "class_weighted", "wgan_gp", "cvae"):
            name = name.replace(key, CONDITION_NAMES[key])
        return name.removeprefix("cnn: ")

    ax.set_yticks(range(len(comparisons)), [pretty(name) for _, name in reversed(comparisons)], fontsize=7)
    ax.locator_params(axis="x", nbins=6)
    ax.set_xlabel("difference in test rare-class macro-F1 (95% CI)")
    ax.set_title(title, loc="left")


def diffusion_mechanism_figure(mech: dict) -> list[Path]:
    """P15 before and after: over how many training patients the synthetic beats spread, for P12's cVAE bank and the
    diffusion bank, beside the real beats themselves."""
    plt = _plt()
    classes = list(mech["classes"])
    series = [("real training beats", "real", INK), ("cVAE bank (P12, the best earlier generator)", "cvae_p12", "#2a78d6"),
              ("diffusion bank (P15)", "diffusion", DIFFUSION_COLOR)]  # fmt: skip
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    width = 0.8 / len(series)
    for j, (label, key, color) in enumerate(series):
        values = [mech["classes"][c][key]["effective_records"] for c in classes]
        x = np.arange(len(classes)) + (j - (len(series) - 1) / 2) * width
        ax.bar(x, values, width=width * 0.9, color=color, label=label)
        for xi, v in zip(x, values):
            ax.annotate(f"{v:.1f}", (xi, v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7, color=INK_2)
    ax.set_xticks(range(len(classes)), [f"{c} beats" for c in classes])
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("effective number of training patients")
    ax.legend(fontsize=7, loc="upper right")
    caption = (f"For each synthetic beat ({mech['beats_per_class']:,} per class from the seed-0 banks), the patient of its nearest real training beat "
               "(Euclidean, the form the cnn sees); bars give exp of the entropy of those patient shares (P8's measure, by patient). Real: the "
               "training beats' own spread. Post hoc (PREREGISTRATION v14, P15).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_diffusion") / "mechanism.png", "Before and after: how many patients the synthetic beats resemble", caption)]


def _real_relative_beats(per_class: int = 2000, seed: int = 0) -> dict[int, np.ndarray] | None:
    """Real DS1 training beats of S, V and F in the cnn's form (P17's shape check uses the same draw); None without local data."""
    from ecgaug.experiments import phase6_diffusion_long as long
    from ecgaug.experiments.common import PreconditionError

    try:
        return long.real_training_beats(per_class, seed)
    except PreconditionError:
        return None


def diffusion_samples_figure() -> list[Path]:
    """P15 before and after: per rare class, the median and middle half of real DS1 training beats, the seed-0 cVAE bank (P12)
    and the seed-0 diffusion bank (P15), in the form the cnn sees them, with three single beats of each (drawn with seed 0)."""
    from ecgaug.generators import bank as banks

    plt = _plt()
    loaded = {}
    for key, gid in (("cvae", "phase6_relative_generators/phase3_cvae-D"), ("diffusion", "phase6_diffusion/diffusion")):
        path = banks.bank_dir(gid) / "seed0.npz"
        if not path.is_file():
            return []
        with np.load(path) as z:
            X, y = np.asarray(z["X"], dtype=np.float32), np.asarray(z["y"])
            loaded[key] = {c: X[y == c] for c in (1, 2, 3)}
    rows = [("cvae", "#2a78d6", "cVAE bank (P12)"), ("diffusion", DIFFUSION_COLOR, "diffusion bank (P15)")]
    longer = banks.bank_dir("phase6_diffusion_long/diffusion") / "seed0.npz"
    if longer.is_file():  # P17: the same model trained five times longer
        with np.load(longer) as z:
            X, y = np.asarray(z["X"], dtype=np.float32), np.asarray(z["y"])
            loaded["diffusion_long"] = {c: X[y == c] for c in (1, 2, 3)}
        rows.append(("diffusion_long", LONG_COLOR, "diffusion, 5x steps (P17)"))
    real = _real_relative_beats()
    if real is not None:
        loaded["real"] = real
        rows.insert(0, ("real", INK, "real DS1 training beats"))
    names = ("N", "S", "V", "F")
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 2.8 * len(rows)), sharex=True, sharey="col")
    rng = np.random.default_rng(0)
    for row, (key, color, label) in enumerate(rows):
        for col, cls in enumerate((1, 2, 3)):
            ax = axes[row, col]
            beats = loaded[key][cls]
            t = np.arange(beats.shape[1])
            lo, mid, hi = np.percentile(beats, [25, 50, 75], axis=0)
            ax.fill_between(t, lo, hi, color=color, alpha=0.25, linewidth=0)
            ax.plot(t, mid, color=color, linewidth=1.6)
            for b in beats[rng.choice(len(beats), 3, replace=False)]:
                ax.plot(t, b, color=INK_2, linewidth=0.6, alpha=0.8)
            ax.set_title(f"{label}: {names[cls]} ({len(beats):,} beats)", loc="left", fontsize=9)
            ax.set_axisbelow(True)
        axes[row, 0].set_ylabel("beat minus the patient's template")
    for ax in axes[-1]:
        ax.set_xlabel("sample (360 Hz, R peak in the window)")
    caption = ("In the form the cnn sees them: the beat minus the patient's ±150 s template (synthetic beats multiplied back by the training scale). "
               "Coloured: median and 25-75% band; grey: three single beats drawn at random. The real row shows up to 2,000 DS1 training beats per "
               "class (the MIT-BIH part of the generators' training patients; INCART and SVDB are not drawn). cVAE: P12; diffusion: P15, the same "
               "training beats and step budget" + ("; P17: P15's model trained for five times as many steps" if "diffusion_long" in loaded else "")
               + ". Post hoc (PREREGISTRATION v14" + (" and v16, P15 and P17)." if "diffusion_long" in loaded else ", P15)."))  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_diffusion") / "samples.png", "Before and after: real and synthetic rare beats", caption)]


def diffusion_results_figure(summary: dict) -> list[Path]:
    """P15 before and after: P13's conditions and the diffusion condition on the test set, alone and with Huang's SVEB stage,
    and the pre-registered comparisons."""
    plt = _plt()
    greys = {"none": "#8f8d88", "class_weighted": "#a6a49f", "random_oversampling": "#bdbbb6", "cvae": "#2a78d6"}
    rows = []
    for key, entry in summary["before"].items():
        c = key.split("/")[1]
        rows += [(f"{CONDITION_NAMES[c]} (P13)", entry, greys.get(c, INK_2)), (f"{CONDITION_NAMES[c]} (P13) + SVEB rule", summary["sveb_rule"][key], "#eb6834")]
    for key, entry in summary["conditions"].items():
        rows += [(f"diffusion, strength {entry.get('strength')} (P15)", entry, DIFFUSION_COLOR), ("diffusion (P15) + SVEB rule", summary["sveb_rule"][key], "#eb6834")]
    fig, axes = plt.subplots(1, 4, figsize=(19, 0.42 * len(rows) + 2.4), gridspec_kw={"width_ratios": [1, 0.7, 0.7, 1.1]})
    for ax, metric, xlabel, title in ((axes[0], "rare_macro_f1", "test rare-class macro-F1 (95% record-level CI)", "each condition (the P7 cnn on P11's patients)"),
                                      (axes[1], "f1_S", "test S F1", "S beats"), (axes[2], "f1_F", "test F F1", "F beats")):  # fmt: skip
        for i, (label, entry, color) in enumerate(reversed(rows)):
            _interval(ax, entry[metric], i, color, marker="s" if "SVEB" in label else "o")
        ax.set_yticks(range(len(rows)), [label for label, _, _ in reversed(rows)] if metric == "rare_macro_f1" else [], fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
    _comparisons_panel(axes[3], summary["families"], "pre-registered comparisons, P15")
    caption = ("P13's conditions (grey; cVAE blue) and the class-conditional diffusion model (purple), same cnn, data and selection rule; squares: Huang et "
               "al.'s SVEB stage on top (descriptive). Right, black: confirmatory (diffusion vs no augmentation and vs the cVAE, P13); grey: "
               "secondary. Post hoc (PREREGISTRATION v14, P15).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_diffusion") / "results.png", "Before and after: a diffusion model as the generator", caption)]


def af_rule_check_figure(check: dict) -> list[Path]:
    """P16's design check: where DS1's false S calls of the rule lie, the vetoed rule's S F1 per training-side database at
    each threshold, and the validation S F1 of each form per condition."""
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.3), gridspec_kw={"width_ratios": [0.75, 1.1, 1.3]})
    ax = axes[0]
    r = check["ds1_rhythm"]
    groups = [("rule's false S calls", r["rule_false_s_in_af"], r["rule_false_s"] - r["rule_false_s_in_af"]), ("real S beats", r["s_beats_in_af"], r["s_beats"] - r["s_beats_in_af"])]
    for i, (label, inside, outside) in enumerate(groups):
        ax.bar(i, inside, color="#eb6834", width=0.55, label="in AF or flutter" if i == 0 else None)
        ax.bar(i, outside, bottom=inside, color="#b8b6b0", width=0.55, label="other rhythms" if i == 0 else None)
        ax.annotate(f"{inside:,} of {inside + outside:,} in AF", (i, inside + outside), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_ylabel("beats (DS1 annotations)")
    ax.set_title("DS1: where the rule goes wrong", loc="left")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    dbs = {"mitdb_ds1": "DS1", "incartdb": "INCART", "svdb": "SVDB"}
    thresholds = list(check["mean_f1_by_threshold"])
    for db, color in zip(check["databases"], ("#2a78d6", "#1a9a6c", "#8a5cc8")):
        e = check["databases"][db]
        ax.plot([float(t) for t in thresholds], [e[f"veto_{t}"]["f1"] for t in thresholds], marker="o", color=color, label=f"{dbs[db]}, vetoed")
        ax.axhline(e["rule"]["f1"], color=color, linestyle=":", linewidth=1.2)
    ax.plot([float(t) for t in thresholds], list(check["mean_f1_by_threshold"].values()), marker="s", color=INK, label="mean of the three, vetoed")
    ax.axvline(check["chosen_threshold"], color=GRID, linewidth=1.2)
    ax.annotate(f"chosen: {check['chosen_threshold']}", (check["chosen_threshold"], 0.04), xytext=(-4, 0), textcoords="offset points", ha="right", fontsize=7, color=INK_2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("AF probability threshold of the veto")
    ax.set_ylabel(f"S F1 of the rule (ratio < {check['rule_threshold']})")
    ax.set_title("training-side annotations (dotted: the rule alone)", loc="left")
    ax.legend(fontsize=7, loc="lower left")
    ax = axes[2]
    v = check["validation"]["conditions"]
    keys = [("cnn", "cnn alone", INK_2), ("rule", "+ SVEB rule", "#eb6834"), ("veto_rule", "+ rule with AF veto (its own S calls)", VETO_COLOR),
            ("veto_all", "+ rule with AF veto (every S call)", "#9ec5f4")]  # fmt: skip
    conditions = list(v)
    width = 0.8 / len(keys)
    for j, (k, label, color) in enumerate(keys):
        x = np.arange(len(conditions)) + (j - (len(keys) - 1) / 2) * width
        ax.bar(x, [float(np.mean(v[c][k]["f1"])) for c in conditions], width=width * 0.9, color=color, label=label)
    ax.set_xticks(range(len(conditions)), [CONDITION_NAMES[c].replace(" ", "\n", 1) for c in conditions], fontsize=7)
    ax.set_ylim(0.5, 0.85)
    ax.set_ylabel("validation S F1 (mean of 3 seeds)")
    ax.set_title(f"validation patients: chosen form '{check['chosen_form']}'", loc="left")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    caption = ("No test data. Left: DS1's annotated rhythms. Middle: the rule on the annotations of DS1, INCART and SVDB, vetoed where the AF detector "
               "(fitted on the MIT-BIH AF database, beat times only) exceeds each threshold; the registered choice is the highest mean. Right: the "
               "training runs' validation predictions; the registered form is the higher mean. Post hoc (PREREGISTRATION v15, P16).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_af_rule") / "check.png", "Why an AF veto: the rule's false S calls sit in atrial fibrillation", caption)]


def af_rule_results_figure(summary: dict) -> list[Path]:
    """P16 before and after: each condition with Huang's SVEB stage alone and with the AF veto, on the test set; S precision
    and recall; the pre-registered comparisons."""
    plt = _plt()
    rows = []
    for key in summary["rule_alone"]:
        c = key.split("/")[1]
        rows += [(f"{CONDITION_NAMES[c]} + SVEB rule", summary["rule_alone"][key], "#eb6834"), (f"{CONDITION_NAMES[c]} + rule with AF veto", summary["with_af_veto"][key], VETO_COLOR)]
    fig, axes = plt.subplots(1, 5, figsize=(21, 0.42 * len(rows) + 2.4), gridspec_kw={"width_ratios": [1, 0.65, 0.65, 0.65, 1.1]})
    panels = ((axes[0], "rare_macro_f1", "test rare-class macro-F1 (95% CI)", "each condition (the P7 cnn)"), (axes[1], "f1_S", "test S F1", "S beats"),
              (axes[2], "ppv_S", "test S +P (precision)", "S calls that are right"), (axes[3], "se_S", "test S Se (recall)", "S beats found"))  # fmt: skip
    for ax, metric, xlabel, title in panels:
        for i, (label, entry, color) in enumerate(reversed(rows)):
            _interval(ax, entry[metric], i, color)
        ax.set_yticks(range(len(rows)), [label for label, _, _ in reversed(rows)] if metric == "rare_macro_f1" else [], fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
    _comparisons_panel(axes[4], summary["families"], "pre-registered comparisons, P16")
    caption = (f"Huang et al.'s SVEB stage (orange) and the same stage with its S calls undone where the AF detector exceeds {summary['af_threshold']} "
               f"(green; form '{summary['veto_form']}'). The detector reads only each test record's beat times. F calls are unchanged. Right: per "
               "condition, the veto vs the stage alone (Holm). Post hoc (PREREGISTRATION v15, P16).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_af_rule") / "results.png", "Before and after: an atrial-fibrillation veto on the SVEB rule", caption)]


def _generator_losses(experiment: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per seed of a diffusion experiment, the training loss averaged over 1,000-step windows (end step, mean); seeds whose
    run log is not on disk are left out."""
    from ecgaug import runs

    curves = []
    for seed in ((_summary(experiment, "diffusion_generators.json") or {}).get("seeds") or {}).values():
        run_dir = runs.find_run_dir(seed["run_id"])
        if run_dir is None or not (run_dir / "train_log.csv").is_file():
            continue
        log = np.genfromtxt(run_dir / "train_log.csv", delimiter=",", names=True)
        window = (log["step"].astype(int) - 1) // 1000
        ends = np.unique(window)
        curves.append(((ends + 1) * 1000, np.array([log["loss"][window == w].mean() for w in ends])))
    return curves


def diffusion_long_check_figure(shape: dict) -> list[Path]:
    """P17 before and after, training data only: the loss of P15's and P17's runs, how closely each bank's class medians
    match real DS1 training beats, and over how many training patients the synthetic beats spread."""
    plt = _plt()
    fig, grid = plt.subplots(2, 2, figsize=(14, 8.6))
    axes = [grid[0, 0], grid[0, 1], grid[1, 1], grid[1, 0]]
    ax = axes[0]
    runs_drawn =(("phase6_diffusion_long", LONG_COLOR, "P17: 100,000 steps", 1.0, 2), ("phase6_diffusion", DIFFUSION_COLOR, "P15: 20,000 steps", 1.8, 3))
    for experiment, color, label, linewidth, z in runs_drawn:  # P15 on top: both start from the same seeds and settings
        for i, (steps, loss) in enumerate(_generator_losses(experiment)):
            ax.plot(steps[1:] / 1000, loss[1:], color=color, linewidth=linewidth, zorder=z, label=f"{label} (3 seeds)" if i == 0 else None)
    ax.axvline(20, color=INK_2, linewidth=0.8, linestyle=":")
    ax.annotate("P15's budget", (20, 1), xycoords=("data", "axes fraction"), xytext=(4, -4), textcoords="offset points", va="top", fontsize=7, color=INK_2)
    ax.set_xlabel("training step (thousands)")
    ax.set_ylabel("loss, mean over 1,000 steps")
    ax.set_title("training loss", loc="left")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, loc="upper right")
    banks_shown = (("cvae_p12", "#2a78d6", "cVAE (P12)"), ("diffusion_p15", DIFFUSION_COLOR, "diffusion (P15)"), ("diffusion_p17", LONG_COLOR, "diffusion, 5x steps (P17)"))
    ax = axes[1]
    classes, width = ("S", "V", "F"), 0.8 / len(banks_shown)
    for j, (key, color, label) in enumerate(banks_shown):
        seeds = shape["banks"].get(key, {})
        if not seeds:
            continue
        x = np.arange(len(classes)) + (j - (len(banks_shown) - 1) / 2) * width
        values = [[s[c]["corr"] for s in seeds.values()] for c in classes]
        ax.bar(x, [float(np.median(v)) for v in values], width=width * 0.9, color=color, label=label)
        for xi, v in zip(x, values):
            ax.scatter([xi] * len(v), v, s=12, color=INK_2, zorder=3)
            ax.annotate(f"{np.median(v):.2f}", (xi, max(max(v), float(np.median(v)))), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7, color=INK_2)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xticks(range(len(classes)), [f"{c} beats" for c in classes])
    ax.set_ylim(min(-0.1, ax.get_ylim()[0]), 1.12)
    ax.set_ylabel("correlation with the real median beat")
    ax.set_title("shape: bank median vs real training beats", loc="left")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax = axes[2]
    mech = {"phase6_diffusion": _summary("phase6_diffusion", "mechanism.json"), "phase6_diffusion_long": _summary("phase6_diffusion_long", "mechanism.json")}
    series = [("real training beats", "phase6_diffusion_long", "real", INK), ("cVAE (P12)", "phase6_diffusion_long", "cvae_p12", "#2a78d6"),
              ("diffusion (P15)", "phase6_diffusion", "diffusion", DIFFUSION_COLOR), ("diffusion, 5x steps (P17)", "phase6_diffusion_long", "diffusion", LONG_COLOR)]  # fmt: skip
    series = [s for s in series if mech[s[1]] is not None]
    mech_classes = list(mech["phase6_diffusion_long"]["classes"]) if mech["phase6_diffusion_long"] is not None else []
    width = 0.8 / max(len(series), 1)
    for j, (label, experiment, key, color) in enumerate(series):
        values = [mech[experiment]["classes"][c][key]["effective_records"] for c in mech_classes]
        x = np.arange(len(mech_classes)) + (j - (len(series) - 1) / 2) * width
        ax.bar(x, values, width=width * 0.9, color=color, label=label)
        for xi, v in zip(x, values):
            ax.annotate(f"{v:.1f}", (xi, v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7, color=INK_2)
    ax.set_xticks(range(len(mech_classes)), [f"{c} beats" for c in mech_classes])
    ax.set_ylabel("effective number of training patients")
    ax.set_title("spread over training patients", loc="left")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, loc="upper right")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax = axes[3]
    align = shape.get("alignment")
    if align is not None:
        series = [("real", INK, "real training beats"), *banks_shown]
        width = 0.8 / len(series)
        for j, (key, color, label) in enumerate(series):
            if key == "real":
                values = [[align["real"][c]] for c in classes]
            else:
                seeds = shape["banks"].get(key, {})
                if not seeds or "aligned" not in next(iter(seeds.values()))["S"]:
                    continue
                values = [[s[c]["aligned"] for s in seeds.values()] for c in classes]
            x = np.arange(len(classes)) + (j - (len(series) - 1) / 2) * width
            ax.bar(x, [float(np.median(v)) for v in values], width=width * 0.9, color=color, label=label)
            for xi, v in zip(x, values):
                if len(v) > 1:
                    ax.scatter([xi] * len(v), v, s=12, color=INK_2, zorder=3)
                ax.annotate(f"{np.median(v):.2f}", (xi, max(v)), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7, color=INK_2)
        ax.set_xticks(range(len(classes)), [f"{c} beats" for c in classes])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"share with the largest deflection within {align['window']} samples")
        ax.set_title("aligned with the real beats' peak (exploratory)", loc="left")
        ax.legend(fontsize=7, loc="upper left", ncol=2)
        ax.set_axisbelow(True)
        ax.grid(axis="x", visible=False)
    else:
        ax.set_visible(False)
    caption = ("No test data. Top left: the diffusion model's training loss (each line one seed; P15's runs drawn on top of P17's, which start "
               "from the same seeds and settings; dotted: P15's budget). Top right: correlation between each bank's median beat and the median "
               "of up to 2,000 real DS1 training beats per class, in the form the cnn sees (bars: median over the three seed banks; dots: single "
               "seeds). Bottom left (exploratory, added after P17's training and before its test read): the share of beats whose largest "
               "deflection lies within 10 samples (28 ms) of where the real beats have it. Bottom right: over how many training patients the "
               "seed-0 synthetic beats spread (P8's measure, as in P15), beside the real beats. Post hoc (PREREGISTRATION v16, P17).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_diffusion_long") / "check.png", "Before and after: five times more training steps for the diffusion model", caption)]


def diffusion_long_results_figure(summary: dict) -> list[Path]:
    """P17 before and after: P15's diffusion condition, P13's conditions and the longer-trained diffusion condition on the
    test set, alone, with Huang's SVEB stage and with the stage plus P16's AF veto; the pre-registered comparisons."""
    plt = _plt()
    colors = {"none": "#8f8d88", "class_weighted": "#a6a49f", "random_oversampling": "#bdbbb6", "cvae": "#2a78d6", "diffusion": DIFFUSION_COLOR, "diffusion_long": LONG_COLOR}
    before = sorted(summary["before"], key=lambda k: k.split("/")[1] == "diffusion")  # P15 last, next to P17
    entries = {**{k: summary["before"][k] for k in before}, **summary["conditions"]}
    rows = []
    for key, entry in entries.items():
        c = key.split("/")[1]
        name = CONDITION_NAMES[c] + (" (P17)" if c == "diffusion_long" else " (P15)" if c == "diffusion" else " (P13)")
        rows += [(name, entry, colors.get(c, INK_2), "o"), (f"{name} + SVEB rule", summary["sveb_rule"][key], "#eb6834", "s"),
                 (f"{name} + rule with AF veto", summary["with_af_veto"][key], VETO_COLOR, "D")]  # fmt: skip
    fig, axes = plt.subplots(1, 4, figsize=(19, 0.36 * len(rows) + 2.4), gridspec_kw={"width_ratios": [1, 0.7, 0.7, 1.1]})
    for ax, metric, xlabel, title in ((axes[0], "rare_macro_f1", "test rare-class macro-F1 (95% record-level CI)", "each condition (the P7 cnn on P11's patients)"),
                                      (axes[1], "f1_S", "test S F1", "S beats"), (axes[2], "f1_F", "test F F1", "F beats")):  # fmt: skip
        for i, (label, entry, color, marker) in enumerate(reversed(rows)):
            _interval(ax, entry[metric], i, color, marker=marker)
        ax.set_yticks(range(len(rows)), [label for label, _, _, _ in reversed(rows)] if metric == "rare_macro_f1" else [], fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
    _comparisons_panel(axes[3], summary["families"], "pre-registered comparisons, P17")
    caption = ("Dots: each condition alone (P13's conditions grey, cVAE blue; P15's diffusion model purple; P17, the same model trained for five times "
               "as many steps, magenta); squares: Huang et al.'s SVEB stage on top; diamonds: the stage with P16's AF veto (both descriptive). Right, "
               "black: confirmatory (P17 vs P15 and vs no augmentation); grey: secondary. Post hoc (PREREGISTRATION v16, P17).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_diffusion_long") / "results.png", "Before and after: the diffusion model trained five times longer", caption)]


ENSEMBLE_COLOR, SINGLE_RUN_COLOR = "#0f6e8c", "#a6a49f"


def seed_ensemble_check_figure(members: dict) -> list[Path]:
    """P18 before and after on the validation patients (the runs' stored predictions, no test data): S and V F1 with Huang's
    SVEB stage and P16's AF veto, for each of the ten single runs, the three-run and the ten-run ensemble of each condition."""
    plt = _plt()
    table = members["validation"]
    conditions = list(table)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 0.5 * len(conditions) + 2.0), sharey=True)
    for ax, cls in zip(axes, ("S", "V")):
        for i, c in enumerate(reversed(conditions)):
            e = table[c]["veto"][cls]
            ax.scatter(e["single"], np.full(len(e["single"]), i), s=14, color=SINGLE_RUN_COLOR, zorder=2, label="single runs (seeds 0-9)" if i == 0 else None)
            ax.scatter([e["three"]], [i + 0.18], s=40, facecolors="none", edgecolors=ENSEMBLE_COLOR, linewidths=1.4, zorder=3,
                       label="three-run ensemble (seeds 0-2)" if i == 0 else None)  # fmt: skip
            ax.scatter([e["ten"]], [i - 0.18], s=40, color=ENSEMBLE_COLOR, zorder=3, label="ten-run ensemble" if i == 0 else None)
        ax.set_yticks(range(len(conditions)), [CONDITION_NAMES[c] for c in reversed(conditions)])
        ax.set_xlabel(f"validation {cls} F1, with the SVEB stage and the AF veto")
        ax.set_title(f"{cls} beats", loc="left")
        ax.grid(axis="y", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles, labels, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    caption = (f"Validation patients only ({len(members['records'])} recordings; the runs' stored predictions). Grey: each of a condition's ten "
               "runs alone; open: the mean class probabilities of seeds 0-2; filled: of all ten. Huang et al.'s SVEB stage and P16's AF veto are "
               "applied after the averaging. Descriptive: nothing is chosen. Post hoc (PREREGISTRATION v17, P18).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_seed_ensemble") / "check.png", "Before and after: ten runs averaged, on the validation patients", caption)]


def seed_ensemble_results_figure(summary: dict) -> list[Path]:
    """P18 before and after on the test set: per condition, P16's single runs (seeds 0-2) and the ten-run ensemble, both with
    Huang's SVEB stage and P16's AF veto, with the ten single runs' scores; the pre-registered comparisons."""
    plt = _plt()
    keys = list(summary["ensemble"]["with_af_veto"])
    rows = []
    for key in keys:
        name = CONDITION_NAMES[key.split("/")[1]]
        rows += [(f"{name}: single runs (P16)", summary["p16"]["with_af_veto"][key], VETO_COLOR, "D", summary["per_run_points"][key]),
                 (f"{name}: ten-run ensemble", summary["ensemble"]["with_af_veto"][key], ENSEMBLE_COLOR, "o", None)]  # fmt: skip
    fig, axes = plt.subplots(1, 4, figsize=(19, 0.36 * len(rows) + 2.4), gridspec_kw={"width_ratios": [1, 0.7, 0.7, 1.1]})
    for ax, metric, xlabel, title in ((axes[0], "rare_macro_f1", "test rare-class macro-F1 (95% record-level CI)", "each condition, stage and veto on top"),
                                      (axes[1], "f1_S", "test S F1", "S beats"), (axes[2], "f1_F", "test F F1", "F beats")):  # fmt: skip
        for i, (label, entry, color, marker, points) in enumerate(reversed(rows)):
            if metric == "rare_macro_f1" and points is not None:
                ax.scatter(points, np.full(len(points), i - 0.22), s=9, color=SINGLE_RUN_COLOR, zorder=2,
                           label="each of the ten single runs" if i == 1 else None)  # fmt: skip
            _interval(ax, entry[metric], i, color, marker=marker)
        ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)] if metric == "rare_macro_f1" else [], fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, fontsize=7, loc="lower right")
    _comparisons_panel(axes[3], summary["families"], "pre-registered comparisons, P18")
    caption = ("Every row has Huang et al.'s SVEB stage and P16's AF veto on top. Diamonds: P16, the mean over three single runs (seeds 0-2); "
               "grey dots: each of the condition's ten runs alone; circles: the ten runs' class probabilities averaged before the argmax. Right, "
               "black: confirmatory (per condition, ensemble vs P16); grey: secondary (the cVAE's ensemble vs each other condition's). "
               "Post hoc (PREREGISTRATION v17, P18).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_seed_ensemble") / "results.png", "Before and after: ten runs averaged per condition", caption)]


FRESH_COLOR = "#c46a1b"


def fresh_patients_figure(summary: dict, p18: dict | None) -> list[Path]:
    """P19 before and after: per condition, S and V F1 on the test records (P18) and on 1,000 unused Icentia11k patients, the ten-run
    ensemble with Huang's stage and P16's veto and the ensemble alone; the pre-registered comparisons on S F1."""
    plt = _plt()
    keys = list(summary["ensemble"]["with_af_veto"])
    rows = []
    for key in keys:
        name = CONDITION_NAMES[key.split("/")[1]]
        if p18 is not None:
            rows.append((f"{name}: test records (P18), stage and veto", p18["ensemble"]["with_af_veto"].get(key), ENSEMBLE_COLOR, "s"))
        rows += [(f"{name}: new patients, stage and veto", summary["ensemble"]["with_af_veto"][key], FRESH_COLOR, "o"),
                 (f"{name}: new patients, cnn alone", summary["ensemble"]["cnn_alone"][key], SINGLE_RUN_COLOR, "o")]  # fmt: skip
    fig, axes = plt.subplots(1, 3, figsize=(17, 0.3 * len(rows) + 2.4), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    for ax, metric, xlabel in ((axes[0], "f1_S", "S F1 (95% patient-level CI)"), (axes[1], "f1_V", "V F1 (95% patient-level CI)")):
        for i, (label, entry, color, marker) in enumerate(reversed(rows)):
            if entry is not None and metric in entry:
                _interval(ax, entry[metric], i, color, marker=marker)
        ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)] if metric == "f1_S" else [], fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_xlabel(xlabel)
        ax.set_title(f"{metric[-1]} beats, ten-run ensembles", loc="left")
    _comparisons_panel(axes[2], summary["families"], "pre-registered comparisons, P19")
    axes[2].set_xlabel("difference in S F1 on the new patients (95% CI)")
    data = summary.get("data", {})
    beats = summary["beats"]
    caption = (f"{summary['patients']:,} Icentia11k patients that no analysis had used (first segment each; {beats['N']:,} N, {beats['S']:,} S and "
               f"{beats['V']:,} V beats; no F labels), scored with P18's sixty runs, Huang et al.'s stage and P16's AF veto unchanged. Squares: P18 "
               "on the test records; orange: the new patients with the stage and the veto; grey: the ensemble alone. Right, black: confirmatory (per condition, "
               "ensemble vs P16's single runs); grey: secondary (the cVAE vs each other condition), both on S F1"
               + (f"; {data['s_dominant_patients']} S-dominant patients" if "s_dominant_patients" in data else "")
               + ". Post hoc (PREREGISTRATION v18, P19).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_fresh_patients") / "results.png", f"Before and after: the same pipeline on {summary['patients']:,} new patients", caption)]


TUNED_COLOR = "#1f9e6e"
RULE_GRID = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
VETO_GRID = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)


def _setting_label(t: float | None, a: float | None) -> str:
    return "ensemble alone" if t is None else f"stage {t:.2f}" + ("" if a is None else f", veto {a:.2f}")


def rule_tuning_choice_figure(choice: dict, summary: dict | None) -> list[Path]:
    """P20's choice: per condition, pooled S F1 of every setting against the stage's threshold, on the 400 tuning patients
    (top) and on P19's patients (bottom, after the choice); the chosen setting and P19's setting marked."""
    plt = _plt()
    conditions = list(choice["chosen"])
    rows = [("tuning patients (P14's 400): the choice", choice["tuning_surface"])]
    if summary is not None and "evaluation_surface" in summary:
        rows.append(("P19's patients: after the choice", summary["evaluation_surface"]))
    fig, axes = plt.subplots(len(rows), len(conditions), figsize=(2.9 * len(conditions), 3.0 * len(rows) + 1.0), squeeze=False, sharey="row")
    shades = BLUES[2:]  # light to dark: low to high veto threshold
    for r, (row_title, table) in enumerate(rows):
        for j, c in enumerate(conditions):
            ax, scores = axes[r][j], table[c]
            ax.axhline(scores["ensemble alone"], color=SINGLE_RUN_COLOR, linestyle="--", linewidth=1.5, label="ensemble alone")
            ax.plot(RULE_GRID, [scores[_setting_label(t, None)] for t in RULE_GRID], color=INK, linewidth=2, label="stage, no veto")
            for a, shade in zip(VETO_GRID, shades):
                ax.plot(RULE_GRID, [scores[_setting_label(t, a)] for t in RULE_GRID], color=shade, linewidth=1.5, label=f"stage, veto {a:.2f}")
            ax.plot(0.80, scores[_setting_label(0.80, 0.95)], "o", color=FRESH_COLOR, markersize=8, label="P19's setting", zorder=4)
            ch = choice["chosen"][c]
            x = 0.80 if ch["rule_threshold"] is None else ch["rule_threshold"]
            ax.plot(x, scores[ch["label"]], "*", color=TUNED_COLOR, markersize=14, markeredgecolor=SURFACE, label="chosen", zorder=5)
            ax.set_title(f"{CONDITION_NAMES[c]}\nchosen: {ch['label']}" if r == 0 else CONDITION_NAMES[c], loc="left", fontsize=8)
            ax.set_xticks(RULE_GRID[::2])
            if j == 0:
                ax.set_ylabel(f"S F1, {row_title}", fontsize=8)
            if r == len(rows) - 1:
                ax.set_xlabel("stage threshold t")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=5, fontsize=7, frameon=False, bbox_to_anchor=(0.995, 1.0))
    caption = ("Pooled S F1 of each condition's ten-run ensemble for the 50 settings: the ensemble alone (dashed), Huang et al.'s stage at threshold t "
               "with no veto (black) and with P16's AF veto at six thresholds (light to dark: 0.50 to 0.95). Green star: the setting chosen on the "
               "tuning patients; orange: P19's setting (t 0.80, veto 0.95). Bottom row: the same settings on P19's patients, which played no part in "
               "the choice. Post hoc (PREREGISTRATION v19, P20).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_rule_tuning") / "choice.png", "The choice: S F1 of every setting, per condition", caption)]


def rule_tuning_results_figure(summary: dict, test: dict | None) -> list[Path]:
    """P20 before and after: per condition, S F1 on P19's patients with P19's setting, the chosen setting and the ensemble
    alone; rare-class macro-F1 on the test records with P18's setting and the chosen one; the pre-registered comparisons."""
    plt = _plt()
    keys = list(summary["conditions"]["chosen"])
    panels = 2 + (2 if test is not None else 0)
    fig, axes = plt.subplots(1, panels, figsize=(4.6 * panels, 0.3 * 3 * len(keys) + 2.2))
    rows = [(k, kind, color, marker) for k in keys for kind, color, marker in
            (("p19_setting", FRESH_COLOR, "o"), ("chosen", TUNED_COLOR, "*"), ("ensemble_alone", SINGLE_RUN_COLOR, "o"))]  # fmt: skip
    names = {"p19_setting": "P19's setting", "chosen": "chosen", "ensemble_alone": "ensemble alone"}
    ax = axes[0]
    for i, (k, kind, color, marker) in enumerate(reversed(rows)):
        _interval(ax, summary["conditions"][kind][k]["f1_S"], i, color, marker=marker)
    ax.set_yticks(range(len(rows)), [f"{CONDITION_NAMES[k.split('/')[1]]}: {names[kind]}" for k, kind, _, _ in reversed(rows)], fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("S F1 on P19's patients (95% patient-level CI)")
    ax.set_title("after: P19's patients", loc="left")
    _comparisons_panel(axes[1], summary["families"], "pre-registered comparisons on P19's patients")
    axes[1].set_xlabel("difference in S F1 (95% CI)")
    if test is not None:
        rows2 = [(k, kind, color, marker) for k in keys for kind, color, marker in (("p18_setting", ENSEMBLE_COLOR, "s"), ("chosen", TUNED_COLOR, "*"))]
        names2 = {"p18_setting": "P18's setting", "chosen": "chosen"}
        ax = axes[2]
        for i, (k, kind, color, marker) in enumerate(reversed(rows2)):
            _interval(ax, test["conditions"][kind][k]["rare_macro_f1"], i, color, marker=marker)
        ax.set_yticks(range(len(rows2)), [f"{CONDITION_NAMES[k.split('/')[1]]}: {names2[kind]}" for k, kind, _, _ in reversed(rows2)], fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_xlabel("rare-class macro-F1 on the test records (95% record-level CI)")
        ax.set_title("the test records, one read", loc="left")
        _comparisons_panel(axes[3], {"confirmatory": test["families"]["ds2"]}, "test records: chosen vs P18's setting")
    caption = (f"{summary['patients']:,} Icentia11k patients from P19 (no F labels). Orange: P19's setting (Huang et al.'s stage at 0.80, P16's veto at 0.95); "
               "green star: the setting chosen per condition on P14's 400 patients; grey: the ten-run ensemble alone. "
               + ("Third panel: the test records, P18's setting (squares) and the chosen one; right, chosen vs P18's setting, Holm over six. " if test is not None else "")
               + "Second panel, black: confirmatory (chosen vs P19's setting); grey: secondary (chosen vs the ensemble alone). Post hoc (PREREGISTRATION v19, P20).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_rule_tuning") / "results.png", "Before and after: the stage and the veto chosen on other patients", caption)]


PHYSICS_NAMES = {"cvae_physics": "cVAE + Euler loss", "wgan_gp_physics": "WGAN-GP + Euler loss", "cvae_fusion": "cVAE, fusion F beats"}
PHYSICS_BASE = {"cvae_physics": "cvae", "wgan_gp_physics": "wgan_gp", "cvae_fusion": "cvae"}


def _beat_axis(n: int = 180) -> np.ndarray:
    return (np.arange(n) - n // 2) / 360 * 1000


def physics_check_figure(physics: dict) -> list[Path]:
    """P21's check, before and after the fit: real training beats and McSharry et al.'s simulator fitted to them (median
    and 10th-percentile R^2 per class), and the fusion model on a real F beat with the spread of alpha-hat."""
    plt = _plt()
    classes = [c for c in ("S", "V", "F") if c in physics["classes"]]
    fig, axes = plt.subplots(2, len(classes) + 1, figsize=(3.9 * (len(classes) + 1), 6.6), squeeze=False)
    for j, c in enumerate(classes):
        entry = physics["classes"][c]
        for r, q in enumerate(("q50", "q10")):
            ex, ax = physics["examples"][c][q], axes[r][j]
            t = _beat_axis(len(ex["real"]))
            ax.plot(t, ex["real"], color=INK, linewidth=1.6, label="real training beat")
            ax.plot(t, ex["fitted"], color=TUNED_COLOR, linewidth=2, linestyle="--", label="simulator fitted to it")
            head = f"{c}: {'median' if q == 'q50' else '10th-percentile'} fit, R² {ex['r2']:.3f}"
            ax.set_title(head + (f"\nmedian R² over {entry['beats']:,} beats: {entry['median_r2']:.3f}" if r == 0 else ""), loc="left", fontsize=8)
            if r == 1:
                ax.set_xlabel("ms from the R peak")
            if j == 0:
                ax.set_ylabel("whole beat (z-scored)")
    fu, ex = physics["fusion"], physics["examples"].get("fusion")
    ax = axes[0][-1]
    if ex:
        t = _beat_axis(len(ex["real"]))
        ax.plot(t, ex["vbar"], color=SINGLE_RUN_COLOR, linewidth=1.2, label="the recording's mean V deviation")
        ax.plot(t, ex["real"], color=INK, linewidth=1.6)
        ax.plot(t, ex["fitted"], color=FRESH_COLOR, linewidth=2, linestyle="--", label="alpha-hat x mean V deviation")
        stop = "" if fu["passed"] else ": below the stop rule, not run"
        ax.set_title(f"fusion model, record {ex['record']}: alpha-hat {ex['alpha']:.2f}\nmedian R² over {fu['f_beats']:,} F beats {fu['median_r2']:.3f}{stop}", loc="left", fontsize=8)
        ax.set_ylabel("patient-relative beat")
    ax = axes[1][-1]
    q = fu.get("alpha_quantiles", {})
    if q:
        ax.axvspan(fu["alpha_low"], fu["alpha_high"], color=BLUES[2], alpha=0.6, label="range the fusion beats draw from")
        keys = sorted(q)
        ax.plot([q[k] for k in keys], [0] * len(keys), "o-", color=INK)
        for k in keys:
            ax.annotate(k.replace("q", "")[:2] + "%", (q[k], 0), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7, color=INK_2)
        ax.set_yticks([])
        ax.set_xlabel("alpha-hat of real training F beats")
        ax.set_title(f"{len(fu['recordings'])} recordings, alpha {fu['alpha_low']:.2f} to {fu['alpha_high']:.2f}", loc="left", fontsize=8)
    handles = [h for a in (axes[0][0], axes[0][-1], axes[1][-1]) for h in a.get_legend_handles_labels()[0]]
    labels = [lab for a in (axes[0][0], axes[0][-1], axes[1][-1]) for lab in a.get_legend_handles_labels()[1]]
    fig.legend(handles, labels, loc="upper right", ncol=5, fontsize=7, frameon=False, bbox_to_anchor=(0.995, 1.0))
    caption = ("Real training beats (black) and McSharry et al.'s simulator fitted to each (green, dashed), at the median and the 10th-percentile R² "
               "of each class; whole beats (the patient's template plus the patient-relative beat). Right: a real F beat against alpha-hat times "
               "its recording's mean V deviation (orange, dashed), and the spread of alpha-hat (shaded: the 10th-90th percentile range the fusion "
               f"beats draw from). Stop rule: median R² below {physics.get('r2_stop', 0.5)}. Training beats only. Post hoc (PREREGISTRATION v20, P21).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_physics_prior") / "physics_fit.png", "Before and after the fit: real beats and the simulator", caption)]


def physics_banks_figure(families: dict[str, dict], fusion: dict | None) -> list[Path]:
    """P21's weight choice (validation S F1 per lambda, 0 beside) and the seed-0 banks: Euler loss per class and how many
    training patients the synthetic S and F beats spread over (P12's bank before, the physics banks after)."""
    plt = _plt()
    fams = [f for f in ("cvae", "wgan_gp") if f in families]
    fig, axes = plt.subplots(1, 1 + len(fams) + 1, figsize=(4.4 * (2 + len(fams)), 4.4), squeeze=False)
    axes = axes[0]
    ax = axes[0]
    for f in fams:
        table = families[f]["lambda_choice"]["table"]
        lams = list(table)
        x = np.arange(len(lams))
        ax.plot(x, [table[k]["mean"] for k in lams], "o-", color=FAMILY_COLORS[f], label=CONDITION_NAMES[f])
        for i, k in enumerate(lams):
            ax.plot([i] * len(table[k]["values"]), table[k]["values"], ".", color=FAMILY_COLORS[f], alpha=0.5)
        chosen = lams.index(f"{families[f]['chosen_lambda']:g}")
        ax.plot(chosen, table[lams[chosen]]["mean"], "*", color=TUNED_COLOR, markersize=15, markeredgecolor=SURFACE, zorder=5)
        ax.set_xticks(x, [f"λ {k}" for k in lams])
    ax.set_ylabel("validation S F1 (cnn seeds 0-2)")
    ax.set_title("the weight choice (star: chosen; λ 0 beside)", loc="left", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    for ax, f in zip(axes[1 : 1 + len(fams)], fams):
        d = families[f]["describe"]
        names = list(d)
        for c in ("S", "V", "F"):
            ax.plot(range(len(names)), [d[n][c]["euler"] for n in names], "o-", color=CLASS_COLORS[c], label=c)
        ax.axhline(1.0, color=INK_2, linewidth=1, dashes=(2, 2))
        ax.set_xticks(range(len(names)), [n.replace("_seed0", "").replace("lambda", "λ ").replace("p12", "P12") for n in names], fontsize=7)
        ax.set_ylabel("Euler loss over κ (dashed: 1)")
        ax.set_title(f"{CONDITION_NAMES[f]}: physics error, before (P12) and after", loc="left", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
    ax = axes[-1]
    rows = []
    for f in fams:
        d, lam = families[f]["describe"], f"lambda{families[f]['chosen_lambda']:g}_seed0"
        for name, label in (("p12_seed0", f"{CONDITION_NAMES[f]}, P12 (before)"), (lam, f"{CONDITION_NAMES[f]} + Euler loss (after)")):
            rows.append((label, d[name]["S"].get("nearest_patients"), d[name]["F"].get("nearest_patients")))
    if fusion is not None:
        rows.append(("cVAE, P12 F beats (before)", None, fusion["nearest_patients_F"]["p12_seed0"]))
        rows.append(("cVAE, fusion F beats (after)", None, fusion["nearest_patients_F"]["fusion_seed0"]))
    seen: set[str] = set()
    for i, (_, s_eff, f_eff) in enumerate(reversed(rows)):
        for cls, value, marker in (("S", s_eff, "o"), ("F", f_eff, "s")):
            if value is not None:
                ax.plot(value, i, marker, color=CLASS_COLORS[cls], label=None if cls in seen else cls)
                seen.add(cls)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=7)
    ax.set_xlabel("effective number of training patients (nearest real beat)")
    ax.set_title("spread over patients", loc="left", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    caption = ("Left: mean validation S F1 of three cnn runs per weight λ of the Euler loss (dots: the runs; star: the chosen λ; λ 0 is P12's own "
               "bank, beside). Middle: the seed-0 banks' Euler loss per class, the value the generators were trained on, over κ (κ: real "
               "whole training beats with their own templates average 1, dashed); 'real': real relative training beats scored as in "
               "training, with a random template of their class. Right: over how many training patients the synthetic S and F beats spread, by the patient "
               "of each beat's nearest real training beat. Training beats only. Post hoc (PREREGISTRATION v20, P21).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_physics_prior") / "banks.png", "The weight choice and the banks, before and after", caption)]


def _physics_family(items: dict) -> dict:
    out = {}
    for name, v in items.items():
        for c, label in PHYSICS_NAMES.items():
            name = name.replace(c, label)
        out[name.replace("cnn: ", "")] = v
    return out


def physics_results_figure(summary: dict, test: dict | None) -> list[Path]:
    """P21 before and after: each new condition against the P18 condition it changes (before: P18's ensemble with P20's
    setting; after: the new one with its chosen setting), on P19's patients (S F1) and on the test records."""
    plt = _plt()
    pairs = [c for c in PHYSICS_NAMES if f"cnn/{c}" in summary["conditions"]["chosen"]]
    panels = 2 + (2 if test is not None else 0)
    fig, axes = plt.subplots(1, panels, figsize=(4.6 * panels, 0.45 * 2 * len(pairs) + 2.4))
    rows = [(c, key, color, marker, label) for c in pairs for key, color, marker, label in
            ((PHYSICS_BASE[c], ENSEMBLE_COLOR, "s", f"{CONDITION_NAMES[PHYSICS_BASE[c]]} (before)"), (c, TUNED_COLOR, "*", f"{PHYSICS_NAMES[c]} (after)"))]  # fmt: skip
    ax = axes[0]
    for i, (_, key, color, marker, _) in enumerate(reversed(rows)):
        _interval(ax, summary["conditions"]["chosen"][f"cnn/{key}"]["f1_S"], i, color, marker=marker)
    ax.set_yticks(range(len(rows)), [r[4] for r in reversed(rows)], fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("S F1 on P19's patients (95% patient-level CI)")
    ax.set_title("P19's patients, chosen settings", loc="left")
    _comparisons_panel(axes[1], {"secondary": _physics_family(summary["families"]["secondary"])}, "after vs before, P19's patients")
    axes[1].set_xlabel("difference in S F1 (95% CI)")
    if test is not None:
        ax = axes[2]
        for i, (_, key, color, marker, _) in enumerate(reversed(rows)):
            _interval(ax, test["conditions"]["chosen"][f"cnn/{key}"]["rare_macro_f1"], i, color, marker=marker)
        ax.set_yticks(range(len(rows)), [r[4] for r in reversed(rows)], fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_xlabel("rare-class macro-F1 on the test records (95% record-level CI)")
        ax.set_title("the test records, one read", loc="left")
        _comparisons_panel(axes[3], {"confirmatory": _physics_family(test["families"]["confirmatory"])}, "after vs before, test records")
        axes[3].locator_params(axis="x", nbins=4)
    caption = ("Squares: P18's ten-run ensemble of the generator a condition changes, with the setting P20 chose for it; stars: the new condition "
               "(ten runs) with the setting chosen for it on P14's patients by P20's rule. "
               + ("Right: the confirmatory family on the test records (Holm over three). " if test is not None else "")
               + "Second panel: the secondary family on P19's 977 patients (Holm over three). Post hoc (PREREGISTRATION v20, P21).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_physics_prior") / "results.png", "Before and after: physics in the generators", caption)]

PAIR_NAMES = {"cnn/none": "cnn, no augmentation", "cnn/cvae": "cnn, cVAE", "cnn/wgan_gp": "cnn, WGAN-GP",
              "resnet/none": "ResNet, no augmentation", "resnet/cvae": "ResNet, cVAE", "resnet/wgan_gp": "ResNet, WGAN-GP",
              "cnn/cvae_fitted": "cVAE + fitted physics loss", "cnn/wgan_gp_fitted": "WGAN-GP + fitted physics loss"}


def _pair_name(name: str) -> str:
    """``resnet/cvae vs cnn/cvae, chosen settings`` -> ``ResNet, cVAE vs cnn, cVAE``."""
    out = name.replace(", chosen settings", "")
    for key in sorted(PAIR_NAMES, key=len, reverse=True):
        out = out.replace(key, PAIR_NAMES[key])
    return out


def ensemble_pairs_figure(summary: dict, test: dict | None, pairs: dict[str, str], experiment: str, title: str, caption: str) -> list[Path]:
    """Before and after for P22 and P23: each new ten-run ensemble against the P18 ensemble it is compared with (before:
    P18's with P20's setting; after: the new one with its chosen setting), on P19's patients (S F1) and on the test
    records (rare-class macro-F1), with the registered comparisons."""
    plt = _plt()
    present = {c: b for c, b in pairs.items() if c in summary["conditions"]["chosen"]}
    panels = 2 + (2 if test is not None else 0)
    fig, axes = plt.subplots(1, panels, figsize=(4.8 * panels, 0.45 * 2 * len(present) + 2.4))
    rows = [(key, color, marker, f"{PAIR_NAMES[key]} ({when})") for c, b in present.items()
            for key, color, marker, when in ((b, ENSEMBLE_COLOR, "s", "before"), (c, TUNED_COLOR, "*", "after"))]  # fmt: skip

    def intervals(ax, block: dict, stat: str, xlabel: str, head: str) -> None:
        for i, (key, color, marker, _) in enumerate(reversed(rows)):
            _interval(ax, block[key][stat], i, color, marker=marker)
        ax.set_yticks(range(len(rows)), [r[3] for r in reversed(rows)], fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_xlabel(xlabel)
        ax.set_title(head, loc="left")

    intervals(axes[0], summary["conditions"]["chosen"], "f1_S", "S F1 on P19's patients (95% patient-level CI)", "P19's patients, chosen settings")
    _comparisons_panel(axes[1], {"secondary": {_pair_name(k): v for k, v in summary["families"]["secondary"].items()}}, "after vs before, P19's patients")
    axes[1].set_xlabel("difference in S F1 (95% CI)")
    if test is not None:
        intervals(axes[2], test["conditions"]["chosen"], "rare_macro_f1", "rare-class macro-F1 on the test records (95% record-level CI)", "the test records, one read")
        _comparisons_panel(axes[3], {"confirmatory": {_pair_name(k): v for k, v in test["families"]["confirmatory"].items()}}, "after vs before, test records")
        axes[3].locator_params(axis="x", nbins=4)
    return [_save(fig, paths.figures_for(experiment) / "results.png", title, caption)]


def bigger_classifier_validation_figure(members: dict, cnn: dict | None) -> list[Path]:
    """P22 before and after on validation: each of the ten runs' validation S, V and F F1 (its kept epoch) for the cnn
    (P18's runs) and the ResNet, per condition."""
    plt = _plt()
    conds = [c.split("/", 1)[1] for c in members["validation"]]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, stat in zip(axes, ("f1_S", "f1_V", "f1_F")):
        for j, c in enumerate(conds):
            for k, (arch, values, color, marker) in enumerate((("cnn", (cnn or {}).get(f"cnn/{c}", {}).get(stat), ENSEMBLE_COLOR, "s"),
                                                               ("ResNet", members["validation"][f"resnet/{c}"][stat], TUNED_COLOR, "*"))):  # fmt: skip
                if values is None:
                    continue
                x = j + (k - 0.5) * 0.3
                ax.plot([x] * len(values), values, marker, color=color, alpha=0.45, markersize=5)
                ax.plot([x - 0.1, x + 0.1], [np.mean(values)] * 2, color=color, linewidth=2.5, label=f"{arch} (bar: mean of ten runs)" if j == 0 else None)
        ax.set_xticks(range(len(conds)), [CONDITION_NAMES[c] for c in conds], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"validation {stat.split('_')[1]} F1")
        ax.set_title(f"{stat.split('_')[1]} F1, each run's kept epoch", loc="left", fontsize=9)
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    p = members.get("classifier", {})
    caption = (f"Validation F1 of the ten runs per condition: P18's cnn (squares) and phase 2's ResNet1D (stars; {p.get('n_parameters_phase2', 0):,} "
               "parameters in phase 2's count against the cnn's 73,226), same data, banks, seeds and budget. Validation patients only. "
               "Post hoc (PREREGISTRATION v21, P22).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_bigger_classifier") / "validation.png", "Before and after on validation: the cnn and the ResNet", caption)]


FITTED_BANKS = {"real": "real", "templates_alone": "templates\nalone", "p12_seed0": "P12", "p21_seed0": "P21",
                "lambda0.1_seed0": "λ 0.1", "lambda1_seed0": "λ 1", "lambda10_seed0": "λ 10"}


def physics_fitted_check_figure(check: dict) -> list[Path]:
    """P23's check: how much of real training beats the simulator explains with the class's positions and fitted heights
    (P21's fit, with every parameter fitted, beside), and the loss of the banks that exist before P23 trains anything."""
    plt = _plt()
    classes = list(check["classes"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    ax = axes[0]
    for j, c in enumerate(classes):
        q = check["classes"][c]["r2_quantiles"]
        ax.plot([j, j], [q["q10"], q["q90"]], color=CLASS_COLORS[c], linewidth=3, alpha=0.5)
        ax.plot(j, q["q50"], "o", color=CLASS_COLORS[c], markersize=9, label=f"{c}: median {q['q50']:.2f} (P21's full fit {check['classes'][c]['p21_median_r2']:.2f})")
        ax.plot(j + 0.15, check["classes"][c]["p21_median_r2"], "s", color=INK_2, markersize=6)
    ax.axhline(check.get("r2_stop", 0.5), color=INK_2, linewidth=1, dashes=(2, 2))
    ax.set_xticks(range(len(classes)), classes)
    ax.set_ylim(min(0, min(check["classes"][c]["r2_quantiles"]["q10"] for c in classes)), 1.02)
    ax.set_ylabel("R² of real training beats")
    ax.set_title("the simulator with the class's wave timing (dashed: stop rule)", loc="left", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax = axes[1]
    d = check["describe"]
    names = list(d)
    for c in classes:
        ax.plot(range(len(names)), [d[n][c]["physics"] for n in names], "o-", color=CLASS_COLORS[c], label=c)
    ax.axhline(1.0, color=INK_2, linewidth=1, dashes=(2, 2))
    ax.set_xticks(range(len(names)), [n.replace("_seed0", "").replace("_", " ").replace("p12", "P12").replace("p21", "P21").replace("wgan gp", "WGAN-GP").replace("cvae", "cVAE") for n in names], fontsize=7)
    ax.set_ylabel("fitted physics loss over κ′ (dashed: 1)")
    ax.set_title("the loss before any P23 training", loc="left", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    caption = ("Left: R² of P21's fitted real training beats when the simulator keeps the class's mean wave positions and widths and fits only "
               "heights and baseline (bar: 10th-90th percentile; grey squares: P21's fit of every parameter). Right: the fitted physics loss of "
               "real relative beats, of the templates alone (a zero deviation), and of P12's and P21's seed-0 banks (each beat plus a random "
               "training template of its class, as in training). κ′ makes real whole beats with their own templates average 1 (dashed); with a "
               "random template of their class real V and F beats score higher. Training beats only. Post hoc (PREREGISTRATION v21, P23).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_physics_fitted") / "check.png", "The fitted simulator on real beats", caption)]


def physics_fitted_banks_figure(families: dict[str, dict]) -> list[Path]:
    """P23's weight choice and the seed-0 banks before (P12, P21) and after: fitted physics loss, roughness and spread."""
    plt = _plt()
    fams = [f for f in ("cvae", "wgan_gp") if f in families]
    fig, axes = plt.subplots(2, 1 + len(fams), figsize=(4.6 * (1 + len(fams)), 8.2), squeeze=False)
    ax = axes[0][0]
    for f in fams:
        table = families[f]["lambda_choice"]["table"]
        lams = list(table)
        ax.plot(range(len(lams)), [table[k]["mean"] for k in lams], "o-", color=FAMILY_COLORS[f], label=CONDITION_NAMES[f])
        for i, k in enumerate(lams):
            ax.plot([i] * len(table[k]["values"]), table[k]["values"], ".", color=FAMILY_COLORS[f], alpha=0.5)
        chosen = lams.index(f"{families[f]['chosen_lambda']:g}")
        ax.plot(chosen, table[lams[chosen]]["mean"], "*", color=TUNED_COLOR, markersize=15, markeredgecolor=SURFACE, zorder=5)
        ax.set_xticks(range(len(lams)), [f"λ {k}" for k in lams])
    ax.set_ylabel("validation S F1 (cnn seeds 0-2)")
    ax.set_title("the weight choice (star: chosen; λ 0 beside)", loc="left", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1][0]
    rows = []
    for f in fams:
        d, lam = families[f]["describe"], f"lambda{families[f]['chosen_lambda']:g}_seed0"
        for name, label in (("p12_seed0", "P12 (before)"), ("p21_seed0", "P21"), (lam, "fitted loss (after)")):
            rows.append((f"{CONDITION_NAMES[f]}, {label}", d[name]["S"].get("nearest_patients"), d[name]["F"].get("nearest_patients")))
    for i, (_, s_eff, f_eff) in enumerate(reversed(rows)):
        ax.plot(s_eff, i, "o", color=CLASS_COLORS["S"], label="S" if i == 0 else None)
        ax.plot(f_eff, i, "s", color=CLASS_COLORS["F"], label="F" if i == 0 else None)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=7)
    ax.set_xlabel("effective number of training patients (nearest real beat)")
    ax.set_title("spread over patients", loc="left", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    for j, f in enumerate(fams):
        d = families[f]["describe"]
        names = [n for n in FITTED_BANKS if n in d]
        for r, (key, ylabel) in enumerate((("physics", "fitted physics loss over κ′ (dashed: 1)"), ("roughness", "roughness over the real beats'"))):
            ax = axes[r][j + 1]
            shown = [n for n in names if key in d[n]["S"]]
            for c in ("S", "V", "F"):
                ax.plot(range(len(shown)), [d[n][c][key] for n in shown], "o-", color=CLASS_COLORS[c], label=c)
            ax.axhline(1.0, color=INK_2, linewidth=1, dashes=(2, 2))
            ax.set_xticks(range(len(shown)), [FITTED_BANKS[n] for n in shown], fontsize=7)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{CONDITION_NAMES[f]}: {'physics loss' if r == 0 else 'roughness'}, before and after", loc="left", fontsize=9)
            ax.legend(fontsize=7, frameon=False)
    caption = ("Top left: mean validation S F1 of three cnn runs per weight λ of the fitted physics loss (dots: the runs; star: the chosen λ; λ 0 is "
               "P12's bank). Right: the seed-0 banks' fitted physics loss and roughness (mean squared first difference) per class, before (P12, "
               "P21) and after (λ 0.1, 1, 10); real relative beats and the templates alone beside; dashed: 1, real whole beats with their own "
               "templates (loss) and the real beats' roughness. Bottom left: over "
               "how many training patients the synthetic S and F beats spread. Training beats only. Post hoc (PREREGISTRATION v21, P23).")  # fmt: skip
    return [_save(fig, paths.figures_for("phase6_physics_fitted") / "banks.png", "The weight choice and the banks, before and after", caption)]


def posthoc_figures() -> list[Path]:
    """Every post-hoc figure whose summary exists (PREREGISTRATION v4, P1-P4); nothing is re-run."""
    final, written = _summary("final_eval"), []
    if final is None:
        return written
    if (s := _summary("phase6_rr_reference")) is not None:
        written += rr_reference_figure(s, final)
    external = {db: s for db in ("incartdb", "svdb") if (s := _summary("phase6_external_validation", f"{db}_summary.json")) is not None}
    if external:
        written += external_validation_figure(external, final)
    if (paths.processed_dir() / "external").is_dir():
        written += external_data_figure()
    if (s := _summary("phase6_extra_seeds")) is not None:
        written += extra_seeds_figure(s, final)
    if (s := _summary("phase6_audit_check")) is not None:
        written += audit_check_figure(s)
    written += rr_baseline_feature_figure()
    if (s := _summary("phase6_rr_baseline")) is not None:
        written += rr_baseline_results_figure(s, final)
    v2_ext = {db: s for db in ("incartdb", "svdb") if (s := _summary("phase6_rr_baseline", f"{db}_summary.json")) is not None}
    if v2_ext and external:
        written += rr_baseline_external_figure(v2_ext, external)
    written += patient_template_input_figure()
    if (d := _summary("phase6_patient_template", "diagnostic.json")) is not None:
        written += patient_template_diagnostic_figure(d, _summary("phase6_patient_template", "adoption.json"))
    if (s := _summary("phase6_patient_template")) is not None:
        written += patient_template_results_figure(s, final, _summary("phase6_rr_baseline"))
    v3_ext = {db: s for db in ("incartdb", "svdb") if (s := _summary("phase6_patient_template", f"{db}_summary.json")) is not None}
    if v3_ext:
        written += patient_template_external_figure({"v1": external, "v2": v2_ext, "v3": v3_ext})
    if (s := _summary("phase6_relative_cnn")) is not None:
        written += relative_cnn_results_figure(s, final, _summary("phase6_rr_baseline"))
    p7_ext = {db: s for db in ("incartdb", "svdb") if (s := _summary("phase6_relative_cnn", f"{db}_summary.json")) is not None}
    if p7_ext:
        written += relative_cnn_external_figure({"v1": external, "v2": v2_ext, "v3": p7_ext})
    mech = (_summary("phase6_augmentation_mechanism", "attribution.json"), _summary("phase6_augmentation_mechanism", "gap.json"))
    if any(m is not None for m in mech):
        written += mechanism_figure(*mech)
    written += patient_transfer_example_figure()
    if (s := _summary("phase6_patient_transfer")) is not None:
        written += patient_transfer_results_figure(s, _summary("phase6_rr_baseline"))
    if (a := _summary("phase6_patient_adaptation", "adoption.json")) is not None:
        written += patient_adaptation_figure(a, _summary("phase6_patient_adaptation"))
    if (s := _summary("phase6_more_patients")) is not None and (p7 := _summary("phase6_relative_cnn")) is not None:
        written += more_patients_figure(s, p7)
    mechs = {f: m for f in ("cvae", "wgan_gp") if (m := _summary("phase6_relative_generators", f"{f}_mechanism.json")) is not None}
    if mechs:
        written += relative_generators_mechanism_figure(mechs)
    if (s := _summary("phase6_relative_generators")) is not None and (p11 := _summary("phase6_more_patients")) is not None:
        written += relative_generators_results_figure(s, p11)
    if (c := _summary("phase6_record_rr", "design_check.json")) is not None:
        written += record_rr_check_figure(c)
    if (s := _summary("phase6_record_rr")) is not None:
        written += record_rr_results_figure(s)
    if (c := _summary("phase6_icentia", "design_check.json")) is not None:
        written += icentia_check_figure(c)
    if (s := _summary("phase6_icentia")) is not None:
        written += icentia_results_figure(s)
    if (m := _summary("phase6_diffusion", "mechanism.json")) is not None:
        written += diffusion_mechanism_figure(m)
    written += diffusion_samples_figure()
    if (s := _summary("phase6_diffusion")) is not None:
        written += diffusion_results_figure(s)
    if (c := _summary("phase6_af_rule", "design_check.json")) is not None and "validation" in c:
        written += af_rule_check_figure(c)
    if (s := _summary("phase6_af_rule")) is not None:
        written += af_rule_results_figure(s)
    if (c := _summary("phase6_diffusion_long", "shape_check.json")) is not None:
        written += diffusion_long_check_figure(c)
    if (s := _summary("phase6_diffusion_long")) is not None:
        written += diffusion_long_results_figure(s)
    if (m := _summary("phase6_seed_ensemble", "members.json")) is not None:
        written += seed_ensemble_check_figure(m)
    if (s := _summary("phase6_seed_ensemble")) is not None:
        written += seed_ensemble_results_figure(s)
    if (s := _summary("phase6_fresh_patients")) is not None:
        written += fresh_patients_figure(s, _summary("phase6_seed_ensemble"))
    if (c := _summary("phase6_rule_tuning", "choice.json")) is not None:
        written += rule_tuning_choice_figure(c, _summary("phase6_rule_tuning"))
    if (s := _summary("phase6_rule_tuning")) is not None:
        written += rule_tuning_results_figure(s, _summary("phase6_rule_tuning", "test_summary.json"))
    if (ph := _summary("phase6_physics_prior", "physics.json")) is not None:
        written += physics_check_figure(ph)
    fams = {f: d for f in ("cvae", "wgan_gp") if (d := _summary("phase6_physics_prior", f"{f}.json")) is not None}
    if fams:
        written += physics_banks_figure(fams, _summary("phase6_physics_prior", "fusion.json"))
    if (s := _summary("phase6_physics_prior")) is not None:
        written += physics_results_figure(s, _summary("phase6_physics_prior", "test_summary.json"))
    if (m := _summary("phase6_bigger_classifier", "members.json")) is not None:
        from ecgaug.experiments.phase6_bigger_classifier import cnn_validation

        written += bigger_classifier_validation_figure(m, cnn_validation())
    if (s := _summary("phase6_bigger_classifier")) is not None:
        from ecgaug.experiments.phase6_bigger_classifier import PAIRS as P22_PAIRS

        written += ensemble_pairs_figure(s, _summary("phase6_bigger_classifier", "test_summary.json"), P22_PAIRS, "phase6_bigger_classifier",
                                         "Before and after: a larger classifier", BIGGER_CAPTION)  # fmt: skip
    if (c := _summary("phase6_physics_fitted", "physics.json")) is not None:
        written += physics_fitted_check_figure(c)
    fitted = {f: d for f in ("cvae", "wgan_gp") if (d := _summary("phase6_physics_fitted", f"{f}.json")) is not None}
    if fitted:
        written += physics_fitted_banks_figure(fitted)
    if (s := _summary("phase6_physics_fitted")) is not None:
        from ecgaug.experiments.phase6_physics_fitted import PAIRS as P23_PAIRS

        written += ensemble_pairs_figure(s, _summary("phase6_physics_fitted", "test_summary.json"), P23_PAIRS, "phase6_physics_fitted",
                                         "Before and after: the fitted physics loss", FITTED_CAPTION)  # fmt: skip
    return written


BIGGER_CAPTION = ("Squares: P18's cnn ensemble (ten runs) with the setting P20 chose for it; stars: phase 2's ResNet1D (ten runs, same data, banks, "
                  "seeds and budget) with the setting chosen for it on P14's patients by P20's rule. Comparisons: the ResNet against the cnn per "
                  "condition, Holm over three (secondary on P19's 977 patients, confirmatory on the test records). Post hoc (PREREGISTRATION v21, P22).")
FITTED_CAPTION = ("Squares: P18's ten-run ensemble of the generator a condition changes, with the setting P20 chose for it; stars: the generator "
                  "trained with the fitted physics loss (ten runs) with the setting chosen for it on P14's patients by P20's rule. Comparisons: "
                  "Holm over two (secondary on P19's 977 patients, confirmatory on the test records). Post hoc (PREREGISTRATION v21, P23).")


def quality_vs_gain_figure(summary: dict, rows: list[dict]) -> list[Path]:
    """RQ3 (Fig 5): each quality metric against the downstream gain, one point per generator checkpoint."""
    plt = _plt()
    metrics = list(summary["metrics"])
    y_key = "gain" if summary.get("reference") else "pilot_selection_value"
    columns = min(4, len(metrics))
    fig, axes = plt.subplots(-(-len(metrics) // columns), columns, figsize=(3.2 * columns, 2.9 * -(-len(metrics) // columns)), squeeze=False)
    for ax, metric in zip(axes.flat, metrics):
        for family, color in FAMILY_COLORS.items():
            points = [(r[metric], r[y_key]) for r in rows if r["family"] == family and np.isfinite(r.get(metric, np.nan)) and np.isfinite(r[y_key])]
            if points:
                x, y = zip(*points)
                ax.scatter(x, y, s=28, color=color, edgecolor=SURFACE, linewidth=0.8, label={"cvae": "cVAE", "wgan_gp": "WGAN-GP"}[family])
        m = summary["metrics"][metric]
        ci = f" [{m['ci_low']:+.2f}, {m['ci_high']:+.2f}]" if m["ci_low"] is not None else ""
        ax.set_title(metric.removeprefix("quality_").removesuffix("_mean").replace("_", " "), loc="left")
        ax.set_title(f"rho {m['rho']:+.2f}{ci}", loc="right", fontsize=7, fontweight="normal", color=INK_2)
        if y_key == "gain":
            ax.axhline(0, color=GRID, linewidth=1.2)
    for ax in list(axes.flat)[len(metrics):]:
        ax.set_visible(False)
    for row in axes:
        row[0].set_ylabel("downstream gain" if y_key == "gain" else "pilot score")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, bbox_to_anchor=(0.995, 1.0))
    caption = (f"{summary['n_checkpoints']} checkpoints from {summary['n_runs']} generator runs. Spearman rho with a 95% interval from "
               "resampling runs. TSTR is analysed apart. Exploratory (H3)." + (" Provisional." if summary["provisional"] else ""))  # fmt: skip
    return [_save(fig, paths.figures_for("phase4_quality_vs_gain") / "quality_vs_gain.png", "RQ3: quality metrics against the downstream gain", caption)]


def phase4_strength_figure(gate: dict) -> list[Path]:
    """Validation selection metric against strength, per condition and architecture (the strength curves)."""
    plt = _plt()
    archs = list(gate["chosen"])
    fig, axes = plt.subplots(1, len(archs), figsize=(6 * len(archs), 3.4), squeeze=False, sharey=True)
    palette = {"random_oversampling": INK_2, "smote": "#7f63b8", "cvae": CLASS_COLORS["S"], "wgan_gp": CLASS_COLORS["V"]}
    for ax, arch in zip(axes[0], archs):
        for condition, c in gate["chosen"][arch].items():
            if condition in ("none", "class_weighted"):
                continue
            pairs = sorted((float(s), v["mean"]) for s, v in c["all_strengths"].items() if s != "None")
            if pairs:
                ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", color=palette.get(condition, INK), label=condition)
        if "none" in gate["chosen"][arch]:
            ax.axhline(gate["chosen"][arch]["none"]["mean"], color=INK, dashes=(4, 2), linewidth=1, label="none")
        ax.set_title(arch, loc="left")
        ax.set_xlabel("strength")
        ax.legend(loc="best")
    axes[0][0].set_ylabel("validation selection metric (seed mean)")
    return [_save(fig, paths.figures_for("phase4_gate") / "strength_curves.png", "Phase 4 gate: validation score against augmentation strength",
                  "Each point is the mean over seeds. Dashed: no augmentation. The final evaluation uses the best strength of each condition.")]  # fmt: skip


def phase3a_gate_figure(gate: dict) -> list[Path]:
    """Both families' variants side by side: pilot score per seed, and whether the hard gates passed."""
    plt = _plt()
    families = [f for f in (PHASE3_CVAE, PHASE3_WGAN_GP) if f in gate["families"]]
    fig, axes = plt.subplots(1, len(families), figsize=(6 * len(families), 3.4), squeeze=False, sharey=True)
    for ax, family in zip(axes[0], families):
        results = gate["families"][family]["results"]
        for i, (letter, r) in enumerate(results.items()):
            passed = r["hard_gates"]["passed"]
            seeds = [s["selection_value"] for s in r["per_seed"]]
            ax.scatter([i] * len(seeds), seeds, s=22, color=CLASS_COLORS["N"], alpha=0.6, zorder=3)
            ax.errorbar(i, r["mean"], yerr=r["sd"], fmt="o", color=INK, markersize=6, capsize=3, zorder=4,
                        markerfacecolor=INK if passed else SURFACE)  # fmt: skip
            ax.annotate(f"{r['mean']:.3f}" + ("" if passed else "\nfails hard gates"), (i, r["mean"]), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=8)  # fmt: skip
        ax.set_xticks(range(len(results)), list(results))
        ax.set_xlim(-0.5, len(results) - 0.3)
        winner = gate["families"][family]["decision"]["winner"]
        ax.set_title(f"{family}: winner {winner or 'none'}", loc="left")
    axes[0][0].set_ylabel("pilot S F1 on validation")
    return [_save(fig, paths.figures_for(PHASE3A_GATE) / "variants.png", "Phase 3a gate: generator variants on the pilot",
                  "Small dots: pilot seeds. Black: mean ± SD; hollow = fails a hard gate (divergence, diversity or memorization).")]  # fmt: skip


def for_result(result_path: Path) -> list[Path]:
    """The figures for a result file, chosen by the experiment it belongs to."""
    path = Path(result_path)
    if path.parent.name == PHASE2_PTBXL_PRETRAINING:
        return phase2_pretraining_figures(path)
    if path.parent.name == PHASE1_INPUT_REPRESENTATION:
        return phase1_variant_figures(path) + phase1_gate_figure()
    if path.parent.name in (PHASE3_CVAE, PHASE3_WGAN_GP):
        return phase3_variant_figures(path)
    if path.parent.name == PHASE2_BASELINE_CLASSIFIERS:
        return phase1_variant_figures(path)
    if path.parent.name == PHASE2_QUALITY_METRICS:
        return phase2_quality_figures(path)
    if path.parent.name == "phase5_explainability":
        return phase5_saliency_figure(path)
    return []


def safe_for_result(result_path: Path) -> list[Path]:
    """Like :func:`for_result`, but a plotting error never fails an experiment; it is reported instead."""
    try:
        return for_result(result_path)
    except Exception as err:  # noqa: BLE001 (figures are a by-product; the run's results are already saved)
        print(f"WARNING: figures for {result_path} were not made: {err}")
        return []


def results_figures() -> list[Path]:
    written = []
    for experiment in EXPERIMENTS:
        for path in sorted(paths.results_for(experiment).glob("[A-Z].json")):
            written += for_result(path)
    return written


def _count(n: int | None) -> str:
    if n is None:
        return ""
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"


def architecture_flow_figure() -> list[Path]:
    """The whole system as a flow: the pre-registered system of phases 1-5 (top), the post-hoc pipeline with the best
    test score (middle) and what else was tested at each stage (bottom). Parameter counts are computed from the configs
    (``ecgaug.architecture``); choices come from the gate records and scores from the result files."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    from ecgaug import architecture as arch
    from ecgaug.experiments import phase6_record_rr as rr

    rows = {r["name"]: r for r in arch.inventory()}
    best = arch.best_pipeline()
    p = {k: _count(rows[k]["parameters"]) for k in rows}
    try:
        strength = rr.strengths()["cvae"]
    except Exception:  # noqa: BLE001 - the strength is only a label here
        strength = None
    score = best["score"]["point"] if best["score"] else None
    columns = ["Data", "Beat", "Augmentation", "Classifier", "Decision", "Explanation", "Note"]
    top = [
        "MIT-BIH Arrhythmia Database\nlead MLII, 360 Hz, paced excluded\nDS1: training + validation\ntest: the other 22 records",
        "window ±0.25 s (180 samples)\nper-beat z-score\nRR: previous, next and ratio\n(representation C)",
        "frozen banks: cVAE D, WGAN-GP A\nSMOTE · random oversampling\nclass-weighted loss\npolicy D (screened, fraction of N)",
        f"cnn: shallow CNN ({p['Shallow CNN (the cnn)']})\nrnn: BiLSTM ({p['BiLSTM (the rnn)']})\n3 seeds, early stopping\non validation S F1",
        "argmax over\nN, S, V, F",
        "Grad-CAM (cnn)\nocclusion (rnn)\nsaliency shares in the\nP, QRS, ST, T windows",
        "JSON: class, probabilities,\nlow confidence, segments\n→ LLM note (variant C)\n→ fact checker; template fallback",
    ]
    middle = [
        "+ INCART and SVDB\npatients in training\n(P11)",
        "patient-relative beat:\nbeat − patient template (P6)\nRR over the recording's\nmean (P13)",
        "cVAE D retrained on\npatient-relative beats (P12)" + (f"\nstrength {strength:g}" if strength is not None else ""),
        "the cnn × 10 seeds\nmean probability\n(P18)",
        f"Huang's SVEB stage: N → S if\nRR < {best['rule_threshold']:g} × recording mean (P13)\n+ AF veto at {best['af_threshold']:g}: logistic\nregression on RR irregularity (P16)"
        if best["rule_threshold"] is not None else "Huang's SVEB stage (P13)\n+ AF veto (P16)",
        "as above", "as above",
    ]
    bottom = [
        "Icentia11k: 400 training\npatients (P14); unseen\npatients for evaluation\n(P19, P20)",
        "long window + band-pass,\n125 Hz, no RR (phase 1)\nlocal RR ratio (P5)",
        f"cVAE A-C, WGAN-GP B-D (phase 3)\ndiffusion U-Net, {p['Diffusion (1D U-Net)']} (P15, P17)\nphysics losses (P21, P23)\npatient transfer (P9)",
        f"ResNet1D, {p['ResNet1D']} (phase 2, P22)\nCNN-BiLSTM, {p['CNN-BiLSTM']} (phase 2)\nPTB-XL pretraining (phase 2)\nper-patient batch norm (P10)\nRR discriminants (P1)",
        "stage and veto\nthresholds chosen on\nIcentia11k patients\n(P20)",
        "integrated gradients\nSmoothGrad\n(phase 5)",
        "template only (A)\nconstrained LLM (B)\nfact-check loop (D)\n(phase 5)",
    ]
    lanes = [("Pre-registered system\n(phases 1-5, chosen on validation,\nread once on the test records)", top, BLUES[1], INK),
             ("Post-hoc pipeline with the best\ntest score (P18" + (f": rare-class\nmacro-F1 {score:.3f})" if score is not None else ")"), middle, "#d7efe4", INK),
             ("Also tested at each stage\n(not in either pipeline)", bottom, "#efeeea", INK_2)]  # fmt: skip
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10.5, 12.2))
    n = len(columns)
    ax.set_xlim(0, 3.55)
    ax.set_ylim(-0.05, n + 0.78)
    ax.axis("off")
    width, height, left = 0.98, 0.8, 0.55
    x0 = [left + k * 1.0 for k in range(3)]
    y0 = [n - 1 - i + 0.1 for i in range(n)]
    for i, name in enumerate(columns):
        ax.text(0.02, y0[i] + height / 2, name, ha="left", va="center", fontsize=11, fontweight="bold", color=INK)
    for k, (title, texts, face, ink) in enumerate(lanes):
        ax.text(x0[k] + width / 2, n + 0.05, title, ha="center", va="bottom", fontsize=10, fontweight="bold", color=ink, linespacing=1.3)
        for i, text in enumerate(texts):
            dashed = text == "as above"
            ax.add_patch(FancyBboxPatch((x0[k], y0[i]), width, height, boxstyle="round,pad=0.0,rounding_size=0.05", linewidth=1.0,
                                        edgecolor=GRID if dashed else INK_2, facecolor=SURFACE if dashed else face, linestyle="--" if dashed else "-"))  # fmt: skip
            ax.text(x0[k] + width / 2, y0[i] + height / 2, text, ha="center", va="center", fontsize=9 if text.count("\n") < 4 else 8.6,
                    color=INK_2 if dashed else ink, linespacing=1.3)  # fmt: skip
            if k == 2 or i == n - 1 or dashed or texts[i + 1] == "as above":
                continue
            ax.add_patch(FancyArrowPatch((x0[k] + width / 2, y0[i]), (x0[k] + width / 2, y0[i + 1] + height), arrowstyle="-|>", mutation_scale=12,
                                         color=INK, linewidth=1.2))  # fmt: skip
    # the post-hoc decision feeds the same explanation and note
    i = columns.index("Decision")
    ax.add_patch(FancyArrowPatch((x0[1] + width / 2, y0[i]), (x0[0] + width, y0[i + 1] + height * 0.75), arrowstyle="-|>", mutation_scale=12,
                                 color=TUNED_COLOR, linewidth=1.3, connectionstyle="arc3,rad=-0.2"))  # fmt: skip
    caption = ("Left: the system as pre-registered and frozen at the phase gates (phases 1-5); every choice was made on validation patients. Middle: "
               "the post-hoc changes that give the highest test score so far (P18), each tested under its own pre-registration; its predictions "
               "feed the same saliency and note (green arrow). Right: the other models and settings tested at each stage. Parameter counts are trainable "
               "parameters with the RR side input, computed from the configs. P-numbers are the post-hoc analyses (Section on post-hoc analyses).")  # fmt: skip
    return [_save(fig, paths.figures_for("architecture") / "flow.png", "Architecture: the full flow and every model tested", caption)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Before-and-after figures for each phase.")
    parser.add_argument("what", choices=["phase1_data", "phase2_ptbxl_data", "results", "final_eval", "leakage_audit", "posthoc", "architecture", "all"])
    parser.add_argument("--ptbxl-root", default=None, help="folder with PTB-XL recordings (default data/raw/ptbxl)")
    args = parser.parse_args(argv)
    written = []
    if args.what in ("phase1_data", "all"):
        written += phase1_data_figures()
    if args.what in ("phase2_ptbxl_data", "all"):
        written += phase2_ptbxl_data_figures(args.ptbxl_root)
    if args.what in ("results", "all"):
        written += results_figures()
    if args.what == "final_eval":  # redraw from the saved summary; the test set is not read
        written += final_eval_figures(json.loads((paths.results_for("final_eval") / "summary.json").read_text()))
    if args.what == "leakage_audit":  # from the audit's summary; nothing is re-run
        written += leakage_audit_figure(json.loads((paths.results_for("phase4_leakage_audit") / "summary.json").read_text()))
    if args.what == "posthoc":  # from the saved post-hoc summaries; nothing is re-run
        written += posthoc_figures()
    if args.what in ("architecture", "all"):  # from the configs and result files; nothing is trained
        written += architecture_flow_figure()
    for path in written:
        print(path.relative_to(paths.repo_root()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
