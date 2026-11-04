"""Before-and-after figures for every phase, so the team can see how the data and models change.

    python -m ecgaug.figures phase1_data          # MIT-BIH: raw signal, filtering, the four inputs, class counts
    python -m ecgaug.figures phase2_ptbxl_data    # PTB-XL: resampling and detected beats, exclusions, labels
    python -m ecgaug.figures results              # every experiment result: curves, confusion matrices, the phase 1 gate
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Before-and-after figures for each phase.")
    parser.add_argument("what", choices=["phase1_data", "phase2_ptbxl_data", "results", "final_eval", "all"])
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
    for path in written:
        print(path.relative_to(paths.repo_root()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
