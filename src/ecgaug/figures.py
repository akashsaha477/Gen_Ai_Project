"""Before-and-after figures for every stage, so the team can see how the data and models change.

    python -m ecgaug.figures phase1        # MIT-BIH: raw signal, filtering, the four inputs, class counts
    python -m ecgaug.figures ptbxl         # PTB-XL: resampling and detected beats, exclusions, labels
    python -m ecgaug.figures results       # every result file: training curves, confusion matrices, Gate 1
    python -m ecgaug.figures all

Figures are written to ``figures/<stage>/`` as PNG files and are committed. The runners also call
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

# Light reference palette (validated: adjacent CVD ΔE >= 9.1; aqua and yellow need direct labels).
CLASS_COLORS = {"N": "#2a78d6", "S": "#eb6834", "V": "#1baf7a", "F": "#eda100"}
ARM_COLORS = {"scratch": "#2a78d6", "finetune": "#eb6834", "pretrain": "#1baf7a"}
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


def figures_dir(*parts: str) -> Path:
    return paths.repo_root().joinpath("figures", *parts)


# --------------------------------------------------------------------------------------------------
# Phase 1: MIT-BIH data, before and after each preprocessing step
# --------------------------------------------------------------------------------------------------


def _m1_specs() -> dict:
    from ecgaug.data.representation import RepresentationSpec
    from ecgaug.experiments.input_representation import load_config
    from ecgaug.workflow import variant_configs

    return {letter: (path.stem, RepresentationSpec(**load_config(path)["representation"])) for letter, path in variant_configs().items()}


def _train_records() -> tuple[int, ...]:
    from ecgaug.data.splits import load_approved_split

    return load_approved_split().train


def phase1_figures() -> list[Path]:
    """Raw versus filtered signal, the four inputs for the same beats, class counts, class averages."""
    import pandas as pd

    from ecgaug.data import raw
    from ecgaug.data.beats import extract_beats
    from ecgaug.data.contract import build_arrays
    from ecgaug.data.records import CLASSES, FS
    from ecgaug.data.representation import bandpass, transform_record

    plt = _plt()
    out, written = figures_dir("phase1"), []
    specs = _m1_specs()
    train = _train_records()
    counts = pd.read_csv(paths.results_dir() / "data" / "record_counts.csv")
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
    all_counts = pd.read_csv(paths.results_dir() / "data" / "record_counts.csv")
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
# PTB-XL preprocessing
# --------------------------------------------------------------------------------------------------


def ptbxl_figures(root: Path | None = None) -> list[Path]:
    """Raw 500 Hz versus resampled signal with detected beats; exclusions; labels; shape versus MIT-BIH."""
    from ecgaug.data import ptbxl, raw
    from ecgaug.data.representation import RepresentationSpec

    plt = _plt()
    out, written = figures_dir("ptbxl"), []
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

    report_path = paths.results_dir() / "data" / "ptbxl_preprocessing.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        # 3. what was kept at each stage
        stages = [("in metadata", report["in_metadata"]), ("eligible (labels, no pacemaker, electrodes ok)", report["eligible"]),
                  ("kept after signal checks", report["recordings_kept"])]  # fmt: skip
        reasons = {**report["excluded_by_metadata"], **report["excluded_by_signal"]}
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.4), gridspec_kw={"width_ratios": [1.1, 1]})
        labels, values = zip(*stages)
        bars = axes[0].barh(labels[::-1], values[::-1], color="#2a78d6", height=0.5)
        for bar, v in zip(bars, values[::-1]):
            axes[0].annotate(f"{v:,}", (v, bar.get_y() + bar.get_height() / 2), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
        axes[0].set_title("recordings kept at each stage", loc="left")
        axes[0].set_xlabel("recordings")
        names, counts = zip(*sorted(reasons.items(), key=lambda kv: kv[1]))
        bars = axes[1].barh([n.replace("_", " ") for n in names], counts, color="#e34948", height=0.5)
        for bar, v in zip(bars, counts):
            axes[1].annotate(f"{v:,}", (v, bar.get_y() + bar.get_height() / 2), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
        axes[1].set_title("recordings excluded, by reason", loc="left")
        axes[1].set_xlabel("recordings")
        written.append(_save(fig, out / "03_recordings_kept_and_excluded.png",
                             f"PTB-XL preprocessing: {report['recordings_kept']:,} recordings and {report['beats_kept']:,} beats kept",
                             "From results/data/ptbxl_preprocessing.json (the full run on Kaggle)."))  # fmt: skip

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
# Results: training curves, confusion matrices, Gate 1
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


def pretraining_figures(result_path: Path) -> list[Path]:
    """Figures for one pretraining-arm result: curves, confusion matrices, per-class F1 before and after."""
    plt = _plt()
    result = json.loads(Path(result_path).read_text())
    letter = result["variant"]
    out, written = figures_dir("ptbxl_pretraining", letter), []
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
    written.append(_save(fig, out / "01_pretraining_curves.png", f"Pretraining on PTB-XL, variant {letter}{tag}",
                         f"One line per seed. Best macro AUROC {result['pretraining']['val_macro_auroc_mean']:.3f}."))  # fmt: skip

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for arm, kind in (("scratch", "scratch"), ("pretrained", "finetune")):
        _curves(axes[0], axes[1], kind, result["arms"][arm]["run_ids"], ARM_COLORS[kind], metric)
    axes[1].legend(loc="upper right")
    written.append(_save(fig, out / "02_finetune_vs_scratch_curves.png", f"MIT-BIH training from scratch versus from pretrained weights, variant {letter}{tag}",
                         "One line per seed at the chosen learning rate. Blue: from scratch. Orange: fine-tuned from PTB-XL."))  # fmt: skip

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    _confusion_panel(axes[0], _confusion(result["arms"]["scratch"]["run_ids"]), "from scratch")
    _confusion_panel(axes[1], _confusion(result["arms"]["pretrained"]["run_ids"]), "fine-tuned from PTB-XL")
    written.append(_save(fig, out / "03_confusion_matrices.png", f"Validation confusion matrices, variant {letter}{tag}",
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
    written.append(_save(fig, out / "04_f1_per_class_before_after_pretraining.png", f"F1 per class: from scratch versus fine-tuned, variant {letter}{tag}",
                         f"Mean ± SD over seeds. Winner by the selection rule: {result['winner']}."))  # fmt: skip
    return written


def module1_figures(result_path: Path) -> list[Path]:
    """Figures for one Module 1 variant: training curves and the confusion matrix."""
    plt = _plt()
    result = json.loads(Path(result_path).read_text())
    letter = result["variant"]
    out, written = figures_dir("phase1", f"variant_{letter}"), []
    metric = "S F1" if result["selection_metric"] == "s_f1" else result["selection_metric"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    _curves(axes[0], axes[1], f"variant {letter}", result["run_ids"], CLASS_COLORS["N"], metric)
    _confusion_panel(axes[2], _confusion(result["run_ids"]), "validation confusion (mean over seeds)")
    written.append(_save(fig, out / "training_and_confusion.png", f"Module 1 variant {letter}: {metric} {result['mean']:.3f} ± {result['sd']:.3f}",
                         f"One line per seed at the chosen learning rate ({result['chosen_lr']:g})."))  # fmt: skip
    return written


def gate1_figure() -> list[Path]:
    """All four Module 1 variants side by side: the evidence Gate 1 decides on."""
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
    return [_save(fig, figures_dir("gates", "gate1_variants.png"), "Gate 1 evidence: Module 1 variants on the validation split",
                  "Small dots: seeds. Black: mean ± SD. The rule picks the best mean; ties within the margin go to fewer inputs.")]  # fmt: skip


def for_result(result_path: Path) -> list[Path]:
    """The figures for a result file, chosen by the experiment it belongs to."""
    path = Path(result_path)
    if path.parent.name == "ptbxl_pretraining":
        return pretraining_figures(path)
    if path.parent.name == "m1_input_representation":
        return module1_figures(path) + gate1_figure()
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
    for path in sorted((paths.results_dir() / "validation").glob("*/[A-Z].json")):
        written += for_result(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Before-and-after figures for each stage.")
    parser.add_argument("what", choices=["phase1", "ptbxl", "results", "all"])
    parser.add_argument("--ptbxl-root", default=None, help="folder with PTB-XL recordings (default data/raw/ptbxl)")
    args = parser.parse_args(argv)
    written = []
    if args.what in ("phase1", "all"):
        written += phase1_figures()
    if args.what in ("ptbxl", "all"):
        written += ptbxl_figures(args.ptbxl_root)
    if args.what in ("results", "all"):
        written += results_figures()
    for path in written:
        print(path.relative_to(paths.repo_root()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
