"""Policy pilot against the run without augmentation, per bank and seed (phase 4a gate figure).

    python scripts/phase4a_policy_figure.py   # writes figures/phase4a_gate/policies.png

Reads only the validation results in results/phase4_augmentation_policy/.
"""

from __future__ import annotations

import json
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecgaug import paths

LETTERS = "ABCD"
BANKS = {"cvae": ("cVAE D", "C0", -0.12), "wgan_gp": ("WGAN-GP A", "C1", 0.12)}
METRICS = {"S F1 (selection)": "selection", "V F1": "tie_break", "N recall": "n_recall"}


def per_seed(cell: dict, key: str) -> list[float]:
    """Per-seed values of one bank's cell; V F1 and N recall are stored only as the seed mean."""
    if key == "selection":
        return cell["per_seed"]
    if key == "tie_break":
        return [cell["tie_break_value"]]
    return [cell["n_recall"]]


def main() -> int:
    folder = paths.results_for("phase4_augmentation_policy")
    results = {v: json.loads((folder / f"{v}.json").read_text()) for v in LETTERS}
    ref = results["A"]["reference"]
    ref_values = {"selection": ref["selection_value"], "tie_break": ref["tie_break_value"], "n_recall": ref["n_recall"]}
    fig, axes = plt.subplots(1, len(METRICS), figsize=(13, 3.8))
    for ax, (title, key) in zip(axes, METRICS.items()):
        ax.axhline(ref_values[key], color="0.4", ls="--", lw=1, label="no augmentation (mean of 3 seeds)")
        for i, letter in enumerate(LETTERS):
            for family, (name, color, dx) in BANKS.items():
                values = per_seed(results[letter]["per_bank"][family], key)
                ax.scatter(np.full(len(values), i + dx), values, color=color, s=22, alpha=0.8, label=name if i == 0 else None)
                ax.plot([i + dx - 0.08, i + dx + 0.08], [np.mean(values)] * 2, color=color, lw=2)
        ax.set_xticks(range(len(LETTERS)), [f"{v}\n" + textwrap.fill(results[v]["policy"]["kind"].replace("_", " "), 12) for v in LETTERS], fontsize=8)
        ax.set_title(title, fontsize=10)
    axes[2].axhline(ref_values["n_recall"] - results["A"]["constraint"]["max_n_recall_drop"], color="C3", lw=1, label="N recall limit")
    axes[0].set_ylabel("validation")
    handles, labels = [], []
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Policy pilot: dots = seeds (S F1) or the seed mean (V F1, N recall), bars = means; dashed = no augmentation", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = paths.repo_root() / "figures" / "phase4a_gate" / "policies.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
