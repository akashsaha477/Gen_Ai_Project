"""Every condition of the full grid at its chosen strength against no augmentation (phase 4 gate figure).

    python scripts/phase4_gate_figure.py   # writes figures/phase4_gate/conditions.png

Reads only validation results (the grid ledger through ``ecgaug.workflow.evaluate_phase4_gate``).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecgaug import paths, workflow
from ecgaug.grid import runner as grid_runner

METRICS = {"S F1 (selection)": "values", "V F1": "tie_break_values", "N recall": "n_recall"}
SHORT = {"none": "none", "class_weighted": "class wt.", "random_oversampling": "ROS", "smote": "SMOTE", "cvae": "cVAE", "wgan_gp": "WGAN-GP"}


def main() -> int:
    gate = workflow.evaluate_phase4_gate()
    summary = grid_runner.summarize(gate["grid"])
    archs = list(gate["chosen"])
    fig, axes = plt.subplots(len(archs), len(METRICS), figsize=(13, 3.6 * len(archs)), squeeze=False)
    for row, arch in enumerate(archs):
        chosen = gate["chosen"][arch]
        conditions = list(chosen)
        for col, (title, key) in enumerate(METRICS.items()):
            ax = axes[row, col]
            for i, condition in enumerate(conditions):
                cell = summary[(condition, chosen[condition]["strength"], arch)]
                values = cell[key] if key != "n_recall" else [cell["n_recall"]]
                color = "0.4" if condition == "none" else ("C1" if condition == gate["best_generative"].get(arch) else "C0")
                ax.scatter(np.full(len(values), i), values, color=color, s=22, alpha=0.8)
                ax.plot([i - 0.2, i + 0.2], [np.mean(values)] * 2, color=color, lw=2)
            none = summary[("none", chosen["none"]["strength"], arch)]
            ax.axhline(np.mean(none[key]) if key != "n_recall" else none["n_recall"], color="0.4", ls="--", lw=1)
            labels = [f"{SHORT.get(c, c)}\n{chosen[c]['strength'] if chosen[c]['strength'] is not None else ''}" for c in conditions]
            ax.set_xticks(range(len(conditions)), labels, fontsize=8)
            ax.set_title(f"{arch}: {title}", fontsize=10)
        axes[row, 0].set_ylabel("validation")
    fig.suptitle("Full grid at the chosen strengths: dots = seeds (S, V F1) or the seed mean (N recall), bars = means; "
                 "dashed = no augmentation; orange = best generative", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = paths.repo_root() / "figures" / "phase4_gate" / "conditions.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
