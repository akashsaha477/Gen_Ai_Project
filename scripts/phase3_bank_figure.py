"""Real training beats next to the frozen synthetic banks, per rare class (phase 3 gate figure).

    python scripts/phase3_bank_figure.py [--seed 0]   # writes figures/phase3_gate/banks.png

Reads training beats only (never validation or test) and bank seed ``--seed`` of each family.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecgaug import paths
from ecgaug.data.loaders import load_candidate

REPRESENTATION = "C_short_window_rr_features"
BANKS = {"cVAE D": "phase3_cvae-D", "WGAN-GP A": "phase3_wgan_gp-A"}
CLASSES = {1: "S", 2: "V", 3: "F"}
N_EXAMPLES = 20


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0, help="bank seed to show")
    args = parser.parse_args()
    data = load_candidate(REPRESENTATION)
    train = data["split"] == "train"
    sources = {"real (training records)": (data["X"][train], data["y"][train])}
    for name, generator_id in BANKS.items():
        with np.load(paths.repo_root() / "data" / "synthetic" / generator_id / f"seed{args.seed}.npz") as bank:
            sources[f"{name}, bank seed {args.seed}"] = (bank["X"], bank["y"])
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(CLASSES), len(sources), figsize=(4 * len(sources), 2.6 * len(CLASSES)), sharex=True, sharey="row")
    for row, (label, cls) in enumerate(CLASSES.items()):
        for col, (title, (X, y)) in enumerate(sources.items()):
            ax = axes[row, col]
            beats = X[y == label]
            t = np.arange(beats.shape[1])
            for i in rng.choice(len(beats), size=min(N_EXAMPLES, len(beats)), replace=False):
                ax.plot(t, beats[i], color="0.75", lw=0.6)
            mean, sd = beats.mean(axis=0), beats.std(axis=0)
            ax.fill_between(t, mean - sd, mean + sd, color="C0", alpha=0.2, lw=0)
            ax.plot(t, mean, color="C0", lw=2)
            ax.set_title(f"{cls}: {title} (n = {len(beats):,})", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{cls} beat (z-score)")
            if row == len(CLASSES) - 1:
                ax.set_xlabel("sample (360 Hz, R peak at the centre)")
    fig.suptitle("Real training beats vs frozen synthetic banks: mean ± SD (blue) and 20 random beats (grey)", fontsize=11)
    fig.tight_layout()
    out = paths.repo_root() / "figures" / "phase3_gate" / "banks.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
