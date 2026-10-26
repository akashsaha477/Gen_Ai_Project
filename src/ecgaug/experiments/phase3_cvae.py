"""Phase 3 experiment phase3_cvae: conditional VAE variants A-D (PLAN Phase 3).

Usage:
    python -m ecgaug.experiments.phase3_cvae --config configs/phase3_cvae/A_conv_concat.yaml            # pilot (seeds 0, 1)
    python -m ecgaug.experiments.phase3_cvae --config configs/phase3_cvae/A_conv_concat.yaml --final    # full budget and banks
    python -m ecgaug.experiments.phase3_cvae --config configs/phase3_cvae/A_conv_concat.yaml --smoke    # fake data, end to end

Add ``--preview`` to run before the phase 1 and phase 2 gates. The runner itself is shared with
phase3_wgan_gp: see ``ecgaug.experiments.generator_runs``; the generator is ``ecgaug.generators.cvae``.
"""

from __future__ import annotations

from ecgaug.experiments import generator_runs
from ecgaug.names import PHASE3_CVAE

EXPERIMENT = PHASE3_CVAE


def main(argv: list[str] | None = None) -> int:
    return generator_runs.main(EXPERIMENT, argv)


if __name__ == "__main__":
    raise SystemExit(main())
