"""Phase 3 experiment phase3_wgan_gp: conditional WGAN-GP variants A-D (PLAN Phase 3).

Usage:
    python -m ecgaug.experiments.phase3_wgan_gp --config configs/phase3_wgan_gp/A_concat_embedding.yaml            # pilot (seeds 0, 1)
    python -m ecgaug.experiments.phase3_wgan_gp --config configs/phase3_wgan_gp/A_concat_embedding.yaml --final    # full budget and banks
    python -m ecgaug.experiments.phase3_wgan_gp --config configs/phase3_wgan_gp/A_concat_embedding.yaml --smoke    # fake data, end to end

Add ``--preview`` to run before the phase 1 and phase 2 gates. The runner itself is shared with
phase3_cvae: see ``ecgaug.experiments.generator_runs``; the generator is ``ecgaug.generators.wgan_gp``.
"""

from __future__ import annotations

from ecgaug.experiments import generator_runs
from ecgaug.names import PHASE3_WGAN_GP

EXPERIMENT = PHASE3_WGAN_GP


def main(argv: list[str] | None = None) -> int:
    return generator_runs.main(EXPERIMENT, argv)


if __name__ == "__main__":
    raise SystemExit(main())
