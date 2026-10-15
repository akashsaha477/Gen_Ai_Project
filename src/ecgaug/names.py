"""The project's one naming scheme (explained for people in docs/NAMING.md).

    Phase 1-6        the six parts of the plan, done in order
    experiment       one comparison inside a phase, named phase<N>_<what>
    variant A-D      one option inside an experiment
    run              one training of one variant with one seed and one learning rate
    phase N gate     the decision at the end of phase N

An experiment's name is used for everything that belongs to it: its config folder, its runner
module, its run folder, its result folder, its figure folder and its Kaggle notebook.
"""

from __future__ import annotations

PHASES: dict[int, str] = {
    1: "data",
    2: "baselines",
    3: "generators",
    4: "comparison",
    5: "explanations",
    6: "paper",
}

# Experiments that exist (later phases add theirs here).
PHASE1_INPUT_REPRESENTATION = "phase1_input_representation"
PHASE2_PTBXL_PRETRAINING = "phase2_ptbxl_pretraining"
EXPERIMENTS: tuple[str, ...] = (PHASE1_INPUT_REPRESENTATION, PHASE2_PTBXL_PRETRAINING)

# Folders for data reports and gate decisions, named the same way.
PHASE1_DATA = "phase1_data"
PHASE1_GATE = "phase1_gate"
PHASE2_PTBXL_DATA = "phase2_ptbxl_data"


def phase_of(name: str) -> int:
    """The phase number in a phase<N>_<what> name."""
    if not name.startswith("phase") or not name[5].isdigit():
        raise ValueError(f"{name!r} does not follow the phase<N>_<what> naming")
    return int(name[5])
