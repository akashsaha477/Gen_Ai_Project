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
PHASE2_BASELINE_CLASSIFIERS = "phase2_baseline_classifiers"
PHASE2_QUALITY_METRICS = "phase2_quality_metrics"
PHASE2_PTBXL_PRETRAINING = "phase2_ptbxl_pretraining"
PHASE3_CVAE = "phase3_cvae"
PHASE3_WGAN_GP = "phase3_wgan_gp"
PHASE4_AUGMENTATION_POLICY = "phase4_augmentation_policy"
PHASE4_LEAKAGE_AUDIT = "phase4_leakage_audit"  # optional, after the final evaluation (PLAN Phase 4, DECISIONS A4)
PHASE5_EXPLAINABILITY = "phase5_explainability"
PHASE5_CLINICAL_NOTES = "phase5_clinical_notes"
EXPERIMENTS: tuple[str, ...] = (
    PHASE1_INPUT_REPRESENTATION, PHASE2_BASELINE_CLASSIFIERS, PHASE2_QUALITY_METRICS, PHASE2_PTBXL_PRETRAINING,
    PHASE3_CVAE, PHASE3_WGAN_GP, PHASE4_AUGMENTATION_POLICY, PHASE4_LEAKAGE_AUDIT, PHASE5_EXPLAINABILITY, PHASE5_CLINICAL_NOTES,
)  # fmt: skip

# Folders for data reports and gate decisions, named the same way.
PHASE1_DATA = "phase1_data"
PHASE1_GATE = "phase1_gate"
PHASE2_PTBXL_DATA = "phase2_ptbxl_data"
PHASE2_GATE = "phase2_gate"
PHASE3A_GATE = "phase3a_gate"  # phase 3 has two gates: 3a chooses one generator per family (Day 12) ...
PHASE3_GATE = "phase3_gate"  # ... and 3 freezes the synthetic banks of the chosen generators (Day 14)
PHASE4A_GATE = "phase4a_gate"  # phase 4 has two gates too: 4a chooses the augmentation policy (Day 17) ...
PHASE4_FULL_GRID = "phase4_full_grid"  # the grid of conditions x strengths x architectures x seeds
PHASE4_GATE = "phase4_gate"  # ... and 4 fixes the strengths and the checkpoints the final evaluation tests
FINAL_EVAL = "final_eval"  # the single test-set evaluation, after the phase 4 gate
PHASE4_QUALITY_VS_GAIN = "phase4_quality_vs_gain"  # RQ3: quality metrics against the downstream gain, from the phase 3 checkpoints
PHASE5_GATE = "phase5_gate"  # explanation method per architecture, note variant, demo: then the code freeze

# Post-hoc analyses pre-registered after the final evaluation (PREREGISTRATION §10 v4, DECISIONS A24).
PHASE6_RR_REFERENCE = "phase6_rr_reference"  # fixed RR-feature reference classifiers
PHASE6_EXTERNAL_VALIDATION = "phase6_external_validation"  # frozen models on INCART and SVDB
PHASE6_EXTRA_SEEDS = "phase6_extra_seeds"  # seeds 3 and 4 of the final evaluation's conditions
PHASE6_AUDIT_CHECK = "phase6_audit_check"  # the leakage audit at the chosen WGAN-GP strength and on more splits
PHASE6_RR_BASELINE = "phase6_rr_baseline"  # the full grid again with a patient-baseline RR ratio (v5)
PHASE6_PATIENT_TEMPLATE = "phase6_patient_template"  # the full grid again with patient-relative beats (v6)
PHASE6_RELATIVE_CNN = "phase6_relative_cnn"  # the cnn alone with patient-relative beats, full comparison (v7)
PHASE6_AUGMENTATION_MECHANISM = "phase6_augmentation_mechanism"  # why augmentation does not carry over to new patients (v7)
PHASE6_PATIENT_TRANSFER = "phase6_patient_transfer"  # rare-class beats moved onto other patients' templates (v8)
PHASE6_PATIENT_ADAPTATION = "phase6_patient_adaptation"  # per-patient batch-normalization statistics at test time (v9)
PHASE6_MORE_PATIENTS = "phase6_more_patients"  # the P7 cnn trained on DS1 plus the INCART and SVDB patients (v10)
PHASE6_RELATIVE_GENERATORS = "phase6_relative_generators"  # the generators trained on patient-relative beats of P11's patients (v11)
PHASE6_RECORD_RR = "phase6_record_rr"  # RR intervals over the mean RR of their recording, P11's and P12's conditions (v12)
PHASE6_ICENTIA = "phase6_icentia"  # the P13 cnn trained on many more patients, from Icentia11k (v13)
PHASE6_DIFFUSION = "phase6_diffusion"  # a class-conditional diffusion model as the generator, on P13's data (v14)
PHASE6_AF_RULE = "phase6_af_rule"  # Huang's SVEB stage with atrial-fibrillation beats vetoed (v15)
POST_HOC: tuple[str, ...] = (PHASE6_RR_REFERENCE, PHASE6_EXTERNAL_VALIDATION, PHASE6_EXTRA_SEEDS, PHASE6_AUDIT_CHECK, PHASE6_RR_BASELINE,
                             PHASE6_PATIENT_TEMPLATE, PHASE6_RELATIVE_CNN, PHASE6_AUGMENTATION_MECHANISM,
                             PHASE6_PATIENT_TRANSFER, PHASE6_PATIENT_ADAPTATION, PHASE6_MORE_PATIENTS, PHASE6_RELATIVE_GENERATORS,
                             PHASE6_RECORD_RR, PHASE6_ICENTIA, PHASE6_DIFFUSION,
                             PHASE6_AF_RULE)

# Result folders that travel back from Kaggle: every experiment's, plus the full grid's ledger.
RESULT_FOLDERS: tuple[str, ...] = (*EXPERIMENTS, PHASE4_FULL_GRID, *POST_HOC)


def phase_of(name: str) -> int:
    """The phase number in a phase<N>_<what> name."""
    if not name.startswith("phase") or not name[5].isdigit():
        raise ValueError(f"{name!r} does not follow the phase<N>_<what> naming")
    return int(name[5])
