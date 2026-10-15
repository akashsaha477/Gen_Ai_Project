# Naming

The project uses one naming scheme everywhere: in folders, files, code, run names, documents and
conversation. Nothing else is used: no "stage", "step", "module" or "arm".

## The four levels

| Level | What it is | How it is named | Example |
|---|---|---|---|
| **Phase** | one of the six parts of the plan, done in order | `phase<N>` plus a one-word name | phase 1 (data), phase 2 (baselines) |
| **Experiment** | one comparison inside a phase | `phase<N>_<what>` | `phase1_input_representation` |
| **Variant** | one option inside an experiment | a letter, plus a short description in file names | A, `A_short_window` |
| **Run** | one training of one variant with one seed and one learning rate | `<experiment>-<variant>.<lr>-s<seed>-<commit>-<time>` | `phase1_input_representation-A.lr1e-03-s0-32b9464-202610151200` |

A **gate** is the decision at the end of a phase: "the phase 1 gate". It is approved by the team.

## The phases

| Phase | Name | Experiments |
|---|---|---|
| 1 | data | data pipeline, evaluation code, `phase1_input_representation` |
| 2 | baselines | `phase2_baseline_classifiers`, `phase2_quality_metrics`, `phase2_ptbxl_pretraining` |
| 3 | generators | `phase3_cvae`, `phase3_wgan_gp` |
| 4 | comparison | `phase4_augmentation_policy`, the full grid (`phase4_full_grid`), the final test evaluation |
| 5 | explanations | `phase5_explainability`, `phase5_clinical_notes`, the demo |
| 6 | paper | report, paper draft |

## One name, every place

An experiment's name is used for everything that belongs to it:

| What | Where |
|---|---|
| settings | `configs/phase1_input_representation/A_short_window.yaml` |
| code that runs it | `src/ecgaug/experiments/phase1_input_representation.py` |
| Kaggle notebook | `kaggle/phase1_input_representation.ipynb` |
| run folders | `runs/phase1_input_representation/A_short_window/<run>/` |
| results | `results/phase1_input_representation/A.json` |
| figures | `figures/phase1_input_representation/A/` |
| one-page brief for the person running a variant | `docs/briefs/phase1_input_representation_A_short_window.md` |

Folders that belong to a whole phase rather than one experiment use the same pattern:
`results/phase1_data/` and `figures/phase1_data/` (MIT-BIH data reports), `results/phase1_gate/` and
`figures/phase1_gate/` (the gate decision), `results/phase2_ptbxl_data/` and `figures/phase2_ptbxl_data/`
(PTB-XL data reports).

## Inside phase2_ptbxl_pretraining

Each variant is trained three ways, and its runs are kept in three folders:

| Folder | Runs |
|---|---|
| `pretrain/` | pretraining on PTB-XL |
| `scratch/` | MIT-BIH from scratch |
| `finetune/` | MIT-BIH starting from the pretrained weights |

## Inside the phase 3 experiments

`phase3_cvae` and `phase3_wgan_gp` keep two kinds of run per variant:

| Folder | Runs |
|---|---|
| `pilot/` | the pilot budget on seeds 0 and 1 (what the phase 3a gate compares) |
| `final/` | the chosen variant at the full budget on the grid seeds; each exports a synthetic bank |

A bank is named after its variant: `data/synthetic/phase3_cvae-A/` (with the setting for cVAE B:
`phase3_cvae-B.beta0.25`). A run ID can carry the setting or `final` as its tag:
`phase3_cvae-B.beta0.1-s0-<commit>-<time>`, `phase3_cvae-A.final-s2-<commit>-<time>`.

## Gates and folders that belong to a phase

| Name | What it is |
|---|---|
| `phase1_gate` | the input representation; freezes the dataset |
| `phase2_gate` | the `cnn` and the `rnn`, the quality composite |
| `phase3a_gate` | one generator per family, from the pilots (Day 12) |
| `phase3_gate` | the synthetic banks, frozen (Day 14) |
| `phase4a_gate` | the augmentation policy; writes the full grid file |
| `phase4_full_grid` | the grid of conditions x strengths x architectures x seeds: configs, runs (`runs/phase4_full_grid/<grid>/<architecture>/`), one ledger file per cell |
| `phase4_gate` | strengths and checkpoints for the final evaluation |
| `final_eval` | the single test evaluation (`results/final_eval/`, `figures/final_eval/`) |
| `phase5_gate` | saliency method per architecture, note variant, demo |
| `error_analysis` | confusions, per-record errors and RR context, from saved predictions |

Each gate has `results/<gate>/report.md` and, once approved, an entry of the same name in
`results/manifest.json` and in `docs/DECISIONS.md`. Grid runs are named by architecture and condition:
`phase4_full_grid-cnn.smote_0.25-s1-<commit>-<time>`.

The paper's generated tables and numbers are in `paper/generated/`, written by `python -m ecgaug.paper`.

## Adding something new

Name it `phase<N>_<what>` after the phase it belongs to, add it to `src/ecgaug/names.py`, and use that
name for its config folder, runner, notebook, results and figures.
