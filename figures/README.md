# figures

Before-and-after figures for every phase. They are made by code (`src/ecgaug/figures.py`): the
runners draw them at the end of every experiment, the phase 1 gate draws its comparison, and
`python -m ecgaug.figures all` redraws everything from what is on disk. Only DS1 and PTB-XL are
plotted; the test set (DS2) never is.

| Folder | Phase and experiment | What it shows |
|---|---|---|
| `phase1_data/` | phase 1: MIT-BIH data before and after preprocessing | raw versus band-passed signal; the same N and S beat in variants A–D; beats per class in training and validation; median beat per class and variant |
| `phase1_input_representation/<letter>/` | phase 1: each variant of phase1_input_representation | training loss and validation score per epoch; validation confusion matrix |
| `phase1_gate/` | phase 1: the gate | `variants.png`: the four variants side by side, seeds and mean ± SD; `per_class.png`: F1 and PR-AUC per class; `early_stopping.png`: epoch kept vs epochs run for every run |
| `phase2_ptbxl_data/` | phase 2: PTB-XL data before and after preprocessing | raw 500 Hz versus resampled signal with detected beats; PTB-XL versus MIT-BIH beat shape; recordings kept and excluded; label shares per split |
| `phase2_ptbxl_pretraining/<letter>/` | phase 2: each variant of phase2_ptbxl_pretraining | pretraining curves; from scratch versus fine-tuned curves; confusion matrices; F1 per class before and after pretraining |

Every figure has a title and a one-line caption saying what to read from it.

Folder names match `results/` and follow `docs/NAMING.md`.
| `phase2_baseline_classifiers/<letter>/` | phase 2: baseline classifiers | training curves and validation confusion matrix per variant |
| `phase2_quality_metrics/<letter>/` | phase 2: quality metrics | `sanity.png`: score against the severity of each degradation |
| `phase2_gate/` | phase 2: the gate | `baselines.png`: the four classifiers by role |
| `phase3_cvae/<letter>/`, `phase3_wgan_gp/<letter>/` | phase 3: generator pilots | `01_generator_training.png` (losses), `02_checkpoints.png` (every checkpoint scored), and for the first seed `03_real_vs_synthetic.png`, `04_class_counts_before_after.png`, `05_synthetic_and_nearest_real.png` |
| `phase3_cvae/final/<bank>/seed<s>/` | phase 3: final banks | the same before-and-after figures for each exported bank |
| `phase3a_gate/` | phase 3a gate | `variants.png`: pilot score per variant, hollow if a hard gate fails |
| `phase4_gate/` | phase 4 gate | `strength_curves.png`: validation score against augmentation strength |
| `final_eval/` | the final evaluation | `forest.png` (pre-registered differences with CIs), `per_record_S_F.png` |
| `architecture/` | the whole system (`python -m ecgaug.figures architecture`) | `flow.png`: the pre-registered system, the post-hoc pipeline with the best test score and every model tested at each stage, with parameter counts computed from the configs |
| `report/` | the course report (`python -m ecgaug.figures report`) | `medical_background.png`: eight seconds of a training recording with the cardiologists' labels and RR intervals, a normal beat with its waves and the saliency windows, and the median S, V and F beats beside the same patient's normal beats (training records only); `progress.png`: the test score after each main change, from the pre-registered system to P20, and the best pipeline's S F1 on new patients |
| `phase5_explainability/<letter>/` | phase 5: saliency | `class_average_saliency.png`: where the models look, none versus augmented |
| `error_analysis/` | phase 5: errors | confusion matrices and the RR context of N/S confusions |
| `demo/` | the demo | one beat with its saliency and probabilities |
