# figures

Before-and-after figures for every phase. They are made by code (`src/ecgaug/figures.py`): the
runners draw them at the end of every experiment, the phase 1 gate draws its comparison, and
`python -m ecgaug.figures all` redraws everything from what is on disk. Only DS1 and PTB-XL are
plotted; the test set (DS2) never is.

| Folder | Phase and experiment | What it shows |
|---|---|---|
| `phase1_data/` | phase 1: MIT-BIH data before and after preprocessing | raw versus band-passed signal; the same N and S beat in variants A–D; beats per class in training and validation; median beat per class and variant |
| `phase1_input_representation/<letter>/` | phase 1: each variant of phase1_input_representation | training loss and validation score per epoch; validation confusion matrix |
| `phase1_gate/` | phase 1: the gate | the four variants side by side, seeds and mean ± SD |
| `phase2_ptbxl_data/` | phase 2: PTB-XL data before and after preprocessing | raw 500 Hz versus resampled signal with detected beats; PTB-XL versus MIT-BIH beat shape; recordings kept and excluded; label shares per split |
| `phase2_ptbxl_pretraining/<letter>/` | phase 2: each variant of phase2_ptbxl_pretraining | pretraining curves; from scratch versus fine-tuned curves; confusion matrices; F1 per class before and after pretraining |

Every figure has a title and a one-line caption saying what to read from it.

Folder names match `results/` and follow `docs/NAMING.md`.
