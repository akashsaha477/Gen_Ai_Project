# figures

Before-and-after figures for every stage. They are made by code (`src/ecgaug/figures.py`): the
runners draw them at the end of every experiment, Gate 1 draws its comparison, and
`python -m ecgaug.figures all` redraws everything from what is on disk. Only DS1 and PTB-XL are
plotted; the test set (DS2) never is.

| Folder | Stage | What it shows |
|---|---|---|
| `phase1/` | MIT-BIH preprocessing | raw versus band-passed signal; the same N and S beat in variants A–D; beats per class in training and validation; median beat per class and variant |
| `phase1/variant_<letter>/` | each Module 1 run | training loss and validation score per epoch; validation confusion matrix |
| `gates/` | Gate 1 | the four variants side by side, seeds and mean ± SD |
| `ptbxl/` | PTB-XL preprocessing | raw 500 Hz versus resampled signal with detected beats; PTB-XL versus MIT-BIH beat shape; recordings kept and excluded; label shares per split |
| `ptbxl_pretraining/<letter>/` | each pretraining-arm run | pretraining curves; from scratch versus fine-tuned curves; confusion matrices; F1 per class before and after pretraining |

Every figure has a title and a one-line caption saying what to read from it.
