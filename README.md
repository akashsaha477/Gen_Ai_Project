# Rare-arrhythmia detection with generative ECG augmentation

Does class-conditional generative augmentation (cVAE, WGAN-GP) improve detection of rare arrhythmia
classes on MIT-BIH when training and test patients are kept apart? A 30-day project (2026-10-07 to
2026-11-05) run phase by phase.

New to the project? Read `PROJECT_GUIDE.md` first: it explains the whole project and what is done.
`CODE_GUIDE.md` then explains what each code file and function does.

Then `docs/RULES.md` (rules and data protocol), then `docs/PLAN.md`, `docs/PAPER_TRACK.md` and
`docs/PREREGISTRATION.md`. Live state is in `docs/PROGRESS.md` and `docs/DECISIONS.md`.

## Status

Updated 2026-10-13 (Day 7). Phase 1 of 6, data foundation.

- Done: environment and lockfile, MIT-BIH download with checksum verification, beat extraction, per-record
  counts, the test lock, evaluation utilities, probe model, Module 1 runner with variants A-D, Kaggle setup.
- Dataset: MIT-BIH Arrhythmia Database (PhysioNet), inter-patient DS1/DS2 split, for training,
  validation and test. PTB-XL is used only to pretrain the classifier. Its preprocessing and the
  pretraining plus fine-tuning runner are built and tested on fake data and a 64-recording sample;
  the full run is meant for Kaggle (`kaggle/README.md`) and has not been done yet.
- Validation split: option V1 (records 115, 118, 201, 205, 223), in `configs/validation_split/approved.yaml`.
  Variants are selected on S F1 with V F1 as tie-break.
- Module 1 variants can be run now, locally or on Kaggle.
- The PTB-XL pretraining arm can be run on Kaggle: one run does preprocessing, pretraining and
  fine-tuning. Its results are provisional until Gate 1 fixes the input representation.
- Next: draw variant assignments, run `/variant 1 A..D`, then `/gate 1`.

## Repository layout

| Path | Contents |
|---|---|
| `src/ecgaug/` | the package: `data` (MIT-BIH pipeline, PTB-XL preprocessing, R-peak detector), `eval`, `models`, `train`, `experiments` (`m1`, `pretrain`) |
| `configs/` | experiment configs (`m1/` variants, `pretrain/` PTB-XL arm, `splits/` validation split) |
| `kaggle/` | everything for running on Kaggle; start with `kaggle/README.md` |
| `docs/` | plan, pre-registration, decisions, progress log, variant briefs |
| `results/` | generated summaries (counts, validation results, gate reports) |
| `tests/` | leakage, test lock, contract, metrics, bootstrap, runner |
| `Papers/` | reference papers |

## Names

| Name | Meaning |
|---|---|
| `configs/validation_split/approved.yaml` | the validation records every experiment uses |
| `configs/m1_input_representation/` | Module 1: four ways to present a beat to the model |
| `A_short_window` | 0.5 s window around the R peak (180 samples) |
| `B_long_window_bandpass` | 1 s window, 0.5–40 Hz band-pass (360 samples) |
| `C_short_window_rr_features` | A plus three RR-interval features |
| `D_short_window_125hz` | A resampled to 125 Hz (63 samples) |
| `shared_budget.yaml` | seeds, learning rates and epochs, identical for A–D |
| `configs/ptbxl_pretraining/` | the same four inputs, with PTB-XL pretraining before MIT-BIH |
| `ecgaug.experiments.input_representation` | runs a Module 1 config |
| `ecgaug.experiments.ptbxl_pretraining` | runs a pretraining config |

Run folders are named `p<phase>-m<module>-<what><letter>...`, for example `scratchA`, `pretrainA`
and `finetuneA` for the three kinds of run in the pretraining arm. Results keep the letter:
`results/validation/m1_input_representation/A.json`.

## Setup

```
uv sync                              # creates .venv with the versions pinned in uv.lock
uv run pytest                        # leakage, test lock, contract, metrics, bootstrap, runner
uv run python -m ecgaug.data.raw     # download MIT-BIH from PhysioNet and verify checksums
uv run python -m ecgaug.data.counts  # per-record class counts for DS1
```


## The test set is locked

DS2 is read only through `ecgaug.data.load_test()`, which raises unless `ALLOW_TEST=1`, and only
`/final-eval` sets that. Counting DS2 labels and building `test.npz` need `ALLOW_TEST_BUILD=1`. Every
access is appended to `docs/TEST_ACCESS_LOG.md`. Do not set either flag by hand outside those steps.

## Running a variant

```
uv run python -m ecgaug.experiments.input_representation --config configs/m1_input_representation/A_short_window.yaml --check   # DS1 inputs, no training
uv run python -m ecgaug.experiments.input_representation --config configs/m1_input_representation/A_short_window.yaml --smoke   # fake data, end to end
uv run python -m ecgaug.experiments.input_representation --config configs/m1_input_representation/A_short_window.yaml           # the experiment
```

The experiment needs the approved validation split (`configs/validation_split/approved.yaml`, option V1) and a
committed tree. Anyone with a clone can run it. To run on Kaggle, follow `kaggle/README.md`.

## PTB-XL pretraining arm

```
uv run python -m ecgaug.experiments.ptbxl_pretraining --config configs/ptbxl_pretraining/A_short_window.yaml --smoke          # fake data
uv run python -m ecgaug.experiments.ptbxl_pretraining --config configs/ptbxl_pretraining/A_short_window.yaml --ptbxl-root DIR  # real run
```

One command preprocesses PTB-XL (lead II, 360 Hz, detected R peaks, quality checks), pretrains the
classifier on the recordings' diagnostic superclasses, fine-tunes it on MIT-BIH, and trains the same
model from scratch for comparison. `DIR` is a PTB-XL copy; `--ptbxl-download` fetches it from PhysioNet
instead. Results go to `results/validation/ptbxl_pretraining/`. Design: `docs/PLAN.md`, Phase 2.

## Troubleshooting

- **`KeyError: '_items'` on `import matplotlib.pyplot` (macOS):** the system font listing returns
  nothing on some macOS versions. Call `ecgaug.eval.plotting.import_pyplot()` once; it builds
  matplotlib's font cache without that step, after which normal imports work.
