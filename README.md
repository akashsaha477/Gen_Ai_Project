# Rare-arrhythmia detection with generative ECG augmentation

Does class-conditional generative augmentation (cVAE, WGAN-GP) improve detection of rare arrhythmia
classes on MIT-BIH when training and test patients are kept apart? A 30-day project (2026-10-07 to
2026-11-05) run phase by phase with Claude Code.

New to the project? Read `PROJECT_GUIDE.md` first: it explains the whole project and what is done.
`CODE_GUIDE.md` then explains what each code file and function does.

Then `CLAUDE.md` (rules and data protocol), then `docs/PLAN.md`, `docs/PAPER_TRACK.md` and
`docs/PREREGISTRATION.md`. Live state is in `docs/PROGRESS.md` and `docs/DECISIONS.md`.

## Where we are and what to do next

One command answers both:

```
uv run python -m ecgaug.workflow status
```

It prints the checklist below, ticked where done, with the command for each item, and one line
starting with `NEXT:`. Run it at the start of every work session.

## The checklist, phase by phase

Names follow `docs/NAMING.md`: **phase** (1–6) → **experiment** (`phase<N>_<what>`) → **variant**
(A–D) → **run**. Each phase ends with a **gate**, where the team approves the winner. Each item needs
the one before it; the code refuses to skip ahead and says what is missing.

| Phase | Item | Command | State |
|---|---|---|---|
| 1 data | download and verify MIT-BIH | `python -m ecgaug.data.raw` | done |
| 1 data | approve the validation split | `configs/validation_split/approved.yaml` | done (V1) |
| 1 data | run `phase1_input_representation`, variants A–D | Kaggle: `kaggle/phase1_input_representation.ipynb`, `VARIANTS = ["A", "B", "C", "D"]` | done (2026-10-15) |
| 1 data | phase 1 gate: report | `python -m ecgaug.workflow phase1-gate` | done: winner C, see [docs/PHASE1_REPORT.md](docs/PHASE1_REPORT.md) |
| 1 data | phase 1 gate: team approves, the winner's dataset is frozen | `ALLOW_TEST_BUILD=1 python -m ecgaug.workflow phase1-gate --approve "Name" "Name"` | **next** (team) |
| 2 baselines | run `phase2_ptbxl_pretraining` for the phase 1 winner | Kaggle: `kaggle/phase2_ptbxl_pretraining.ipynb` with the winner's letter | one preview (variant D) |
| 2 baselines | `phase2_baseline_classifiers`, `phase2_quality_metrics` | not built yet | |

After each Kaggle run the notebook pushes the results to GitHub itself: `git pull`, then run `status`
again. (If the push did not happen: download the zip from the notebook's Output and run
`python kaggle/ingest_runs.py <zip>`.)

**Figures for every phase:** each run, and the phase 1 gate, draws before-and-after figures into
`figures/<experiment or phase folder>/` (data before and after preprocessing, training curves,
confusion matrices, before and after pretraining). They travel back from Kaggle in the zip.
`python -m ecgaug.figures all` redraws everything; `figures/README.md` lists what each folder shows.

**Current state (2026-10-16, Day 10):** all four `phase1_input_representation` variants have run.
The phase 1 gate report picks **variant C** (short window + RR features), with validation S F1
0.296 ± 0.038 against 0.164 for A, the next best. It is waiting for the team's approval. The full
analysis is in [docs/PHASE1_REPORT.md](docs/PHASE1_REPORT.md). Read section 6 before phase 2: an
early-stopping flaw cut 10 of the 24 runs short and should be fixed first. The earlier
`phase2_ptbxl_pretraining` preview used variant D and must be re-run with C.

**Previews:** the pretraining runner refuses to run before the phase 1 gate, or for a variant that the phase 1 gate did not
pick, unless you add `--preview` (`PREVIEW = True` in the notebook). Preview results are marked
provisional and do not count.

**Datasets:** MIT-BIH (inter-patient DS1/DS2 split) for training, validation and test. PTB-XL only
for pretraining.

## Repository layout

| Path | Contents |
|---|---|
| `src/ecgaug/` | the package: `data` (MIT-BIH pipeline, PTB-XL preprocessing, R-peak detector), `eval`, `models`, `train`, `experiments` (`phase1_input_representation`, `phase2_ptbxl_pretraining`), `workflow` (status and gates) |
| `configs/` | experiment configs (`validation_split/`, `phase1_input_representation/`, `phase2_ptbxl_pretraining/`) |
| `kaggle/` | everything for running on Kaggle; start with `kaggle/README.md` |
| `runs/` | one folder per training run, grouped by experiment, variant and kind of run |
| `figures/` | before-and-after figures for each phase, made by code |
| `results/` | generated summaries: counts, variant results, gate reports, `manifest.json` (what is frozen) |
| `docs/` | plan, pre-registration, decisions, progress log, variant briefs |
| `tests/` | leakage, test lock, contract, metrics, bootstrap, runners, workflow |
| `Papers/` | reference papers |

## Names

| Name | Meaning |
|---|---|
| `configs/validation_split/approved.yaml` | the validation records every experiment uses |
| `configs/phase1_input_representation/` | phase1_input_representation: four ways to present a beat to the model |
| `A_short_window` | 0.5 s window around the R peak (180 samples) |
| `B_long_window_bandpass` | 1 s window, 0.5–40 Hz band-pass (360 samples) |
| `C_short_window_rr_features` | A plus three RR-interval features |
| `D_short_window_125hz` | A resampled to 125 Hz (63 samples) |
| `shared_budget.yaml` | seeds, learning rates and epochs, identical for A–D |
| `configs/phase2_ptbxl_pretraining/` | the same four inputs, with PTB-XL pretraining before MIT-BIH |
| `ecgaug.experiments.phase1_input_representation` | runs a phase1_input_representation config |
| `ecgaug.experiments.phase2_ptbxl_pretraining` | runs a pretraining config |
| `ecgaug.workflow` | `status` (where we are) and `gate1` (pick and freeze the winning input) |

Runs are saved in folders by experiment, variant and kind of run:

```
runs/
  phase1_input_representation/
    A_short_window/              <- one folder per phase1_input_representation run of variant A
    B_long_window_bandpass/ ...
  phase2_ptbxl_pretraining/
    D_short_window_125hz/
      pretrain/                  <- pretraining on PTB-XL, one run per seed
      scratch/                   <- MIT-BIH from scratch, 6 runs
      finetune/                  <- MIT-BIH from the pretrained weights, 6 runs
```

Each run folder is named `<experiment>-<kind><letter>.<learning rate>-s<seed>-<commit>-<time>`
and holds the config, software versions, training curve, validation metrics and predictions.
`results/runs_index.csv` lists every run with its folder. Variant results keep the letter:
`results/phase2_ptbxl_pretraining/D.json`.

## Setup

```
uv sync                              # creates .venv with the versions pinned in uv.lock
uv run pytest                        # leakage, test lock, contract, metrics, bootstrap, runner
uv run python -m ecgaug.data.raw     # download MIT-BIH from PhysioNet and verify checksums
uv run python -m ecgaug.data.counts  # per-record class counts for DS1
```

Open Claude Code in this directory (not its parent) so `CLAUDE.md` and the `/phase`, `/variant`,
`/gate`, `/status`, `/final-eval` and `/paper` commands load. See `KICKOFF_PROMPT.md` for the phase loop.

## The test set is locked

DS2 is read only through `ecgaug.data.load_test()`, which raises unless `ALLOW_TEST=1`, and only
`/final-eval` sets that. Counting DS2 labels and building `test.npz` need `ALLOW_TEST_BUILD=1`. Every
access is appended to `docs/TEST_ACCESS_LOG.md`. Do not set either flag by hand outside those steps.

## Running a variant

```
uv run python -m ecgaug.experiments.phase1_input_representation --config configs/phase1_input_representation/A_short_window.yaml --check   # DS1 inputs, no training
uv run python -m ecgaug.experiments.phase1_input_representation --config configs/phase1_input_representation/A_short_window.yaml --smoke   # fake data, end to end
uv run python -m ecgaug.experiments.phase1_input_representation --config configs/phase1_input_representation/A_short_window.yaml           # the experiment
```

The experiment needs the approved validation split (`configs/validation_split/approved.yaml`, option V1) and a
committed tree. Anyone with a clone can run it. To run on Kaggle, follow `kaggle/README.md`.

## phase2_ptbxl_pretraining experiment

```
uv run python -m ecgaug.experiments.phase2_ptbxl_pretraining --config configs/phase2_ptbxl_pretraining/A_short_window.yaml --smoke          # fake data
uv run python -m ecgaug.experiments.phase2_ptbxl_pretraining --config configs/phase2_ptbxl_pretraining/A_short_window.yaml --ptbxl-root DIR  # real run
```

One command preprocesses PTB-XL (lead II, 360 Hz, detected R peaks, quality checks), pretrains the
classifier on the recordings' diagnostic superclasses, fine-tunes it on MIT-BIH, and trains the same
model from scratch for comparison. `DIR` is a PTB-XL copy; `--ptbxl-download` fetches it from PhysioNet
instead. Results go to `results/phase2_ptbxl_pretraining/`. Design: `docs/PLAN.md`, Phase 2.
It runs only for the variant the phase 1 gate chose; add `--preview` to run it earlier or for another variant.

## Troubleshooting

- **`KeyError: '_items'` on `import matplotlib.pyplot` (macOS):** the system font listing returns
  nothing on some macOS versions. Call `ecgaug.eval.plotting.import_pyplot()` once; it builds
  matplotlib's font cache without that step, after which normal imports work.
