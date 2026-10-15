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

It prints a checklist of every step (done or not, with the command for each) and one line starting
with `NEXT:`. Run it at the start of every work session.

## The steps, in order

Each step needs the one before it. The code refuses to skip ahead and says what is missing.

| Step | What | Command | Who |
|---|---|---|---|
| 1 | Download and verify MIT-BIH | `python -m ecgaug.data.raw` | done |
| 2 | Approve the validation split | `configs/validation_split/approved.yaml` | done (V1) |
| 3 | Run Module 1 variants A, B, C, D | Kaggle: `kaggle/module1_input_representation.ipynb`, `VARIANTS = ["A", "B", "C", "D"]` | one person, one session |
| 4 | Gate 1 report: the rule picks the winning input | `python -m ecgaug.workflow gate1` | anyone |
| 5 | Team reads `results/gates/gate1.md` and approves; the winner's dataset is frozen | `ALLOW_TEST_BUILD=1 python -m ecgaug.workflow gate1 --approve "Name" "Name"` | after the team agrees |
| 6 | Phase 2: PTB-XL pretraining for the winner, then the baseline classifiers | Kaggle: `kaggle/ptbxl_pretraining.ipynb` with the winner's letter | after step 5 |

After each Kaggle run: download the zip, `python kaggle/ingest_runs.py <zip>`, commit and push, then
run `status` again.

**Current state (2026-10-15, Day 9):** steps 1 and 2 are done. Step 3 has not been run. One preview of
step 6 exists (pretraining with variant D): pretraining did not help, and without RR timing the model
detects S beats at chance level. It is marked provisional because Gate 1 has not happened.

**Previews:** the pretraining runner refuses to run before Gate 1, or for a variant Gate 1 did not
pick, unless you add `--preview` (`PREVIEW = True` in the notebook). Preview results are marked
provisional and do not count.

**Datasets:** MIT-BIH (inter-patient DS1/DS2 split) for training, validation and test. PTB-XL only
for pretraining.

## Repository layout

| Path | Contents |
|---|---|
| `src/ecgaug/` | the package: `data` (MIT-BIH pipeline, PTB-XL preprocessing, R-peak detector), `eval`, `models`, `train`, `experiments` (`input_representation`, `ptbxl_pretraining`), `workflow` (status and gates) |
| `configs/` | experiment configs (`validation_split/`, `m1_input_representation/`, `ptbxl_pretraining/`) |
| `kaggle/` | everything for running on Kaggle; start with `kaggle/README.md` |
| `runs/` | one folder per training run, grouped by experiment, variant and kind of run |
| `results/` | generated summaries: counts, variant results, gate reports, `manifest.json` (what is frozen) |
| `docs/` | plan, pre-registration, decisions, progress log, variant briefs |
| `tests/` | leakage, test lock, contract, metrics, bootstrap, runners, workflow |
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
| `ecgaug.workflow` | `status` (where we are) and `gate1` (pick and freeze the winning input) |

Runs are saved in folders by experiment, variant and kind of run:

```
runs/
  m1_input_representation/
    A_short_window/              <- one folder per Module 1 run of variant A
    B_long_window_bandpass/ ...
  ptbxl_pretraining/
    D_short_window_125hz/
      pretrain/                  <- pretraining on PTB-XL, one run per seed
      scratch/                   <- MIT-BIH from scratch, 6 runs
      finetune/                  <- MIT-BIH from the pretrained weights, 6 runs
```

Each run folder is named `p<phase>-m<module>-<kind><letter>.<learning rate>-s<seed>-<commit>-<time>`
and holds the config, software versions, training curve, validation metrics and predictions.
`results/runs_index.csv` lists every run with its folder. Variant results keep the letter:
`results/validation/ptbxl_pretraining/D.json`.

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
It runs only for the variant Gate 1 chose; add `--preview` to run it earlier or for another variant.

## Troubleshooting

- **`KeyError: '_items'` on `import matplotlib.pyplot` (macOS):** the system font listing returns
  nothing on some macOS versions. Call `ecgaug.eval.plotting.import_pyplot()` once; it builds
  matplotlib's font cache without that step, after which normal imports work.
