# Rare-arrhythmia detection with generative ECG augmentation

Does class-conditional generative augmentation (cVAE, WGAN-GP) improve detection of rare arrhythmia
classes on MIT-BIH when training and test patients are kept apart? A 30-day project (2026-10-07 to
2026-11-05) run phase by phase.

New to the project? Read `PROJECT_GUIDE.md` first: it explains the whole project and what is done.
`CODE_GUIDE.md` then explains what each code file and function does.

Then `docs/RULES.md` (rules and data protocol), then `docs/PLAN.md`, `docs/PAPER_TRACK.md` and
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
| 1 data | run `phase1_input_representation`, variants A–D | `kaggle/phase1_input_representation.ipynb` | done (2026-10-15) |
| 1 data | phase 1 gate: report | `python -m ecgaug.workflow phase1-gate` | done: winner C, see [docs/PHASE1_REPORT.md](docs/PHASE1_REPORT.md) |
| 1 data | phase 1 gate: team approves, the winner's dataset is frozen | `ALLOW_TEST_BUILD=1 python -m ecgaug.workflow phase1-gate --approve "Name" "Name"` | **next** (team) |
| 2 baselines | `phase2_baseline_classifiers` A–D (shallow CNN, ResNet1D, BiLSTM, CNN+BiLSTM) | `kaggle/phase2_baseline_classifiers.ipynb` | done: A 0.349, B 0.231, C 0.315, D 0.252 (S F1) |
| 2 baselines | `phase2_quality_metrics` A–D (DTW, spectral, FID-analog, TSTR) | `kaggle/phase2_quality_metrics.ipynb` | done: DTW and spectral pass; FID-analog fails; TSTR could not be scored |
| 2 baselines | `phase2_ptbxl_pretraining` for the phase 1 winner | `kaggle/phase2_ptbxl_pretraining.ipynb` | done for C: tied, scratch wins by the rule |
| 2 baselines | phase 2 gate: the `cnn`, the `rnn`, the quality composite | `python -m ecgaug.workflow phase2-gate [--approve ...]` | approved 2026-11-01: cnn A, rnn C, composite DTW + spectral |
| 3 generators | `phase3_cvae` A–D and `phase3_wgan_gp` A–D, pilots | `kaggle/phase3_cvae.ipynb`, `kaggle/phase3_wgan_gp.ipynb` (`MODE = "pilot"`) | done 2026-11-01 |
| 3 generators | phase 3a gate: one generator per family | `python -m ecgaug.workflow phase3a-gate [--approve ...]` | report written, pending approval: cVAE D, WGAN-GP A by rule |
| 3 generators | final runs of the two winners, synthetic banks | the same notebooks with `MODE = "final"` | built |
| 3 generators | phase 3 gate: banks frozen | `python -m ecgaug.workflow phase3-gate [--approve ...]` | built |
| 4 comparison | `phase4_augmentation_policy` A–D, seeds 0–2 (DECISIONS A19) | `kaggle/phase4_augmentation_policy.ipynb`, all four letters in one session | built |
| 4 comparison | phase 4a gate: the policy; writes the full grid | `python -m ecgaug.workflow phase4a-gate [--approve ...]` | built |
| 4 comparison | the full grid (96 runs at 3 seeds, resumable) | `kaggle/phase4_full_grid.ipynb` | built |
| 4 comparison | phase 4 gate: strengths and checkpoints, on validation | `python -m ecgaug.workflow phase4-gate [--approve ...]` | built |
| 4 comparison | the single final evaluation on the test set | `ALLOW_TEST=1 python -m ecgaug.final_eval --confirm "run final eval"` | built |
| 5 explanations | `phase5_explainability` A–D, `phase5_clinical_notes` A–D, error analysis, demo | `python -m ecgaug.experiments.phase5_*`, `python -m ecgaug.demo` | built |
| 5 explanations | phase 5 gate, then `git tag v1.0` | `python -m ecgaug.workflow phase5-gate [--approve ...]` | built |
| 4 comparison | RQ3 analysis; leakage audit (optional, after the final evaluation) | `python -m ecgaug.eval.rq3`; `kaggle/phase4_leakage_audit.ipynb` | built |
| 6 paper | `paper/main.tex` and the course report `paper/report.tex`: tables, numbers and results sentences | `python -m ecgaug.paper` | built |

"Built" means the code, configs, notebook, tests and brief exist. A dress rehearsal runs every stage in
order on fake records (`ECGAUG_REHEARSAL=1 uv run pytest tests/test_rehearsal.py`, about 8 minutes). Every
runner refuses an official run until the gates before it are approved, and says which. `--preview`
runs earlier with stand-ins and marks the result provisional; a gate never approves a preview.

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
early-stopping flaw cut 10 of the 24 runs short; the fix is the `min_epochs` setting (team decision).
The code of phases 2-6 is built on the branch `finish-project` (docs/DECISIONS.md A14 lists the
design choices the team should confirm). The earlier `phase2_ptbxl_pretraining` preview used variant D
and must be re-run with C.

**Previews:** every runner after phase 1 refuses to run before the gates it needs unless you add
`--preview` (`PREVIEW = True` in the notebooks). Preview results are marked provisional and do not count.

**Artifacts that are not in git:** model weights and synthetic banks are too big for GitHub. They
travel in each Kaggle run's zip (`python kaggle/ingest_runs.py <zip>` brings them into `runs/` and
`data/synthetic/`). Phase 4 and 5 notebooks take them from an attached Kaggle dataset
(`BANKS_DATASET_DIR`, `ARTIFACTS_DIR`); see `kaggle/README.md`.

**Datasets:** MIT-BIH (inter-patient DS1/DS2 split) for training, validation and test. PTB-XL only
for pretraining.

## Repository layout

| Path | Contents |
|---|---|
| `src/ecgaug/` | the package: `data` (MIT-BIH pipeline, PTB-XL preprocessing, R-peak detector), `eval` (metrics, bootstrap, selection rules), `models` (probe and the four baselines), `train`, `generators` (cVAE, WGAN-GP, banks), `quality` (diagnostics, metric families, sanity harness), `augment` (policies, SMOTE, oversampling, class weights), `grid` (the grid runner), `experiments` (one runner per experiment), `explain` (saliency and its scores), `report` (notes and fact checker, error analysis), `final_eval`, `demo`, `paper`, `workflow` (status and gates) |
| `configs/` | one folder per experiment (`<experiment>/<letter>_<name>.yaml` plus a shared settings file), `validation_split/`, `phase4_full_grid/` |
| `paper/` | the LaTeX draft (`main.tex`), `generated/` tables and numbers, `claims_vs_evidence.md` |
| `scripts/reproduce.sh` | the whole pipeline in order |
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
| `ecgaug.workflow` | `status` (where we are) and the gates: `phase1-gate`, `phase2-gate`, `phase3a-gate`, `phase3-gate`, `phase4a-gate`, `phase4-gate`, `phase5-gate` |
| `ecgaug.final_eval` | the single test evaluation (`--plan` reads no test data) |
| `ecgaug.grid.runner` | the full grid (`--dry-run` lists the cells and which are done) |
| `ecgaug.demo` | one validation beat: prediction, saliency plot, fact-checked note |
| `ecgaug.paper` | the paper's tables and numbers from the result files |

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

## Phases 2-6

Each experiment's runner has the same flags: `--config` (a variant file), `--smoke` (fake data, end to
end, writes nothing to the repository) and `--preview` (before the gates it needs; provisional):

```
uv run python -m ecgaug.experiments.phase2_baseline_classifiers --config configs/phase2_baseline_classifiers/A_shallow_cnn.yaml --smoke
uv run python -m ecgaug.experiments.phase2_quality_metrics --config configs/phase2_quality_metrics/A_dtw.yaml --smoke
uv run python -m ecgaug.experiments.phase3_cvae --config configs/phase3_cvae/A_conv_concat.yaml --smoke            # add --final for the full budget and banks
uv run python -m ecgaug.experiments.phase3_wgan_gp --config configs/phase3_wgan_gp/A_concat_embedding.yaml --smoke
uv run python -m ecgaug.experiments.phase4_augmentation_policy --config configs/phase4_augmentation_policy/B_fraction_of_n.yaml --smoke
uv run python -m ecgaug.grid.runner --config configs/phase4_full_grid/dry_run_baselines.yaml --smoke
uv run python -m ecgaug.experiments.phase5_explainability --config configs/phase5_explainability/B_integrated_gradients.yaml --smoke
uv run python -m ecgaug.experiments.phase5_clinical_notes --config configs/phase5_clinical_notes/A_template.yaml --smoke
uv run python -m ecgaug.demo --smoke
uv run python -m ecgaug.paper
```

What each variant is, how to run it and what to check: `docs/briefs/<experiment>_<variant>.md`.
`scripts/reproduce.sh` lists the whole order. The paper's results sentences and hypothesis verdicts are
written by `python -m ecgaug.paper` from the result files (`src/ecgaug/paper_text.py`); compile
`paper/main.tex` and `paper/report.tex` with pdflatex and bibtex. The LLM note variants need `LLM_API_KEY` (and
optionally `ECGAUG_LLM_MODEL`) in the environment; without it the template note is used.

## Troubleshooting

- **`KeyError: '_items'` on `import matplotlib.pyplot` (macOS):** the system font listing returns
  nothing on some macOS versions. Call `ecgaug.eval.plotting.import_pyplot()` once; it builds
  matplotlib's font cache without that step, after which normal imports work.
