# Project guide: rare-arrhythmia detection with generative ECG augmentation

This is the long explanation of the project: what we are trying to find out, how the work is
organised, what each part does, and what is finished. For commands only, see `README.md`. For the
day-by-day log, see `docs/PROGRESS.md`. Last updated 2026-10-16 (Day 10 of 30).

## Contents

1. [The project in one page](#1-the-project-in-one-page)
2. [The data](#2-the-data)
3. [The rules we work by](#3-the-rules-we-work-by)
4. [The plan: six phases](#4-the-plan-six-phases)
5. [What is built, part by part](#5-what-is-built-part-by-part)
6. [Built but not run yet](#6-built-but-not-run-yet)
7. [How to run things](#7-how-to-run-things)
8. [Where everything lives](#8-where-everything-lives)
9. [Decisions made so far](#9-decisions-made-so-far)
10. [What has been checked and what has not](#10-what-has-been-checked-and-what-has-not)
11. [Glossary](#11-glossary)

---

## 1. The project in one page

**The problem.** In ECG recordings most heartbeats are normal. The abnormal ones a doctor cares about
are rare, so a classifier sees very few examples of them and detects them poorly.

**The idea being tested.** Train a generative model on the few rare beats we have, let it produce
extra synthetic ones, add those to the training data, and see whether the classifier gets better at
the rare classes.

**The question.** Does this generative augmentation really help when the model is tested on patients
it has never seen? We compare two generators (a conditional VAE and a WGAN-GP) against simpler
options: no augmentation, class-weighted loss, random oversampling and SMOTE.

**Why it is worth a paper.** Many published results on this topic split the data by beat, so the same
patient appears in training and test, which inflates the scores. We split by patient, lock the test
set, and choose everything on validation data by rules fixed in advance. The contribution is a
careful answer, not a new generator.

**What we deliver.**

1. A course deliverable in 30 days (2026-10-07 to 2026-11-05): an ECG beat goes in; a rare-arrhythmia
   prediction, a saliency map and a short clinician-readable note come out.
2. Evidence solid enough to publish, including negative results.

**Team.** Four members. Each experiment has four variants (A–D); each member takes one, and the winner is
chosen at a "gate" by a written rule.

---

## 2. The data

### MIT-BIH Arrhythmia Database (the main dataset)

- 48 half-hour ECG recordings, with every beat labelled by cardiologists.
- We use lead MLII and the 44 recordings without a pacemaker.
- Beats are grouped into four classes (the AAMI standard):

| Class | Meaning | Beats in training half (DS1) | Beats in test half (DS2) |
|---|---|---|---|
| N | normal | 45,866 | 44,259 |
| S | supraventricular ectopic (premature, from above the ventricles) | 944 | 1,837 |
| V | ventricular ectopic | 3,788 | 3,221 |
| F | fusion of a normal and a ventricular beat | 415 | 388 |

S and F are the "rare classes" the project is about.

**The split is by patient** (the standard de Chazal split):

| Part | Records | Used for |
|---|---|---|
| Training | 17 records of DS1 | fitting models and generators |
| Validation | 115, 118, 201, 205, 223 | choosing between variants, early stopping |
| Test | the 22 records of DS2 | one final evaluation at the end |

Two facts about this data shape the whole project:

- **Rare beats sit in very few patients.** Record 208 holds 373 of the 415 F beats in DS1. In the test
  half, record 213 holds 362 of 388 F beats and record 232 holds 1,382 of 1,837 S beats. So results on
  rare classes depend heavily on one or two patients, and our confidence intervals will be wide.
- **Records 201 and 202 are the same person**, one in each half. We keep the standard split but put 201
  in validation, so nothing trains on that patient.

### PTB-XL (pretraining only)

- About 21,800 ten-second, 12-lead recordings from about 18,900 patients, each with diagnostic
  labels for the whole recording. It has no beat-level labels.
- We use it for one thing: to pretrain the classifier before fine-tuning it on MIT-BIH. It is never
  used for validation or testing.

---

## 3. The rules we work by

These come from `CLAUDE.md`. In plain terms:

| Rule | What it means in practice |
|---|---|
| Split by patient | No record is in two of training, validation and test. Anything fitted (generators, oversamplers) sees training records only. |
| The test set is locked | Code cannot read DS2 without an explicit permission, and every access is written to `docs/TEST_ACCESS_LOG.md`. The test set is evaluated once, at the end. |
| Choose on validation, by written rules | Which variant wins is decided by a rule fixed beforehand and applied by code, not by looking at results and picking. |
| Equal budgets | Variants of an experiment get the same data, seeds, epochs and learning rates. |
| No invented numbers | Every number in a document comes from code output. Unknown values are written as unknown. |
| Report what happened | Failures and negative results are recorded and go in the paper. |
| Humans approve gates | A phase ends with a gate report; the team approves it before the next phase starts. |

**How a winner is chosen** (the selection rule, in `src/ecgaug/eval/selection.py`):

1. Each variant is trained with seeds 0, 1 and 2; its score is the mean over seeds.
2. The score is S F1 on validation. (The planned score was the mean of S F1 and F F1, but validation
   has only 27 F beats, too few to measure F reliably, so the fallback applies.)
3. Variants within a tie margin of the best one are treated as tied. The margin is 0.02 or the
   seed-to-seed spread, whichever is larger.
4. Ties are broken by V F1, then by simplicity.

---

## 4. The plan: six phases

| Phase | Planned days | What happens | State |
|---|---|---|---|
| 1 data | 1–3 | data pipeline, evaluation code, choose the input representation (phase1_input_representation) | infrastructure built; variants not run |
| 2 baselines | 4–8 | baseline classifiers (phase2_baseline_classifiers), metrics for judging synthetic beats (phase2_quality_metrics), PTB-XL pretraining (phase2_ptbxl_pretraining) | one preview of phase2_ptbxl_pretraining; the rest not started |
| 3 generators | 9–14 | cVAE (phase3_cvae) and WGAN-GP (phase3_wgan_gp), synthetic beat banks | not started |
| 4 comparison | 16–21 | augmentation policy (phase4_augmentation_policy), the full comparison grid, the single test evaluation | not started |
| 5 explanations | 22–26 | saliency maps (phase5_explainability), clinical-note generator (phase5_clinical_notes), demo | not started |
| 6 paper | 27–30 | paper draft, course report, reproducibility package | not started |

We are on Day 7 and about four days behind: Phase 1 was due on Day 3.

**Where are we?** `python -m ecgaug.workflow status` prints every step, ticked or not, and the next
action. The code enforces the order: Phase 1 gate needs all four phase1_input_representation results, and the phase2_ptbxl_pretraining experiment
needs an approved phase 1 gate (or `--preview`, which marks the result provisional).

**The phase loop.** One person builds the shared code for a phase. Each member runs one variant on
their own branch. A gate report applies the selection rule. The team approves, and the next phase
starts. Details are in `KICKOFF_PROMPT.md`.

---

## 5. What is built, part by part

### 5.1 Environment

- Python 3.11 with every package pinned in `uv.lock`. `uv sync` recreates it exactly.
- 159 automated tests (`uv run pytest`). They also pass in a fresh clone.

### 5.2 MIT-BIH data pipeline (`src/ecgaug/data/`)

| Step | What it does | File |
|---|---|---|
| Download and verify | Fetches the database from PhysioNet and checks all 144 files against PhysioNet's published checksums. All match. | `raw.py` |
| Beat extraction | Reads the cardiologists' annotations, maps them to N, S, V, F, and measures the time to the previous and next beat (RR intervals). | `beats.py`, `records.py` |
| Counts | Counts beats per record and class. Our counts equal the table published with the database for all 44 records. | `counts.py` |
| Splits | Defines training, validation and test, and refuses any split where a record appears twice. | `splits.py` |
| Input representations | Cuts a window around each beat. One code path serves all four variants (below). | `representation.py` |
| Dataset files | Writes the arrays with a SHA-256 hash; loaders check the hash before use. | `contract.py`, `loaders.py` |

A beat is used only if its window fits inside the record and both RR intervals exist. That gives
50,969 usable beats in DS1, identical for all four variants.

### 5.3 The test lock (`src/ecgaug/data/access.py`, `raw.py`)

- All raw-file reading goes through one file. It refuses any DS2 record unless the caller declares
  why and a matching environment flag is set.
- Two kinds of access exist: *construction* (counting labels, building files; no model involved) and
  *evaluation* (the final test, once).
- So far the log has one entry: counting DS2 labels for the dataset table. No model has seen DS2.

### 5.4 Evaluation code (`src/ecgaug/eval/`) — this is the evaluation code

- **Metrics:** per-class precision, recall and F1, PR-AUC, the AAMI sensitivity and positive
  predictivity used in older papers, and the confusion matrix. Tested against hand-computed cases.
- **Bootstrap:** confidence intervals that resample whole patients, not single beats, because beats
  from one patient are not independent. Includes paired comparisons between two methods, p-values and
  the Holm correction for multiple comparisons.
- **Selection rule:** the tie margin and tie-breaks described in section 3.

### 5.5 phase1_input_representation: input representation (`configs/phase1_input_representation/`)

The first real experiment. It asks: how should a beat be presented to the model?

| Variant | Config | Input per beat |
|---|---|---|
| A | `A_short_window.yaml` | 0.5 s window around the R peak, 180 samples |
| B | `B_long_window_bandpass.yaml` | 1 s window, 0.5–40 Hz band-pass filter, 360 samples |
| C | `C_short_window_rr_features.yaml` | A plus three RR-interval features |
| D | `D_short_window_125hz.yaml` | A resampled to 125 Hz, 63 samples |

- All four are compared with the same small "probe" network (about 19,000 parameters), the same seeds
  and the same budget (`shared_budget.yaml`): 3 seeds, 2 learning rates, at most 40 epochs, early
  stopping after 5 epochs without improvement.
- The winner defines the dataset every later phase uses.
- **State:** ready to run, not run. The inputs build correctly on real data and the whole pipeline runs
  on fake data.

### 5.6 Training and run logging (`src/ecgaug/train/`, `src/ecgaug/runs.py`)

- One training loop with early stopping, fixed seeds and deterministic settings.
- Every run gets its own folder, grouped by experiment, variant and kind of run (for example
  `runs/phase2_ptbxl_pretraining/D_short_window_125hz/finetune/`), with its config, software versions, training curve,
  validation metrics, predictions and model checkpoint. `results/runs_index.csv` lists all runs.

### 5.7 PTB-XL preprocessing (`src/ecgaug/data/ptbxl.py`, `rpeaks.py`, `checks.py`)

Turns PTB-XL recordings into beats that look like MIT-BIH beats:

1. Drop recordings with no diagnostic superclass, a pacemaker, or flagged electrode problems.
2. Take lead II (closest to MLII), check the signal, resample from 500 Hz to 360 Hz.
3. Find the beats with an R-peak detector, since PTB-XL has no beat annotations.
4. Keep beats with a full window and plausible RR intervals (0.3–2.0 s).
5. Give each beat its recording's diagnostic superclasses (normal, infarction, ST/T change, conduction
   disturbance, hypertrophy).
6. Split by patient using PTB-XL's official folds: 1–8 train, 9 validation, 10 unused.
7. Cut windows with the same code as MIT-BIH, after mirror-padding each recording by 2 s.

What was measured to justify these choices:

| Check | Result |
|---|---|
| R-peak detector against MIT-BIH DS1 annotations | sensitivity 99.63%, positive predictivity 99.33%, median timing offset 0 ms |
| Mirror padding, for beats within 1 s of a recording's end (band-passed variant) | correlation with the properly filtered beat, 5th percentile: 0.990 with padding, 0.854 without |
| Shape match on a 64-recording sample | average PTB-XL beat correlates 0.93 with the average MIT-BIH normal beat |
| Sample files against PhysioNet's checksums | 128 of 128 match |

- **State:** built and tested on 64 real recordings (59 kept, 587 beats). The full dataset has not been
  processed; that happens in the Kaggle run.

### 5.8 phase2_ptbxl_pretraining experiment (`src/ecgaug/experiments/phase2_ptbxl_pretraining.py`)

Asks: does pretraining on a large ECG dataset help rare-beat detection on MIT-BIH?

1. **Pretrain** the model on PTB-XL beats to predict the five superclasses (once per seed).
2. **Fine-tune** it on MIT-BIH: replace the output layer with a 4-class one and train on DS1.
3. **Train the same model from scratch** on MIT-BIH, with the same budget, for comparison.
4. **Decide** with the selection rule. Pretraining has to beat from-scratch by more than the tie margin.

- One command does all four steps, including the PTB-XL preprocessing.
- **State:** one real run, variant D, done on Kaggle on 2026-10-14 by kartheek2159
  (`results/phase2_ptbxl_pretraining/D.json`). Pretraining reached a macro AUROC of 0.83 on PTB-XL,
  but fine-tuning did not beat from-scratch on MIT-BIH, so the rule picked from-scratch. Both methods
  detect S beats at chance level with this input (S F1 about 0.03). The result is provisional and uses
  the small probe network.

### 5.9 Kaggle setup (`kaggle/`)

- `phase1_input_representation.ipynb` and `phase2_ptbxl_pretraining.ipynb`: one notebook per job. You set the
  variant letters and run all cells.
- `make_raw_dataset.py`: packs the verified MIT-BIH files into a 66 MB zip to upload once as a private
  Kaggle dataset, so sessions do not re-download from PhysioNet.
- Nothing is pushed from Kaggle by default (`PUSH_RESULTS = False`): download the result zip, bring it
  into your clone with `ingest_runs.py`, and commit it under your own name.
- **State:** rehearsed on a laptop against a fresh clone. The phase 1 notebook has run on Kaggle (all four
  variants); the later notebooks are tested locally only.

### 5.10 Phase 2: baseline classifiers and quality metrics (`models/baselines.py`, `quality/`)

- Four classifiers (shallow CNN, 1D-ResNet, BiLSTM, CNN+BiLSTM), each with the RR side input, a
  feature hook (`embed`) and, for the convolutional ones, `last_conv` for Grad-CAM. The runner trains
  them like a phase1_input_representation variant; the phase 2 gate picks the `cnn` (A or B) and the
  `rnn` (C or D).
- Four quality-metric families: DTW to the nearest real beats, Welch spectra (Jensen-Shannon), a
  Fréchet distance on a small frozen feature network, and train-on-synthetic-test-on-real. A sanity
  harness degrades real beats (noise, label shuffle, time shift, smoothing; three severities each) and
  keeps the families that rank real beats best, worsen with severity and are stable.
- Diagnostics used by phase 3's hard gates: diversity ratio, memorization fraction (leave-one-out
  threshold), and a classifier two-sample test (C2ST).

### 5.11 Phase 3: generators and synthetic banks (`generators/`, `experiments/generator_runs.py`)

- A conditional VAE (variants: plain, low KL weight, spectral loss, residual with FiLM) and a
  conditional WGAN-GP (label embedding, projection critic, auxiliary classifier, one GAN per class),
  behind one interface: `fit` on training beats only (asserted), `sample(class, n, seed)`.
- The pilot harness scores each quarter of the step budget: diagnostics, quality metrics, and the
  pilot classifier trained with synthetic S, V and F beats up to 50% of N. Every checkpoint is logged
  (the RQ3 data); the final checkpoint counts. Synthetic beats get RR features copied from real beats.
- The phase 3a gate applies the hard gates and the pilot score; the winners are retrained at the full
  budget and export banks (`data/synthetic/`), which the phase 3 gate freezes by hash.

### 5.12 Phase 4: policy, grid, final evaluation (`augment.py`, `grid/`, `final_eval.py`)

- Augmentation conditions: generator banks, SMOTE, random oversampling, class-weighted loss; four
  policies for how many beats to add (to N, to a fraction of N, a multiple of the real count, or a
  DTW-screened fraction).
- The grid runner trains every cell (condition x strength x architecture x seed), skips cells already
  done, and keeps one small ledger file per cell, so sessions can share the work.
- The final evaluation refuses unless the phase 4 gate is approved and no evaluation has happened;
  it applies the pre-registered statistics (paired record-level bootstrap, Holm per family,
  sensitivity without record 202 and leaving out each S/F record) and writes tables, a forest plot and
  LaTeX macros.

### 5.13 Phase 5: explanations, notes, demo (`explain/`, `report/`, `demo.py`)

- Saliency: Grad-CAM, integrated gradients, SmoothGrad, occlusion; scored for faithfulness (deletion
  against random order), stability across seeds and the share of saliency in P/QRS/ST/T windows.
- Notes: a template and three LLM variants restate the model's output; a fact checker rejects any
  number, class, segment or diagnosis not in the input, and requires the disclaimer and, for
  low-confidence beats, "Clinician review is required."
- Error analysis from saved predictions only; a demo for one validation beat.

### 5.14 Phase 6: the paper (`paper.py`, `paper/`)

- `python -m ecgaug.paper` writes every table and number of `paper/main.tex` from the result files;
  missing results show as red placeholders, so the draft always compiles. `scripts/reproduce.sh` lists
  the whole pipeline in order.

---

## 6. Built but not run yet

Everything below exists as code, configs, Kaggle notebook, tests and a brief per variant
(`docs/briefs/`). None of it has run officially: each runner waits for the gates before it.

| Part | Phase | Waits for |
|---|---|---|
| phase 1 gate approval (frozen dataset) | 1 | the team (report: `results/phase1_gate/report.md`, analysis: `docs/PHASE1_REPORT.md`) |
| phase2_baseline_classifiers, phase2_quality_metrics, phase2_ptbxl_pretraining (for C) | 2 | the phase 1 gate |
| phase3_cvae, phase3_wgan_gp pilots, then final banks | 3 | the phase 2 gate (pilot classifier, quality composite), then the phase 3a gate |
| phase4_augmentation_policy, the full grid | 4 | the phase 3 gate (frozen banks), then the phase 4a gate |
| the final test evaluation | 4 | the phase 4 gate |
| phase5_explainability, phase5_clinical_notes, demo | 5 | the phase 4 gate (models); an `ANTHROPIC_API_KEY` for the LLM notes |
| paper text, verified references, course report | 6 | results |

Design choices the team should confirm before the official runs are listed in `docs/DECISIONS.md` A14;
the values still to fix in the pre-registration are in `docs/PREREGISTRATION.md` §10.

---

## 7. How to run things

**Set up once**

```
uv sync
uv run pytest
uv run python -m ecgaug.data.raw      # download and verify MIT-BIH (about 30 minutes)
```

**phase1_input_representation, your variant**

```
uv run python -m ecgaug.experiments.phase1_input_representation --config configs/phase1_input_representation/A_short_window.yaml
```

Add `--smoke` for a one-minute check on fake data, or `--check` to build the inputs without training.

**phase2_ptbxl_pretraining experiment**

```
uv run python -m ecgaug.experiments.phase2_ptbxl_pretraining --config configs/phase2_ptbxl_pretraining/A_short_window.yaml --ptbxl-download
```

**Phases 2-6:** the commands are in `README.md` ("Phases 2-6") and in each variant's brief.

**On Kaggle:** follow `kaggle/README.md`.

A real run needs two things: the approved validation split (`configs/validation_split/approved.yaml`,
already in the repo) and no uncommitted code changes.

---

## 8. Where everything lives

| Path | Contents |
|---|---|
| `README.md` | quick start and current status |
| `PROJECT_GUIDE.md` | this file |
| `CODE_GUIDE.md` | what each code file and function does |
| `CLAUDE.md` | the project rules and data protocol |
| `docs/PLAN.md` | every phase, experiment and variant in detail |
| `docs/PAPER_TRACK.md` | research questions, hypotheses, statistics plan, figures |
| `docs/PROGRESS.md` | what was done each day |
| `docs/DECISIONS.md` | every decision and plan change, with the reason |
| `docs/PREREGISTRATION.md` | the selection and statistics rules written out |
| `docs/briefs/` | one page per variant of every experiment, for the person running it |
| `docs/PHASE1_REPORT.md` | the analysis of the phase 1 results |
| `docs/TEST_ACCESS_LOG.md` | every access to the test set |
| `configs/` | experiment settings |
| `src/ecgaug/data/` | MIT-BIH pipeline, test lock, PTB-XL preprocessing |
| `src/ecgaug/eval/` | metrics, bootstrap, selection rule |
| `src/ecgaug/models/`, `train/` | the probe, the four baseline classifiers and the training loop |
| `src/ecgaug/generators/`, `quality/`, `augment.py`, `grid/` | generators and banks, quality metrics, augmentation conditions, the grid runner |
| `src/ecgaug/explain/`, `report/`, `demo.py`, `final_eval.py`, `paper.py` | saliency, notes and error analysis, demo, final evaluation, paper tables |
| `src/ecgaug/experiments/` | one runner per experiment |
| `paper/` | the LaTeX draft and its generated tables and numbers |
| `kaggle/` | notebooks, scripts and guide for Kaggle |
| `results/phase1_data/` | counts, checks and validation reports produced by code |
| `results/<experiment>/` | variant results (empty until variants are run) |
| `runs/` | training runs, in folders by experiment, variant and kind of run |
| `tests/` | the automated tests |
| `Papers/` | reference papers |

---

## 9. Decisions made so far

Full reasons are in `docs/DECISIONS.md`.

| Decision | Why |
|---|---|
| Validation split V1 (records 115, 118, 201, 205, 223) | Keeps record 208, which holds most F beats, in training; spreads validation S over three records; keeps the patient shared with the test set out of training. |
| Selection on S F1, with V F1 as tie-break | Validation has only 27 F beats, too few to rank variants on. |
| Two families of statistical tests | The main claim (generators vs no augmentation) is tested on its own, 4 tests; the other comparisons are secondary. |
| Add random oversampling to the comparison | Without it, a gain from synthetic beats cannot be separated from simply showing the model more rare beats. |
| MIT-BIH is the main dataset | The task, the classes and the generators are all beat-level, and PTB-XL has no beat labels. |
| PTB-XL for supervised pretraining | Simple to do; each beat takes its recording's diagnosis. The labels are noisy at beat level, which is stated as a limitation. |
| The pre-registration document is not enforced by code | The runner needs only the approved split. The document stays as a description of the rules; the paper cannot call it a time-stamped pre-registration. |
| phase2_ptbxl_pretraining built before the phase 1 gate | To use Kaggle time now. Its results are provisional until the input representation is fixed. |
| The phase 1 gate report picks variant C (pending approval) | Best S F1 by a clear margin; RR timing is what makes S beats detectable (docs/PHASE1_REPORT.md). |
| Phases 2-6 built before their gates (A14) | At Akash's instruction; official runs still wait for every gate, and A14 lists the design choices to confirm. |

---

## 10. What has been checked and what has not

**Checked**

- All MIT-BIH files match PhysioNet's checksums; beat counts match the published table.
- The four input variants build correctly on real DS1 data and produce the same 50,969 beats.
- The R-peak detector and the padding choice were measured on DS1 (section 5.7).
- Both runners work end to end on fake data, also through the Kaggle runner script in a fresh clone.
- 271 tests pass, including tests that the test set cannot be read by accident, and an end-to-end final
  evaluation on fake data that runs once and refuses a second time.
- Every phase 2-5 runner runs end to end on fake data; the phase 3 pilot and the DTW and spectral
  quality metrics also ran on real DS1 beats (short CPU previews, numbers not used).
- The phase2_ptbxl_pretraining experiment ran on Kaggle for variant D, including full PTB-XL preprocessing (20,918 recordings kept).

**Not checked yet**

- The phase 2-5 notebooks have not run on Kaggle; run times of generators and the grid are not measured.
- The LLM note variants have not been tried with a real API key.
- Only one real experiment exists: the phase2_ptbxl_pretraining experiment for variant D (section 5.8).
- PTB-XL file checksums were not compared on Kaggle (PhysioNet's list could not be fetched there).
- One metric convention (AAMI positive predictivity for V) is implemented from memory of the original
  paper and must be verified before it is quoted.

**Known risks**

- The schedule: four days behind, and the one buffer day before Phase 4 is already used.
- Rare classes depend on one or two patients, so the final intervals will be wide and the main test
  may be inconclusive even if augmentation helps.
- The probe network is very small; pretraining may show little effect until a larger baseline exists.

---

## 11. Glossary

| Term | Meaning |
|---|---|
| AAMI classes | The standard grouping of beat types into N, S, V, F (and Q, which we exclude). |
| DS1, DS2 | The two halves of MIT-BIH in the standard patient-wise split: DS1 for training and validation, DS2 for testing. |
| Inter-patient | Training and test data come from different patients. |
| RR interval | Time between two consecutive beats. A premature beat has a short one before it. |
| R peak | The tallest point of a heartbeat in the ECG; beats are cut around it. |
| F1 | A score between 0 and 1 that balances missed beats against false alarms. |
| Validation set | Data used to choose between options; never the test set. |
| Seed | The number that fixes a run's randomness. We use 0, 1 and 2 and average. |
| Variant | One of four alternatives (A–D) within an experiment. |
| Experiment | One comparison inside a phase, named `phase<N>_<what>`, with its own variants and selection rule. |
| Gate | The checkpoint at the end of a phase where the winner is chosen and the team approves. |
| Probe network | The small fixed classifier used only to compare input representations. |
| cVAE, WGAN-GP | The two kinds of generative model we will train to produce synthetic beats. |
| SMOTE | A classical method that creates new rare examples by interpolating between real ones. |
| Pretraining, fine-tuning | Training first on a large dataset (PTB-XL), then continuing on the target dataset (MIT-BIH). |
| Bootstrap | Estimating uncertainty by recomputing a result on many resampled versions of the data. |
| Smoke run | A quick run on fake data to check that the code works; its numbers mean nothing. |
