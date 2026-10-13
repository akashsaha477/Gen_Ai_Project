# Project guide: rare-arrhythmia detection with generative ECG augmentation

This is the long explanation of the project: what we are trying to find out, how the work is
organised, what each part does, and what is finished. For commands only, see `README.md`. For the
day-by-day log, see `docs/PROGRESS.md`. Last updated 2026-10-13 (Day 7 of 30).

## Contents

1. [The project in one page](#1-the-project-in-one-page)
2. [The data](#2-the-data)
3. [The rules we work by](#3-the-rules-we-work-by)
4. [The plan: six phases](#4-the-plan-six-phases)
5. [What is built, part by part](#5-what-is-built-part-by-part)
6. [What is not built yet](#6-what-is-not-built-yet)
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

**Team.** Four members. Each module has four variants (A–D); each member takes one, and the winner is
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

These come from `docs/RULES.md`. In plain terms:

| Rule | What it means in practice |
|---|---|
| Split by patient | No record is in two of training, validation and test. Anything fitted (generators, oversamplers) sees training records only. |
| The test set is locked | Code cannot read DS2 without an explicit permission, and every access is written to `docs/TEST_ACCESS_LOG.md`. The test set is evaluated once, at the end. |
| Choose on validation, by written rules | Which variant wins is decided by a rule fixed beforehand and applied by code, not by looking at results and picking. |
| Equal budgets | Variants of a module get the same data, seeds, epochs and learning rates. |
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
| 1. Data foundation | 1–3 | data pipeline, evaluation code, choose the input representation (Module 1) | infrastructure built; variants not run |
| 2. Baselines and quality metrics | 4–8 | baseline classifiers (Module 3), metrics for judging synthetic beats (Module 4) | not started, except the PTB-XL arm |
| 3. Generators | 9–14 | cVAE (Module 5) and WGAN-GP (Module 6), synthetic beat banks | not started |
| 4. Integration and ablation | 16–21 | augmentation policy (Module 7), the full comparison grid, the single test evaluation | not started |
| 5. Explainability, notes, demo | 22–26 | saliency maps (Module 8), clinical-note generator (Module 9), demo | not started |
| 6. Report and paper | 27–30 | paper draft, course report, reproducibility package | not started |

We are on Day 7 and about four days behind: Phase 1 was due on Day 3.

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

- All raw-file reading goes through one module. It refuses any DS2 record unless the caller declares
  why and a matching environment flag is set.
- Two kinds of access exist: *construction* (counting labels, building files; no model involved) and
  *evaluation* (the final test, once).
- So far the log has one entry: counting DS2 labels for the dataset table. No model has seen DS2.

### 5.4 Evaluation code (`src/ecgaug/eval/`) — this is Module 2

- **Metrics:** per-class precision, recall and F1, PR-AUC, the AAMI sensitivity and positive
  predictivity used in older papers, and the confusion matrix. Tested against hand-computed cases.
- **Bootstrap:** confidence intervals that resample whole patients, not single beats, because beats
  from one patient are not independent. Includes paired comparisons between two methods, p-values and
  the Holm correction for multiple comparisons.
- **Selection rule:** the tie margin and tie-breaks described in section 3.

### 5.5 Module 1: input representation (`configs/m1_input_representation/`)

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
- Every run gets its own folder under `runs/` with its config, software versions, training curve,
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

### 5.8 PTB-XL pretraining arm (`src/ecgaug/experiments/ptbxl_pretraining.py`)

Asks: does pretraining on a large ECG dataset help rare-beat detection on MIT-BIH?

1. **Pretrain** the model on PTB-XL beats to predict the five superclasses (once per seed).
2. **Fine-tune** it on MIT-BIH: replace the output layer with a 4-class one and train on DS1.
3. **Train the same model from scratch** on MIT-BIH, with the same budget, for comparison.
4. **Decide** with the selection rule. Pretraining has to beat from-scratch by more than the tie margin.

- One command does all four steps, including the PTB-XL preprocessing.
- **State:** built and run end to end on fake data. No real run yet. Results will be marked
  provisional until the input representation is fixed, and the arm uses the small probe network for
  now.

### 5.9 Kaggle setup (`kaggle/`)

- `module1_input_representation.ipynb` and `ptbxl_pretraining.ipynb`: one notebook per job. You set the
  variant letters and run all cells.
- `make_raw_dataset.py`: packs the verified MIT-BIH files into a 66 MB zip to upload once as a private
  Kaggle dataset, so sessions do not re-download from PhysioNet.
- `ingest_runs.py`: brings a result zip from Kaggle back into your clone.
- **State:** rehearsed on a laptop against a fresh clone. Not yet run on Kaggle itself.

---

## 6. What is not built yet

| Part | Phase | What it will be |
|---|---|---|
| Module 1 results and Gate 1 | 1 | the four variant runs, then the choice of input representation and the frozen dataset |
| Module 3: baseline classifiers | 2 | shallow CNN, 1D ResNet, BiLSTM, CNN+BiLSTM; pick one CNN and one recurrent model |
| Module 4: synthetic-quality metrics | 2 | DTW, spectral distance, a Fréchet-style distance, train-on-synthetic test |
| Modules 5 and 6: generators | 3 | conditional VAE and conditional WGAN-GP, four variants each |
| Module 7: augmentation policy and the grid | 4 | how many synthetic beats to add; then 96 runs comparing all conditions |
| Final test evaluation | 4 | the single DS2 evaluation with patient-level statistics |
| Module 8: explainability | 5 | Grad-CAM, integrated gradients and others, compared for faithfulness |
| Module 9: clinical notes | 5 | template and LLM note generators with an automatic fact check |
| Demo, report, paper | 5–6 | command-line demo, course report, paper draft |

---

## 7. How to run things

**Set up once**

```
uv sync
uv run pytest
uv run python -m ecgaug.data.raw      # download and verify MIT-BIH (about 30 minutes)
```

**Module 1, your variant**

```
uv run python -m ecgaug.experiments.input_representation --config configs/m1_input_representation/A_short_window.yaml
```

Add `--smoke` for a one-minute check on fake data, or `--check` to build the inputs without training.

**PTB-XL pretraining arm**

```
uv run python -m ecgaug.experiments.ptbxl_pretraining --config configs/ptbxl_pretraining/A_short_window.yaml --ptbxl-download
```

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
| `docs/RULES.md` | the project rules and data protocol |
| `docs/PLAN.md` | every phase, module and variant in detail |
| `docs/PAPER_TRACK.md` | research questions, hypotheses, statistics plan, figures |
| `docs/PROGRESS.md` | what was done each day |
| `docs/DECISIONS.md` | every decision and plan change, with the reason |
| `docs/PREREGISTRATION.md` | the selection and statistics rules written out |
| `docs/briefs/` | one page per Module 1 variant for the person running it |
| `docs/TEST_ACCESS_LOG.md` | every access to the test set |
| `configs/` | experiment settings |
| `src/ecgaug/data/` | MIT-BIH pipeline, test lock, PTB-XL preprocessing |
| `src/ecgaug/eval/` | metrics, bootstrap, selection rule |
| `src/ecgaug/models/`, `train/` | the probe network and the training loop |
| `src/ecgaug/experiments/` | the runners: `input_representation`, `ptbxl_pretraining` |
| `kaggle/` | notebooks, scripts and guide for Kaggle |
| `results/data/` | counts, checks and validation reports produced by code |
| `results/validation/` | variant results (empty until variants are run) |
| `runs/` | one folder per training run |
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
| Pretraining arm built before Gate 1 | To use Kaggle time now. Its results are provisional until the input representation is fixed. |

---

## 10. What has been checked and what has not

**Checked**

- All MIT-BIH files match PhysioNet's checksums; beat counts match the published table.
- The four input variants build correctly on real DS1 data and produce the same 50,969 beats.
- The R-peak detector and the padding choice were measured on DS1 (section 5.7).
- Both runners work end to end on fake data, also through the Kaggle runner script in a fresh clone.
- 159 tests pass, including tests that the test set cannot be read by accident.

**Not checked yet**

- No model has been trained on real data. There are no results of any kind.
- Nothing has been run on Kaggle; run times are unknown.
- The full PTB-XL dataset has not been processed.
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
| Variant | One of four alternatives (A–D) within a module. |
| Module | One piece of the project with its own four variants and selection rule. |
| Gate | The checkpoint at the end of a phase where the winner is chosen and the team approves. |
| Probe network | The small fixed classifier used only to compare input representations. |
| cVAE, WGAN-GP | The two kinds of generative model we will train to produce synthetic beats. |
| SMOTE | A classical method that creates new rare examples by interpolating between real ones. |
| Pretraining, fine-tuning | Training first on a large dataset (PTB-XL), then continuing on the target dataset (MIT-BIH). |
| Bootstrap | Estimating uncertainty by recomputing a result on many resampled versions of the data. |
| Smoke run | A quick run on fake data to check that the code works; its numbers mean nothing. |
