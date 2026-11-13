# Project guide: rare-arrhythmia detection with generative ECG augmentation

This is the long explanation of the project: what we are trying to find out, how the work is
organised, what each part does, and what is finished. For commands only, see `README.md`. For the
day-by-day log, see `docs/PROGRESS.md`. Sections 1 to 5.14 were written on 2026-10-16 (Day 10 of 30), before the
experiments ran, and describe the design. Section 5.15 (the post hoc analyses) and section 6 (what has run) were added
on 2026-11-09. The checklist in `docs/CHECKLIST.md` gives the state and the result of every item.

## Contents

1. [The project in one page](#1-the-project-in-one-page)
2. [The data](#2-the-data)
3. [The rules we work by](#3-the-rules-we-work-by)
4. [The plan: six phases](#4-the-plan-six-phases)
5. [What is built, part by part](#5-what-is-built-part-by-part)
6. [What has run](#6-what-has-run)
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

These come from `docs/RULES.md`. In plain terms:

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
| 1 data | 1–3 | data pipeline, evaluation code, choose the input representation (phase1_input_representation) | done; gate approved (variant C) |
| 2 baselines | 4–8 | baseline classifiers (phase2_baseline_classifiers), metrics for judging synthetic beats (phase2_quality_metrics), PTB-XL pretraining (phase2_ptbxl_pretraining) | done; gate approved |
| 3 generators | 9–14 | cVAE (phase3_cvae) and WGAN-GP (phase3_wgan_gp), synthetic beat banks | done; banks frozen |
| 4 comparison | 16–21 | augmentation policy (phase4_augmentation_policy), the full comparison grid, the single test evaluation | done; final evaluation run once |
| 5 explanations | 22–26 | saliency maps (phase5_explainability), clinical-note generator (phase5_clinical_notes), demo | done; gate approved |
| 6 paper | 27–30 | paper draft, course report, reproducibility package | done: course report in `report/` (the paper draft is kept local); follow-up analyses P1-P23 |

The State column was updated on 2026-11-12, when every phase was done; the rest of this section describes the plan as made.

**Where are we?** `python -m ecgaug.workflow status` prints every step, ticked or not, and the next
action. The code enforces the order: Phase 1 gate needs all four phase1_input_representation results, and the phase2_ptbxl_pretraining experiment
needs an approved phase 1 gate (or `--preview`, which marks the result provisional).

**The phase loop.** One person builds the shared code for a phase. Each member runs one variant on
their own branch. A gate report applies the selection rule. The team approves, and the next phase
starts. Details are in `docs/PLAN.md`.

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
- Error analysis from saved predictions only; a demo for one validation beat in three parts: the offline analysis by our
  models with no LLM (prediction, saliency by segment, RR intervals, the beat in all 36 saved runs, a figure), a review
  by Gemini written as a cardiologist's short report from that result only (added after the gate for the demo, A44), and
  each claim of the review checked against it. Without `GEMINI_API_KEY` the same report is filled in offline and the
  demo says why; `--notes gate` writes the phase 5 note variant (C) instead.

### 5.14 Phase 6: the paper and the course report (`paper.py`, `report_text.py`, `report_folder.py`, `report/`)

- `python -m ecgaug.paper` writes every table and number of `paper/main.tex` (kept local) from the result files, and
  the course report's share of them into `report/generated/` and `report/results.md`; missing results show as red
  placeholders, so every version always compiles. The course report is `report/report.pdf` (29 pages),
  `report/report_short.pdf` (20 pages) and `report/report_brief.pdf` (a 10-page short paper). `scripts/reproduce.sh` lists
  the whole pipeline in order.

### 5.15 After the final evaluation: post hoc analyses P1-P23 (`src/ecgaug/experiments/phase6_*.py`)

The single final evaluation (2026-11-04) is the study's primary result and stays so: the cnn without augmentation
reached a test rare-class macro-F1 of 0.088, and no generator was detectably better than the simpler options. The
analyses below came after it, mostly to find out why test scores fell so far below validation and what would raise
them. Each was pre-registered as a new version of `docs/PREREGISTRATION.md` §10 (v4 to v17) before it ran, reads the
test records at most once (logged in `docs/TEST_ACCESS_LOG.md`), and is reported as post hoc; none replaces the primary
result.

| | Experiment | What it asks |
|---|---|---|
| P1 | `phase6_rr_reference` | how fixed RR-feature reference classifiers compare with the cnn |
| P2 | `phase6_external_validation` | how the frozen models do on two databases never used before (INCART, SVDB) |
| P3 | `phase6_extra_seeds` | whether the final evaluation holds with seeds 3 and 4 |
| P4 | `phase6_audit_check` | two checks of the leakage audit |
| P5 | `phase6_rr_baseline` | the full grid again with an RR ratio relative to the patient's baseline |
| P6 | `phase6_patient_template` | the full grid again with patient-relative beats (beat minus the patient's template) |
| P7 | `phase6_relative_cnn` | the cnn alone with patient-relative beats, the full comparison |
| P8 | `phase6_augmentation_mechanism` | why augmentation does not carry over to new patients |
| P9 | `phase6_patient_transfer` | rare-class beats moved onto other patients as augmentation |
| P10 | `phase6_patient_adaptation` | per-patient batch-normalization statistics at test time |
| P11 | `phase6_more_patients` | the P7 cnn trained on more patients (DS1 plus INCART and SVDB) |
| P12 | `phase6_relative_generators` | the generators retrained on what the P7 cnn sees |
| P13 | `phase6_record_rr` | RR intervals over the mean RR of their recording; Huang et al.'s SVEB rule on top |
| P14 | `phase6_icentia` | the P13 cnn trained on many more patients, from Icentia11k |
| P15 | `phase6_diffusion` | a class-conditional diffusion model as a third generator family |
| P16 | `phase6_af_rule` | Huang et al.'s SVEB rule with its S calls undone inside atrial fibrillation |
| P17 | `phase6_diffusion_long` | P15's diffusion model trained five times longer |
| P18 | `phase6_seed_ensemble` | each condition's class probabilities averaged over ten independently trained runs, with the SVEB rule and the AF veto on top |
| P19 | `phase6_fresh_patients` | P18's frozen pipeline on 1,000 Icentia11k patients no analysis had used, with no test record read |
| P20 | `phase6_rule_tuning` | the SVEB rule's and the veto's thresholds chosen on P14's 400 Icentia11k patients, scored on P19's patients and once on the test records |
| P21 | `phase6_physics_prior` | McSharry et al.'s ECG simulator as a prior for the cVAE and WGAN-GP (SimGANs' Euler loss), and physiological fusion beats; the loss barely moved the generators, no detectable change on P19's patients or DS2 |
| P22 | `phase6_bigger_classifier` | phase 2's ResNet1D (6.7 times the cnn's parameters) in P18's pipeline against P18's cnn ensembles; better S F1 on P19's patients with augmentation, no detectable change on DS2 |
| P23 | `phase6_physics_fitted` | P21 with the simulator fitted to each generated beat (the loss is what it cannot explain); a small S gain for the cVAE on P19's patients, no detectable change on DS2 |

The highest post hoc test score is the cVAE with Huang et al.'s SVEB rule and the atrial-fibrillation veto, its class
probabilities averaged over ten runs (P18): rare-class macro-F1 0.587 (S F1 0.647, F F1 0.527). Against the same set-up with
single runs (P16, 0.557) that is not a detectable change (Holm p 1.000). On 977 new Icentia11k patients (P19) the same pipeline reached S F1 0.294 and V F1 0.648, below its DS2 scores; there the ensemble alone (S F1 0.348) did better than with the SVEB rule and the veto on top. Choosing the two thresholds on 400 other Icentia11k patients (P20) raised S F1 on those 977 patients in all six conditions (cVAE 0.294 to 0.414, Holm p 0.001), but on the test records it lowered rare-class macro-F1 in five of six, none detectably (cVAE 0.587 to 0.459); the test records' S beats sit mostly in record 232, and none of the 388 tuning patients is like it. The result of every analysis is in the `docs/CHECKLIST.md` checklist, the
detail in `docs/PROGRESS.md`, and the before-and-after figures in `figures/phase6_*/`.

---

## 6. What has run

Phases 1 to 5, the full grid, the single final evaluation (2026-11-04), the leakage audit and the post hoc
analyses P1-P23 (section 5.15) have all run. The checklist in `docs/CHECKLIST.md` gives the state of every item, including
any approval still open; `python -m ecgaug.workflow status` prints it.

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

**Phases 2-6:** the smoke commands are in `README.md` ("Reproducing"), the full commands in `docs/CHECKLIST.md` and in each variant's brief.

**On Kaggle:** follow `kaggle/README.md`.

A real run needs two things: the approved validation split (`configs/validation_split/approved.yaml`,
already in the repo) and no uncommitted code changes.

---

## 8. Where everything lives

| Path | Contents |
|---|---|
| `README.md` | overview for readers: question, setup, system, results, demo, how to reproduce |
| `docs/CHECKLIST.md` | every item of every phase with its command and outcome, and the working notes |
| `PROJECT_GUIDE.md` | this file |
| `CODE_GUIDE.md` | what each code file and function does |
| `docs/RULES.md` | the project rules and data protocol |
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
| The phase 1 gate picked variant C (approved) | Best S F1 by a clear margin; RR timing is what makes S beats detectable (docs/PHASE1_REPORT.md). |
| Phases 2-6 built before their gates (A14) | At Akash's instruction; official runs waited for every gate, and A14 lists the design choices that were confirmed. |

---

## 10. What has been checked and what has not

Updated on 2026-11-12 for the final submission; `docs/PROGRESS.md` has every check with its date.

**Checked**

- All MIT-BIH files match PhysioNet's checksums; beat counts match the published table.
- The frozen dataset (`data/processed/v1/trainval.npz`) and the six synthetic banks match the SHA-256 hashes in
  `results/manifest.json`. The phase 2 checkpoints are kept in the Kaggle run bundles; their hashes in the manifest match
  the ones each run recorded.
- Every phase ran on the real data (training on Kaggle), and every gate was approved before the next phase; the final evaluation
  read the test records once, and every later read is logged in `docs/TEST_ACCESS_LOG.md`.
- The full test suite passes in a fresh clone on Kaggle at the submitted code (9f13bfb): 447 passed, 1 skipped (the paper
  draft is kept local), 0 failed. It includes the dress rehearsal, which runs every phase, gate, the final evaluation,
  phase 5, RQ3, the leakage audit and the paper on fake records, and tests that the test set cannot be read by accident.
- Every smoke command in `README.md` runs end to end in a fresh clone and leaves the tracked files unchanged;
  `python -m ecgaug.paper` on a fresh clone rewrites `report/generated/` and `report/results.md` byte for byte; the demo
  runs on a real validation beat with the frozen model, and its note comes from Gemini (variant C) when the key is set.
- Both versions of the course report compile from the committed files with no undefined reference and no overfull line,
  and every number typed by hand in them was checked against the configs, the code or the documents.

**Open points**

- Positive predictivity for V follows the AAMI rule that fusion beats called V are neither penalised nor rewarded, so
  they do not count as false positives. Huang et al. (2014) count them (their published confusion matrix gives
  3025 / 3329 = 0.909), so the literature table compares slightly different definitions; on the seed-mean confusion
  matrices of the twelve pre-registered models the two definitions differ by at most 0.017, and no comparison in the report changes direction.
- PTB-XL file checksums were not compared on Kaggle (PhysioNet's list could not be fetched there).
- The test set holds 22 patients, and the rare beats come from a few of them, so intervals are wide (the report's
  limitations section).

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
