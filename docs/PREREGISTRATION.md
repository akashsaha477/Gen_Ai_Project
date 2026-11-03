# PREREGISTRATION

**Version:** 2 (committed 2026-11-01 at the phase 2 gate, see §10)
**Status:** reference document (not enforced by code since 2026-10-13)
**Drafted:** 2026-10-08 (Day 2)

**How it is used.** This file describes the rules the code applies (selection metric, tie rule,
statistics). The runner no longer checks it; it only needs `configs/validation_split/approved.yaml`.

Sources: `docs/RULES.md` §3–4, `docs/PLAN.md` (global conventions, Phase 1), `docs/PAPER_TRACK.md`
(hypotheses, statistics) and the amendments A1–A11 in `docs/DECISIONS.md`.
Items marked **(new)** were not in the kit and need the team's explicit agreement.

## 1. Data and protocol

- **Database:** MIT-BIH Arrhythmia Database 1.0.0 from PhysioNet, lead MLII selected by signal name.
- **Split (de Chazal et al., 2004):** DS1 for training and validation, DS2 for test, the four paced
  records excluded. The record lists are in `docs/RULES.md` §4 and `src/ecgaug/data/records.py`.
- **Known overlap:** records 201 (DS1) and 202 (DS2) are the same subject. The standard split is kept
  for comparability. The overlap is disclosed in the paper, and test results are also reported without
  record 202 (§7).
- **Labels:** reference annotations mapped to AAMI classes N, S, V, F, Q as in `docs/RULES.md` §4.
- **Q policy:** Q beats are excluded from the arrays and from every metric. Their counts are reported.
- **Unmapped QRS annotations (new):** QRS-type annotations outside the AAMI map, such as the
  ventricular flutter waves `!` of record 207, are excluded and counted the same way.
- **RR intervals (new):** measured between consecutive QRS-type annotations (WFDB `isqrs`), including
  Q beats and flutter waves, before any class is dropped. `rr_ratio` is `rr_prev` divided by the mean
  of the up to 10 RR intervals that precede it, and is 1 for a beat with no earlier interval.
- **Usable beats (new):** a beat is kept if the window from 0.4 s before to 0.6 s after the R peak lies
  inside the record and both RR intervals exist. The rule is the same for every representation, so
  the beat set never depends on the input representation.
- **Counts:** `results/phase1_data/record_counts.csv` and `results/phase1_data/sanity_check.json`, produced by
  `python -m ecgaug.data.counts`.

## 2. Validation split

Options and their DS1 per-record counts are in
`results/phase1_data/val_options.md`. The choice is justified from DS1 counts alone; DS2 counts are not read
by the proposal tool.

- Chosen option: V1
- Validation records: 115, 118, 201, 205, 223
- Selection metric given by §3: S F1, with V F1 as tie-break (validation holds 27 F beats)
- Approved by: Akash Saha on 2026-10-13

Generators, oversamplers and any fitted preprocessing never see validation records.

## 3. Metrics

- **Primary metric (test):** rare-class macro-F1, the mean of the F1 of S and the F1 of F.
- **Selection metric (validation):** the primary metric, if validation holds at least **100 F beats
  from at least 2 records (new)**. Otherwise the fallback: S F1, with V F1 as tie-break. The numbers
  are chosen so that one F beat moves F recall by at most 0.01, below the tie margin of §4. The
  selection metric is fixed once, at approval of §2, and is the same for every experiment.
- **Secondary metrics:** per-class precision, recall and F1; one-vs-rest PR-AUC; AAMI-style Se and +P
  for S and V; N recall; macro-F1 over N, S, V, F.
- **Decision rule:** argmax of the class probabilities in every condition. No threshold is tuned.
  PR-AUC for S and F is read next to F1 to separate ranking quality from operating-point shifts.
- **Undefined values:** a class with no true beats has no recall or F1 (NaN, not 0). A class that is
  present but never predicted has F1 = 0.

## 4. Selection conventions (all experiments)

- **Seeds:** [0, 1, 2]. The selection value of a variant is the mean over seeds.
- **Tuning grid:** learning rate in {1e-3, 3e-4}, nothing else. **(new)** For each variant, every seed
  is trained at both rates, the rate with the higher seed-mean selection metric is used, and both are
  reported.
- **Early stopping:** patience 5 on the selection metric, evaluated on validation after every epoch.
  The best epoch's checkpoint is kept. From phase 2 on, no early stop before epoch 10 (`min_epochs: 10`,
  amendment of 2026-10-26, §11); phase 1 keeps the original rule.
- **Tie margin (new):** two variants are tied if their seed-mean selection metrics differ by at most
  max(0.02, pooled SD), where pooled SD = sqrt((s_a² + s_b²) / 2) and s is the sample SD over seeds.
- **Tie-break order (new):**
  1. Take the variant with the highest seed-mean selection metric; every variant within the tie
     margin of it is tied with it.
  2. Under the fallback metric, keep the tied variants whose V F1 is within the tie margin of the
     best V F1 among them.
  3. Among the remaining variants the simplest wins, as defined per experiment; then lower wall-clock.
- **"Best" always means validation.** Any "best" condition, checkpoint, strength or variant is the one
  with the highest validation selection metric, recorded in `docs/DECISIONS.md` before `/final-eval`.
- **Equal budgets:** variants of an experiment get identical data, seeds, epoch cap, tuning grid and
  early-stopping rule. Deviations are flagged in the gate report.
- **Losing variants are kept** and reported in the variation-study tables.
- The rule is applied by `ecgaug.eval.selection.select_winner`. If it is ambiguous for the actual
  results, the gate stops and asks.

## 5. phase1_input_representation (input representation)

- **Variants:** A, B, C and D as defined in `docs/PLAN.md` and `configs/phase1_input_representation/`. The band-pass of B is a
  4th-order Butterworth applied forward and backward **(new: the order was unspecified)**. D resamples
  the continuous record to 125 Hz with a polyphase filter before windowing **(new)**.
- **Probe:** the fixed probe CNN of `src/ecgaug/models/probe.py`, unweighted cross-entropy, AdamW,
  batch 256, weight decay 0.01, at most 40 epochs **(new: epoch cap)**; budget in
  `configs/phase1_input_representation/shared_budget.yaml`.
- **Selection:** §4 with the selection metric of §3. Simplicity = fewer input values per beat:
  D 63, A 180, C 183, B 360.
- The winner defines contract v1. If C wins, every downstream classifier takes the RR side input and
  synthetic beats get RR features sampled from the class-conditional empirical distribution of
  training beats; this is reported as a limitation.

## 6. Hypotheses

- **H1 (confirmatory):** a generative condition (cVAE or WGAN-GP) improves test rare-class macro-F1
  over no augmentation for at least one architecture: positive difference with Holm-adjusted two-sided
  bootstrap p < 0.05 within the confirmatory family (§7).
- **H1c (secondary):** generative augmentation beats SMOTE, random oversampling and class-weighted loss
  on the primary metric.
- **H1b, H2, H3, H4 (exploratory):** as stated in `docs/PAPER_TRACK.md`. They are reported with
  effect sizes and intervals and are not used for significance claims.

## 7. Conditions, comparisons and statistics

- **Conditions:** none, class-weighted cross-entropy, random oversampling of real minority beats,
  SMOTE, cVAE, WGAN-GP. Each condition with strengths enters the comparisons at the strength chosen
  on validation.
- **Confirmatory family (4 tests):** cVAE vs none and WGAN-GP vs none, for each of the two
  architectures.
- **Secondary family (8 tests):** cVAE vs WGAN-GP, best generative vs SMOTE, best generative vs
  class-weighted, best generative vs random oversampling, for each architecture. "Best generative" is
  fixed on validation.
- **Resampling:** cluster bootstrap over the 22 test records with replacement, 10,000 replicates,
  metrics on the pooled beats of each replicate, seeds averaged inside each replicate, the same
  resampled records for both conditions of a comparison. Beat-level bootstrap is not used.
- **Undefined replicates:** a replicate in which S or F has no true beats is excluded from intervals
  and p-values. The excluded fraction is reported.
- **p-value:** two-sided, p = min(1, 2 · min(#{Δ* ≤ 0} + 1, #{Δ* ≥ 0} + 1) / (B + 1)) over the B
  defined replicates.
- **Multiplicity:** Holm–Bonferroni on the p-values within each family at α = 0.05. Unadjusted 95%
  percentile intervals are reported next to the adjusted p-values.
- **Reported with every interval:** the fraction of replicates in which each class is absent, and, for
  S and F, the fraction in which the record holding most of that class's test beats is absent.
- **Sensitivity analyses:** leave-one-record-out for S and F; per-record results; all test results
  recomputed without record 202; five seeds if run.
- **Seed variance** (mean ± SD over seeds) is reported separately from record-level uncertainty. No
  significance claim rests on seed-level tests with three seeds.
- Implemented in `src/ecgaug/eval/bootstrap.py` and tested in `tests/test_bootstrap.py`.

## 8. Test-set access

- All raw-file reads go through `ecgaug.data.raw`, which refuses DS2 records without a declared access
  kind and its environment flag. `ecgaug.data.load_test()` is the only reader of `test.npz`.
- **Construction** (`ALLOW_TEST_BUILD=1`): counting DS2 labels and building `test.npz`. No model sees
  the data. Each use is logged.
- **Evaluation** (`ALLOW_TEST=1`): exactly one, by `/final-eval`, after the phase 4 gate approval.
- **Audit:** the optional intra-patient leakage-audit experiment runs only after `/final-eval`, in its own
  directories, logged, and its artifacts cannot enter the main grid.
- Every access is appended to `docs/TEST_ACCESS_LOG.md`. No decision of any kind uses test data.
- After `/final-eval` nothing is retrained, re-selected or re-tuned. A bug found later is documented
  and any re-run is disclosed.

## 9. Generators and RQ3 (rules fixed now, thresholds in v2)

- Each synthetic bank is exported from the final checkpoint of its step budget. Earlier checkpoints
  are scored for RQ3 only and are never selectable.
- Generator selection follows `docs/PLAN.md` Phase 3: hard gates, then the downstream pilot selection
  metric, then the composite quality rank for ties.
- In RQ3, TSTR is analysed separately from the fidelity and diversity metrics, and intervals come from
  resampling generator training runs (variant × seed), not checkpoints.

## 10. Deferred to later versions

- **v2, at the phase 2 gate and before any Phase 3 run:** the composite quality metric, the diversity floor, the
  memorization threshold, the class-weight exponents, and the rule that decides 3 or 5 seeds.
- **v3, before the phase4_augmentation_policy pilot:** the three augmentation strengths, how the chosen policy applies to
  SMOTE and random oversampling, and the quality screen of policy D.
- **Before Phase 5 runs:** the low-confidence rule for notes and the explainability evaluation sets.

Proposed values, as the code reads them now (2026-10-16, DECISIONS A14). Each becomes a rule only when the
team commits it in the version named above:

| item | proposed | read from |
|---|---|---|
| composite quality metric (v2) | the metric families that pass the sanity harness (`quality/sanity.py`) | `phase2-gate` result |
| diversity floor (v2) | 0.7 x real, every generated class and pilot seed | `configs/phase3_*/generator_settings.yaml` |
| memorization ceiling (v2) | 10%, threshold = 5th percentile of leave-one-out real distances | same |
| class-weight exponents (v2) | gamma in {0.5, 0.75, 1} | `workflow.PROPOSED_V2`, written into the full grid |
| seeds (v2) | 3; 5 if the measured run times fit the quota | phase 3 configs `final_seeds` |
| strengths (v3) | B and D: {0.1, 0.25, 0.5} x N; C: {1, 2, 4} x real count; A: none | `configs/phase4_augmentation_policy/` |
| policy pilot seeds and tie margin (v3) | seeds 0, 1, 2; ties within max(0.02, pooled SD) (DECISIONS A19) | `configs/phase4_augmentation_policy/pilot_settings.yaml` |
| SMOTE and random oversampling (v3) | same policy and strengths as the generators; no quality screen | `grid/runner.py` |
| quality screen of D (v3) | of 2 x the needed bank beats, keep those closest to real beats by DTW | same |
| low-confidence rule (phase 5) | top probability below 0.6 | `configs/phase5_clinical_notes/shared_settings.yaml` |
| explainability evaluation set (phase 5) | 200 validation beats per class (all if fewer), seed 0 | `configs/phase5_explainability/shared_settings.yaml` |

**v2, committed 2026-11-01 at the phase 2 gate, before any official phase 3 run:**

| item | rule |
|---|---|
| composite quality metric | equal-weight average of the ranks of DTW and spectral, the families that passed the sanity harness (`results/phase2_gate/report.md`) |
| excluded families | FID-analog (median CV 0.126, limit 0.10). TSTR (never scored: its fixed recipe gives S F1 = 0 on real beats, so every score is NaN; fails by the rule, recipe unchanged, DECISIONS A15.1) |
| diversity floor | 0.7 x real, every generated class and pilot seed |
| memorization ceiling | 10%, threshold = 5th percentile of leave-one-out real distances |
| class-weight exponents | gamma in {0.5, 0.75, 1} |
| seeds | 3 (`final_seeds: [0, 1, 2]`; pilots use 0 and 1). 5 only if approved at the phase 3 gate (docs/RULES.md rule 8) |

**v3, committed 2026-11-03 at the phase 3 gate, before any phase4_augmentation_policy run:**

| item | rule |
|---|---|
| policies | A balance each minority class to the training N count; B each minority class up to a fraction of N; C add a multiple of each minority class's real count; D as B, keeping of 2 x the needed bank beats those closest to real beats by DTW |
| pilot strengths | A none; B 0.5 x N; C 2 x real count; D 0.5 x N |
| grid strengths of the winner | B and D {0.1, 0.25, 0.5} x N; C {1, 2, 4} x real count; A none |
| pilot classifier and seeds | the phase 2 `cnn` (variant A); seeds 0, 1, 2, classifier seed s with bank seed s of both banks |
| pilot score | per seed the mean over the two banks of validation S F1; a policy's score is the mean over seeds |
| constraint | a policy whose mean N recall over the two banks is more than 0.01 below the no-augmentation reference is out (A14.11; per-bank drops are reported, not enforced) |
| selection | best score; ties within max(0.02, pooled SD) go to V F1, then to lower complexity (A 1, B 2, C 2, D 3) (A19) |
| SMOTE and random oversampling | same policy and strengths as the generators; no quality screen |
| seeds of the full grid | 3 (`[0, 1, 2]`); 5 not adopted (decided at the phase 3 gate, DECISIONS A20) |

Each version is committed before the experiments it governs, tagged `prereg-vK` and pushed that day.

## 11. Deviations and amendments

| date | section | change | reason | approved by |
|---|---|---|---|---|
| 2026-10-26 | §4 early stopping | `min_epochs: 10` for every MIT-BIH classifier from phase 2 on | 10 of 24 phase 1 runs stopped at epoch 6 on an all-N model (PHASE1_REPORT §6); fixed before any phase 2 run | Akash Saha (DECISIONS A15), pending team confirmation |
| 2026-11-01 | §9 RQ3 | The separate TSTR analysis has no data: TSTR was not scored in the sanity harness and is not in the composite | Fixed recipe gives S F1 = 0 on real beats (PROGRESS 2026-11-01); to be stated in the paper | Akash Saha (phase 2 gate) |
| 2026-11-03 | §10 v2 diversity floor | cVAE D bank seed 1 has lowest diversity 0.694 (class S), under the 0.7 floor; the bank is kept and frozen | The floor is pre-registered for pilot seeds only; final banks have no rule, and adding one after seeing the result would be post hoc. Disclosed in the paper | Akash Saha (phase 3 gate, DECISIONS A20) |
