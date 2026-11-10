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
- **Post hoc** (`ALLOW_TEST_POSTHOC=1`, v4 to v17): the analyses listed in §10 v4 to v17 only, after `/final-eval`, logged;
  they add evidence beside the primary results and replace none of them.

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

**v4, committed 2026-11-05, after the final evaluation and the leakage audit, before any of the analyses below
(DECISIONS A24).** These are post-hoc analyses asked for to answer expected review objections. None replaces a
reported result: the primary results stay the final evaluation of 2026-11-04 (3 seeds, DS2). Nothing below selects,
tunes or changes a frozen model, strength, policy or rule, and no result below feeds any decision. All are exploratory
and labelled post hoc in the paper.

| item | rule |
|---|---|
| test access | a new kind `posthoc` (`ALLOW_TEST_POSTHOC=1`), allowed only when the log holds the evaluation entry; each DS2 read is logged. Used by P1 and P3 only |
| P1 `phase6_rr_reference` | two fixed reference classifiers with no tuning: `rr_lda`, linear discriminant analysis on (rr_prev, rr_next, rr_ratio); `rr_morph_lda`, the same plus every 10th sample of the frozen representation's z-scored window (18 values). Uniform class priors, features standardised on training beats. Trained on the DS1 training records (the frozen v1 split), deterministic, one fit each |
| P1 report | validation and DS2 rare-class macro-F1, per-class F1, S Se and +P, record-level bootstrap (10,000, seed 0); paired difference (reference minus cnn none, and minus cnn cVAE) with 95% CI, no test. Wording: "the cnn is weaker than the reference" only if the interval of reference minus cnn none excludes zero above |
| P2 `phase6_external_validation` | the 36 frozen checkpoints of the final evaluation and the two P1 references, unchanged, on two databases never used before: INCART (`incartdb`, primary; lead II by name; all records without paced beats) and SVDB (`svdb`, secondary; the first signal, `ECG1`, by name; all records). Signals resampled to 360 Hz (polyphase), annotation positions scaled; RR features in seconds from the annotations at the native rate; the AAMI map, Q policy, usable-beat rule and the frozen representation (v1, variant C) unchanged. Records whose annotations hold a symbol outside the WFDB table are reported and excluded |
| P2 report | per database: per-condition statistics as in §7 with record-level bootstrap; the 4 confirmatory comparisons replicated with Holm within the database. Primary metric rare-class macro-F1; if a database holds fewer than 50 usable F beats, S F1 is reported as its primary metric instead (stated before any model is run on it). "Replicated" means the same Holm verdict as on DS2 |
| P3 `phase6_extra_seeds` | seeds 3 and 4 added for the 12 conditions at the strengths of the phase 4 gate: generator banks for seeds 3 and 4 from the frozen variants (phase3_cvae-D, phase3_wgan_gp-A) at the full budget, then the grid cells; classifier seed s with bank seed s. Seeds 0-2 are the existing checkpoints. 5-seed DS2 statistics with the same families and Holm, as a sensitivity analysis. "Changed" means any Holm verdict differs from the 3-seed result |
| P4 `phase6_audit_check` | the leakage audit repeated on its frozen split (split seed 0, the existing audit banks) with WGAN-GP at the validation-chosen strength (cnn 0.1), and with `none` on two more random splits (split seeds 1 and 2); seeds 0-2; access kind `audit`. Reported beside the audit |

**v5, committed 2026-11-05, after v4's analyses, before any of the work below (DECISIONS A25).** A follow-up study
with one change to the input: the RR ratio is measured against the patient's baseline rhythm instead of the previous
ten beats. It is post hoc. The primary results stay the final evaluation of 2026-11-04 (3 seeds, v1 features) and are
not rewritten; the v2 study is reported beside them, labelled post hoc, with its own verdicts. The motivation is a
diagnostic on DS1 only (no validation-based selection, no test data): with v1's `rr_ratio`, 51% of training S beats
have a ratio below 0.9 (record 209, where S beats come in runs: 53%), because the ten preceding intervals are
themselves short during a run; with the baseline ratio defined below, 87% (record 209: 98%), and N beats are unchanged
(4%). The definition follows the average RR interval of de Chazal et al. (2004), with a median for robustness to runs.

| item | rule |
|---|---|
| P5 `phase6_rr_baseline`, dataset v2 | identical to the frozen v1 dataset except one field: `rr_ratio` = `rr_prev` divided by the median of the RR intervals (between consecutive QRS-type annotations of the beat's own record) whose midpoints lie within 150 s before or after the beat. Annotation times only; no labels. X, y, record, split, r_idx, rr_prev, rr_next and the beat set are checked equal to v1. Saved as `data/processed/v2/trainval.npz` with a contract and hashes |
| DS2 under v2 | the baseline ratios of DS2 beats are computed from DS2 annotations under a logged `construction` access (`ALLOW_TEST_BUILD=1`), saved apart and keyed by (record, r_idx); no model sees them; `test.npz` of v1 is not changed |
| banks | the frozen phase 3 banks (phase3_cvae-D, phase3_wgan_gp-A, seeds 0-2), waveforms unchanged. Each synthetic beat carries the v1 (rr_prev, rr_next, rr_ratio) of a real training beat of its class (§5); its `rr_ratio` becomes that beat's v2 value (exact match on the three float32 values within the class; the first in dataset order on ties; refused if none). Saved and hashed under `data/synthetic/phase6_rr_baseline/` |
| conditions and grid | the phase 4a gate's full grid unchanged (policy D, DTW-screened, pool factor 2; strengths 0.1, 0.25, 0.5; class weights gamma 0.5, 0.75, 1.0; SMOTE k 5; seeds 0-2), the phase 2 gate's cnn and rnn with their settings unchanged (no re-tuning): 96 runs. SMOTE and random oversampling act on the v2 features |
| selection | the phase 4 gate rule exactly as implemented (`workflow.evaluate_phase4_gate`: per architecture and condition the strength with the best validation S F1 seed mean, ties within max(0.02, pooled SD) to V F1 then the lower strength; the best generative condition per architecture likewise), applied by code with no human choice, and written to `results/phase6_rr_baseline/` before any DS2 read |
| test | one `posthoc` access: DS2 predictions of the 36 selected checkpoints with the v2 ratio; the analysis of §7 (confirmatory family of 4 and secondary family, Holm within each, record-level bootstrap 10,000, seed 0) with the wording rules of §7. Also, per architecture and condition, the paired difference v2 minus v1 in rare-class macro-F1 (record-level bootstrap, 95% CI, descriptive, no test) |
| references | `rr_lda` and `rr_morph_lda` of P1 refitted on the v2 training beats, same rules, scored on DS2 in the same access |
| external | INCART and SVDB as in P2, with the baseline ratio computed from the annotations at the native rate with the same window; the 36 v2 checkpoints and the two v2 references; the P2 report and primary-metric rules |
| reporting | the paper keeps every v1 result and verdict; the v2 study gets its own section and verdicts, labelled post hoc. Compute: the full grid on a Kaggle T4 (the v1 grid took 1.2 GPU-hours) |

**v6, committed 2026-11-05, after v5's DS2 read, before any of the work below (DECISIONS A26).** A second follow-up
with one more change to the input: each beat is given relative to the patient's own usual beat. It is post hoc. The
primary results stay the final evaluation of 2026-11-04; v5 and v6 are reported beside them, labelled post hoc. The
motivation is an exploratory look at DS1 made before this version (validation data, no test data, no selection):
the frozen v1 cnn without augmentation (seed 0) reaches S PR-AUC 0.92 on the training records and 0.20 on the
validation records (V: 0.995 and 0.61); all 96 validation S beats of record 118 (right bundle branch block) and 83 of
128 of record 201 are called V or N, which points to the networks learning the training patients' beat shapes. An
LDA with uniform priors on the v2 RR features reached validation S F1 0.194, with every 10th sample of the raw beat
added 0.073, and with every 10th sample of the beat minus the patient's local median beat 0.275. A quick cnn check of
the same inputs was stopped before it gave a number. These looks are disclosed here; no rule below depends on them.

| item | rule |
|---|---|
| P6 `phase6_patient_template`, dataset v3 | dataset v2 (v5) with one more change: `X` is replaced by `X` minus the beat's template, the element-wise median of the z-scored windows (the frozen representation) of every QRS-type annotation of the same record whose R peak lies within 150 s of the beat's R peak and whose window fits in the record (the beat itself included). Annotation times and the signal only; no labels. Everything else equals v2. Saved as `data/processed/v3/trainval.npz`, hashed in `results/phase6_patient_template/data.json` |
| banks | the frozen phase 3 banks, matched to a real training beat of their class exactly as in v5; each synthetic beat takes that beat's v2 ratio and has that beat's template subtracted from its waveform. Saved and hashed under `data/synthetic/phase6_patient_template/` |
| grid | the phase 4a gate's full grid unchanged (as in v5), the phase 2 gate's cnn and rnn with their settings unchanged, on v3: 96 runs. SMOTE and random oversampling act on the v3 beats |
| adoption (validation) | the 6 cells without augmentation (cnn, rnn; seeds 0-2) run first. v3 is adopted if the mean over the two architectures of (v3 minus v2 seed-mean validation selection value, S F1) is larger than max(0.02, pooled seed SD), the pooled SD being the root mean of the four seed variances (v2, v3; cnn, rnn), the v2 values being the v5 grid's cells. Applied by code (`adoption`), written to `results/phase6_patient_template/adoption.json`. If v3 is not adopted, the other 90 cells are not run, nothing is selected, DS2 is not read, and the outcome is reported |
| selection | as in v5: the phase 4 gate rule by `workflow.choose_strengths`, written before any DS2 read |
| test | one `posthoc` access, on Kaggle: the frozen DS2 beats (rebuilt from the raw records if absent, checked against the frozen hash), their v2 ratios and templates; DS2 predictions of the 36 selected v3 checkpoints; the analysis of §7 with its wording rules; per architecture and condition the paired differences v3 minus v1 and v3 minus v2 in rare-class macro-F1 (record-level bootstrap, descriptive). The v1 and v2 predictions are recomputed from their checkpoints, and the run refuses unless the recomputed v1 scores equal the final evaluation's |
| external | INCART and SVDB as in P2 and P5, with templates from each record's signal resampled to 360 Hz and all its QRS annotations; the 36 v3 checkpoints |
| diagnostic (validation only) | descriptive, no decision: per architecture, the seed means of per-class PR-AUC and F1 on the training and validation beats of the no-augmentation networks of v1 (the final evaluation's), v2 (v5) and v3, per-record validation S recall, and LDAs with uniform priors on three fixed feature sets (v2 RR; v2 RR plus every 10th raw sample; v2 RR plus every 10th sample of the v3 beat) |
| reporting | its own section and verdicts, labelled post hoc, beside v1 and v5. Compute: one Kaggle T4 session (the v5 grid took about 2.8 GPU-hours); CPU sessions for the rest |

**v7, committed 2026-11-05, after v6's adoption result, before any of the work below (DECISIONS A27).** Two analyses,
both post hoc. P7: v6's adoption rule averaged over the two architectures and did not adopt v3 (mean difference
-0.0015 against a margin of 0.026), but per architecture v3 raised the cnn's validation S F1 (seed means 0.472 against
0.350; seed SDs 0.021 and 0.042) and lowered the rnn's (0.181 against 0.306). P7 takes the cnn with v3 inputs, a choice
made on validation after seeing v6's result and reported as such, and repeats the comparison for the cnn alone, so that
the study's question is also answered on a baseline that is stronger on validation. P8 describes, on DS1 only, why
augmentation does not carry over to new patients.

| item | rule |
|---|---|
| P7 `phase6_relative_cnn`, data | dataset v3 and its banks exactly as v6 builds them (hash-checked against `results/phase6_patient_template/data.json`) |
| P7 grid | the phase 4a gate's full grid restricted to the cnn of the phase 2 gate, settings unchanged: none; class-weighted gamma 0.5, 0.75, 1.0; random oversampling, SMOTE, cVAE, WGAN-GP at 0.1, 0.25, 0.5; seeds 0-2; 48 runs, all run in this experiment (v6's 3 cnn cells are not reused) |
| P7 selection | the phase 4 gate rule by `workflow.choose_strengths` on the 48 cells, written before any DS2 read |
| P7 test | one `posthoc` access, as in v6: frozen DS2 beats (rebuilt and hash-checked if absent), v2 ratios, templates; the analysis of §7 for the cnn alone, with the families `final_eval.comparisons` gives for one architecture (confirmatory: cVAE vs none, WGAN-GP vs none; secondary: cVAE vs WGAN-GP and the best generative condition vs SMOTE, class-weighted and random oversampling), Holm within each family, the wording rules of §7; per condition the paired differences P7 minus v1 and P7 minus v2 (descriptive); refused unless the recomputed v1 scores equal the final evaluation's |
| P7 external | INCART and SVDB as in v6, with the 18 selected checkpoints |
| P8 `phase6_augmentation_mechanism`, attribution | for each frozen bank (cVAE-D, WGAN-GP-A; seeds 0-2) and each class S, V, F, every synthetic beat is attributed to the training record of its nearest real training beat of the same class (Euclidean distance between the z-scored windows); likewise the beats the policy adds at the strength the phase 4 gate chose for each architecture. References: the records of the real training beats themselves, each real training beat's nearest other real training beat (same record or not), and each validation beat's nearest real training beat. Reported per class: share per record and the effective number of records, exp(entropy) |
| P8 gap | for every condition of the final evaluation (v1, the gate's strengths, seeds 0-2) and both architectures: seed means of S and V PR-AUC on the real training beats and on the validation beats, and their difference |
| reporting | own sections, labelled post hoc, beside v1, v5 and v6; descriptive only for P8. Compute: one Kaggle T4 session for the 48 P7 runs (v6's cells took about a minute each; estimated well under 2 GPU-hours); CPU sessions for everything else |

**v8, committed 2026-11-05, after v7's results, before any of the work below (DECISIONS A28).** One analysis, post hoc.
P8 found that the networks fit the shapes of the training patients, that the generators reproduce the same patients
(record 209: 0.60 of real and 0.57-0.65 of synthetic S beats), and that validation patients resemble training patients
with few S beats; it also found that the quality screen narrows the patients behind the added beats. P9 asks whether
augmentation helps when it adds patients rather than beats.

| item | rule |
|---|---|
| P9 `phase6_patient_transfer`, transfer | a rare-class training beat x of patient p is written as its template t_p (v6's definition, ±150 s, no labels) plus the deviation d = x - t_p; the transferred beat is zscore(t_q + d), where q is a training record other than p drawn uniformly among the training records and t_q is the template of a random N beat of q; rr_prev and rr_next are multiplied by m_q / m_p, the patients' median RR at the two beats (rr_prev / v2 rr_ratio), and rr_ratio is unchanged. Training records only |
| conditions | `none`; `transfer_real` (deviations of real training beats of the class, drawn with replacement); `transfer_cvae` and `transfer_wgan_gp` (deviations of the frozen bank beats, each minus the template of the real beat whose v1 RR values it carries, v5's matching). Banks of 18,000 beats per rare class per seed (bank seed = classifier seed), built and hashed by code |
| data and grid | dataset v2 (v5, hash-checked); the phase 2 gate's cnn and rnn with their settings; strengths 0.1, 0.25, 0.5 under the `fraction_of_n` policy, unscreened (the screen would keep beats that look like the existing patients, which P8 found narrows them); seeds 0-2; 60 runs |
| selection | the phase 4 gate rule by `workflow.choose_strengths` (strength per condition and architecture), written before any DS2 read |
| test | one `posthoc` access on Kaggle (DS2 rebuilt and hash-checked, v2 ratios). Confirmatory family (Holm, 4 tests): per architecture, transfer_cvae vs none and transfer_wgan_gp vs none, in rare-class macro-F1, record-level bootstrap 10,000, seed 0, with the wording rules of §7. Secondary family (Holm, 4 tests): per architecture, transfer_real vs none, and the generative transfer condition with the better validation seed mean vs transfer_real. Descriptive: every P9 condition minus v5's none, cVAE and WGAN-GP of the same architecture; the recomputed v5 none score is reported beside v5's |
| external | INCART and SVDB as in P5, the P9 checkpoints, the confirmatory family within each database |
| reporting | its own section, labelled post hoc; the primary results stay the v1 final evaluation. Compute: one Kaggle T4 session (estimated under 2 GPU-hours); CPU sessions for the test and external parts |

**v9, committed 2026-11-05, after v8's results, before any of the work below (DECISIONS A29).** One analysis, post hoc.
P6 to P9 point to patient variation rather than class imbalance as the bottleneck: patient-relative inputs raised the
cnn's test rare-class macro-F1 from 0.088 to 0.371, while no rebalancing method was detectably better than none. P10
applies the same idea inside the network without retraining: adaptive batch normalization (Li et al., 2016), in which
each batch-normalization layer of a frozen checkpoint normalizes with the statistics of the patient being classified.

| item | rule |
|---|---|
| P10 `phase6_patient_adaptation`, models | the frozen checkpoints of the final evaluation (v1 inputs; cnn and rnn; the 12 chosen conditions) and of P7 (cnn, v3 inputs; the 6 chosen conditions); nothing is retrained, re-selected or re-tuned |
| adaptation | each record separately: the running statistics of every BatchNorm1d layer of a copy of the checkpoint are reset and recomputed in one forward pass over all of that record's beats in the array being scored (training-mode statistics, cumulative average), then the record's beats are classified in evaluation mode. No labels; records are adapted independently. The cnn has three feature-level layers and one on the RR inputs, the rnn only the one on the RR inputs |
| adoption (validation) | per (version, architecture) = v1/cnn, v1/rnn, P7/cnn, on the no-augmentation checkpoints (seeds 0-2): adaptation is adopted if the seed-mean validation S F1 with adaptation exceeds that without by more than max(0.02, root mean of the two seed variances). Written to `results/phase6_patient_adaptation/adoption.json` before any DS2 read. If no pair is adopted, DS2 is not read and the outcome is reported |
| test | one `posthoc` access on Kaggle (DS2 rebuilt and hash-checked; v2 ratios and templates for P7). For every adopted pair, every chosen condition with adaptation. Family "adaptation" (Holm): no augmentation with adaptation vs without, per adopted pair. Family "augmentation under adaptation" (Holm): cVAE vs none and WGAN-GP vs none, both with adaptation, per adopted pair. Rare-class macro-F1, record-level bootstrap 10,000, seed 0, the wording rules of §7. The recomputed scores without adaptation are reported against the earlier summaries |
| external | INCART and SVDB, each record adapted on its own beats; the adopted pairs; the same two families within each database (P2 primary-metric rule) |
| reporting | its own section, labelled post hoc. CPU only; no GPU |

**v10, committed 2026-11-06, after v9's results and the A30 fix, before any of the work below (DECISIONS A31).** One
analysis, post hoc. Every analysis so far points to the few training patients as the limit (record 209 holds 0.60 of
the training S beats, record 208 0.96 of the F beats; the networks fit the training patients; the generators copy them).
P11 tests the remedy directly: train on more patients.

| item | rule |
|---|---|
| P11 `phase6_more_patients`, data | dataset v3 for DS1 exactly as v6 builds it (hash-checked), plus every record of INCART and SVDB in v3 form (v2 ratio from the annotations at the native rate, template from the signal resampled to 360 Hz; P2's beat set). In INCART and SVDB every S, V and F beat is kept and at most 600 N beats per record (random, seed 0), so that the added data are patients rather than normal beats |
| splits | DS1 keeps its approved split. INCART patients (from the headers) and SVDB records (no patient named) are sorted; every 5th goes to validation (positions 5, 10, ...), the rest to training. DS2 stays the only test set; no external database is left for testing |
| model and conditions | the phase 2 gate's cnn with its settings unchanged; none, class-weighted loss, random oversampling, cVAE and WGAN-GP at the strengths P7 chose on validation (no new search; SMOTE left out to keep the run under 2 GPU-hours), the cVAE and WGAN-GP banks being v6's (generators trained on DS1 only), the full grid's screened policy; seeds 0-2: 15 runs; early stopping on the enlarged validation set as before |
| test | one `posthoc` access on Kaggle (DS2 rebuilt and hash-checked, v2 ratios, templates). Confirmatory (1 test): P11 none vs P7 none, the same cnn trained on DS1 only, in rare-class macro-F1, record-level bootstrap 10,000, seed 0. Secondary family (Holm, 4 tests): each P11 condition vs P11 none. Descriptive: each P11 condition minus the same P7 condition; the recomputed P7 none score is checked against P7's summary |
| reporting | its own section, labelled post hoc. Compute: one Kaggle T4 session, estimated under 2 GPU-hours; CPU for the test |

**v11, committed 2026-11-06, after P11's test result, before any of the work below (DECISIONS A32).** One analysis, post
hoc. P11 trained the P7 cnn on 106 patients but augmented it with banks of the phase 3 generators, which learned ordinary
beats of DS1's 17 training records (their beats made patient-relative afterwards); on the test records those conditions
(0.245, 0.217) stayed below random oversampling (0.383) and class weighting (0.407). P12 asks whether the generators fall
short because of what they learned from, not because they are generators.

| item | rule |
|---|---|
| P12 `phase6_relative_generators`, generators | the phase 3a gate's cVAE (D) and WGAN-GP (A), settings, 20,000-step budget and seeds 0-2 unchanged, trained on P11's training beats (its data, hash-checked) in v3 form divided by one scale, the SD of all training beat samples; no pilot harness, no hard gates (the variants are fixed) |
| banks | from the final checkpoint; samples multiplied back by the scale and not z-scored per beat; RR features copied from real training beats of the class (phase 3 rule); size: what the unscreened policy adds at strength 0.5 |
| classifier | the P7 cnn on P11's data, settings unchanged; each bank at 0.1, 0.25 and 0.5 x N, unscreened (`fraction_of_n`, as P9), seeds 0-2: 18 runs; one strength per generator chosen on P11's validation by the phase 4 gate rule |
| test | one `posthoc` access on Kaggle (DS2 rebuilt and hash-checked, v3 form). Confirmatory (Holm, 2 tests): each P12 generator vs the same generator in P11, rare-class macro-F1, record-level bootstrap 10,000, seed 0. Secondary (Holm, 4 tests): each P12 generator vs P11 none and vs P11 random oversampling. P11's scores are recomputed and must match its summary |
| descriptive | for S and F beats, the patient of the nearest real training beat (P8's measure, by patient; 5,000 beats per class from the seed-0 banks) for the P12 bank, the bank P11 used and the real beats |
| reporting | its own section, labelled post hoc. Compute: two Kaggle T4 sessions, one per generator, about 2.5 GPU-hours in all (approved by Akash Saha, 2026-11-06); CPU for the test |

**v12, committed 2026-11-06, after P12's test result, before any of the work below (DECISIONS A33).** One analysis, post
hoc, asked for after a literature search on why published inter-patient methods detect S beats better. Published analyses
of DS2 report that record 232 holds three quarters of DS2's S beats, that S beats outnumber N beats there and come without
a compensatory pause, and that RR intervals normalized by local averages miss almost all of them (a CNN with such features:
S F1 0.280, sensitivity 0.186); Huang et al. (2014) detect S beats by the RR interval over the mean RR interval of the
whole recording (S sensitivity 91.1%, +P 42.2% on DS2), and Lin and Yang (2014) and Mondejar-Guerra et al. (2019)
normalize RR intervals by the patient's average. Our features are local: v1's ratio uses the mean of the 10 preceding
intervals, v2's the median within +/-150 s. The same failure is in P11's training data: SVDB record 865 (a training
patient) has S 1,818 of 2,919 N and S beats with a median post/pre ratio of 1.00, and the rule "ratio < 0.8" finds 2.5% of
its S beats with v1's ratio, 1.4% with v2's and 93.6% with the recording mean (`--part check`, annotations of DS1, INCART
and SVDB only, `results/phase6_record_rr/design_check.json`). Because the idea was motivated in part by published analyses
of DS2, P13's test result is not blind to its motivation; its validation results are reported beside it.

| item | rule |
|---|---|
| P13 `phase6_record_rr`, data | P11's data (hash-checked) with `rr_prev` and `rr_next` divided by the mean RR interval of their recording (every interval between consecutive QRS annotations, pauses included, no label used; MIT-BIH, INCART and SVDB alike; offline, as the +/-150 s templates); `rr_ratio` stays v2's; beats, split and caps unchanged |
| banks | P12's banks (seeds 0-2, hash-checked); each synthetic beat takes its RR features from the same real training beat as in P12 (the draws are recomputed from the bank seed and must reproduce the stored values), in the new form |
| classifier | the P7 cnn, settings unchanged; the five conditions at the strengths already chosen, none chosen anew: none; class weighting 0.5 and random oversampling 0.25 (P11); cVAE 0.1 and WGAN-GP 0.25 (P12); P12's unscreened bank policy; seeds 0-2: 15 runs |
| test | one `posthoc` access on Kaggle (DS2 rebuilt and hash-checked; v3 beats, v2 ratio, `rr_prev` and `rr_next` over the recording's mean). Confirmatory (Holm, 5 tests): each condition with the new features vs the same condition in P11 (none, class weighting, random oversampling) or P12 (cVAE, WGAN-GP), rare-class macro-F1, record-level bootstrap 10,000, seed 0. Secondary (Holm, 6 tests): with the new features, each generator vs none, vs random oversampling and vs class weighting. P11's and P12's scores are recomputed and must match their summaries |
| descriptive | validation S F1 of each condition beside P11's and P12's; Huang et al.'s SVEB stage on top of each new condition (a beat the cnn calls N becomes S when `rr_prev` over its recording's mean is below 0.8, Huang's threshold, not tuned), test scores with intervals, no test |
| reporting | its own section, labelled post hoc, framed as limited by scale. Compute: one Kaggle T4 session, under 1 GPU-hour; CPU for the test |

**v13, committed 2026-11-06, after P13's test result, before any of the work below (DECISIONS A34).** One analysis, post
hoc, asked for to "find the solution". A design check on training-side annotations only (`phase6_icentia --part check`,
`results/phase6_icentia/design_check.json`; DS1, INCART and SVDB with P11's split, SVDB 865 left out of training as an
unseen S-dominant patient) finds a trade-off no rule or interval model escapes: references that are local (the mean of the
10 preceding intervals, the mean or median within +/-150 s) give validation S F1 up to 0.738 but find 2-8% of the unseen
patient's S beats; the recording mean finds 94-98% of them (Huang's rule, or a logistic regression on the recording-mean
intervals alone) at validation S F1 0.627-0.650; vetoing irregular rhythm costs validation recall (0.248-0.288). With 865
in training, every model finds its beats, which says nothing about an unseen patient. A network can only learn which
reference to trust from training patients that hold these patterns, and P11's 106 hold one. P14 adds many more.

| item | rule |
|---|---|
| P14 `phase6_icentia`, patients | 400 patients of Icentia11k (PhysioNet; 11,000 patients; technician-labelled N, S, V and Q beats; one chest lead at 250 Hz; CC BY-NC-SA 4.0) drawn at random with seed 0, no label looked at; the first segment (about 70 minutes) of each; files from PhysioNet's open-data mirror (AWS) or PhysioNet, each file's SHA-256 recorded |
| beats | as P11 built INCART and SVDB (resampled to 360 Hz, the frozen representation, the usable-beat rule, the AAMI map, Q dropped), in P13's form (minus the +/-150 s template from all QRS annotations, v2's ratio, `rr_prev` and `rr_next` over the segment's mean RR); at most 600 N beats per record (P11's cap, seed 0), chosen before the templates; every S and V beat kept; every 5th patient in sorted order to validation (P11's rule) |
| data | P13's data (hash-checked) plus these beats; the combined hash is recorded in `data.json` |
| classifier | the P7 cnn, settings unchanged; four of P13's five conditions at P13's strengths: none, class weighting 0.5, random oversampling 0.25, cVAE 0.1 (P12's bank, RR features as in P13); seeds 0-2: 12 runs. WGAN-GP is left out: at 0.25 of the larger N count its bank holds too few F beats |
| test | one `posthoc` access on Kaggle (DS2 in P13's form). Confirmatory (Holm, 4 tests): each condition with the Icentia11k patients vs the same condition in P13, rare-class macro-F1, record-level bootstrap 10,000, seed 0. Secondary (Holm, 3 tests): with them, the cVAE vs none, vs random oversampling and vs class weighting. P13's scores are recomputed and must match its summary |
| descriptive | validation S F1 beside P13's; Huang's SVEB stage on top (P13's rule, threshold 0.8); per-class F1 |
| reporting | its own section, labelled post hoc, framed as limited by scale. Compute: one Kaggle T4 session (download and build included), under 2 GPU-hours; CPU for the test |

**v14, committed 2026-11-08, after P14's test result, before any of the work below (DECISIONS A36).** One analysis, post
hoc. The study's generators are a cVAE and a WGAN-GP; the current standard for synthetic ECG is a diffusion model (SSSD-ECG,
Alcaraz and Strodthoff 2023; conditional diffusion for imbalanced ECG classes, Kwak and Jung 2024). P15 adds it as the third
generator family under the same rules.

| item | rule |
|---|---|
| P15 `phase6_diffusion`, generator | `ecgaug.generators.diffusion`: a class-conditional DDPM (cosine schedule, 1,000 levels) with a 1D U-Net (widths 64/128/256, kernel 5, FiLM conditioning on noise level and class), classifier-free guidance (10% null-class training, scale 1.5 at sampling), an exponential moving average of the weights (0.999) used for sampling, DDIM sampling with 50 levels; AdamW, learning rate 2e-4; the phase 3 budget (batch 128 with 32 beats per class, 20,000 steps, seeds 0-2); a fit with a non-finite or exploding loss, or non-finite beats, is diverged |
| data | P13's training beats (P11's patients, hash-checked) in the form the cnn sees them, divided by one scale (the SD of all training samples), as P12; banks multiplied back by the scale, not z-scored per beat; RR features copied from real training beats of the class in P13's form; bank size: what the unscreened policy adds at 0.5 |
| classifier | the P7 cnn on P13's data, settings unchanged; the bank at 0.1, 0.25 and 0.5 x N (unscreened), seeds 0-2: 9 runs; one strength chosen on validation by the phase 4 gate rule |
| test | one `posthoc` access on Kaggle (DS2 in P13's form). Confirmatory (Holm, 2 tests): the diffusion condition vs P13's cnn without augmentation, and vs P13's cVAE (the best earlier generator); rare-class macro-F1, record-level bootstrap 10,000, seed 0. Secondary (Holm, 2 tests): vs P13's class weighting and random oversampling. P13's scores are recomputed and must match its summary |
| descriptive | Huang et al.'s SVEB stage on top of the diffusion condition and of P13's compared conditions (as P13); the nearest-patient check of P12 for the seed-0 bank beside P12's cVAE bank; validation S F1 |
| reporting | its own section, labelled post hoc, framed as limited by scale. Compute: one Kaggle T4 session, under 2 GPU-hours (approved: "Run whatever you want", 2026-11-08); CPU for the test |

**v15, committed 2026-11-09, after P15's training and validation selection, before P15's test read and before any of the
work below (DECISIONS A37).** One analysis, post hoc. P13's best test scores come from Huang et al.'s SVEB stage on top of
the cnn, a rule that reads only a beat's RR interval over its recording's mean. Irregular rhythms make normal beats look
premature to it. On DS1's annotations (`phase6_af_rule --part check`, `results/phase6_af_rule/design_check.json`), 1,251
of the stage's 1,412 false S calls lie in atrial fibrillation or flutter, where 16 of DS1's 944 S beats lie: without
organized atrial activity a beat is not labelled S. P16 undoes the stage's S calls where an AF detector fires. The detector
sees only beat times. The idea and every setting come from training-side data; the wish to improve the stage came after
P13's test result, so P16 is reported as post hoc.

| item | rule |
|---|---|
| P16 `phase6_af_rule`, AF detector | RR intervals only. Per beat, five irregularity measures of the 64 intervals around it (normalized RMSSD; coefficient of variation; RMSSD without the 4 largest jumps; the share of successive differences above 10% of the mean; the entropy of a 16-bin histogram of the intervals over their mean), after removing every interval below 0.85 of the median of its 21 neighbours and the interval after it (premature beats and their pauses); no value, and probability 0, where fewer than 16 intervals remain. A logistic regression (balanced classes) on their logarithms, fitted on every 5th beat of the 25 records of the MIT-BIH Atrial Fibrillation Database (other subjects; its beat detections and rhythm labels): 244,288 beats, 43.5% in AF or flutter, training sensitivity 0.967 and specificity 0.954 (`results/phase6_af_rule/af_detector.json`) |
| threshold | the candidate in {0.5, 0.8, 0.9, 0.95, 0.99} with the highest mean S F1 of the vetoed stage over DS1, INCART and SVDB (annotations only; ties to the higher threshold): 0.95 (the stage alone 0.490, vetoed 0.575) |
| form | "rule" (undo only the stage's own N-to-S conversions) or "all" (undo every S call where the detector fires): the higher mean validation S F1 over the compared conditions and seeds (P13's validation patients, the runs' stored validation predictions; ties to "rule"): "rule" 0.753, "all" 0.730; the stage alone 0.727; the cnn alone 0.755 |
| conditions | P13's five (none, class weighting 0.5, random oversampling 0.25, cVAE 0.1, WGAN-GP 0.25) and P15's diffusion condition (0.25, chosen on validation): their selected runs, seeds 0-2; nothing is retrained |
| test | one `posthoc` access on Kaggle, after P15's test (DS2 in P13's form; the detector reads each DS2 record's QRS annotation times, as the RR features already do, never a label; every test beat must sit on one). Confirmatory (Holm, 6 tests): per condition, the stage with the veto vs the stage alone, rare-class macro-F1 (only S calls change), record-level bootstrap 10,000, seed 0. The stage-alone scores must match P13's summary and, for the diffusion condition, P15's |
| descriptive | S sensitivity, +P and F1 with and without the veto; the share of each DS2 record's beats the detector flags; without record 202 |
| reporting | its own section, labelled post hoc, framed as limited by scale. Compute: CPU only (the fit and the check ran on annotations; the test on a Kaggle CPU session) |

**v16, committed 2026-11-09, after P15's training and P16's design, before P15's and P16's test results were read and
before any of the work below (DECISIONS A38).** One analysis, post hoc. P15 trained its diffusion model with the phase 3
budget of 20,000 steps. Two training-side signs say the budget was too small for it. Its training loss was still falling
at the end: a mean of 0.117-0.118 over steps 1,001-2,000, 0.098-0.099 over 15,001-16,000 and 0.096 over 19,001-20,000, in
all three seeds (the runs' `train_log.csv`). And the median beat of each class in its banks matches the median of real
DS1 training beats far less closely than the cVAE's does: correlation S 0.10-0.30, V 0.57-0.59, F 0.72-0.73, against the
cVAE's 0.58-0.74, 0.94-0.96 and 0.95-0.96 (`phase6_diffusion_long --part shape`, `results/phase6_diffusion_long/shape_check.json`).
Sampling P15's seed-0 model with its raw weights instead of their moving average, or without guidance, did not close the
gap in an exploratory check. P15's and P16's test summaries had been written and committed by then, but not read.

| item | rule |
|---|---|
| P17 `phase6_diffusion_long`, generator | P15's model, data, sampler, batch, seeds, bank sizes, strengths and selection rule, unchanged (the config is checked against P15's when it is loaded); 100,000 steps instead of 20,000 |
| classifier | the P7 cnn on P13's data with each P17 bank at 0.1, 0.25 and 0.5 x N (unscreened), seeds 0-2: 9 runs; one strength chosen on validation by the phase 4 gate rule |
| test | one `posthoc` access on Kaggle (DS2 in P13's form; the AF detector reads only each DS2 record's QRS annotation times). Confirmatory (Holm, 2 tests): P17 vs P15's diffusion condition (the budget) and vs P13's cnn without augmentation; rare-class macro-F1, record-level bootstrap 10,000, seed 0. Secondary (Holm, 3 tests): vs P13's cVAE, class weighting and random oversampling. P13's, P15's and P16's scores are recomputed and must match their summaries |
| descriptive | Huang et al.'s SVEB stage, and the stage with P16's AF veto (threshold 0.95, form "rule"), on top of every compared condition; the nearest-patient check (as P15); the shape check for P17's banks; validation S F1 |
| reporting | its own section, labelled post hoc, framed as limited by scale. Only the diffusion model gets the longer budget; the cVAE and WGAN-GP keep the phase 3 budget, so P17 against them is not an equal-budget comparison, and the paper says so. Compute: one Kaggle T4 session, about 6 GPU-hours (P15 took about 18 minutes per 20,000 steps; approved: "Run whatever you want", 2026-11-08); CPU for the test |

**v17, committed 2026-11-09, after P17's test result, before any of the work below (DECISIONS A39).** One analysis, post
hoc. Every run of the cnn keeps the epoch with the highest validation S F1. Within one run that score moves by 0.09 to
0.34 between consecutive epochs (median 0.20; the 18 runs P16 compares, their `train_log.csv`), and 6 of the 18 kept their
first epoch, so the model a run keeps is a noisy draw. Averaging the class probabilities of runs trained independently (a
deep ensemble; Lakshminarayanan et al., 2017) is the standard way to reduce that noise, and such ensembles held up best
under dataset shift in Ovadia et al. (2019). On P13's validation patients the average of the three existing runs raised S
F1 with Huang et al.'s stage and P16's veto in all six of P16's conditions (mean 0.753 to 0.767; cVAE 0.765 to 0.777) and V
F1 in all six (cVAE 0.796 to 0.823): the runs' stored validation predictions, no test data. The wish to raise the cVAE's
score came after P16's and P17's test results, so P18 is reported as post hoc.

| item | rule |
|---|---|
| P18 `phase6_seed_ensemble`, runs | the P7 cnn on P13's data in P16's six conditions at their selected strengths (none; class weighting 0.5; random oversampling 0.25; cVAE 0.1; WGAN-GP 0.25; diffusion 0.25), every setting unchanged. Seeds 0-2 are P13's and P15's selected runs; seeds 3-9 are trained (42 runs). A run of seed s uses bank seed s mod 3 (each generator's three frozen banks, P12's and P15's); random oversampling draws with the run's seed |
| ensemble | per condition, the mean of its ten runs' class probabilities, then argmax; Huang et al.'s SVEB stage (0.8) and P16's AF veto (threshold 0.95, form "rule") on top, as in P16. Nothing is chosen: the ten seeds and the averaging are fixed now |
| test | one `posthoc` access on Kaggle (DS2 in P13's form; the AF detector reads only each DS2 record's QRS annotation times). Confirmatory (Holm, 6 tests): per condition, the ten-run ensemble vs P16's three single runs (mean F1 over the runs), both with the stage and the veto; rare-class macro-F1, record-level bootstrap 10,000, seed 0. Secondary (Holm, 5 tests): the cVAE's ensemble vs each other condition's ensemble, with the stage and the veto. P13's, P15's and P16's scores are recomputed from seeds 0-2 and must match their summaries |
| descriptive | the ensembles alone and with the stage alone; the ten single runs (mean and spread); the three-run ensemble of seeds 0-2; without record 202; validation S, V and F F1 of the single runs and both ensembles (`--part check`, written with the run IDs before the test read) |
| reporting | its own section, labelled post hoc, framed as limited by scale. P16's cVAE score (0.557) stays as reported; the paper says an ensemble needs ten trained models where P16 used one. Compute: one Kaggle T4 session, under 1 GPU-hour (P13's runs took 37-52 s each; requested: "can you make the cvae results even better", 2026-11-09); CPU for the test |

**v18, committed 2026-11-10, after P18's test result, before any of the work below (DECISIONS A40).** One analysis, post
hoc, reading no test records. P18's highest score (the cVAE's ten-run ensemble with Huang et al.'s SVEB stage and P16's AF
veto, rare-class macro-F1 0.587) is the best of 14 post hoc reads of the same 22 DS2 patients, and one of them (record 232)
holds 1,381 of DS2's 1,836 S beats. P19 asks whether the result holds on patients that played no part in any choice. The
wish came after comparing P18 with published results; the design below was fixed before any Icentia11k beat of these
patients was downloaded.

| item | rule |
|---|---|
| P19 `phase6_fresh_patients`, data | Icentia11k (PhysioNet, CC BY-NC-SA 4.0): the first segment (about 70 minutes) of 1,000 patients drawn at random with seed 1 from the 10,600 that P14 did not draw (P14: 400 with seed 0); no label is looked at to draw them. Built as P14 built its patients (the frozen representation after resampling to 360 Hz, minus the local template of all QRS annotations within ±150 s; `rr_prev` and `rr_next` over the segment's mean RR interval; v2's ratio) but every usable N, S and V beat is kept (no cap on N beats). Q beats are dropped as everywhere; Icentia11k has no F labels. Segments whose download fails or that hold no usable labelled beat are left out and reported |
| models | P18's sixty runs as listed in its `members.json` (ten per condition: none, class weighting, random oversampling, cVAE, WGAN-GP, diffusion), P16's AF detector (`af_detector.json`) with its threshold 0.95 and form "rule", Huang et al.'s threshold 0.8; each beat's AF probability from its segment's QRS annotation times. Nothing is fitted, chosen or tuned |
| measures | S F1 is the primary measure (no F labels); V F1, S Se, S +P and N recall with it. Patient-level bootstrap (each patient one cluster), 10,000 resamples, seed 0 |
| confirmatory (Holm, 6) | per condition, the ten-run ensemble vs P16's single runs (mean over seeds 0-2), both with the stage and the veto, on S F1 |
| secondary (Holm, 5) | the cVAE's ten-run ensemble vs each other condition's, with the stage and the veto, on S F1 |
| descriptive | the ensembles alone and with the stage alone; per condition, the stage and the stage with the veto against the ensemble alone (paired intervals, no correction); the three-run ensemble; the ten single runs; the share of beats called F; how many patients are S-dominant and the largest patient's share of S beats; P18's DS2 values beside |
| reporting | its own section, labelled post hoc and as an external check on a different device and annotation process; P18's DS2 results stay as reported. Compute: one Kaggle T4 session, about 1-1.5 GPU-hours (the beats are built on the CPU cores; prediction on the CPU would take about 50 hours, 639 µs per beat and run measured locally); requested: "so improve these" and the choice "fresh-patient check", 2026-11-10 |

**v19, committed 2026-11-10, after P19's result, before any of the work below (DECISIONS A41).** One analysis, post hoc.
On P19's 977 unused Icentia11k patients, Huang et al.'s SVEB stage lowered S F1 in five of six conditions against the
ten-run ensembles alone (cVAE: S Se 0.303 to 0.629, S +P 0.408 to 0.157). Its threshold 0.8 is Huang et al.'s MIT-BIH
value, and P16's veto threshold 0.95 came from MIT-BIH, INCART and SVDB annotations. P20 chooses the stage's threshold and
the veto's threshold, including the choice of no stage at all, on a larger and more varied group of patients that none of
P18's runs has seen, then scores the choice on P19's patients and, once, on DS2. The wish came after P19's result; P19's
patients play no part in the choice.

| item | rule |
|---|---|
| P20 `phase6_rule_tuning`, tuning patients | P14's 400 Icentia11k patients (seed 0), the first segment of each, built exactly as P19 built its patients (every usable N, S and V beat, no cap; each beat's AF probability from the segment's QRS annotation times). P18's sixty runs were trained on P13's data (DS1, INCART and SVDB patients) and never saw an Icentia11k beat |
| evaluation patients | P19's patients, rebuilt the same way; the rebuilt beats must reproduce P19's `content_sha256`, else nothing is scored. They play no part in the choice |
| models | P18's sixty runs (`members.json`), each condition's ten-run ensemble (mean probability); P16's AF detector and form "rule" |
| settings | the ensemble alone; Huang's stage with threshold t in {0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90}, with no veto or with P16's veto at AF threshold a in {0.50, 0.60, 0.70, 0.80, 0.90, 0.95}: 50 settings |
| choice | per condition, the setting with the highest S F1 over all the tuning patients' beats pooled. Ties: the ensemble alone first, then no veto, then the t nearest 0.80 (the lower when two are as near), then the higher a. Written to `choice.json` before any evaluation patient is scored |
| measures | on Icentia11k S F1 (no F labels), with V F1, S Se, S +P and N recall; on DS2 rare-class macro-F1, with per-class F1. Patient-level bootstrap (each patient or record one cluster), 10,000 resamples, seed 0 |
| confirmatory (Holm, 6) | on P19's patients, per condition, the ten-run ensemble with its chosen setting vs with P19's setting (t 0.8, a 0.95), on S F1 |
| secondary (Holm, 6) | on P19's patients, per condition, the chosen setting vs the ensemble alone, on S F1 (a condition whose choice is the ensemble alone has difference 0 and p 1) |
| DS2 (Holm, 6) | one posthoc read, logged: per condition, the ten-run ensemble with its chosen setting vs with P18's setting (t 0.8, a 0.95), on rare-class macro-F1; P18's values must reproduce exactly |
| descriptive | S F1 of every setting on the tuning and on the evaluation patients (no intervals); the cVAE's ensemble vs each other condition's with the chosen settings on P19's patients (paired intervals, no correction); DS2 per-class F1, S Se and +P, the ensembles alone and without record 202; the difference from P19's own S F1 for P19's setting (GPU prediction may differ in the last digits; a difference above 0.001 stops the run) |
| reporting | its own section, labelled post hoc; DS2 is read for the 15th time after the final evaluation and the primary results stay those of the final evaluation. Compute: the beats are built on a Kaggle CPU session (about 1.5 hours, 1,400 segments); prediction with the sixty runs on one T4 session, about 0.7 GPU-hours (P19 predicted about 3.7 million beats with them in about 20 minutes); the DS2 read on a CPU session. Requested: "do it", 2026-11-10, after the proposal to choose the stage and the veto on more varied patients |

Each version is committed before the experiments it governs, tagged `prereg-vK` and pushed that day.

## 11. Deviations and amendments

| date | section | change | reason | approved by |
|---|---|---|---|---|
| 2026-10-26 | §4 early stopping | `min_epochs: 10` for every MIT-BIH classifier from phase 2 on | 10 of 24 phase 1 runs stopped at epoch 6 on an all-N model (PHASE1_REPORT §6); fixed before any phase 2 run | Akash Saha (DECISIONS A15), pending team confirmation |
| 2026-11-01 | §9 RQ3 | The separate TSTR analysis has no data: TSTR was not scored in the sanity harness and is not in the composite | Fixed recipe gives S F1 = 0 on real beats (PROGRESS 2026-11-01); to be stated in the paper | Akash Saha (phase 2 gate) |
| 2026-11-03 | §10 v2 diversity floor | cVAE D bank seed 1 has lowest diversity 0.694 (class S), under the 0.7 floor; the bank is kept and frozen | The floor is pre-registered for pilot seeds only; final banks have no rule, and adding one after seeing the result would be post hoc. Disclosed in the paper | Akash Saha (phase 3 gate, DECISIONS A20) |
| 2026-11-09 | §10 v16 motivation | Two motivating correlations were rounded by hand when v16 was written: P15's lowest S value 0.105 was given as 0.11 (0.10) and the cVAE's F range 0.955-0.963 as 0.96 (0.95-0.96); corrected in place | Transcription error found when the paper text was generated from `shape_check.json`; no rule, threshold or analysis depends on these numbers | to be confirmed by Akash Saha |
