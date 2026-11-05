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
- **Post hoc** (`ALLOW_TEST_POSTHOC=1`, v4 to v7): the analyses listed in §10 v4 to v7 only, after `/final-eval`, logged;
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

Each version is committed before the experiments it governs, tagged `prereg-vK` and pushed that day.

## 11. Deviations and amendments

| date | section | change | reason | approved by |
|---|---|---|---|---|
| 2026-10-26 | §4 early stopping | `min_epochs: 10` for every MIT-BIH classifier from phase 2 on | 10 of 24 phase 1 runs stopped at epoch 6 on an all-N model (PHASE1_REPORT §6); fixed before any phase 2 run | Akash Saha (DECISIONS A15), pending team confirmation |
| 2026-11-01 | §9 RQ3 | The separate TSTR analysis has no data: TSTR was not scored in the sanity harness and is not in the composite | Fixed recipe gives S F1 = 0 on real beats (PROGRESS 2026-11-01); to be stated in the paper | Akash Saha (phase 2 gate) |
| 2026-11-03 | §10 v2 diversity floor | cVAE D bank seed 1 has lowest diversity 0.694 (class S), under the 0.7 floor; the bank is kept and frozen | The floor is pre-registered for pilot seeds only; final banks have no rule, and adding one after seeing the result would be post hoc. Disclosed in the paper | Akash Saha (phase 3 gate, DECISIONS A20) |
