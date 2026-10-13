# PLAN — Phases, Modules, Variants, Gates

Day 1 = 2026-10-07 · Day 30 = 2026-11-05 · Buffer days: 15, 29, 30.
Amended 2026-10-08 after the kickoff review: see `docs/DECISIONS.md` (A1–A11).

## Global selection conventions (copied into PREREGISTRATION.md in Phase 1)

- **Primary selection metric:** validation rare-class macro-F1 (mean F1 of S and F), averaged over
  seeds [0, 1, 2]. Fallback if validation holds too few F beats: see docs/RULES.md §4.
- **Ties:** variants within the tie margin of the best one → the simpler/cheaper one wins (fewer
  parameters, then less wall-clock). Tie margin = max(0.02, pooled across-seed SD of the two variants).
  Under the fallback, a tie on S F1 is broken first by V F1, then by simplicity.
- **"Best" always means validation:** any "best" condition, checkpoint, strength or variant is the one
  with the highest validation selection metric, recorded in DECISIONS.md before `/final-eval`.
- **Decision rule:** argmax of the class probabilities in every condition; no threshold is tuned.
- **Equal budget per variant:** same epoch/step cap, early stopping (patience 5 on the primary
  metric), and tuning grid (learning rate ∈ {1e-3, 3e-4}, nothing else) unless a module says otherwise.
- **Result file per variant:** `results/validation/m<M>/<X>.json` with per-seed metrics, mean ± sd,
  run IDs, wall-clock, budget deviations; plus `<X>_notes.md`.
- **Losing variants are kept.** They form the variation-study tables in the paper appendix.

---

## Phase 1 — Data foundation (Days 1–3, Oct 7–9)

**Goal:** a frozen, leakage-free data contract, shared eval utilities, and a committed pre-registration.

### Shared infrastructure (`/phase 1`)
1. Repo scaffold (layout in docs/RULES.md), environment + lockfile, pytest, `.gitignore`, docs templates
   (bottom of this file).
2. Download MIT-BIH via wfdb; verify all 48 records; record checksums.
3. Beat extraction: reference R-peak annotations, MLII selected by name, AAMI mapping, paced records
   excluded, Q policy, `rr_prev` / `rr_next`.
4. Per-record class counts → `results/data/record_counts.csv`; compare totals with docs/RULES.md sanity counts.
   Counting DS2 labels is a logged construction access. The validation proposal uses DS1 counts only.
5. **Validation proposal:** 2–3 options for which DS1 records form validation, each with class counts
   for train and val, and a per-record concentration table. Goal: S, V and (if at all possible) F
   represented in validation while training keeps enough of each. **STOP for a human decision.**
6. Guarded test loader, `docs/TEST_ACCESS_LOG.md`, tests: zero record overlap; loader raises without
   `ALLOW_TEST=1`; contract hashes stable.
7. **Module 2 — eval utilities (no variants):**
   - `evaluate(probs, y, records) -> dict`: per-class precision / recall / F1, one-vs-rest PR-AUC,
     macro averages, AAMI-style Se and +P for S and V, confusion matrix.
   - `cluster_bootstrap(metric_fn, preds, records, n=10_000)`: resamples **records, not beats**;
     reports the fraction of replicates where a class is absent.
   - `paired_cluster_bootstrap_diff(a, b, ...)` for condition differences.
   - Unit tests against hand-computed cases.
8. Probe CNN (small, fixed) used only to compare Module 1 variants. Supports an optional RR side input.
9. Kaggle runner notebook template + `kaggle/ingest_runs.py`.
10. `docs/PREREGISTRATION.md` v1 generated from `docs/PAPER_TRACK.md` (hypotheses, metrics,
    comparisons, statistics, thresholds). **Committed before any Module 1 variant runs.**

### Module 1 — Data pipeline variants
Only the input representation differs. Splits and labels are identical across variants.

| Variant | Definition |
|---|---|
| A | Window −0.25 s to +0.25 s around R (180 samples at 360 Hz); per-beat z-score; no filtering. |
| B | Window −0.4 s to +0.6 s (360 samples); 0.5–40 Hz zero-phase Butterworth band-pass on the continuous record before segmentation; per-beat z-score. |
| C | Variant A plus RR features (`rr_prev`, `rr_next`, `rr_prev` / mean RR of the previous 10 beats) fed to the probe model's side branch. |
| D | Variant A resampled to 125 Hz (about 63 samples). |

**Selection:** probe CNN, seeds [0, 1, 2], validation primary metric; tie → fewer samples per beat.
The winner defines contract v1.
**If C wins:** every downstream classifier takes the RR side input, and synthetic beats need RR
features. Default: sample RR features from the class-conditional empirical distribution of training
beats; document this as a limitation (S beats are largely defined by prematurity, not morphology).

### Gate 1 checklist
- [ ] All 44 non-paced records processed; class totals within ±1% of sanity counts (deviations explained).
- [ ] Zero record overlap across train/val/test (test passes).
- [ ] Test loader raises without `ALLOW_TEST`; TEST_ACCESS_LOG holds construction entries only (zero
      evaluation accesses).
- [ ] Validation split approved; record lists written to DECISIONS.md and contract.json.
- [ ] Selection metric fixed by the minimum-support rule (primary or fallback) and recorded.
- [ ] PREREGISTRATION.md v1 committed (SHA recorded), tagged `prereg-v1` and pushed before Module 1 runs.
- [ ] Module 1 winner selected by rule; contract v1 frozen with hashes in `results/manifest.json`.
- [ ] Eval utilities and bootstrap tested.

**Paper artifacts:** dataset/protocol table (records and class counts per split), per-record counts
(appendix), Module 1 variation table, protocol diagram source.

---

## Phase 2 — Baselines and shared quality metrics (Days 4–8, Oct 10–14)

**Goal:** frozen reference classifiers and a frozen, sanity-tested synthetic-quality metric suite.

### Shared infrastructure (`/phase 2`)
1. Trainer: unweighted cross-entropy, AdamW, max 40 epochs, early stopping (patience 5) on the
   validation primary metric, best checkpoint saved, deterministic seeding, logging per docs/RULES.md.
2. `build_model(name, input_spec)` with a penultimate-feature hook (needed by FID-analog) and
   access to the last conv layer (needed by Grad-CAM).
3. Quality interface `score(real, synth, cls) -> dict` and a **sanity harness** that builds degraded
   "synthetic" sets: Gaussian noise at 3 levels, label-shuffled beats, time-shifted beats, low-pass
   smoothed beats, plus a real-vs-real reference (held-out training records vs remaining training records).
4. Always-on diagnostics (not variants):
   - **Diversity:** mean within-class pairwise distance, relative to real.
   - **Memorization:** fraction of synthetic beats whose nearest-neighbour distance to training beats is
     below the 5th percentile of real-held-out-to-train distances (≈5% expected without copying).
   - **C2ST:** accuracy of a small classifier separating real from synthetic (0.5 = indistinguishable).
5. Toy cVAE and toy WGAN-GP smoke tests on one minority class (plumbing only; not selected).
6. Measure wall-clock per run to re-estimate Phases 3–4.

### PTB-XL pretraining arm (A12 and A13, built 2026-10-13)

One extra condition for the baseline classifier: pretrain on PTB-XL, then fine-tune on MIT-BIH, compared
with the same model trained from scratch. Runner: `python -m ecgaug.experiments.pretrain --config
configs/pretrain/<A-D>.yaml`; one command does preprocessing, pretraining and both fine-tuning arms.

1. **Data (`src/ecgaug/data/ptbxl.py`):** PTB-XL 500 Hz files, lead II by name (closest to MLII),
   resampled to 360 Hz. Recordings are dropped if they have no diagnostic superclass, a pacemaker,
   flagged electrode problems, a flat or implausible signal, an implausible number of peaks, or peaks
   that do not stand out of the noise. R peaks come from `src/ecgaug/data/rpeaks.py`, validated against
   the MIT-BIH DS1 annotations (`results/data/rpeak_validation_ds1.json`). Beats need both RR intervals
   inside 0.3–2.0 s and the same usable-beat rule as MIT-BIH. Files are compared with PhysioNet's
   checksums and the counts are reported.
2. **Windows:** cut with the representation of the config's letter, through the same code as the
   MIT-BIH beats. Each 10-second recording is mirror-padded by 2 s before filtering; that padding was
   the best of four treatments measured on DS1 (`results/data/ptbxl_method_checks_ds1.json`).
3. **Labels:** each beat gets its recording's diagnostic superclasses (NORM, MI, STTC, CD, HYP,
   multi-label). Noisy at beat level by construction.
4. **Pretraining:** same architecture with a 5-output head, binary cross-entropy, AdamW, learning rate
   1e-3, at most 30 epochs, patience 5 on the macro AUROC of PTB-XL fold 9. Folds 1–8 train, 10 unused.
   One pretraining run per seed; fine-tuning seed s starts from pretraining seed s.
5. **Fine-tuning:** the head is replaced by a new 4-class head and all layers are trained on the DS1
   training records. Budget, seeds, learning-rate grid and early stopping are read from the Module 1
   config, for both arms. The from-scratch arm is exactly a Module 1 run of that representation.
6. **Comparison:** the selection rule on the validation selection metric, with from scratch as the
   simpler option, so pretraining has to win by more than the tie margin. It joins the Phase 4 grid as
   a second starting point only if it wins; otherwise it is reported as a negative result.
7. **Status of results:** provisional until Gate 1 freezes the representation. The architecture is the
   probe CNN for now; the arm is rerun with the `cnn` selected in Module 3.
8. **Checks:** PTB-XL code cannot read MIT-BIH records or the test file; no patient is in two PTB-XL
   splits; the arm never asks for test access.

### Module 3 — Baseline classifier variants (real, imbalanced data, no reweighting)

| Variant | Definition |
|---|---|
| A | Shallow 1D-CNN: 3 conv blocks (conv–BN–ReLU–pool), global average pooling, linear head. |
| B | 1D-ResNet: 4 residual stages, global average pooling, linear head. |
| C | BiLSTM: 2 layers, hidden 64–128, concatenated last and mean-pooled states, linear head. |
| D | CNN + BiLSTM hybrid: 2 conv blocks, then 1 BiLSTM layer, linear head. |

**Selection:** validation primary metric (seeds 0–2); tie → fewer parameters.
Best of {A, B} → `cnn`. Best of {C, D} → `rnn` (the grid's recurrent model).
Frozen checkpoints for every seed become the "none" condition.

### Module 4 — Quality metric variants (metric families)

| Variant | Definition |
|---|---|
| A | DTW: mean DTW distance from each synthetic beat to its 5 nearest real training beats of the same class, normalized by the real-held-out reference. |
| B | Spectral: Welch power spectrum per beat; distance between class-mean normalized spectra (e.g. Jensen–Shannon) plus the per-beat distribution. |
| C | FID-analog: Fréchet distance between Gaussians fitted to frozen `cnn` penultimate features, per class, equal sample sizes, with a real-vs-real reference. |
| D | TSTR: replace real minority training beats with an equal number of synthetic ones, train the probe CNN, report validation rare-class F1 relative to train-on-real. |

**Selection (sanity only — no generator results are used):** a metric passes if it
(1) ranks the real-held-out reference best and recovers the degradation ordering (Spearman ≥ 0.8 with
severity), and (2) is stable across 3 resamples (coefficient of variation < 10%).
Passing metrics form the **pre-registered composite**: equal-weight average of per-metric ranks across
candidate generators. Fixed at Gate 2, before any generator is evaluated.
Whether metrics predict downstream gains is a paper analysis (RQ3), not a selection rule.

### Gate 2 checklist
- [ ] Module 3 variants complete with equal budgets; `cnn` and `rnn` chosen by rule; checkpoints for
      all seeds frozen and hashed.
- [ ] Metric sanity table; composite definition, diversity floor and memorization threshold committed
      to PREREGISTRATION.md v2 before Phase 3 runs.
- [ ] Measured run times → compute estimate for Phases 3–4; seed count (3 or 5) proposed.
- [ ] Toy generators run end to end.

**Paper artifacts:** baseline validation table (test numbers only after final eval), Module 3
variation table, metric sanity figure (score vs degradation level).

---

## Phase 3 — Generators (Days 9–14, Oct 15–20; Day 15 buffer)

**Goal:** one frozen synthetic bank per generator family (cVAE, WGAN-GP) per seed, trained on training
records only, with quality reports.

### Shared infrastructure (`/phase 3`)
1. Generator interface: `fit(X, y)`, `sample(cls, n, seed)`; checkpoint at each quarter of the step budget.
   The bank is exported from the final checkpoint. Earlier checkpoints are scored for RQ3 only and are
   never selectable.
2. Bank exporter in contract format for classes S, V, F.
3. **Pilot harness:** at every checkpoint, compute the quality suite + diagnostics and a downstream pilot
   score (train `cnn`, seed 0, with the bank at a fixed pilot policy: minority classes balanced to 50% of
   N; validation primary metric). Log every checkpoint — this is the RQ3 dataset.
4. Classical comparisons: SMOTE (imbalanced-learn, fitted on training records only), random
   oversampling of real minority beats (the matched-count control: same counts, no new information)
   and class-weighted cross-entropy with three weightings w_c = (n_max / n_c)^γ, γ ∈ {0.5, 0.75, 1}
   (values fixed in PREREGISTRATION v2 before Phase 3 runs). DeepSMOTE optional.
5. Grid runner: conditions × strengths × architectures × seeds from YAML; resumable; skips completed
   runs; updates `results/runs_index.csv`; dry-run on baselines.
6. Assertion in every generator: fitted data contains only `split == 'train'` records.

### Module 5 — Conditional VAE variants

| Variant | Definition |
|---|---|
| A | Conv encoder/decoder, class one-hot concatenated to input and latent, latent dim 16, KL annealing over the first 30% of steps. |
| B | As A with β ∈ {0.1, 0.25} (both tried inside the pilot budget; report both). |
| C | As A plus a spectral reconstruction term (L1 on FFT magnitude) or soft-DTW. |
| D | Residual encoder/decoder with FiLM class conditioning. |

### Module 6 — Conditional WGAN-GP variants
Shared settings (Gulrajani et al., 2017): n_critic = 5, gradient-penalty weight 10, Adam lr 1e-4,
β1 = 0, β2 = 0.9, no batch norm in the critic.

| Variant | Definition |
|---|---|
| A | Label embedding concatenated to generator noise and critic input. |
| B | Projection critic (class embedding inner product with critic features). |
| C | AC-WGAN-GP: auxiliary classification head on the critic. |
| D | Separate unconditional WGAN-GP per minority class (S, V, F). |

### Schedule
Days 9–11 pilots (fixed step budget, seeds 0 and 1) → Day 12 `/gate 3a` selection → Days 12–14 train
each family's winner at full budget with the grid seeds, export banks.

### Selection (pre-registered, both families)
1. **Hard gates:** both pilot seeds train without divergence; within-class diversity ≥ 0.7 × real;
   memorization fraction ≤ 10%.
2. **Rank** passing variants by downstream pilot validation primary metric (mean of 2 seeds).
3. **Ties** (within 0.02) broken by the composite quality rank.

**Fallback:** if no WGAN-GP variant passes the hard gates by Day 12, restrict to the 2–3 rarest classes
or adopt variant D, and record the failure. It is a result, not a secret.

### Gate 3 checklist (3a on Day 12 = selection; 3 on Day 14 = freeze)
- [ ] Banks exported per family × seed; hashes recorded; train-only assertion passed.
- [ ] Quality + diagnostics table for all variants and for the final banks.
- [ ] Checkpoint-level RQ3 data saved.
- [ ] Failure-mode notes (collapse, instability, artifacts) with figures.
- [ ] Compute re-estimate; grid seeds decided (3 or 5).

**Paper artifacts:** real vs synthetic beats per class (median and interquartile band aligned on R),
quality table, training curves, failure-mode figure.

---

## Phase 4 — Integration and ablation (Days 16–21, Oct 22–27)

**Goal:** complete validation-level grid, then a single test evaluation with pre-registered statistics.

### Module 7 — Augmentation policy variants (pilot Days 16–17: `cnn`, seed 0, both banks)

| Variant | Definition |
|---|---|
| A | Balance each minority class up to the N count. |
| B | Balance each minority class up to a fixed fraction of N. |
| C | Add a fixed multiple of the real minority count. |
| D | As B, but quality-screened: keep the best synthetic beats by DTW to real training beats. |

**Selection:** validation primary metric, with N recall drop ≤ 0.01 versus no augmentation; tie →
simpler. The chosen policy gets three pre-registered strengths (low / medium / high), e.g. fractions
{0.1, 0.25, 0.5} for B, or multiples {1×, 2×, 4×} for C. The chosen policy and strengths apply
identically to SMOTE and to random oversampling.

### Full grid (Days 18–21)
- Conditions: none · class-weighted CE · random oversampling · SMOTE · cVAE · WGAN-GP.
- Strengths: three each for class-weighted CE (the three weightings), random oversampling, SMOTE,
  cVAE and WGAN-GP; one for none.
- Architectures: `cnn`, `rnn`. Seeds: grid seeds; classifier seed s uses bank seed s.
- Runs at 3 seeds: (1 + 5 × 3) × 2 × 3 = 96.
- Synthetic beats are added to training records only. Validation and test are always 100% real.

### Optional (if compute allows; otherwise post-deadline): leakage-audit arm
The same pipeline under the common intra-patient protocol (random beat-level split pooling DS1 and DS2):
{none, cVAE, WGAN-GP} at medium strength, `cnn`, 3 seeds. Never used for any decision.
It trains and evaluates on DS2 beats, so it runs **only after `/final-eval` has completed**, in its own
namespace (`runs/audit/`, `data/synthetic/audit/`), with every run logged in TEST_ACCESS_LOG as kind
`audit`. The main grid runner refuses any bank or checkpoint under an audit path.

### Order of operations
`/gate 4` (validation summary) → human approval → `/final-eval` (the single test access) →
pre-registered statistics, tables, figures.

### Gate 4 checklist
- [ ] All grid runs complete or failures documented; `runs_index.csv` consistent with `runs/`.
- [ ] Validation summary table; strength selected per condition on validation.
- [ ] Approval → `/final-eval` executed once; TEST_ACCESS_LOG entry.
- [ ] Main results table; forest plot of pre-registered differences with record-level bootstrap CIs
      (Holm-adjusted); strength curves; per-record breakdown for S and F.

**Paper artifacts:** main table, forest plot, strength curves, leakage-audit table (if run), RQ3 scatter
(quality metric vs downstream gain, Spearman with bootstrap CI).

---

## Phase 5 — Explainability, LLM layer, demo (Days 22–26, Oct 28–Nov 1)

**Goal:** one explanation method per architecture, a grounded note generator, a working demo, and error
analysis. All method selection uses validation beats. Error analysis reuses saved test predictions
(no new test access). The demo uses validation records.

### Module 8 — Explainability variants
Run on the best checkpoint per architecture for: none, best cVAE, best WGAN-GP ("best" = highest
validation selection metric, recorded before `/final-eval`).

| Variant | Definition |
|---|---|
| A | Grad-CAM 1D on the last conv layer (`cnn` only). |
| B | Integrated gradients (captum), zero and per-beat-mean baselines, 50 steps. |
| C | Gradient × input, or SmoothGrad (n = 25, σ = 0.1 × signal std). |
| D | Occlusion with sliding windows (both architectures), or attention weights if `rnn` has attention. |

**Evaluation** (validation beats, stratified, ≥ 200 per class where available):
- **Faithfulness:** deletion curve — progressively replace the top-k% salient samples by linear
  interpolation; area under the probability-drop curve vs a random-order baseline.
- **Stability:** Spearman correlation of saliency maps for the same beats across seed models.
- **Physiological alignment (descriptive):** saliency mass in R-relative windows
  P [−250, −100] ms, QRS [−50, +50] ms, ST [+50, +150] ms, T [+150, +400] ms (approximate).

**Selection:** faithfulness gain over random (primary); tie → stability. One method per architecture.
**RQ4 analysis:** baseline vs augmented models on faithfulness and alignment; whether augmented models'
saliency differs on real vs synthetic beats (possible artifact reliance).

### Module 9 — LLM note variants

| Variant | Definition |
|---|---|
| A | Deterministic template, no LLM — permanent fallback. |
| B | LLM with JSON input and a constrained system prompt (restate only; fixed disclaimer). |
| C | B plus few-shot examples and a fixed output schema (JSON rendered to a note). |
| D | B plus automated fact-check and regenerate loop (max 2 retries, then fall back to A). |

**Evaluation set:** 50 validation cases, stratified by class, ≥ 10 low-confidence cases.
**Hard gate:** zero unsupported claims across the set. The checker verifies every class name, number and
segment mentioned appears in the input JSON, the disclaimer is present, and low-confidence cases state
that clinician review is required.
**Then:** coverage of required fields, blinded teammate readability rating (1–5 on clarity, accuracy of
emphasis, brevity), latency, cost. Provider and model come from env config. On Kaggle, enable internet
for the API cell.

### Also in Phase 5
Error analysis (confusion matrices per condition; per-record errors for S and F; N↔S confusions with RR
context); ethics section; demo CLI: record + beat index → prediction, probabilities, saliency plot, note.

### Gate 5 checklist
- [ ] Explainability method per architecture chosen by rule; figures generated.
- [ ] LLM variant chosen; zero unsupported claims verified; template fallback works offline.
- [ ] Demo runs from a fresh clone + lockfile.
- [ ] All tests pass; code frozen; git tag `v1.0`.

**Paper artifacts:** class-average saliency aligned on R (baseline vs augmented), faithfulness table, LLM
evaluation table, error-analysis figure.

---

## Phase 6 — Report, paper draft, buffer (Days 27–30, Nov 2–5)

1. `paper/` LaTeX draft: abstract, introduction (contributions), related work, data and protocol,
   methods, results (RQ1–RQ4), grounded reporting, limitations and ethics, reproducibility statement.
   All numbers come from `paper/generated/` (tables + `numbers.tex` macros written by scripts).
2. Course report derived from the paper draft.
3. Reproducibility package: `scripts/reproduce.sh` (download → contract → baselines → banks → grid →
   final eval → figures), lockfile, configs, seeds, `runs_index.csv` mapping every table cell to run IDs.
4. Days 29–30: buffer, claims-vs-evidence pass, demo rehearsal, submission.

---

## Cut order (decided at a gate, in this order)
1. Seeds 5 → 3 (if 5 were planned).
2. Strengths 3 → 2.
3. `rnn` out of the grid (keep it in baselines).
4. Leakage-audit arm → post-deadline.

**Never cut:** inter-patient splits, the test lock, the quality metrics, and the class-weighted, random
oversampling and SMOTE comparisons (cutting strengths 3 → 2 applies to them like any other condition).

---

## Templates

`docs/PROGRESS.md` entry
```
## YYYY-MM-DD (Day N) — Phase P
Done:
Run IDs:
Open issues / risks:
Next:
```

`docs/DECISIONS.md` entry
```
## Gate N — YYYY-MM-DD — status: PENDING APPROVAL | APPROVED by <names>
Module M: winner X — evidence: results/gates/gateN.md — rule: PREREGISTRATION §k
Deviations / exceptions:
Frozen artifacts (hashes):
```

`docs/TEST_ACCESS_LOG.md`
```
| date | kind | git SHA | command | purpose | run IDs | approved by |
|---|---|---|---|---|---|---|
```

`docs/ASSIGNMENTS.md`
```
| phase | module | variant | member | drawn on |
|---|---|---|---|---|
```
