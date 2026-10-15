# PAPER TRACK

Amended 2026-10-08 after the kickoff review: see `docs/DECISIONS.md` (A1–A11).

## Working titles
- Does Generative Augmentation Help Rare-Arrhythmia Detection Under Inter-Patient Evaluation?
  A Controlled Comparison of Conditional VAEs and WGAN-GP
- Leakage-Free Evaluation of Generative ECG Augmentation for Rare Arrhythmia Classes

## Positioning (state this honestly in the paper)
GAN/VAE augmentation for MIT-BIH beat classification is a crowded topic, and many reported gains come
from intra-patient splits that leak patient identity. The contribution is not a new generator. It is a
careful answer to whether, when and why generative augmentation helps, under conditions closer to
deployment.

**Contributions**
- **C1 Controlled, leakage-free comparison:** cVAE vs WGAN-GP vs SMOTE vs random oversampling vs
  class-weighted loss vs none;
  two architectures; multiple seeds; inter-patient DS1/DS2; record-level uncertainty; pre-registered selection.
- **C2 Leakage audit:** how much intra-patient evaluation inflates baseline performance and the apparent
  benefit of augmentation.
- **C3 Metric validity:** which synthetic-quality metrics (DTW, spectral, FID-analog, TSTR, diversity,
  C2ST) predict downstream rare-class gains, and which don't.
- **C4 Explainability lens:** whether augmentation changes the faithfulness and physiological alignment
  of explanations, including reliance on synthetic artifacts.
- **C5 System:** grounded clinical-note generation with automated fact verification
  (zero-unsupported-claim gate).

## Research questions and pre-registered hypotheses

H1 is the only confirmatory hypothesis. H1c is tested in the secondary family. H1b, H2, H3 and H4 are
exploratory and are reported as such.

**RQ1 — Does generative augmentation improve rare-class detection under inter-patient evaluation?**
- H1 (confirmatory): a generative condition (cVAE or WGAN-GP) improves test rare-class macro-F1 over
  no augmentation for at least one architecture: positive difference with Holm-adjusted two-sided
  bootstrap p < 0.05 within the confirmatory family of 4 tests.
- H1b (exploratory): gains are larger for F (morphology-defined) than for S (largely defined by
  timing/prematurity), because the generators model beat morphology, not RR context.
- H1c (secondary): generative augmentation beats SMOTE, random oversampling and class-weighted loss on
  the primary metric. It may well not. Report either way.

**RQ2 — How much does intra-patient evaluation inflate results?**
- H2 (exploratory): intra-patient splits give higher baseline scores and larger apparent augmentation
  gains than inter-patient splits.

**RQ3 — Do synthetic-quality metrics predict downstream benefit?**
- H3 (exploratory): across generator checkpoints and variants, at least one metric correlates with
  downstream validation gain (Spearman ρ, bootstrap CI excluding 0); fidelity-only metrics
  (FID-analog) correlate more weakly than diversity-aware ones.
- TSTR is analysed separately from the fidelity and diversity metrics: it and the downstream gain are
  both validation scores of a CNN trained on the same synthetic beats, so some correlation is built in.
- Intervals come from resampling generator training runs (variant × seed), not checkpoints, because
  checkpoints of one run are dependent.

**RQ4 — Does augmentation change what the classifier relies on?**
- H4 (exploratory): augmented models show equal or higher saliency faithfulness than the baseline,
  with saliency in physiologically relevant windows. Any shift toward regions where synthetic beats
  differ from real ones is reported as potential artifact reliance.

## Metrics
- **Primary:** rare-class macro-F1 (mean of S and F F1).
- **Secondary:** per-class F1; one-vs-rest PR-AUC; AAMI-style Se and +P for S (SVEB) and V (VEB);
  N recall; macro-F1 over N/S/V/F. Q excluded (counts reported).

## Statistical analysis plan
- **Unit of resampling = test record (patient).** Cluster bootstrap, 10,000 replicates, resampling the
  22 DS2 records with replacement; metrics on pooled beats within each replicate; seeds averaged within
  each replicate. Beat-level bootstrap is not used: beats within a patient are correlated, and it would
  understate uncertainty.
- Report the fraction of replicates where a class is absent, and, for S and for F, the fraction of
  replicates that lack the record holding most of that class's test beats (S and F are concentrated in
  few records, so the second number is the informative one).
- A replicate in which S or F has no true beats has an undefined primary metric. Such replicates are
  excluded from intervals and p-values, and their fraction is reported.
- **Pre-registered comparisons, per architecture**, each condition at the strength chosen on validation.
  "Best generative" is the generative condition with the higher validation selection metric, fixed
  before `/final-eval`.
  - *Confirmatory family (H1, 2 × 2 architectures = 4 tests):* (1) cVAE vs none, (2) WGAN-GP vs none.
  - *Secondary family (4 × 2 architectures = 8 tests):* (3) cVAE vs WGAN-GP, (4) best generative vs
    SMOTE, (5) best generative vs class-weighted CE, (6) best generative vs random oversampling.
- **Multiplicity:** each comparison gets a two-sided percentile-bootstrap p-value,
  p = min(1, 2 · min(#{Δ* ≤ 0} + 1, #{Δ* ≥ 0} + 1) / (B + 1)). Holm–Bonferroni is applied to the
  p-values within each family at α = 0.05. Unadjusted 95% percentile intervals are reported next to
  the adjusted p-values.
- Effect sizes: Δ rare-class macro-F1 with 95% CI. Seed variance (mean ± sd across seeds) is reported
  separately from record-level uncertainty.
- **Sensitivity analyses:** leave-one-record-out for S and F; per-record results; test results without
  record 202 (same subject as training record 201); 5 seeds if run.
- No significance claims from seed-level tests with 3 seeds.
- F1 is computed from argmax decisions with no threshold tuning. Rebalancing moves the operating
  point, so PR-AUC for S and F is read alongside F1 to separate ranking quality from threshold effects.

## Figures and tables (all generated by scripts)
| Item | Content | Phase |
|---|---|---|
| Table 1 | Dataset and protocol: records and class counts per split | 1 |
| Table 2 | Baselines (validation in P2, test after final eval) | 2 / 4 |
| Table 3 | Synthetic quality and diagnostics, all variants | 3 |
| Table 4 | Main grid results with record-level CIs | 4 |
| Table 5 | Leakage audit: intra- vs inter-patient | 4 / post |
| Table 6 | Explainability faithfulness and stability | 5 |
| Table 7 | LLM note evaluation | 5 |
| Fig 1 | Pipeline and protocol diagram | 1 |
| Fig 2 | Real vs synthetic beats per class (median + IQR on R) | 3 |
| Fig 3 | Forest plot of pre-registered differences | 4 |
| Fig 4 | Augmentation-strength curves | 4 |
| Fig 5 | RQ3: quality metric vs downstream gain | 4 |
| Fig 6 | Class-average saliency aligned on R, baseline vs augmented | 5 |
| Appendix | Per-record counts, variation-study tables for every experiment, training curves, failure modes | all |

## Threats to validity (write these into the paper, specifically)
- Small test population (22 records: 21 from patients not seen in training, plus record 202, whose
  subject also contributes DS1 record 201); rare classes concentrated in few records → wide CIs,
  results may reflect individual patients.
- Single database from one hospital, recorded decades ago; narrow demographics; no external validation
  unless added post-deadline.
- Single lead; beat-level classification on reference R-peaks (no detector errors).
- Generators learn from few minority-class patients, so synthetic diversity is bounded by them.
- Selection relies on a small validation set of DS1 patients.
- Saliency methods are approximate; segment windows are fixed R-relative approximations.
- LLM notes evaluated by team members, not clinicians.

## Reproducibility checklist
- Public data, download script, record lists and label mapping in `contract.json`.
- Lockfile, configs, seeds, git SHA per run, `runs_index.csv`.
- `PREREGISTRATION.md` with commit SHA and date, committed before the governed experiments; each
  version tagged `prereg-vK` and pushed the day it is committed.
- `TEST_ACCESS_LOG.md` showing a single final evaluation.
- Code release; check PhysioNet licence terms before releasing synthetic banks derived from the data.

## Related-work anchors
Verify every citation from its DOI or arXiv page and fetch real BibTeX. Never invent an entry or field.
- de Chazal et al., 2004 — inter-patient DS1/DS2 protocol.
- Systematic review of ECG arrhythmia classification: adherence to standards and fair evaluation,
  arXiv:2503.07276 (2025).
- Shaker et al., IEEE Access 8 (2020) — GAN augmentation vs SMOTE/ADASYN on MIT-BIH.
- Arrhythmia classification using CGAN-augmented ECG signals (AC-WGAN-GP), arXiv:2202.00569.
- Gulrajani et al., 2017 (WGAN-GP); Arjovsky et al., 2017 (WGAN); Kingma & Welling, 2014 (VAE);
  Sohn et al., 2015 (CVAE).
- Chawla et al., 2002 (SMOTE); Dablain, Krawczyk & Chawla, IEEE TNNLS 2022 (DeepSMOTE).
- Alcaraz & Strodthoff, Computers in Biology and Medicine 163 (2023) — SSSD-ECG, the diffusion baseline
  reviewers will expect to see discussed.
- Wagner et al., Scientific Data 2020 (PTB-XL); Strodthoff et al., IEEE JBHI 2021 (PTB-XL benchmarks).
- Storås et al., JAMIA 2025 (evaluating gradient-based ECG heatmaps); Hicks et al., Scientific Reports 2021.
- Zhao et al., ECG-Chat, arXiv:2408.08849 (2024) — LLM ECG reports and hallucination.
- Moody & Mark, IEEE EMB Magazine 2001 — MIT-BIH database.

## Venue notes (team decides)
Biomedical signal-processing journals (e.g. IEEE JBHI, Computers in Biology and Medicine, Biomedical
Signal Processing and Control, Physiological Measurement) or ML-for-health workshops and symposia.
Check scope, page limits and deadlines directly; don't assume them.

## Post-deadline extensions (ranked by reviewer value)
1. Five seeds everywhere (if not already).
2. External validation: train on MIT-BIH DS1, test on another beat-annotated database (e.g. INCART on
   PhysioNet), with class mapping and resampling documented.
3. Diffusion baseline as a third generator family.
4. Leakage-audit experiment, if skipped.
5. Clinician rating of a sample of LLM notes.
