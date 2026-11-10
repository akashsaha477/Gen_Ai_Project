# Rare-arrhythmia detection with generative ECG augmentation

Does class-conditional generative augmentation (cVAE, WGAN-GP) improve detection of rare arrhythmia
classes on MIT-BIH when training and test patients are kept apart? A 30-day project (2026-10-07 to
2026-11-05) run phase by phase.

New to the project? Read `PROJECT_GUIDE.md` first: it explains the whole project and what is done.
`CODE_GUIDE.md` then explains what each code file and function does.

Then `docs/PLAN.md` (phases, rules and the data protocol), `docs/PAPER_TRACK.md` and
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
| 3 generators | phase 3a gate: one generator per family | `python -m ecgaug.workflow phase3a-gate [--approve ...]` | approved 2026-11-02: cVAE D, WGAN-GP A |
| 3 generators | final runs of the two winners, synthetic banks | the same notebooks with `MODE = "final"` | built |
| 3 generators | phase 3 gate: banks frozen | `python -m ecgaug.workflow phase3-gate [--approve ...]` | approved 2026-11-03: banks frozen (cVAE D seed 1 S diversity 0.694 kept and disclosed, DECISIONS A20) |
| 4 comparison | `phase4_augmentation_policy` A–D, seeds 0–2 (DECISIONS A19) | `kaggle/phase4_augmentation_policy.ipynb`, all four letters in one session | built |
| 4 comparison | phase 4a gate: the policy; writes the full grid | `python -m ecgaug.workflow phase4a-gate [--approve ...]` | approved 2026-11-04: policy D (0.5 x N, DTW-screened); full grid strengths 0.1, 0.25, 0.5 |
| 4 comparison | the full grid (96 runs at 3 seeds, resumable) | `kaggle/phase4_full_grid.ipynb` | done 2026-11-04: 96 of 96 runs (Kaggle T4, one session) |
| 4 comparison | phase 4 gate: strengths and checkpoints, on validation | `python -m ecgaug.workflow phase4-gate [--approve ...]` | approved 2026-11-04 (Akash Saha): cVAE best generative for cnn and rnn |
| 4 comparison | the single final evaluation on the test set | `ALLOW_TEST=1 python -m ecgaug.final_eval --confirm "run final eval"` | done 2026-11-04 (once; `results/final_eval/`, `figures/final_eval/`) |
| 5 explanations | `phase5_explainability` A–D, `phase5_clinical_notes` A–D, error analysis, demo | `python -m ecgaug.experiments.phase5_*`, `python -m ecgaug.demo` | done 2026-11-04 (local CPU; notes with gemini-3.5-flash-lite) |
| 5 explanations | phase 5 gate, then `git tag v1.0` | `python -m ecgaug.workflow phase5-gate [--approve ...]` | approved 2026-11-04 by Akash Saha; code tagged `v1.0` |
| 4 comparison | RQ3 analysis; leakage audit (optional, after the final evaluation) | `python -m ecgaug.eval.rq3`; `kaggle/phase4_leakage_audit.ipynb` | RQ3 done; leakage audit done 2026-11-04 on Kaggle (`results/phase4_leakage_audit/`, `python -m ecgaug.figures leakage_audit`) |
| 6 paper | post-hoc analyses for the review (PREREGISTRATION v4, P1-P4; reported apart, never replacing the primary results) | `python -m ecgaug.experiments.phase6_rr_reference`, `phase6_external_validation`, `phase6_audit_check`; `kaggle/phase6_extra_seeds.ipynb`; `python -m ecgaug.figures posthoc` | done 2026-11-05: P1 RR references below the cnn; P2 INCART: generative augmentation lowered 1 of 4 confirmatory comparisons (cnn cVAE; 4 of 4 before intervals resampled INCART's 32 patients instead of its 75 records, A30), SVDB: 1 of 4 improved (rnn cVAE); P3 five seeds: no verdict changed; P4: inflation holds on two more splits (0.803, 0.777 vs 0.088), WGAN-GP at the gate strength 0.769, H2 verdict unchanged |
| 6 paper | P5 follow-up: patient-baseline RR ratio, dataset v2, full grid retrained (PREREGISTRATION v5, post hoc) | `python -m ecgaug.experiments.phase6_rr_baseline --part {data,test-ratios,select,references,test,external}`; `kaggle/phase6_rr_baseline.ipynb` (`--part grid`) | done 2026-11-05: training S beats below 0.9 rose from 0.51 to 0.87, but test rare-class macro-F1 did not detectably change (v2 − v1 from −0.042 to +0.046, no CI excludes zero); generative augmentation still not detectably better than none; RR-only LDA 0.022 → 0.074; external (Kaggle CPU): v2 point above v1 in 12 of 12 conditions on INCART and 10 of 12 on SVDB, generative augmentation lowered rare-class F1 in 0 of 4 on INCART (3 of 4 before A30) and 1 of 4 on SVDB |
| 6 paper | P6 patient-relative beats (dataset v3, PREREGISTRATION v6) and diagnostic | `kaggle/phase6_patient_template.ipynb` (`PART` = data, grid, diagnostic) | done 2026-11-05: validation S F1 cnn 0.350 → 0.472, rnn 0.306 → 0.181; adoption rule (mean over both) not met, so no full grid and no test read; diagnostic: cnn S PR-AUC training 0.94 vs validation 0.26 (v1), validation 0.40 with v3 |
| 6 paper | P7 cnn alone with v3 inputs, full comparison; P8 mechanism (PREREGISTRATION v7) | `kaggle/phase6_relative_cnn.ipynb` (`PART` = grid on GPU; test, external on CPU), `--part select` locally; `kaggle/phase6_augmentation_mechanism.ipynb` (CPU) | done 2026-11-05: test rare-class macro-F1 cnn none 0.371 (v1 0.088), S F1 0.312 (v1 0.148), F F1 0.429 (v1 0.029); generative vs none still not better (cVAE -0.067, WGAN-GP -0.059, Holm p 0.352); INCART 0.200 (v1 0.256), SVDB S F1 0.368 (v1 0.314); P8: synthetic S beats 0.57-0.65 nearest to record 209 (real 0.60), validation S beats nearest to records with 1-2% of training S beats |
| 6 paper | P9 patient-transfer augmentation (PREREGISTRATION v8) | `kaggle/phase6_patient_transfer.ipynb` (`PART` = grid on GPU; test, external on CPU), `--part select` locally | done 2026-11-05: validation S F1 rnn none 0.306 -> transfer 0.46-0.53; test rare-class macro-F1 rnn none 0.075 -> transfer 0.105-0.109, cnn 0.096 -> 0.089-0.116, no Holm rejection vs none; transfer above plain generative in 4 of 4 pairs; external: INCART lowered in 4 of 4, SVDB rnn improved in 2 |
| 6 paper | P10 per-patient batch normalization on frozen checkpoints (PREREGISTRATION v9) | `kaggle/phase6_patient_adaptation.ipynb` (`PART` = validation, test, external; CPU only) | done 2026-11-06: validation S F1 v1 cnn 0.349 -> 0.510 (adopted), v1 rnn and P7 cnn not adopted; test v1 cnn 0.088 -> 0.080 (n.s.; S F1 0.148 -> 0.160, F F1 0.029 -> 0.000); external not detectably changed |
| 6 paper | P11 the P7 cnn trained on DS1 plus the INCART and SVDB patients (PREREGISTRATION v10) | `kaggle/phase6_more_patients.ipynb` (`PART` = grid on GPU; test on CPU), `--part select` locally | done 2026-11-06: test rare-class macro-F1 none 0.147 (P7 0.371; F F1 0.000), class-weighted 0.407 (best of the study), random oversampling 0.383, cVAE 0.245, WGAN-GP 0.217; no Holm rejection |
| 6 paper | P12 the cVAE and WGAN-GP trained on patient-relative beats of P11's patients (PREREGISTRATION v11) | `kaggle/phase6_relative_generators.ipynb` (`PART` = train with `FAMILY` on GPU; test on CPU), `--part select` locally | done 2026-11-06: synthetic S beats over 31.6 / 29.2 effective patients (real 32.0; P11's banks 10.9 / 14.3); test rare-class macro-F1 cVAE 0.368 (P11 0.245; F F1 0.175 to 0.475), WGAN-GP 0.243 (P11 0.217); no Holm rejection |
| 6 paper | P13 RR intervals over the mean RR of their recording, P11's and P12's conditions (PREREGISTRATION v12) | `kaggle/phase6_record_rr.ipynb` (`PART` = grid on GPU; test on CPU), `--part check` and `--part select` locally | done 2026-11-06: as inputs to the cnn, no detectable change (none 0.147 to 0.216, S F1 0.294 to 0.399; Holm p 1.000); with Huang et al.'s rule on top (descriptive) S F1 0.531-0.578 and rare-class macro-F1 up to 0.523 (cVAE), the study's highest |
| 6 paper | P14 the P13 cnn trained on P13's data plus 400 Icentia11k patients drawn at random (PREREGISTRATION v13) | `kaggle/phase6_icentia.ipynb` (`PART` = grid on GPU, which downloads and builds; test on CPU), `--part check` and `--part select` locally | done 2026-11-07: 388 usable patients (S 13,412 and V 6,722 training beats, no F, no S-dominant recording); test rare-class macro-F1 fell in all four conditions (none 0.216 to 0.121, class weighting 0.380 to 0.247, random oversampling 0.374 to 0.159, cVAE 0.356 to 0.304; Holm p 0.543-1.000); the cVAE ranked first (vs none +0.184, Holm p 0.357) |
| 6 paper | P15 a class-conditional diffusion model as the third generator family, on P13's data (PREREGISTRATION v14) | `kaggle/phase6_diffusion.ipynb` (`PART` = train on GPU; test on CPU), `--part select` locally | done 2026-11-09: 3 fits, none diverged (about 18 minutes each on a T4); validation chose strength 0.25 (S F1 0.741); test rare-class macro-F1 0.280 (S F1 0.308, F F1 0.252), vs none +0.064 and vs the cVAE -0.076 (Holm p 0.806 both); with Huang et al.'s rule 0.410. At the phase 3 budget its loss was still falling and its class medians were far from the real beats'; P17 trained it five times longer |
| 6 paper | P16 Huang et al.'s SVEB stage with an atrial-fibrillation veto: a detector from RR intervals only, fitted on the MIT-BIH AF database; threshold and form chosen on training-side data (PREREGISTRATION v15) | `kaggle/phase6_af_rule.ipynb` (test on CPU), `--part fit` and `--part check` locally | done 2026-11-09: the veto raised rare-class macro-F1 in all six conditions (+0.018 to +0.034, every 95% CI above 0, Holm p 0.018-0.057) by removing false S calls (cVAE: S +P 0.415 to 0.496, Se 0.920 to 0.903). The study's highest score until P18: cVAE + rule + veto 0.557 (S F1 0.640, F F1 0.474; 0.583 without record 202) |
| 6 paper | P17 P15's diffusion model trained five times longer (100,000 steps), nothing else changed (PREREGISTRATION v16) | `kaggle/phase6_diffusion_long.ipynb` (`PART` = train on GPU; test on CPU), `--part shape` and `--part select` locally | done 2026-11-09: 3 fits of 100,000 steps, none diverged (about 87 minutes each on a T4); the loss fell further (0.096 to 0.084-0.085) but the class medians stayed far from the real beats' (correlation V 0.60-0.64, F 0.75-0.80; cVAE 0.94-0.96, 0.95-0.96); exploratory: more V peaks where real beats have theirs (0.42-0.49 to 0.61-0.66; real 0.75). Validation chose strength 0.1 (S F1 0.754); test rare-class macro-F1 0.267 (S F1 0.307, F F1 0.226), vs P15 -0.013 and vs none +0.050 (Holm p 0.924 both); with Huang et al.'s rule 0.404, with the rule and the AF veto 0.440. More steps alone did not close the gap to the cVAE |
| 6 paper | P18 each condition's class probabilities averaged over ten independently trained runs (seeds 0-2 existing, 3-9 new), with Huang et al.'s SVEB stage and P16's AF veto on top (PREREGISTRATION v17) | `kaggle/phase6_seed_ensemble.ipynb` (`PART` = grid on GPU; test on CPU), `--part check` locally | done 2026-11-09: 42 new runs (38-74 s each on a T4); validation: the ten-run ensemble beat its single runs' mean S F1 and V F1 in all six conditions. Test (one posthoc read, CPU), rule and veto on top, P16's single runs to the ten-run ensemble: none 0.319 to 0.316, class weighting 0.512 to 0.515, random oversampling 0.525 to 0.541, cVAE 0.557 to 0.587, WGAN-GP 0.375 to 0.474, diffusion 0.436 to 0.433; detectable only for the WGAN-GP (+0.099, Holm p 0.006; cVAE +0.030, Holm p 1.000). The study's highest score: the cVAE's ensemble 0.587 (S F1 0.647, F F1 0.527; 0.615 without record 202); its ten single runs 0.451-0.610, mean 0.536 |
| 6 paper | P19 P18's frozen pipeline (ten runs per condition, Huang et al.'s SVEB stage, P16's AF veto) on the first segment of 1,000 Icentia11k patients no analysis had used, every usable beat; nothing fitted or chosen; primary measure S F1, as Icentia11k has no F labels (PREREGISTRATION v18) | `kaggle/phase6_fresh_patients.ipynb` (`PART` = run on a T4; no test records) | done 2026-11-10: 977 segments usable (3,675,681 N, 34,072 S, 29,473 V beats), one 72-minute T4 session. With the stage and the veto, S F1 0.286-0.295 in every condition (cVAE 0.294 [0.237, 0.350], V F1 0.648; on DS2 0.647 and 0.936). Ten-run ensemble vs P16's single runs: +0.006 to +0.016 S F1, detectable in 5 of 6 conditions (Holm p 0.001; none +0.001, p 0.815). cVAE vs the others: -0.001 to +0.008, none detectable. The ensembles alone scored higher, S F1 0.259-0.382 (class weighting 0.382, cVAE 0.348): the stage raised S Se (cVAE 0.303 to 0.629) but cut S +P (0.408 to 0.157), and with the veto it lowered S F1 against the ensemble alone in 4 of 6 conditions (descriptive) |
| 6 paper | P20 Huang et al.'s SVEB stage and P16's AF veto chosen per condition among 50 settings (the ten-run ensemble alone; the stage at t 0.60-0.90 with no veto or the veto at 0.50-0.95) by S F1 on P14's 400 Icentia11k patients, which P18's runs never saw; scored on P19's patients and once on DS2 (PREREGISTRATION v19) | `kaggle/phase6_rule_tuning.ipynb` (`PART` = build on CPU, run on GPU, test on CPU), `--part score` locally | done 2026-11-10: chosen on 388 tuning patients: stage 0.65-0.75 with veto 0.70-0.80 in every condition. On P19's 977 patients S F1 rose in all six conditions against P19's setting (cVAE 0.294 to 0.414, +0.085 to +0.124, Holm p 0.001) and against the ensemble alone (+0.027 to +0.119, Holm p 0.001-0.002). On the test records (one read, CPU) rare-class macro-F1 fell in five of six against P18's setting, none detectably (cVAE 0.587 to 0.459, S F1 0.647 to 0.391; WGAN-GP 0.474 to 0.497; Holm p 0.502-1.000): record 232 holds 1,381 of 1,836 test S beats and no tuning patient is S-dominant |
| 6 paper | P21 McSharry et al.'s ECG simulator as a prior for P12's cVAE and WGAN-GP (SimGANs' Euler loss on each generated beat plus a real training patient's template, weight chosen on validation among 0.1, 1 and 10), and P12's cVAE banks with physiological fusion beats (a fraction of a real training V deviation); each through P18's ten-run pipeline with the stage and the veto chosen on P14's patients (PREREGISTRATION v20) | `kaggle/phase6_physics_prior.ipynb` (`PART` = check on CPU, train and fusion on GPU, score on GPU, test on CPU) | done: the Euler loss barely moved the generators (banks score as P12's); fusion model below its stop rule, not run. P19's patients, S F1 vs P18: cVAE −0.002 (Holm p 1.0), WGAN-GP +0.022 (Holm p 0.12). DS2 (16th read), rare-class macro-F1: cVAE 0.481 vs 0.459 (Holm p 0.62), WGAN-GP 0.339 vs 0.497 (Holm p 0.49) |
| 6 paper | P22 a larger classifier: phase 2's ResNet1D (6.7 times the cnn's parameters) in P18's pipeline for no augmentation, the cVAE and the WGAN-GP, compared with P18's cnn ensembles on P19's patients and once on DS2 (PREREGISTRATION v21) | `kaggle/phase6_bigger_classifier.ipynb` (`PART` = run on GPU, test on CPU) | pre-registered 2026-11-10 |
| 6 paper | P23 P21 with a fitted physics loss: each generated beat compared with the simulator's best fit for its class's wave timing (signal form, chosen after a check on 60 training beats per class), weight chosen on validation (PREREGISTRATION v21) | `kaggle/phase6_physics_fitted.ipynb` (`PART` = check on CPU, train and score on GPU, test on CPU) | pre-registered 2026-11-10 |
| 6 paper | the paper and the course report (kept out of this repository): tables, numbers and results sentences | `python -m ecgaug.paper` | drafted 2026-11-04 with every result in (`paper/main.pdf`, `paper/report.pdf`); claims table filled; open: author names and team contributions |

"Built" means the code, configs, notebook, tests and brief exist. A dress rehearsal runs every stage in
order on fake records (`ECGAUG_REHEARSAL=1 uv run pytest tests/test_rehearsal.py`, about 8 minutes). Every
runner refuses an official run until the gates before it are approved, and says which. `--preview`
runs earlier with stand-ins and marks the result provisional; a gate never approves a preview.

Since 2026-11-05 the post-hoc work runs entirely on Kaggle (the local machine could not carry it): CPU sessions for everything that trains nothing, one T4 session per training grid. `kaggle_run.py --tests` runs the full test suite first and stops if it fails; `--data-dir` copies an attached `data/` folder (external databases), never a test array. Attach checkpoint datasets as a `runs/` folder next to a README (Kaggle drops the top folder of a dataset that holds only one).

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
| `paper/` | the LaTeX draft, its `generated/` tables and numbers and `claims_vs_evidence.md`; written locally by `python -m ecgaug.paper` and kept out of this repository |
| `scripts/reproduce.sh` | the whole pipeline in order |
| `scripts/phase3_bank_figure.py` | real training beats next to each frozen bank, per class (`figures/phase3_gate/banks.png`) |
| `scripts/phase4a_policy_figure.py` | the policy pilot against no augmentation, per bank (`figures/phase4a_gate/policies.png`) |
| `scripts/phase4_gate_figure.py` | every grid condition at its chosen strength against no augmentation (`figures/phase4_gate/conditions.png`) |
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
After the final evaluation two more kinds exist: `ALLOW_TEST_AUDIT=1` for the leakage audit and its checks, and
`ALLOW_TEST_POSTHOC=1` for the post-hoc analyses of pre-registration v4 (`load_test_posthoc` refuses unless the final
evaluation is already logged). Neither may feed a decision.

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
written by `python -m ecgaug.paper` from the result files (`src/ecgaug/paper_text.py`) into `paper/generated/`; the
draft itself (`paper/main.tex`, `paper/report.tex`) is kept out of this repository. The LLM note variants need `GEMINI_API_KEY` (free
tier from aistudio.google.com; optionally `ECGAUG_LLM_MODEL`) in the environment; without it the template note is used.

## Troubleshooting

- **`KeyError: '_items'` on `import matplotlib.pyplot` (macOS):** the system font listing returns
  nothing on some macOS versions. Call `ecgaug.eval.plotting.import_pyplot()` once; it builds
  matplotlib's font cache without that step, after which normal imports work.
