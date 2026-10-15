# PROGRESS

## 2026-10-08 (Day 2) — Phase 1

Done:
- Kit installed at the repo root (c1bec44). Plan amended after the kickoff review, A1–A11 (39f4f29,
  `docs/DECISIONS.md`); A2 and A7 change the plan and await the whole team.
- Environment: Python 3.11 with `uv.lock` (torch 2.14.1, numpy 2.4.6, scipy 1.17.1,
  scikit-learn 1.9.1, wfdb 4.3.1).
- MIT-BIH 1.0.0 downloaded from PhysioNet. All 144 files (48 records × hea, dat, atr) match
  PhysioNet's published SHA-256 list (`results/phase1_data/raw_checksums.json`).
- Beat extraction with the AAMI map and RR intervals; per-record counts for 44 records
  (`results/phase1_data/record_counts.csv`). Totals: DS1 N 45,866, S 944, V 3,788, F 415, Q 8; DS2 N 44,259, S 1,837, V 3,221, F 388, Q 7.
  Sanity check passes: the largest deviation from the docs/RULES.md reference is S in DS1,
  944 against 942 (+0.21%)
  (`results/phase1_data/sanity_check.json`). DS1 also has 472 flutter-wave
  annotations outside the AAMI map (record 207), excluded like Q.
- Test lock in place: raw reads go through one guarded file, `load_test()` needs `ALLOW_TEST=1`.
  `docs/TEST_ACCESS_LOG.md` holds 1 construction entry (DS2 label counts) and 0 evaluation entries.
- the evaluation code: metrics, record-level cluster bootstrap, paired differences with bootstrap p-values, Holm,
  and the selection rule with its tie margin, all with hand-computed tests.
- Probe CNN (18,628 parameters; 18,762 with the RR side input), training loop with
  early stopping, run logging.
- phase1_input_representation: configs A–D, one brief per variant, and the runner. `--check` on DS1 gives 50,969 usable
  beats for every variant (N 45,824, S 943, V 3,788, F 414; dropped at record edges: N 42, S 1,
  V 0, F 1), an identical beat set across variants, and no non-finite values.
  Shapes: A (50969, 180), B (50969, 360), C (50969, 180), D (50969, 63) (`results/phase1_data/representation_check_*.json`).
  Smoke runs on fake data pass for A–D.
- Validation-split options with DS1 per-record counts: `results/phase1_data/val_options.md`.
- Kaggle runner notebook and `kaggle/ingest_runs.py`.
- `docs/PREREGISTRATION.md` v1 drafted, status DRAFT.
- Tests: 123 pass, also in a fresh clone installed with `uv sync --frozen`.

Run IDs: none. No variant experiment has been run and no model has seen real data.

Open issues / risks:
- **Blocked on the team:** validation split, the (new) items of PREREGISTRATION v1, and plan changes
  A2 and A7.
- Nothing is pushed: `main` is ahead of `origin/main` by every commit since "Papers".
- The Kaggle notebook has not been executed on Kaggle. The repository is private, so it needs a
  `GITHUB_TOKEN` Kaggle secret.
- `wfdb.dl_database` took about 29 minutes here. The last files were fetched in parallel with curl from
  the same PhysioNet URLs; the checksum match covers every file. On Kaggle, attach the verified raw
  files as a private dataset (`RAW_DATASET_DIR` in the notebook) instead of downloading each session.
- The AAMI +P convention for V (fusion beats called V are not false positives) is implemented from
  memory of de Chazal et al. (2004) and must be checked against the paper before it is quoted.
  The plain precision for V is reported next to it.
- Contract arrays can differ in the last bit between platforms (filtering, resampling). The frozen
  contract must be built once and distributed as files; loaders verify the hash.
- Wall-clock of a phase1_input_representation variant run is not measured yet.
- This laptop has about 11 GB of free disk.
- Commit 2f18fa1 lacks the data package because `.gitignore` matched `src/ecgaug/data/`; fixed in
  410986b. The commits in between do not pass their tests on their own.
- matplotlib's font scan fails on this macOS version; `ecgaug.eval.plotting.import_pyplot()` works
  around it (see README).

Next:
1. Team decides the validation split and approves PREREGISTRATION v1 (with or without A2 and A7).
2. Then: write `configs/validation_split/approved.yaml`, fill §2, set the status to IN FORCE, commit, tag
   `prereg-v1`, push.
3. Draw the phase1_input_representation assignments (`docs/ASSIGNMENTS.md`); each member runs `/variant 1 <letter>`.
4. `/gate 1`.

## 2026-10-09 (Day 3) — Phase 1

Done:
- Commit messages rewritten as short one-liners; history pushed to GitHub.
- Kaggle setup for the team: one notebook, a runner script, a raw-data zip builder (66 MB zip of the
  verified MIT-BIH files) and a guide. Rehearsed in a fresh clone on a laptop; not yet run on Kaggle.
- All Kaggle files moved to a top-level `kaggle/` folder (guide is `kaggle/README.md`).
- README gained a status section and a folder map.

Run IDs: none.

Open issues / risks:
- Validation split and variant assignments still undecided; Phase 1 was due today.

Next:
- Decide the validation split, draw assignments, run variants A–D.

## 2026-10-13 (Day 7) — Phase 1

Done:
- Validation split decided: option V1 (validation records 115, 118, 201, 205, 223), stored in
  `configs/validation_split/approved.yaml`. Selection metric: S F1 with V F1 as tie-break.
- The runner no longer checks `docs/PREREGISTRATION.md`; a variant runs with the approved split and a
  committed tree. The document is kept as a description of the rules.
- Dataset decision recorded: MIT-BIH only. PTB-XL was considered and rejected as the main dataset
  because it has no beat-level labels.
- Tests: 121 pass.

Run IDs: none. No variant has been run yet.

Open issues / risks:
- About four days behind the plan (Phase 2 was due to end on Day 8). The Day 15 buffer is used up.
- PTB-XL pretraining decided (DECISIONS A12): supervised, each beat labelled with its recording's
  diagnosis, then fine-tuning on MIT-BIH; one extra baseline experiment in Phase 2. Design is in
  `docs/PLAN.md`. Nothing is built for it yet; it waits for the phase 1 gate because it needs the winning beat window.
- Variant assignments are still TBD in `docs/ASSIGNMENTS.md`.
- The Kaggle notebook has not been executed on Kaggle. Wall-clock of a variant run is unmeasured.
- Two commits are local only (approved split, dataset decision) until pushed.

Next:
1. Push, draw assignments, run `/variant 1 A..D` (locally or on Kaggle).
2. `/gate 1`: pick the input representation and freeze the dataset.
3. Phase 2: baselines, quality metrics and the phase2_ptbxl_pretraining experiment.

## 2026-10-13 (Day 7), later — PTB-XL preprocessing

Done:
- R-peak detector (`src/ecgaug/data/rpeaks.py`) for data without beat annotations. Validated on MIT-BIH
  DS1 against the reference annotations: sensitivity 0.9963, positive predictivity
  0.9933 (TP 51,304, FN 189, FP 346); median timing offset 0.0 ms,
  97% of matches within 10 ms (`results/phase2_ptbxl_data/rpeak_validation_ds1.json`). Parameters were set
  once and not tuned. DS2 was not read.
- PTB-XL pipeline (`src/ecgaug/data/ptbxl.py`): metadata exclusions, lead II by name, signal checks,
  resampling to 360 Hz, peak detection, QRS-contrast check, RR and usable-beat filters, superclass
  labels per beat, patient-wise folds (1–8 train, 9 validation, 10 unused), hashed output.
- Run on a real sample of 59 PTB-XL recordings (ecg_id 1–64, 5 without a diagnostic superclass):
  59 kept, 587 beats (`results/phase2_ptbxl_data/ptbxl_preprocessing_sample.json`).
- Alignment with MIT-BIH on that sample (`results/phase2_ptbxl_data/ptbxl_alignment_sample.json`): correlation of the
  mean PTB-XL beat with the mean DS1 normal beat is 0.93 (A), 0.89 (B), 0.93 (D); the largest
  deflection sits at the window centre for 96% of PTB-XL beats against 97% of MIT-BIH beats (A).
- Tests: 133 pass.

Run IDs: none.

Open issues / risks:
- Only 64 of about 21,800 PTB-XL recordings have been processed. The full run should happen on Kaggle
  with a PTB-XL dataset attached; PhysioNet is too slow from this laptop and disk is short.
- The QRS-contrast threshold (4.0) was set from DS1 segments, the 59-recording sample and white noise.
  Check how many recordings it removes on the full data.
- Pretraining and fine-tuning are not built; they follow phase 1 gate.

Next:
1. phase1_input_representation variants and the phase 1 gate (unchanged).
2. Full PTB-XL preprocessing on Kaggle, then the pretraining runner in Phase 2.

## 2026-10-13 (Day 7), evening — PTB-XL on Kaggle: preprocessing and phase2_ptbxl_pretraining experiment

Done:
- PTB-XL preprocessing made ready for the full data: worker processes (same result for any number of
  workers), search for the dataset folder below an attached Kaggle input, optional download of PTB-XL
  1.0.3 from PhysioNet as one zip, comparison of every file read with PhysioNet's checksum list, and
  heart-rate, QRS-contrast and 50 Hz statistics in the report.
- Edge effect measured on MIT-BIH DS1 (`python -m ecgaug.data.checks`,
  `results/phase2_ptbxl_data/method_checks_ds1.json`): when a 10-second piece is band-pass filtered on its own,
  beats 0.4-1.0s from an end correlate with the same beats filtered inside the full record at a 5th
  percentile of 0.854 without padding, 0.922 with odd reflection, 0.976 with
  edge-value padding and 0.990 with mirror padding (3,061 beats). Mirror padding of 2 s is now used
  for PTB-XL. It only matters for the band-passed representation (B).
- Mains interference: 60 Hz is clearly present in 55% of DS1 records (median ratio 11.9);
  the PTB-XL sample shows no 50 Hz interference (share above 10: 0%). No notch filter was added.
- phase2_ptbxl_pretraining (`src/ecgaug/experiments/phase2_ptbxl_pretraining.py`, `configs/phase2_ptbxl_pretraining/`): one command
  preprocesses PTB-XL, pretrains once per seed (5 superclasses per beat, binary cross-entropy, early
  stopping on PTB-XL fold 9), fine-tunes on MIT-BIH from the pretrained weights and from scratch with
  the phase1_input_representation budget, applies the selection rule and writes `results/phase2_ptbxl_pretraining/<letter>.json`.
- Shared runner code moved to `src/ecgaug/experiments/common.py`; the phase1_input_representation runner uses it and
  gives the same results as before (a test checks that the from-scratch runs equals a phase1_input_representation run).
- Kaggle: notebook parameters and guide for the experiment; the bundle and `ingest_runs.py` now carry
  `results/phase2_ptbxl_data/ptbxl_preprocessing.json`.
- Verified here: the experiment end to end on fake data (A and C); preprocessing and the pretraining part on
  the real 64-recording sample (59 recordings, 587 beats, median heart rate 73 bpm).
- Tests: 157 pass.

Run IDs: none (no real run of the experiment; the sample check was discarded).

Open issues / risks:
- No full PTB-XL run and no real pretraining or fine-tuning run exists yet. Run times on Kaggle are unknown.
- The experiment uses the probe CNN and a representation that the phase 1 gate has not frozen, so its results are
  provisional (DECISIONS A13) and must be rerun after the phase 1 gate and after phase2_baseline_classifiers.
- Which PTB-XL copy on Kaggle to attach is not settled; the download option avoids the question.
- The download from PhysioNet inside the runner was tested with a local zip only, not against PhysioNet.

Next:
1. Push. On Kaggle: smoke run of the experiment, then the real run for the four letters.
2. phase1_input_representation variants and the phase 1 gate (still the critical path).

## 2026-10-13 (Day 7), night — clearer file names

Done:
- Config folders and files renamed to say what they hold: `configs/validation_split/`,
  `configs/phase1_input_representation/{A_short_window, B_long_window_bandpass, C_short_window_rr_features,
  D_short_window_125hz, shared_budget}.yaml`, `configs/phase2_ptbxl_pretraining/` (same four names plus
  `pretraining_settings.yaml`).
- Runner files renamed: `ecgaug.experiments.phase1_input_representation` (was `m1`) and
  `ecgaug.experiments.phase2_ptbxl_pretraining` (was `pretrain`). Briefs: `docs/briefs/phase1_input_representation_<letter>_<name>.md`.
- Run labels in the phase2_ptbxl_pretraining experiment are now `pretrain<letter>`, `scratch<letter>`, `finetune<letter>`.
- Result folders: `results/phase1_input_representation/` and `results/phase2_ptbxl_pretraining/`.
- README has a table of names. Every reference in code, tests, docs and the Kaggle guide was updated.
- Tests: 159 pass; both runners were smoke-run by their new names.

Run IDs: none.

Open issues / risks:
- Anyone with an older clone or notebook must use the new paths and file names.

Next: unchanged (push, phase1_input_representation variants, phase 1 gate, Kaggle run of the phase2_ptbxl_pretraining experiment).

## 2026-10-13 (Day 7), night — Kaggle notebooks

Done:
- The single generic notebook is replaced by one per job: `kaggle/phase1_input_representation.ipynb`
  and `kaggle/phase2_ptbxl_pretraining.ipynb`. The user sets variant letters (`VARIANTS = ["A"]`), a
  `SMOKE_TEST` switch and, for pretraining, `PTBXL_SOURCE`; config paths are resolved in the notebook.
- `kaggle/README.md` rewritten for the two notebooks and the new file names.
- The run cell of both notebooks was executed on this laptop against a fresh clone in smoke mode.

Run IDs: none.

Open issues / risks: the notebooks have still not been run on Kaggle itself.

- 2026-10-13: added `PROJECT_GUIDE.md`, a full explanation of the project and of every part built so far, linked from the README.
- 2026-10-13: added `CODE_GUIDE.md`, a file-by-file explanation of the code.

## 2026-10-14 (Day 8) — first real run: phase2_ptbxl_pretraining experiment, variant D

Done (run by kartheek2159 on Kaggle, CPU, commit 32b9464; reviewed here):
- Full PTB-XL preprocessing: 21,837 recordings in the metadata (so the attached copy is version
  1.0.1, not 1.0.3), 20,918 kept, 213,133 beats. Excluded: {'no_diagnostic_superclass': 407, 'electrodes_problems': 28, 'pacemaker': 15}
  and {'low_qrs_contrast': 447, 'no_usable_beat': 16, 'implausible_peak_count': 4, 'implausible_amplitude': 2}. The QRS-contrast check removed 447 recordings (2%).
  The checksum list could not be fetched on Kaggle, so files were not compared.
- Pretraining: macro AUROC 0.830 ± 0.0004 on PTB-XL fold 9 (170,083 training beats).
- MIT-BIH validation, S F1 (selection metric): from scratch 0.0335 ± 0.0013, pretrained 0.0295 ± 0.0116;
  V F1: 0.335 vs 0.261. Tied on S F1, so V F1 decided: winner `scratch`. Provisional.
- Alignment of PTB-XL normal beats with MIT-BIH training N beats: mean-beat correlation
  0.945.
- Runs are now saved in folders by experiment, variant and kind of run
  (`runs/phase2_ptbxl_pretraining/D_short_window_125hz/{pretrain,scratch,finetune}/`); the 15 runs were moved
  there and `results/runs_index.csv` has a `folder` column. Tests: 162 pass.

Run IDs: the 15 runs listed in `results/runs_index.csv` (pretrainD, scratchD, finetuneD; commit 32b9464).

What the run shows:
- S detection is at chance level in both methods: S PR-AUC 0.02-0.03, about the share of S beats in
  validation. With input D (morphology only, no RR timing) the model cannot recognise premature beats,
  which matches the literature. Many validation N beats are called S (2,600 of 10,360 in one
  from-scratch run), and N recall is only 0.62-0.86.
- Because S F1 is near zero and noisy, early stopping on it picks epochs almost at random (best epochs
  5-11). This is the risk noted at kickoff; it will be smaller for variant C, which has RR features.
- Pretraining did not transfer to S or V detection with this input and this small network.

Open issues / risks:
- Run the phase1_input_representation variants, C in particular, before drawing conclusions about the selection metric.
- Re-fetching PhysioNet's checksum list failed on Kaggle; investigate or accept unverified PTB-XL files.

Next: phase1_input_representation variants A–D and the phase 1 gate.

## 2026-10-15 (Day 9) — streamlined workflow

Done:
- `src/ecgaug/workflow.py`: `status` (checklist and next action), `gate1` (report), `gate1 --approve`
  (freeze the winner's dataset, record it in `results/manifest.json`, log the decision).
- The pretraining runner waits for the phase 1 gate unless `--preview`; its notebook has a `PREVIEW` switch.
- README rewritten around a phase-by-phase checklist and the status command.
- Tests: 169 pass.

Run IDs: none.

Open issues / risks: unchanged. Phase 1 still needs the four phase1_input_representation runs.

Next: run the phase1_input_representation variants A–D (one Kaggle session).

## 2026-10-15 (Day 9) — before-and-after figures

Done:
- `src/ecgaug/figures.py` and `figures/`: 12 figures now (MIT-BIH preprocessing, PTB-XL
  preprocessing, the variant D pretraining preview). Every runner draws its figures at the end, phase 1 gate
  draws the variant comparison, and the Kaggle bundle and ingest carry figures back.
- What the variant D figures show: S beats are split almost evenly between "N" and "V" predictions
  (about 10% correct from scratch, 5% fine-tuned), and 14% of normal beats are called S from scratch;
  PTB-XL pretraining plateaus at macro AUROC 0.83 from the second epoch.
- Tests: 173 pass.

Run IDs: none new.

Next: the phase1_input_representation variants A–D; their figures will appear in `figures/phase1_input_representation/<letter>/`.

## 2026-10-15 (Day 9) — one naming scheme

Done:
- `docs/NAMING.md` and `src/ecgaug/names.py`: phase → experiment (`phase<N>_<what>`) → variant → run.
- Renamed everywhere: configs, runners, Kaggle notebooks, briefs, run folders and run IDs, results
  (`results/phase1_data/`, `results/phase1_input_representation/`, `results/phase1_gate/`,
  `results/phase2_ptbxl_data/`, `results/phase2_ptbxl_pretraining/`), figures (same folder names), the
  gate command (`python -m ecgaug.workflow phase1-gate`), and every document.
- Tests: 174 pass, including one that checks every experiment uses its name for config, runner and
  notebook.

Run IDs: the 15 preview runs renamed, contents unchanged.

Next: run phase1_input_representation, variants A–D (`kaggle/phase1_input_representation.ipynb`).

## 2026-10-15 (Day 9) — Kaggle notebooks push their own results

Done:
- `kaggle/kaggle_run.py --push-to <branch>`: after the runs, commits the run outputs (never weights) and
  pushes them; a new branch is used if the target moved on; nothing is pushed for failed runs, smoke
  tests or runs with test predictions. Both notebooks call it (`PUSH_RESULTS`, `GIT_NAME`, `GIT_EMAIL`).
  The GitHub token secret now needs Contents: read and write. The zip and `ingest_runs.py` stay as backup.
- Fix: figures drawn by one variant no longer make the next variant in the same session refuse to run
  (`figures/` now counts as output, like `runs/` and `results/`).
- Tests: 175 pass, including the push against a local Git remote.

Run IDs: none.

Next: run phase1_input_representation, variants A–D, with the updated notebook.

## 2026-10-16 (Day 10) — phase 1 results analysed, gate report written

Done:
- The four `phase1_input_representation` variants arrived through PRs #1 and #2 (A by
  Manindrathatuthuri, B, C, D by kartheek2159, all on Kaggle T4).
- `python -m ecgaug.workflow phase1-gate`: all checks pass; **winner C**, S F1 0.296 ± 0.038.
  A is 0.164 ± 0.078, D 0.035 ± 0.007 and B 0.001 ± 0.002; the tie margin is 0.061. Pending team
  approval.
- Detailed analysis in `docs/PHASE1_REPORT.md`. Two new gate figures:
  `figures/phase1_gate/per_class.png` and `figures/phase1_gate/early_stopping.png`.
- Audit:
  - code is identical across the four run commits;
  - all 29 run configs match the committed configs;
  - A, C and D datasets rebuild bit-for-bit on macOS, B does not (band-pass, likely platform floating
    point, not confirmed);
  - variant A's folder also holds an earlier, incomplete CPU attempt (5 runs), which is disclosed and
    not used.
- Small renames of leftover old names (`_m1_specs`, `M1_RESULTS`). Tests: 176 pass.

Run IDs: listed in `docs/PHASE1_REPORT.md`, appendix.

Open issues:
- Early stopping: 10 of 24 runs kept epoch 1 because validation S F1 was 0 for the first epochs and
  patience ran out at epoch 6. B's score is mostly this. Proposed fix (count patience only after the
  metric is above 0, or a minimum number of epochs) needs a DECISIONS entry before phase 2 runs.
- All 24 reported GPU runs are flagged `git_dirty`; the changed file is not recorded. Ask whoever ran
  them.
- The variant notes files are empty.

Next:
- The team approves the phase 1 gate.
- Decide the early-stopping fix.
- Re-run phase2_ptbxl_pretraining with C.

## 2026-10-16 (Day 10) — phases 2-6 built (branch `finish-project`)

Done (at Akash's instruction: "do the phase 3 coding ... push to a different branch", then "finish the project"):
- Phase 2: four baseline classifiers (`models/baselines.py`) and their runner; four quality-metric
  families (DTW, spectral, FID-analog with a frozen feature network, TSTR), the sanity harness and its
  runner; `phase2-gate`; `min_epochs` option for the early-stopping fix (off by default).
- Phase 3: cVAE and WGAN-GP with four variants each, the pilot harness (diagnostics, quality metrics,
  pilot classifier at every quarter checkpoint), final runs and synthetic banks, `phase3a-gate`,
  `phase3-gate`; the grid runner (resumable, one ledger file per cell).
- Phase 4: augmentation policies and the DTW quality screen, `phase4_augmentation_policy`, `phase4a-gate`
  (writes the full grid), `phase4-gate`, the final evaluation (`python -m ecgaug.final_eval`) with the
  pre-registered statistics.
- Phase 5: saliency methods and their scores, `phase5_explainability`; notes with the fact checker and
  an LLM client from the environment, `phase5_clinical_notes`; error analysis from saved predictions;
  the demo; `phase5-gate`.
- Phase 6: `python -m ecgaug.paper` (tables and numbers from results), `paper/main.tex` (compiles; red
  placeholders where results are missing), `scripts/reproduce.sh`.
- Kaggle: a notebook per new experiment plus the full grid; banks and checkpoints travel in the zips
  and come back with `ingest_runs.py` or an attached artifacts dataset.
- Docs: a brief per variant (32 in all), DECISIONS A14 (the design choices to confirm), the proposed
  v2/v3/phase 5 values in PREREGISTRATION §10, README, guides, NAMING, Kaggle guide.
- Tests: 271 pass (95 new), including an end-to-end final evaluation on fake data that runs once and
  refuses a second time. Real-data checks: a short CPU preview of a cVAE pilot, and the DTW and
  spectral metrics on real DS1 beats (DTW passes the sanity harness; spectral passes with one swap).

Fixed on the way: the memorization threshold now uses leave-one-out distances (a 10% hold-out was too
noisy for F); a rounding bug made Spearman 0.8 fail the >= 0.8 rule; generator weights were not seeded
before construction.

Run IDs: none official (smoke and scratch previews only, outside the repository).

Open issues:
- Every run waits for the phase 1 gate approval. The team should confirm DECISIONS A14 and fix the
  PREREGISTRATION v2 values at the phase 2 gate.
- `git_dirty` on the phase 1 Kaggle runs is still unexplained (docs/PHASE1_REPORT.md §7).
- Kaggle run times of phases 2-5 are not measured; the LLM notes are untried with a real key.

Next: the team approves the phase 1 gate (and decides `min_epochs`); then phase 2 on Kaggle.
