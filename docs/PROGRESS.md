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

## 2026-10-26 (Day 20): Kaggle uses the attached datasets only

A phase2_ptbxl_pretraining session on Kaggle (commit 1c0b3b9) downloaded MIT-BIH from PhysioNet and then
refused to run: "the phase 1 gate has not chosen the input variant yet".

Causes:
- The download happened because `RAW_DATASET_DIR` pointed at the PTB-XL dataset, which has no MIT-BIH
  files, and the notebooks fell back to `python -m ecgaug.data.raw` (download plus verification).
- The refusal is correct: the phase 1 gate report exists (`results/phase1_gate/report.md`, winner C),
  but the gate was never approved, so there is no `results/manifest.json`. No data is missing. Every
  non-preview run after phase 1 needs that manifest on the branch the notebook checks out.

Done:
- `kaggle/kaggle_run.py` no longer downloads MIT-BIH and no longer calls `ecgaug.data.raw`. It looks
  for the record files under `--raw-dir` (default `/kaggle/input`, including the new
  `/kaggle/input/datasets/<owner>/<name>` layout). It checks each of the 144 files against the
  committed `results/phase1_data/raw_checksums.json`, which matched PhysioNet in phase 1, and then
  copies them. No network is used. A missing or changed file stops the run with exit code 2 and a
  message to attach `mitdb-raw`. Smoke runs skip this step.
- All ten notebooks: the raw-data fallback is removed. The PTB-XL notebook finds an attached PTB-XL copy
  under `/kaggle/input` by default (`"download"` is still an option) and defaults to variant C.
- `ecgaug.data.raw` stays as a module: it is the only reader of raw files (test lock, DS2 guard). It
  is just no longer run in Kaggle sessions.
- Tests: the stricter file search, and the checksum copy (a match is copied; a changed file is refused).
- kaggle/README.md: parameters and troubleshooting updated.

Open issues:
- Phase 2 cannot run without `PREVIEW = True` until the team approves the phase 1 gate and the resulting
  `results/manifest.json` is pushed to the branch the notebooks run (`GIT_REF`).
- The notebook that was run uses the fork `Manindrathatuthuri/Gen_Ai_Project` at `main`. These fixes
  are on `akashsaha477/Gen_Ai_Project`, branch `finish-project`.

Next: approve the phase 1 gate (team), push the manifest, and run phase2_ptbxl_pretraining with C.

## 2026-10-26 (Day 20): the `git_dirty` puzzle solved; what blocks the rest

Done:
- Cause of `git_dirty = true` on the phase 1 runs (docs/PHASE1_REPORT.md §7): a logging bug. In every
  session the first run logged `false` and every later run `true` (GPU: 4 sessions of 1 + 5; CPU: 1 + 4).
  `runs._git` stripped the `git status --porcelain` output, removing the leading space of
  ` M results/runs_index.csv`, which was then read as `esults/runs_index.csv`, a non-output path. Fixed
  (`rstrip("\n")`) with a test that fails on the old code. Logged flags are not rewritten.
- Branches checked: only `main` and `finish-project` exist on `akashsaha477/Gen_Ai_Project`; `main` has
  nothing that `finish-project` lacks. The forks (`Manindrathatuthuri`, `kartheek2159`) are not reachable
  from this session, so any Kaggle results pushed there are not here.
- Tests: 273 pass (locked environment, CPU).

Not possible from this session: PhysioNet is blocked by the network policy (no real-data preview), there
is no GPU, and every official run waits for the phase 1 gate approval by named team members.

Open issues:
- All ten notebooks default to `GIT_REF = "main"`, but phases 2-6 exist only on `finish-project`:
  merge it into `main` (or set `GIT_REF`) before the Kaggle runs.
- Phase 1 gate approval and the `min_epochs` decision (PHASE1_REPORT §6) are with the team.

Next: team approves the phase 1 gate; merge `finish-project` into `main`; phase 2 on Kaggle.

## 2026-10-26 (Day 20): finish-project merged; early-stopping fix; no Kaggle pushes

Done:
- `finish-project` merged into `main` (PR #3), so the notebooks' default `GIT_REF = "main"` has phases 2-6.
- DECISIONS A15.1: `min_epochs: 10` from phase 2 on (PREREGISTRATION §4, §11). The phase 3 pilot
  classifier dropped `min_epochs`; fixed with a test.
- DECISIONS A15.2: notebooks default to `PUSH_RESULTS = False`; results come back via `ingest_runs.py`.
- Tests: 274 pass.

Open issues: the phase 1 gate approval (needs the exact reply `approve gate 1` and a machine with MIT-BIH).

Next: approve the phase 1 gate; phase 2 on Kaggle.

## 2026-10-26 (Day 20): phase 1 gate approved

- Approved by Akash Saha: winner C. `data/processed/v1` frozen, hashes in `results/manifest.json`.
- Raw files from PhysioNet's AWS mirror (`physionet-open/mitdb/1.0.0`); all 144 match the committed
  checksums and the mirror's SHA256SUMS. Train/val content hash equals the one the Kaggle C runs used.
- Counts: train N 35,464 / S 643 / V 3,030 / F 387; val 10,360 / 300 / 758 / 27; test 44,217 / 1,836 / 3,219 / 388.
- Test file built under `ALLOW_TEST_BUILD=1` (construction access, logged).

Next: phase 2 on Kaggle (baselines, quality metrics, PTB-XL with C).

## 2026-10-26 (Day 20): Kaggle gets MIT-BIH from PhysioNet's AWS mirror

- Kaggle run refused: the attached `kartheekdasari/mitbh-ecg` is a CSV/TXT conversion, not the record files.
- `kaggle_run.py`: with no attached record files, the 144 originals are downloaded from
  `physionet-open.s3.amazonaws.com/mitdb/1.0.0` and checked against the committed checksums (attached
  files that do not match are still refused; `--no-download` restores the old behaviour). Tested with a
  local mirror, and once against the real mirror (144 files, 33 s, all match). Tests: 275 pass.

Next: rerun the phase 2 notebooks with `GIT_REF = "main"` and Internet on.

## 2026-10-27 (Day 21): the last code: RQ3 analysis and the leakage audit

Done (branch `finish-project`; results will be added as they come):
- RQ3 (`python -m ecgaug.eval.rq3`): Spearman of each quality metric with the downstream gain across
  generator checkpoints, interval from resampling runs, TSTR apart; table in the paper, Fig 5 figure.
- Leakage audit (`phase4_leakage_audit`, optional, after the final evaluation): intra-patient split of
  DS1+DS2, generators retrained on it, cnn with none / cVAE / WGAN-GP at medium strength; runs in
  `runs/audit/`, banks in `data/synthetic/audit/`; new access kind `audit` (`ALLOW_TEST_AUDIT=1`);
  final evaluation, grid and ingest refuse audit artifacts. Notebook, config, paper table (Table 5).
- DECISIONS A16 (choices pending team confirmation). Tests: 290 pass.

Not runnable yet: both need phase 3 (RQ3) or the final evaluation (audit) results.

## 2026-10-27 (Day 21): dress rehearsal of the whole project

- `tests/test_rehearsal.py` (run with `ECGAUG_REHEARSAL=1`, about 8 min on CPU): on fake records with
  shrunk budgets, every runner, gate (approved as "Rehearsal") and the final evaluation in order, then
  RQ3, the leakage audit, phase 5, error analysis, the demo and the paper build. It passes; the paper
  build has no table left waiting. The numbers mean nothing.
- It found two bugs, fixed (DECISIONS A17): colliding policy run IDs, and the selection rule crashing on
  an undefined tie-break value. Tests: 291 pass (+1 skipped: the rehearsal).

## 2026-10-27 (Day 21): the paper and the course report, written to fill themselves in

- `paper/main.tex`: every section written (abstract, introduction, related work, data, methods,
  statistics, results, notes, limitations and ethics, reproducibility). Results sentences, the abstract's
  results sentence and a hypothesis-verdict table are generated from the result files by
  `src/ecgaug/paper_text.py` with the pre-registered criteria (DECISIONS A18.1); until a result exists the
  draft shows a red placeholder. No number or outcome was assumed.
- `paper/refs.bib`: 29 verified references (A18.2). `paper/report.tex`: the course report from the same
  generated files. Team contributions stay a placeholder until `docs/ASSIGNMENTS.md` is filled.
- Not done here: compiling LaTeX (no TeX in this container); run pdflatex + bibtex on main.tex and report.tex.
- Tests: 297 pass; the dress rehearsal fills every results sentence from its (fake) results.

Next: real runs (phase 2 onward on Kaggle); after each gate, `python -m ecgaug.paper` updates the draft.

## 2026-11-01 (Day 26): phase 2 results reviewed before the phase 2 gate

All phase 2 results are in. Gate report: `results/phase2_gate/report.md`. Status: PENDING APPROVAL.

Results (validation S F1, mean ± SD over 3 seeds, lr 1e-3):
- Baselines: A shallow CNN 0.3486 ± 0.0448; B ResNet 0.2314 ± 0.0969; C BiLSTM 0.3148 ± 0.0195; D CNN+BiLSTM
  0.2516 ± 0.0288. By rule: cnn = A (lead 0.117, margin 0.076), rnn = C (lead 0.063, margin 0.025).
- PTB-XL pretraining, C: fine-tuned 0.3184 ± 0.0270, scratch 0.2961 ± 0.0381. Tied (difference 0.0223, margin
  0.0330), so scratch wins on simplicity and pretraining stays out of the grid.
- Quality metrics: DTW and spectral pass. FID-analog fails (median CV 0.126, limit 0.10; it does get worse with
  every degradation). TSTR fails without being measured (below). Composite: dtw, spectral.

Audit: every official run uses the frozen dataset hash (2f6f9843889f); no code, config or lockfile change between
the run commits; none dirty; baselines and pretraining C on Tesla T4; no epoch-1 picks (the phase 1 early-stopping
flaw is gone); no `preds_test` files; no test access except the logged phase 1 gate construction. Tests pass.

TSTR was never scored. All 13 sets are NaN in all 3 resamples and the run took 14 s. Its real-data classifier
stops at epoch 4 with S F1 = 0 (patience 3, no `min_epochs`), so the guard in `tstr_score` returns NaN.
Reproduced on the frozen dataset on CPU: S F1 0.0000 in all 3 resamples. With `min_epochs: 10` the same recipe
reaches only 0.0497-0.0679, still rising at its last epoch (`max_epochs: 10`). A15.1 left this recipe out of the
early-stopping fix on purpose, so the failure stands by the rule and the recipe was not changed. Consequence:
PREREGISTRATION §9's separate RQ3 analysis of TSTR will have no data unless the team changes the recipe.

Approval dry run in a throwaway worktree (nothing recorded here): the entry writes cnn = A, rnn = C, composite
dtw + spectral, gammas 0.5/0.75/1, and phase 3 preconditions then pass for the official pilots.

Still open: variant notes are empty templates; A14-A18 are pending team confirmation; PREREGISTRATION v2 must be
committed and tagged `prereg-v2` before any phase 3 run.

Next: the team approves the phase 2 gate; then the phase 3 pilots on Kaggle.

Later the same day: the phase 2 gate was approved ("approve gate 2", Akash Saha) and recorded in
`results/manifest.json` and DECISIONS: cnn = A, rnn = C, composite dtw + spectral. PREREGISTRATION v2 committed and
tagged `prereg-v2`. Next: the phase 3 pilots on Kaggle (`phase3_cvae`, `phase3_wgan_gp`, `MODE = "pilot"`).

## 2026-11-02 (Day 27) — Phase 3 pilots reviewed; phase 4 adjusted

All 8 phase 3 pilots are in `main` (18 runs, no source, config or doc change between the run commits). Gate report
written: `results/phase3a_gate/report.md`, figure `figures/phase3a_gate/variants.png`. Status: PENDING APPROVAL.

Pilot results (validation S F1, mean ± SD over generator seeds 0 and 1, classifier seed 0; no-augmentation cnn: 0.3486
over 3 seeds, 0.2980 at seed 0):
- cVAE: D 0.4854 ± 0.0392, A 0.3961 ± 0.1010, C 0.3868 ± 0.0622, B (beta 0.1) 0.3685 ± 0.0711. By rule: D.
- WGAN-GP: D 0.4680 ± 0.0453, A 0.4310 ± 0.1090, B 0.4241 ± 0.0481, C 0.3664 ± 0.0371 (fails: S diversity 0.551). A, B
  and D tie; A and B tie on quality rank; A has fewer parameters. By rule: A.

Worth carrying forward:
- V F1 falls with cVAE D (0.679 to 0.542) and holds with WGAN-GP A (0.675). Selection does not read V F1.
- C2ST is at least 0.876 for every class of every variant: all synthetic beats are easy to tell from real ones.
- F F1 stays between 0 and 0.129; validation holds only 27 F beats.
- cVAE D's lowest diversity is 0.720, just above the 0.7 floor. Check it again on the 20,000-step banks.
- Variant notes are still empty templates, and the real-versus-synthetic sample figures were not committed; they
  should be in the teammates' Kaggle zips.

Phase 4 changed from these results before any policy run (DECISIONS A19, pending team confirmation): three seeds and
the general tie margin in the policy pilot; per-bank N recall and V F1 in the gate reports; all policies in one session.

Run-time estimates from the measured pilots and baselines: final runs about 2 GPU-hours in total (cVAE D about 0.8,
WGAN-GP A about 1.2); policy pilot under 1 hour; full grid at 3 seeds about 2 GPU-hours. These are estimates, not runs.

Report for the team: phases 1 to 3 summary with architecture diagrams (shared with the team separately)

Next: the team approves the phase 3a gate; then the final runs (`MODE = "final"`), one notebook per family.


## 2026-11-02 (Day 27) — Phase 3a gate approved

Approved by Akash Saha: phase3_cvae variant D (default), phase3_wgan_gp variant A (default). Recorded in
`results/manifest.json` and `docs/DECISIONS.md`. The final runs can now start.

Next: `kaggle/phase3_cvae.ipynb` and `kaggle/phase3_wgan_gp.ipynb` with `MODE = "final"` (about 2 GPU-hours in total,
estimated), ingest the zips, then the phase 3 gate. Still open: A19 confirmation, per-bank or average N recall limit,
3 or 5 seeds (decided at the phase 3 gate).

## 2026-11-03 (Day 28) — Phase 3 final runs done; phase 3 gate report written

Final runs on Kaggle T4 (private notebooks on Akash's account; the code came from a git bundle of `main` at d68d316,
uploaded as the private dataset `akashsaha477/ecgaug-repo`, because the GitHub repository is private). 20,000 steps,
seeds 0-2, no divergence, every bank's train-only check passed, hashes recorded.

| generator | seed | run ID | val S F1 | V F1 | F F1 | N recall | wall-clock (s) |
|---|---|---|---|---|---|---|---|
| cVAE D | 0 | phase3_cvae-D.final-s0-d68d316-202611031658 | 0.5245 | 0.5385 | 0.079 | 0.9788 | 1035 |
| cVAE D | 1 | phase3_cvae-D.final-s1-d68d316-202611031715 | 0.5064 | 0.5900 | 0.000 | 0.9888 | 1006 |
| cVAE D | 2 | phase3_cvae-D.final-s2-d68d316-202611031732 | 0.3598 | 0.4449 | 0.000 | 0.9768 | 1024 |
| WGAN-GP A | 0 | see results/phase3_wgan_gp/A_final.json | 0.3411 | 0.6777 | 0.056 | 0.9776 | 1394 |
| WGAN-GP A | 1 | same | 0.4966 | 0.2718 | 0.000 | 0.9871 | 1366 |
| WGAN-GP A | 2 | same | 0.3636 | 0.4303 | 0.058 | 0.8252 | 1385 |

Means over seeds (S F1): cVAE D 0.4636 ± 0.0903, WGAN-GP A 0.4005 ± 0.0841. These come from the pilot classifier inside
each run and are diagnostics, not the phase 4 comparison. Total about 2.3 GPU-hours (sum of wall-clock), close to the
estimate.

Gate report: `results/phase3_gate/report.md`; figure `figures/phase3_gate/banks.png` (`scripts/phase3_bank_figure.py`:
real training beats next to bank seed 0 of each family, per class).

Needs a team decision before approval:
- cVAE D bank seed 1: lowest diversity ratio 0.694 (class S), below the 0.7 floor. The run's own diagnostic sample gave
  0.701. The pre-registered floor covers pilot seeds only (PLAN Phase 3, PREREGISTRATION §10), so the gate checks pass
  as written; the rule does not say what to do with a final bank under the floor.
- WGAN-GP A: V F1 varies widely across seeds (0.272 to 0.678), and N recall at seed 2 is 0.825.
- cVAE D F banks contain a few beats with late artefacts (visible in the figure); WGAN-GP A F beats look closer to real.

Next: team decision on the diversity point, then `approve gate 3`; then the banks dataset and the phase 4 policy pilot.

## 2026-11-03 (Day 28) — Phase 3 gate approved; PREREGISTRATION v3 committed

Phase 3 gate approved by Akash Saha: banks phase3_cvae-D and phase3_wgan_gp-A, seeds 0-2, frozen (hashes in
`results/manifest.json`). DECISIONS A20: the cVAE D seed 1 bank (diversity 0.694) is kept and disclosed; 3 seeds; A19
confirmed. PREREGISTRATION v3 (policy pilot and grid strengths) committed and tagged `prereg-v3` before any policy run.

Next: banks as a private Kaggle dataset, then the phase4_augmentation_policy pilot (all four letters, one session).

## 2026-11-04 (Day 29) — RQ3 run; policy pilot started

RQ3 (`python -m ecgaug.eval.rq3`, exploratory, no decision uses it): 96 checkpoints from 24 generator runs (pilots and
finals). Spearman rho between each quality metric and the downstream gain, 95% interval from resampling runs: DTW -0.20
[-0.44, +0.06], spectral -0.00 [-0.25, +0.23], diversity -0.12 [-0.34, +0.13], memorization +0.18 [-0.07, +0.40],
C2ST +0.04 [-0.20, +0.25]. Every interval includes 0: no quality metric predicted the gain (H3 not supported at this
sample size). Figure: `figures/phase4_quality_vs_gain/quality_vs_gain.png`.

Banks uploaded as the private Kaggle dataset `akashsaha477/ecgaug-banks` (6 banks, sizes match); repo bundle updated to
aa1bb1d. phase4_augmentation_policy pilot started on Kaggle (all four letters, one session, T4).

## 2026-11-04 (Day 29) — Policy pilot done; phase 4a gate report written

phase4_augmentation_policy on Kaggle (T4, one session, run commit aa1bb1d): 24 policy runs and 3 shared reference runs,
ingested. Reference (cnn, no augmentation, seeds 0-2): S F1 0.3486, V F1 0.6786, N recall 0.9790, identical to the
phase 2 baseline. Policy scores (S F1, mean over seeds of the mean over banks; N recall drop on the mean over banks):

| policy | S F1 | V F1 | N recall drop | within limit |
|---|---|---|---|---|
| A balance to N | 0.3959 ± 0.0496 | 0.5581 | +0.0422 | no |
| B 0.5 x N | 0.4463 ± 0.0458 | 0.4574 | +0.0156 | no |
| C 2 x real | 0.4008 ± 0.0461 | 0.5615 | +0.0348 | no |
| D 0.5 x N, DTW-screened | 0.3826 ± 0.0284 | 0.5694 | -0.0064 | yes |

By rule (PREREGISTRATION v3): D, the only policy within the N recall limit. Every policy raised S F1 above the reference
and lowered V F1 (by 0.11 to 0.22). WGAN-GP A with policy A dropped N recall to 0.887. Report:
`results/phase4a_gate/report.md`; figure `figures/phase4a_gate/policies.png` (`scripts/phase4a_policy_figure.py`).

Full grid if D is approved: 96 runs (cnn and rnn x 16 cells x 3 seeds) with 18 DTW screens. Estimate from the measured
pilot times: about 3 hours in one Kaggle session (rnn runs dominate). Estimate, not a run.

Next: `approve gate 4a`, then the full grid.

## 2026-11-04 (Day 29) — Phase 4a gate approved; full grid started

Approved by Akash Saha: policy D (fraction_of_n_screened), strengths 0.1, 0.25, 0.5 x N. `configs/phase4_full_grid/full_grid.yaml`
written by the gate: none, class-weighted (gamma 0.5, 0.75, 1), random oversampling, SMOTE, cVAE D and WGAN-GP A, for the
cnn and rnn, seeds 0-2: 96 runs. Started on Kaggle in one session.

## 2026-11-04 (Day 29) — Note LLM switched to Gemini (A21)

`src/ecgaug/report/notes.py`: `LLMClient` now speaks Gemini (the default; the other provider was removed on 2026-11-07), with the key in a header and waits on
rate-limit replies (429/503); usage maps to the same input/output token fields (Gemini thinking tokens count as output).
Tests updated (`tests/test_phase5.py`, all pass). Notebook, kaggle/README.md and README.md name `GEMINI_API_KEY`.
Model probe (one made-up beat, no project data): default set to `gemini-3.5-flash` with thinking level `minimal`;
at the default thinking level the note was cut off after 396 thinking tokens (details in A21). Key is in the macOS
Keychain; the `~/.zshrc` export line still has to be added by Akash.

## 2026-11-04 (Day 29) — Full grid done; phase 4 gate report written

phase4_full_grid on Kaggle (T4, one session, run commit 382e1b1): 96 of 96 runs done, ingested
(`results/phase4_full_grid/`, `runs/phase4_full_grid/`). `python -m ecgaug.workflow phase4-gate`: all checks pass.
Validation S F1 at the chosen strength (seed mean ± SD, seeds 0-2; V F1 and N recall are seed means):

| condition | cnn strength | cnn S F1 | cnn V F1 | cnn N recall | rnn strength | rnn S F1 | rnn V F1 | rnn N recall |
|---|---|---|---|---|---|---|---|---|
| none | - | 0.3486 ± 0.0448 | 0.6786 | 0.9790 | - | 0.3148 ± 0.0195 | 0.7227 | 0.9805 |
| class-weighted (gamma) | 0.5 | 0.3701 ± 0.0347 | 0.4967 | 0.9616 | 0.75 | 0.4253 ± 0.0410 | 0.7762 | 0.8764 |
| random oversampling | 0.1 | 0.4396 ± 0.0180 | 0.5777 | 0.9105 | 0.1 | 0.4141 ± 0.0635 | 0.6124 | 0.9328 |
| SMOTE | 0.1 | 0.3690 ± 0.0382 | 0.4402 | 0.9905 | 0.1 | 0.3700 ± 0.0661 | 0.6835 | 0.8881 |
| cVAE D | 0.25 | 0.4234 ± 0.0659 | 0.5446 | 0.9832 | 0.1 | 0.3592 ± 0.0201 | 0.2985 | 0.9879 |
| WGAN-GP A | 0.1 | 0.3870 ± 0.0261 | 0.4816 | 0.9301 | 0.5 | 0.2578 ± 0.0206 | 0.7098 | 0.9391 |

Best generative condition by rule: cVAE for both architectures. Several strengths were picked by the tie rule (tied on
S F1 within max(0.02, pooled SD), then V F1, then the lower strength): e.g. rnn WGAN-GP 0.5 (all three tied on S F1; 0.5
had the best V F1). The report now lists the selection steps per condition. Seed spread is large (S F1 SD up to 0.066 at
the chosen strengths), so most differences between conditions are within noise on validation. rnn cVAE at 0.1 has V F1
0.2985 (none: 0.7227). N recall drops more than 0.01 below `none` for cnn class-weighted, ROS, WGAN-GP and rnn
class-weighted, ROS, SMOTE, WGAN-GP (reported, not enforced at this gate). Figure: `figures/phase4_gate/conditions.png`
(`scripts/phase4_gate_figure.py`). Test pipeline: `tests/test_phase4.py` passes.

Next: `approve gate 4`, then `/final-eval` (the only test-set read).

## 2026-11-04 (Day 29) — Phase 4 gate approved; final evaluation done (the only test-set read)

Gate 4 approved by Akash Saha (commit ce45db3). `/final-eval`: checks passed (gate approved, no prior evaluation entry,
36 checkpoints present with matching hashes); plan confirmed with "run final eval".

First attempt stopped before any test read: `data/processed/v1/` did not exist on this machine (it was built on
2026-10-26 in another environment; data/ is not in git). Rebuilt with `build_frozen("v1", ...)` from the raw files under
`ALLOW_TEST_BUILD=1` (logged construction access); both files are byte-identical to the frozen ones (file and content
SHA-256 match `results/manifest.json`). Then the evaluation ran once (logged `evaluation` entry, commit ce45db3).
`runs/.../preds_test.npz` are on this Mac only (gitignored).

Test rare-class macro-F1 (mean of S and F F1; seeds averaged; 95% record-level bootstrap CI, 10,000 replicates):

| condition | cnn | rnn |
|---|---|---|
| none | 0.088 [0.037, 0.194] | 0.075 [0.014, 0.232] |
| class-weighted | 0.113 [0.057, 0.207] | 0.061 [0.022, 0.168] |
| random oversampling | 0.136 [0.061, 0.247] | 0.090 [0.028, 0.196] |
| SMOTE | 0.070 [0.025, 0.172] | 0.086 [0.025, 0.206] |
| cVAE | 0.113 [0.049, 0.217] | 0.073 [0.035, 0.157] |
| WGAN-GP | 0.079 [0.033, 0.164] | 0.069 [0.027, 0.156] |

Confirmatory family (Holm over 4): cnn cVAE − none +0.025 [−0.018, +0.047]; cnn WGAN-GP − none −0.009 [−0.045, +0.015];
rnn cVAE − none −0.002 [−0.100, +0.032]; rnn WGAN-GP − none −0.006 [−0.138, +0.035]; all Holm p = 1.000. **H1 (generative
augmentation beats no augmentation on test) is not supported.** Secondary family (Holm over 8): no comparison significant;
smallest Holm p 0.550 (cnn cVAE vs WGAN-GP, +0.034 [−0.002, +0.068]). Without record 202: same conclusions (all Holm p ≥
0.456). Undefined bootstrap replicates: 0.0001 of each comparison.

Test scores are far below validation (e.g. cnn none S F1 0.148 on test vs 0.349 on validation). Record 232 holds 1,381 of
the 1,836 test S beats; S recall there is 0.000-0.100 for every condition. F F1 is 0.002-0.067 everywhere. Figures:
`figures/final_eval/forest.png`, `figures/final_eval/per_record_S_F.png` (the per-record figure's 12 blue shades are hard to
tell apart; redraw for the paper).

Next: phase 5 (explainability, notes with Gemini), then the paper.

## 2026-11-04 (Day 29) — Review and fixes before phase 5 (A22)

Gemini key now loads in new shells. Fixed: note variant C repeated the disclaimer (test added); notes cases now use the
saliency method the phase 5 rule picks once explainability has run (was always the default); the final-evaluation
per-record figure is a heatmap (record 232: 1,381 S beats, recall ≤ 0.10 for every condition; record 213: 362 F beats,
recall ≤ 0.35); notes docstring names Gemini; two lint findings. Error analysis written (`results/error_analysis/`,
`figures/error_analysis/`) from saved predictions: on test, cnn none labels 1,367 of 1,836 S beats and 333 of 388 F beats
as N (seed-mean counts). Full test suite passes.

Next: phase5_explainability A-D (local CPU, validation only), then phase5_clinical_notes A-D with Gemini.

## 2026-11-04 (Day 29) — Phase 5 runs done; phase 5 gate report written

phase5_explainability A-D on this Mac (CPU, validation beats only: 627 per model, stratified, seed 0). Faithfulness gain
over random deletion order (mean over none, cVAE, WGAN-GP) and stability across seeds:

| variant | cnn gain | cnn stability | rnn gain | rnn stability |
|---|---|---|---|---|
| A Grad-CAM | +0.1423 | 0.389 | n/a | n/a |
| B integrated gradients | +0.0279 | 0.558 | +0.0600 | 0.610 |
| C SmoothGrad | +0.0382 | 0.401 | +0.0297 | 0.498 |
| D occlusion | +0.0988 | 0.747 | +0.0862 | 0.515 |

By rule: cnn Grad-CAM, rnn occlusion (no ties). RQ4 (descriptive): on real S beats the rnn's saliency mass on the QRS is
0.263 without augmentation and 0.137 (cVAE) / 0.123 (WGAN-GP) with it (occlusion); on synthetic F beats the augmented rnn
models put 0.035-0.039 on the QRS and 0.42-0.44 on the P segment. The cnn's QRS mass on real S beats barely moves
(Grad-CAM 0.287 none, 0.344 cVAE and WGAN-GP). Figures: `figures/phase5_explainability/<letter>/class_average_saliency.png`.

phase5_clinical_notes: 50 validation cases (16 low-confidence) from cnn cVAE 0.25 seed 1 with Grad-CAM (A22.2). First
Gemini run failed without results (free-tier daily limit of 20 requests on gemini-3.5-flash; A23); rerun with
gemini-3.5-flash-lite. Unsupported claims / coverage: A 0 / 1.00, B 0 / 0.98, C 0 / 1.00, D 0 / 0.98; no fallbacks. By
rule (hard gate, coverage, ratings, latency, tokens) variant C; no blinded ratings were filled in.

`python -m ecgaug.workflow phase5-gate`: all 5 checks pass (incl. the demo smoke test). Next: `approve gate 5`, then
`git tag v1.0` and phase 6 (paper).

## 2026-11-04 (Day 29) — Phase 5 gate approved; code freeze

Akash Saha approved gate 5 (`approve gate 5`). Recorded in `results/manifest.json` and `docs/DECISIONS.md`: saliency cnn
Grad-CAM, rnn occlusion; notes variant C. Code tagged `v1.0`. Next: phase 6 (paper: regenerate tables and numbers, write
the negative H1/H1c result, records 232/213, limitations, v1 rebuild, Gemini switch A21/A23, phase 5 results).

## 2026-11-04 (Day 29) — Phase 6 started: paper drafted with every result

`python -m ecgaug.paper` regenerated all tables, numbers and results sentences. New generated paragraph
`results_concentration` (`src/ecgaug/paper_text.py`, from `results/error_analysis/test.json`): record 232 holds 1,381 of
1,836 test S beats and record 213 holds 362 of 388 F beats; S recall on 232 ranges 0.00-0.10 and F recall on 213 ranges
0.00-0.35 across the 12 models (seed means). `paper/main.tex`: figures added (forest plot, per-record recall, test
confusion matrices, Grad-CAM and occlusion class averages), a "Deviations from the plan" section (Gemini note model A21/A23,
note cases A22.2, v1 test arrays rebuilt with matching hashes), and the notes limitation corrected (no person rated the
notes). `paper/report.tex` likewise. Both compile with tectonic (`paper/main.pdf` 15 pages, `paper/report.pdf`).

Open: leakage audit (H2/RQ2: run, or report as not run), author names and team contributions, the claims-vs-evidence table.

## 2026-11-04 (Day 29) — Leakage audit done (H2/RQ2); paper complete except authors

Akash Saha: "run whatever needs". phase4_leakage_audit ran on Kaggle (T4, kernel phase4-leakage-audit, commit e1708f2,
`PART = "all"`, about 2 h: generators 15:26-17:11 UTC, classifiers 17:12-17:29). One audit access, merged into
`docs/TEST_ACCESS_LOG.md` by ingest (new: Kaggle bundles carry the access log and ingest appends rows it lacks; test in
`tests/test_runs.py`). Ingested: 9 cnn runs `phase4_leakage_audit-cnn.{none,cvae_0.25,wgan_gp_0.25}-s{0,1,2}-e1708f2-*`,
6 generator runs, 6 audit banks (`data/synthetic/audit/`, not in git); no conflicts or refusals.

Random beat-level split (44 of 44 records in both training and test), cnn rare-class macro-F1, mean ± SD over seeds:
none 0.837 ± 0.026, cVAE 0.820 ± 0.034, WGAN-GP 0.785 ± 0.060; inter-patient (final evaluation) 0.088, 0.113, 0.079.
Intra-patient evaluation raised the baseline by 0.748; the generative gains were -0.017 (cVAE) and -0.052 (WGAN-GP)
intra-patient. H2 verdict by the registered criterion: not consistent (baseline inflated, gains not). The audit uses the
grid's middle strength (0.25) for both generators, the final evaluation the validation-chosen strength (cVAE 0.25,
WGAN-GP 0.1); stated in the table caption. Figure: `figures/phase4_leakage_audit/intra_vs_inter.png`.

Paper: every placeholder filled except authors; audit figure, deviation entry and claims row added; claims table (13 rows)
now survives `python -m ecgaug.paper`. `paper/main.pdf` 16 pages. Open: author names and team contributions.

## 2026-11-05 (Day 30) — Post-hoc analyses for the review (PREREGISTRATION v4, P1-P4)

Akash Saha: "can you fix all these" (the weaknesses of the publication review). Rules committed first as pre-registration
v4 (commit 8786d6b, tag `prereg-v4`; DECISIONS A24), with a new test access kind `posthoc` (`ALLOW_TEST_POSTHOC=1`,
`load_test_posthoc`, refuses before the final evaluation is logged). Nothing here retrains, re-selects or re-tunes a model
of the main study; the 3-seed final evaluation stays the primary result. Not addressable here: blinded human ratings of
the notes (need people) and diffusion generators (future work).

- **P1 RR reference** (`phase6_rr_reference`, local CPU, fitted at 1baec70, one posthoc access at 9df5c94). Runs
  `phase6_rr_reference-rrlda-s0-1baec70-202611050043`, `phase6_rr_reference-rrmorphlda-s0-1baec70-202611050043`.
  Test rare-class macro-F1: rr_lda 0.022 [0.011, 0.038], rr_morph_lda 0.058 [0.015, 0.114]; minus cnn none
  -0.066 [-0.161, -0.021] and -0.031 [-0.109, +0.014]. The "cnn is weaker" wording does not apply.
- **P2 external validation** (`phase6_external_validation`, local CPU, about 3 h each; no MIT-BIH test read). INCART (primary,
  lead II, 75 records, none paced, S 1,958, F 219) and SVDB (secondary, ECG1, 78 records, S 12,194, F 23 so S F1 is
  primary); all 459 downloaded files checked against PhysioNet SHA256SUMS. INCART: generative augmentation lowered
  rare-class macro-F1 in all 4 confirmatory comparisons (cnn cVAE -0.078, Holm p 0.001; cnn WGAN-GP -0.065, p 0.023;
  rnn cVAE -0.073, p 0.009; rnn WGAN-GP -0.104, p 0.006); no augmentation was the best condition for both architectures.
  SVDB: only rnn cVAE vs none rejected (+0.120 [+0.041, +0.195], Holm p 0.009; rnn none is weak there, 0.187, and random
  oversampling reached 0.345). Neither database matches the null verdicts of the final evaluation. Class-mean beats show
  INCART close to MIT-BIH and SVDB clearly different (128 Hz, annotation offset), so these results measure transfer too.
- **P3 seeds 3 and 4** (`phase6_extra_seeds`, Kaggle T4 kernel phase6-extra-seeds at d33fcd6: 4 generator runs and 24
  classifier runs `phase6_extra_seeds-*-s{3,4}-d33fcd6-*`; one posthoc access at 65010c7). Five seeds: no Holm verdict
  changed in either family; cnn cVAE vs none +0.018 [-0.043, +0.038], Holm p 1.0 (3 seeds: +0.025).
- **P4 audit checks** (`phase6_audit_check`, local CPU about 3.3 h, one audit access at 044a3a5). WGAN-GP at the gate
  strength 0.1 on the audit split: 0.769 ± 0.082 (none 0.837 ± 0.026); none on random splits 1 and 2: 0.803 ± 0.074 and
  0.777 ± 0.059, against 0.088 inter-patient. H2 verdict unchanged. Runs `phase6_audit_check-cnn.{wgan_gp_0.1,none}-s{0,1,2}-*`
  (listed in `results/phase6_audit_check/summary.json`). Their run IDs carry four different git SHAs (7e3a954, 65010c7,
  a79f796, 1c9425c) because commits landed while the job ran; the process loaded its code once at 044a3a5, and the later
  commits changed only `figures.py`, `paper_text.py` and results, which it does not import.

Figures (`python -m ecgaug.figures posthoc`): `figures/phase6_rr_reference/references_vs_cnn.png`,
`figures/phase6_external_validation/{class_mean_beats,conditions_external_vs_ds2}.png`,
`figures/phase6_extra_seeds/three_vs_five_seeds.png`, `figures/phase6_audit_check/audit_checks.png`.
Paper: new section "Post-hoc analyses" (generated `results_posthoc.tex`) and one generated abstract sentence
(`abstract_posthoc.tex`), a deviation entry, the external-database limitation rewritten, four claim rows; the report gets
the same paragraphs and figures, and its missing `tab:test` label is fixed. `paper/main.pdf` 21 pages, `paper/report.pdf`
10 pages, no unresolved references.

Mistakes on the way: commit 9e63e4b went in with a failing test (wfdb read outside `raw.py`; fixed in de4add4, commits are
now gated on the pytest exit code); the RR references were first fitted from a dirty tree (removed and refitted at
1baec70); the INCART download missed 12 files (refetched and verified); the Kaggle CLI hung after the 308 MB bundle
download (stopped; the zip was complete).

Open: author names and team contributions; blinded note ratings need people.

## 2026-11-05 (Day 30) — P5 follow-up: patient-baseline RR ratio (pre-registration v5)

Asked for better results; agreed a pre-registered follow-up (A25, `prereg-v5` at 9339a95). The v1 `rr_ratio` (previous RR over
the mean of the ten before it) hides early beats inside runs: only 0.51 of training S beats fall below 0.9 (record 209 holds
383 of 643). v2 divides by the median RR whose interval midpoints lie within ±150 s: 0.87 of training S beats below 0.9,
N unchanged at 0.04 (`figures/phase6_rr_baseline/rr_ratio_v1_vs_v2.png`). Only this field changes (dataset v2); the frozen
banks' ratios were remapped by exact match on their copied real RR triple.

- **Data** (`phase6_rr_baseline --part data`, 220f4c6; hashes in `results/phase6_rr_baseline/data.json` at 71a5cea). DS2 v2
  ratios: one construction access at 10:45 (annotation times only, no labels, no model).
- **Grid** (Kaggle T4 kernel phase6-rr-baseline at 71a5cea, 96 cells, run IDs `phase6_rr_baseline-*-71a5cea-*`, timestamps
  05:20 to 08:06 UTC, about 2.8 GPU-hours; approved above 2). Ingested and committed at 4b80f13.
- **Selection** on validation by `workflow.choose_strengths` (the phase 4 gate rule), committed at 4ba1c4f before the test read:
  cnn class-weighted 0.5, random oversampling 0.1, SMOTE 0.25, cVAE 0.1, WGAN-GP 0.1; rnn 0.5, 0.1, 0.1, 0.1, 0.5.
- **References refitted on v2** (local): `phase6_rr_baseline-rrlda-s0-71a5cea-202611051048` (validation S F1 0.193, v1 0.114),
  `phase6_rr_baseline-rrmorphlda-s0-71a5cea-202611051048`.
- **Test** (one posthoc access at 13:54, 4ba1c4f; `results/phase6_rr_baseline/summary.json`). Rare-class macro-F1 with v2:
  cnn none 0.096 [0.037, 0.226] (v2 − v1 +0.008 [−0.018, +0.065]), rnn none 0.075 [0.018, 0.201] (−0.001 [−0.050, +0.014]);
  across the 12 conditions v2 − v1 ranged −0.042 to +0.046 and no CI excluded zero. Best condition rnn class-weighted 0.107.
  Confirmatory (generative vs none): cnn cVAE −0.022 (Holm p 0.421), cnn WGAN-GP −0.019 (0.421), rnn cVAE −0.013 (0.168),
  rnn WGAN-GP −0.019 (0.457); no secondary rejection. S Se stays low (cnn none 0.136, S +P 0.267); record 232 holds 0.75 of
  DS2 S beats. RR-only LDA rose from 0.022 to 0.074 [0.017, 0.141] (S Se 0.068 → 0.088), still below the networks.
- **Reading:** the better RR feature helps the linear RR model, but not the networks, and it does not change the study's answer:
  generative augmentation is not detectably better than none. The primary results stay the v1 final evaluation.
- Paper: subsection "A follow-up with a patient-baseline RR ratio" (generated `results_rr_baseline.tex`), an abstract sentence,
  a deviation entry and a limitation sentence; the report gets the same paragraph and figures.

## 2026-11-05 (Day 30, evening) — options 2 and 3: diagnosis, v6 and v7, all on Kaggle

The laptop crashed under local load (two external evaluations, the test suite and three trainings at once); everything
heavy now runs on Kaggle (CPU sessions, one T4 session per training grid). Kaggle tooling added: `kaggle_run.py --data-dir`
(copies an attached `data/` folder, refuses guarded test arrays) and `--tests` (full suite first; stops on failure),
`MPLBACKEND=Agg`, private datasets `ecgaug-v5-external` (INCART, SVDB raw and arrays, v5 checkpoints), `ecgaug-v1-checkpoints`,
`ecgaug-v6-runs`, the repo bundle `ecgaug-repo`. Kernel folders and logs live in `~/ecgaug_kaggle/` (the scratchpad was
wiped once, losing the first external runs). Commits are pushed to GitHub only after the full suite passes on Kaggle at
that commit: it failed first at e33c82e (about 20 tests from Kaggle's matplotlib backend, and one real catch:
`test_test_lock` flagged the new module for naming the test file), passed at 51dfa49 (2 skipped, as locally).

- **Diagnosis (exploratory, validation only, disclosed in v6):** frozen v1 cnn without augmentation, seed 0: S PR-AUC 0.92 on
  training records, 0.20 on validation records (V 0.995, 0.61); record 118's 96 validation S beats all called V or N, 83 of
  128 of record 201 called V. LDA on v2 RR 0.194 validation S F1, + raw samples 0.073, + patient-relative samples 0.275.
- **v6 (`prereg-v6`, e33c82e; A26):** dataset v3 = v2 with each beat minus the patient's median beat within ±150 s (no labels;
  hash 69a5ab1b7cba). Grid kernel phase6-patient-template-grid at 51dfa49 (T4, about 30 min incl. build): the 6 cells without
  augmentation `phase6_patient_template-{cnn,rnn}.none-s{0,1,2}-51dfa49-*`. Adoption rule (validation S F1 seed means): cnn
  0.472 vs 0.350 (+0.122), rnn 0.181 vs 0.306 (-0.125); mean -0.0015 against margin 0.026: **not adopted**, so the other 90
  cells were not run and DS2 was not read. Diagnostic part running (CPU kernel phase6-patient-template-diagnostic at 7d82942).
- **v7 (`prereg-v7`, 91bc56d; A27):** P7 `phase6_relative_cnn`, the cnn alone with v3 inputs (chosen on validation after v6's
  result, disclosed), 48 runs on a T4 (kernel phase6-relative-cnn-grid), then selection, one posthoc DS2 read and the external
  databases on CPU. P8 `phase6_augmentation_mechanism` (DS1 only, descriptive, CPU kernel with the full suite): synthetic
  beats attributed to the training patient of their nearest real beat, and the training-versus-validation gap of every
  frozen condition.
- **v5 external:** SVDB kernel at 41443e0 still running; the first INCART kernel stopped at the failing tests without
  evaluating, rerun at 51dfa49.
- Paper: v6, P7 and P8 subsections, a generated table of all post-hoc analyses (P1-P8), a generated Discussion and
  Conclusion (every number from result files; claims conditional on them), deviation entries; figures
  `figures/phase6_patient_template/input_v2_vs_v3.png` (before/after input) and, when their results land, the diagnostic,
  P7 and P8 figures. main.pdf 25 pages, report.pdf 12, no unresolved references.

### Results that landed later the same evening

- **v6 diagnostic** (CPU kernel at 7d82942 after a dataset fix: Kaggle drops the top folder of a one-folder dataset, so the
  v1 checkpoints were missing on the first try; datasets now hold `runs/` next to a README). Seed-mean S PR-AUC training ->
  validation: cnn v1 0.942 -> 0.255, v2 0.959 -> 0.232, v3 0.923 -> 0.401; V: cnn v3 validation 0.905 (v1 0.717); rnn v3
  0.767 -> 0.249. LDA validation S F1: RR 0.193, RR + raw 0.060, RR + relative 0.261. `results/phase6_patient_template/diagnostic.json`.
- **v5 external** (CPU): v2 point above v1 in 12 of 12 conditions on INCART and 10 of 12 on SVDB; generative vs none lowered
  in 3 of 4 (INCART, Holm) and 1 of 4 (SVDB).
- **P7** (`phase6_relative_cnn`, T4 grid at 91bc56d, 48 runs `phase6_relative_cnn-cnn.*-91bc56d-*`; selection at 5cbc350
  before the test read: class-weighted 0.5, random oversampling 0.25, SMOTE 0.1, cVAE 0.1, WGAN-GP 0.1; one posthoc access on
  Kaggle at 5cbc350, DS2 rebuilt and hash-checked, recomputed v1 scores equal to the final evaluation). Test rare-class
  macro-F1, cnn: none 0.371 [0.123, 0.412] (v1 0.088; +0.283 [+0.029, +0.307]), class-weighted 0.366, random oversampling
  0.361, SMOTE 0.335, cVAE 0.304, WGAN-GP 0.312; the P7 - v1 CI is above zero in 5 of 6 conditions. cnn none S F1 0.312
  (v1 0.148), S Se 0.291, S +P 0.350, F F1 0.429 (v1 0.029), V F1 0.931. Confirmatory: cVAE vs none -0.067 [-0.080, +0.007],
  WGAN-GP vs none -0.059 [-0.073, +0.019], Holm p 0.352 both; no secondary rejection. External: INCART cnn none 0.200 (v1
  0.256), SVDB S F1 0.368 (v1 0.314); no confirmatory rejection on either; SMOTE collapses on SVDB (0.047).
- **P8** (`phase6_augmentation_mechanism`, CPU, suite passed at 91bc56d; gap rerun at 5cbc350). Attribution: record 209
  holds 0.60 of real training S beats; synthetic S beats nearest to it 0.65 (cVAE) and 0.57 (WGAN-GP); a real training S
  beat's nearest other S beat comes from the same patient in 0.98 of cases; validation S beats are nearest to records 114
  (0.29) and 108 (0.26), which hold 0.02 and 0.01 of the real training S beats. Quality-screened added V beats come from
  3.70 effective records against 6.74 for the real V beats; F beats 0.96-1.00 from record 208. Gap (S PR-AUC training minus
  validation): augmentation narrowed it in 4 of 4 generative cases (cnn cVAE 0.687 -> 0.574 via validation 0.255 -> 0.375;
  rnn cVAE 0.704 -> 0.219 mostly via training 0.884 -> 0.492), none with a detectably better test result.
- Reading: the patient-relative cnn is the first baseline that is clearly better on the test records (0.088 -> 0.371), and on
  it generative augmentation is still not better than none (both point estimates negative). The gain does not carry over
  evenly to other databases (INCART lower, SVDB higher than v1). The primary results stay the v1 final evaluation.

## 2026-11-05 (Day 30, night) — P9 patient-transfer augmentation (pre-registration v8)

Asked for a way that generative augmentation could help. From P8: the generators add beats of the same few patients.
P9 (`phase6_patient_transfer`, `prereg-v8` at 9cac590, A28) writes each rare-class training beat as its patient's template
plus a deviation and puts the deviation on the template of another training patient (RR scaled by the ratio of median RR);
deviations from real beats (`transfer_real`) or from the frozen cVAE and WGAN-GP banks. Dataset v2, unscreened
`fraction_of_n`, cnn and rnn, strengths 0.1/0.25/0.5, seeds 0-2: 60 runs on a T4 (kernel phase6-patient-transfer-grid at
9cac590; suite passed at 9cac590 in ecgaug-final-tests). Selection (e3a2b8c, before any DS2 read), validation S F1 seed
means: cnn none 0.350, transfer_real 0.428 (0.1), transfer_cvae 0.385 (0.1), transfer_wgan_gp 0.326 (0.1); rnn none
0.306, transfer_real 0.528 (0.25), transfer_cvae 0.461 (0.25), transfer_wgan_gp 0.504 (0.25). The none cells reproduce
v5's values exactly. Test read and external databases running on Kaggle CPU.

Also: the paper and report were reframed at the user's request (b8424d7) to state that the results are bounded by model
size (73k and 135k parameters), runs (3 seeds, settings fixed in phase 2) and patients, with the CI widths of the
differences (0.06-0.17) and the P7 jump (0.088 to 0.371) as the evidence; the report gained a conclusion. Unrelated to the
project: a nanopore MD script was pushed to Kaggle on request and deleted again on request.
- **P9 test** (one posthoc access on Kaggle at e3a2b8c; suite passed there; recomputed v2 none scores identical to v5's).
  Rare-class macro-F1: cnn none 0.096, transfer_real 0.116, transfer_cvae 0.095, transfer_wgan_gp 0.089; rnn none 0.075,
  transfer_real 0.105, transfer_cvae 0.107, transfer_wgan_gp 0.109. Confirmatory (Holm): cnn -0.001, -0.007; rnn +0.032
  (p 0.937), +0.035 (p 0.564); none rejected; no secondary rejection. Transfer above plain generative (v5) in 4 of 4 pairs
  (rnn WGAN-GP +0.054 [+0.003, +0.154]). External: INCART all 4 lowered (Holm p 0.001); SVDB rnn cVAE +0.119 (p 0.005),
  rnn WGAN-GP +0.185 (p 0.001), cnn not detectably changed. Reading: adding patients moves the generative conditions above
  plain generation and, for the rnn, above none on validation and in point estimate on test, but not detectably on the test
  patients at this scale; transfer helps on SVDB and hurts on INCART.

## 2026-11-05/06 (Day 30-31) — P10 per-patient batch normalization (pre-registration v9)

Asked again for a way to make the results better. P1-P9 show methods acting on class counts did not beat none while
patient-level methods helped; P10 (`phase6_patient_adaptation`, `prereg-v9` at 2a7556e, A29) applies adaptive batch
normalization to the frozen checkpoints (each record classified by a copy whose batch-norm statistics come from its own
beats; no labels, no retraining; CPU only). Validation (kernel at 2a7556e, suite passed there), S F1 seed means without ->
with: v1 cnn 0.349 -> 0.510 (+0.162, margin 0.059, **adopted**), v1 rnn 0.315 -> 0.248 (not adopted), P7 cnn 0.472 -> 0.441
(not adopted); `adoption.json` committed at a4fa3a5 before the test read. Test (one posthoc access at a4fa3a5; plain score
reproduced exactly): v1 cnn none 0.080 with adaptation vs 0.088 (-0.008 [-0.046, +0.079], Holm p 0.822); S F1 0.160 (0.148),
**F F1 0.000 (0.029)**: adaptation erased the fusion-beat signal, which the validation records (few F beats, adoption on S
F1) could not show. Under adaptation, cVAE -0.010 and WGAN-GP +0.004 vs none (Holm p 1.0). External: adaptation -0.050
(INCART), +0.046 (SVDB), neither rejected; under adaptation generative vs none: INCART cVAE -0.110 (Holm p < 0.001), SVDB
cVAE +0.112 (p 0.023), WGAN-GP +0.048 (p 0.013). Reading: normalizing the patient at the input (P7) carries over to new
patients; normalizing inside the network did not. The discussion gained a generated paragraph "Patient variation, not
class imbalance" that states both.

## 2026-11-06 (Day 31) — A30: INCART intervals by patient; P11 more patients (pre-registration v10)

Asked whether a better dataset is needed. Two pieces of work.

**A30, a bug in the external intervals.** INCART's 75 records come from 32 patients (the `.hea` headers name them); every
external bootstrap had resampled records as if independent. Fixed at 153c9ab (`ext.clusters`; SVDB names no patients,
so its records stay the unit). The five INCART parts were re-run on Kaggle CPU at 153c9ab (kernels
`phase6-{external-validation,rr-baseline-external,relative-cnn-external,patient-transfer-external,patient-adaptation-external}-incartdb`;
the full suite passed in the last of them, exit code 0). Point estimates are unchanged; intervals widened (e.g. v1 cnn
none 0.256 [0.085, 0.364] -> [0.038, 0.395]). Holm verdicts on INCART, old -> new (commit 46dcea9 replaced the summaries):
- P2 (frozen models): generative vs none lowered 4 of 4 -> **1 of 4** (cnn cVAE -0.078, p 0.001 -> 0.037; cnn WGAN-GP
  -0.065, p 0.023 -> 0.167; rnn cVAE -0.073, p 0.009 -> 0.162; rnn WGAN-GP -0.104, p 0.006 -> 0.125).
- P5 (v2): 3 of 4 -> **0 of 4** (cnn cVAE p 0.016 -> 0.149; rnn cVAE 0.016 -> 0.149; rnn WGAN-GP 0.005 -> 0.089).
- P7: 0 of 2 -> 0 of 2 (cVAE p 0.051 -> 0.093). P9: 4 of 4 -> 4 of 4 (Holm p 0.001). P10: adaptation vs plain n.s.
  (p 0.220 -> 0.521); cVAE under adaptation still lowered (p < 0.001); WGAN-GP n.s.
Reading: the earlier claim that generative augmentation reliably lowered the INCART score was overstated by the record
clustering; only patient transfer (P9) and one frozen cVAE cnn stay detectably lower. README, claims and the paper's
deviation note were corrected; the paper text is generated from the new summaries.

**P11, more patients** (`phase6_more_patients`, `prereg-v10` at b06953b, A31). The P7 cnn (v3 inputs, settings and P7's
strengths unchanged) trained on DS1 plus every INCART and SVDB record (all S, V, F beats, at most 600 N per record),
every 5th external patient to validation. Training beats: DS1 S 643 / F 387, INCART S 253 / F 211, SVDB S 9,404 / F 23
(16 times DS1's S beats); validation: DS1 S 300, INCART S 1,705 (the every-5th rule put most INCART S beats there), SVDB
S 2,790; F 35 in all, so selection uses S F1. Grid: 15 runs on a T4 (kernel phase6-more-patients-grid at b06953b;
13 minutes of training in all, from the runs' `env.json`). Selection (89baa0c, before any DS2 read), validation S F1 seed means: none 0.753, class-weighted
0.767, random oversampling 0.783, cVAE 0.778, WGAN-GP 0.770 (not comparable with MIT-BIH-only validation numbers: different
validation patients). **Test** (one posthoc access at 46dcea9, kernel phase6-more-patients-test; the recomputed P7 none
score matched exactly), rare-class macro-F1 (S F1, F F1), P11 vs P7: none 0.147 (0.294, **0.000**) vs 0.371; class-weighted
**0.407** (0.293, 0.521) vs 0.366; random oversampling 0.383 (0.330, 0.436) vs 0.361; cVAE 0.245 (0.316, 0.175) vs 0.304;
WGAN-GP 0.217 (0.333, 0.101) vs 0.312. Confirmatory, none with more patients vs DS1 only: -0.224 [-0.240, +0.010], Holm p
0.164 (not detectably changed). Secondary, each condition vs none (Holm): class-weighted +0.260 (p 0.193), random
oversampling +0.237 (p 0.193), cVAE +0.099 [+0.005, +0.104] (p 0.067), WGAN-GP +0.070 (p 0.193); none rejected.
Reading: with 106 training patients F beats are 0.4% of training (621 of about 150,000) and the cnn without rebalancing
never predicts F on the test records; every rebalancing method restores part of it, class weighting the most (the best
test score of the project). S F1 stayed at 0.29-0.33 in every condition, as in P7: more patients did not move S on DS2.

**P12, generators trained on what the cnn sees** (`phase6_relative_generators`, pre-registration v11, A32). Asked to
"think of a solution". P11's cVAE and WGAN-GP conditions used the phase 3 generators, which learned ordinary beats of
DS1's 17 training records; their beats were made patient-relative afterwards. P12 trains the same generators (cVAE D,
WGAN-GP A; settings, 20,000 steps and seeds unchanged) on P11's training beats in v3 form, divided by one training SD;
banks are multiplied back and not z-scored per beat. Then the P7 cnn on P11's data with each bank at 0.1, 0.25 and 0.5,
unscreened, 3 seeds (18 runs), strengths chosen on P11's validation. Confirmatory: P12 vs the same generator in P11;
secondary: vs P11 none and vs P11 random oversampling. Plus P8's nearest-patient check on the new banks. Approved at about
2.5 T4-hours (two sessions, one per generator). An end-to-end test trains both real generators for 2 steps and exports
their banks.

**P12 training and selection** (kernels `phase6-relative-generators-{cvae,wgan-gp}` at 99451a2, two T4 sessions,
about 11:16-12:50 IST). All six generators ran their 20,000 steps and none diverged (cVAE 874-879 s per seed, WGAN-GP
1,236-1,241 s; training SD 0.570). Banks per seed: S 44,332, V 25,675, F 54,011. Nearest-patient check (seed-0 banks,
5,000 beats per class), effective number of training patients: S real 32.0, P11's banks cVAE 10.9 / WGAN-GP 14.3, P12's
banks 31.6 / 29.2; F real 4.9, P11's 1.3 / 1.3, P12's 3.4 / 3.3 (figure `figures/phase6_relative_generators/mechanism.png`).
So the retrained generators spread over almost as many patients as the real beats. Grid: 18 runs of 40-89 s. Selection
(committed at fb146a9, before any DS2 read), validation S F1 seed means: cVAE 0.1 chosen (0.768 +/- 0.014; 0.25: 0.733,
0.5: 0.743); WGAN-GP 0.25 chosen (0.740 +/- 0.023; all three strengths within the 0.02 margin, V F1 tie-break 0.832 vs
0.774 at 0.1 and 0.712 at 0.5). On the same validation P11's phase 3 generators scored cVAE 0.778 and WGAN-GP 0.770, so
the wider patient coverage did not raise the validation score. f765ac6 adds a fake-data test of the DS2 report (P12's 9
tests pass). The one DS2 read runs on Kaggle CPU at f765ac6 (kernel `phase6-relative-generators-test`).

**P12 test** (one posthoc DS2 access at f765ac6, kernel `phase6-relative-generators-test` on CPU, logged in
`docs/TEST_ACCESS_LOG.md`; P11's none score recomputed exactly, difference 0.0). Rare-class macro-F1 (S F1, F F1):
cVAE at 0.1 **0.368** [0.087, 0.480] (0.260, 0.475), against 0.245 (0.316, 0.175) for the phase 3 cVAE in P11; WGAN-GP at
0.25 0.243 [0.117, 0.288] (0.279, 0.206), against 0.217 (0.333, 0.101). Confirmatory, P12 vs the same generator in P11:
cVAE +0.122 [-0.047, +0.177], WGAN-GP +0.026 [-0.044, +0.051], Holm p 1.000 for both. Secondary (Holm): cVAE vs P11
none +0.221 (p 0.931), WGAN-GP vs none +0.096 (p 0.821), cVAE vs P11 random oversampling -0.016 [-0.085, +0.049]
(p 0.931), WGAN-GP vs random oversampling -0.141 (p 0.312); none rejected. Reading: training the generators on what
the cnn sees spread the synthetic beats over the training patients and moved the cVAE from well below random
oversampling (0.245 vs 0.383) to close to it (0.368; difference -0.016), the gain coming from F (0.175 to 0.475) while S fell slightly;
with 22 test records and wide intervals none of this is detectable, and more test patients and seeds are needed to tell
a gain of this size from noise. Class weighting in P11 (0.407) stays the best test score of the study. Figures:
`figures/phase6_relative_generators/{mechanism,results}.png`. Paper and report regenerated (main 34 pages, report 21,
no unresolved references); the discussion, conclusion and abstract now report P11 and P12; one overfull line in the
deviations section fixed by rewording.

Full test suite on Kaggle CPU at d3b4c86 (kernel `phase6-relative-generators-suite`, `tests_exit_code` 0; its recomputed
P12 selection is identical to the committed one). Pushed `main` (8fc42b4..d3b4c86) and the tags `prereg-v10` (b06953b)
and `prereg-v11` (99451a2).

**P13, RR intervals over the recording's mean** (`phase6_record_rr`, pre-registration v12, A33). Asked to search the
literature for how others do better and fix it. Finding: our S F1 on DS2 (about 0.3) matches a published CNN with local
RR normalization (0.280), while methods that normalize RR by the whole recording reach S sensitivity 76-91% (de Chazal
2004; Huang 2014). Published DS2 analyses put 75% of DS2's S beats in record 232, an S-dominant recording without
compensatory pauses, where local ratios fail. The same pattern is in P11's training data: SVDB 865, where the 0.8 rule finds
2.5% / 1.4% / 93.6% of the S beats with v1 / v2 / the recording mean (`--part check`, training-side annotations only;
`results/phase6_record_rr/design_check.json`). P13 divides `rr_prev` and `rr_next` by the recording's mean RR (v2's ratio
kept) and retrains the P7 cnn on P11's data in all five conditions at P11's and P12's strengths (15 runs; under 1 T4-hour).
P12's banks are on Kaggle as `ecgaug-p12-banks`; Kaggle's bank copy now keeps nested bank folders apart (it used to flatten
them). P13's 9 tests and the two existing bank-copy tests pass locally.

**P13 grid** (kernel `phase6-record-rr-grid` at 3a8561d on a T4, the full test suite first: `tests_exit_code` 0). P13's
data hash ef3cecfa...; recording means for 175 records. The bank check passed: every P12 synthetic beat's stored RR
features were reproduced by the recomputed draws before being replaced. 15 runs of 37-52 s. Validation S F1, before ->
with the recording-mean RR: none 0.753 -> 0.760, class weighting 0.767 -> 0.770, random oversampling 0.783 -> 0.776,
cVAE 0.768 -> 0.769, WGAN-GP 0.740 -> 0.715 (seed values 0.658, 0.745, 0.743). Unchanged on validation, as expected:
no validation patient is S-dominant (the one such training-side recording, SVDB 865, is a training patient). Runs and
selection committed at a26d941 before the DS2 read; the test kernel `phase6-record-rr-test` runs on CPU at b971794.

**P13 test** (one posthoc DS2 access at b971794, kernel `phase6-record-rr-test` on CPU, logged; all five earlier scores
recomputed exactly). Rare-class macro-F1 (S F1), before -> with `rr_prev`, `rr_next` over the recording's mean: none 0.147
(0.294) -> 0.216 (0.399); class weighting 0.407 (0.293) -> 0.380 (0.360); random oversampling 0.383 (0.330) -> 0.374
(0.323); cVAE 0.368 (0.260) -> 0.356 (0.239); WGAN-GP 0.243 (0.279) -> 0.264 (0.345). Confirmatory, each vs before: +0.070,
-0.027, -0.009, -0.011, +0.021, Holm p 1.000 for all five. Secondary: none rejected (WGAN-GP vs class weighting -0.116,
Holm p 0.511). Descriptive, Huang et al.'s SVEB stage on top (N calls with `rr_prev` below 0.8 of the recording's mean
become S): S F1 0.531-0.578, S sensitivity 0.920-0.955, S +P 0.369-0.419, N recall 0.917-0.946; rare-class macro-F1 none
0.300, class weighting 0.489, random oversampling 0.501, **cVAE 0.523** (the highest test score of the study), WGAN-GP
0.356. Reading: the published remedy works on our pipeline as published (Huang: Se 91.1%, +P 42.2%), and with it S F1
reaches the published range (0.51-0.61 for de Chazal, Huang, Mondejar-Guerra); the cnn given the same feature as an input
did not learn to use it from the one S-dominant training patient. The rule numbers are descriptive and DS2 has been read
many times, so they are reported as such. Figures `figures/phase6_record_rr/{check,results}.png`; paper and report
regenerated (38 and 24 pages, no unresolved references); the discussion, conclusion and abstract report P13, and the
"highest test score" scan now includes the rule results.

Full test suite on Kaggle CPU at e5ba5b1 (kernel `phase6-record-rr-suite`, `tests_exit_code` 0; its recomputed P13
selection is identical to the committed one). Pushed `main` (d3b4c86..e5ba5b1) and the tag `prereg-v12` (3a8561d).

**P14, many more training patients** (`phase6_icentia`, pre-registration v13, A34). Asked to "make it better ... find the
solution". Training-side design check with SVDB 865 held out (`--part check`): local RR references reach validation S F1
up to 0.738 but find 2-8% of an unseen S-dominant patient's S beats; the recording mean finds 94-98% at 0.627-0.650; an
irregularity veto loses too much recall (0.248-0.288). So the choice of reference has to be learned from many patients;
P11 has one S-dominant training patient. P14 adds the first segment of 400 Icentia11k patients drawn at random (11,000
patients, technician-labelled N/S/V, one chest lead at 250 Hz), built like INCART and SVDB in P13's form, and retrains the
cnn in four of P13's conditions (WGAN-GP's bank too small at the larger N count). A real segment builds locally in 0.7 s
(patient 2: S 494, N capped at 600). 7 tests for P14 pass, one of them building a synthetic Icentia-format record end to
end. Compute: one T4 session including download and build.
First P14 grid session failed fast (exit code 1, before any download): the frozen spec was read from a contract file that
is not in the Kaggle session. Fixed (P13's data source spec); downloads and segment builds made robust (reported
exclusions). The kernel log was fetched with one API request (`ListKernelSessionOutput`, page size 1); the CLI pages
through every output file of the working folder and got rate-limited.

## 2026-11-07 (Day 32) — repository cleanup (A35)

Asked to stop committing four local-only paths and to remove them from GitHub, history included, and to reword every
mention. Done: untracked and ignored locally (`.git/info/exclude`); rules moved to `docs/RULES.md` with the same numbering;
54 files reworded; the note generator is Gemini-only; the related tests pass. Next: P14's test, then the history rewrite
(git filter-repo, once, at the end), a commit map, the full suite on Kaggle at the rewritten head, and the force-push of
`main`, `finish-project` and every tag. GitHub keeps its own refs for the 10 pull requests, which only GitHub Support can purge.

**P14 test** (one posthoc DS2 access, kernel `phase6-icentia-test` on CPU; P13's scores reproduced exactly). Adding 388
Icentia11k patients lowered the test score in all four conditions, none detectably (Holm p 0.543-1.000): none 0.216 ->
0.121, class weighting 0.380 -> 0.247, random oversampling 0.374 -> 0.159, cVAE 0.356 -> 0.304. F F1 fell most (no F beats
in Icentia11k), S F1 stayed 0.23-0.27 (no S-dominant patient among the 388). The cVAE ranked first (vs none +0.184, Holm p
0.357); with Huang's rule it reached 0.460 (P13's 0.523 stays the highest). Ideas from SimGANs and PGANs, checked on
training-side data first: simulating S-dominant rhythm contexts did not transfer to the held-out S-dominant patient (SVDB
865 turns out to be paroxysmal SVT episodes of up to 665 beats, not runs with pauses; by RR alone such episodes look like
sinus tachycardia), so it was not run. A physiology-based fusion-beat synthesis (a fraction of each training patient's own
V deviation; 137 of 175 training-side recordings hold at least 5 V beats, against about 5 effective patients for real F
beats) is the most promising transfer and is not run yet.

History rewrite force-pushed 2026-11-07 on request (A35): `main` and `finish-project` at 45e75be, all 16 tags rewritten
(`prereg-v13` new on GitHub). A fresh clone from GitHub shows no removed path in any branch or tag history and no mention
in the current files. GitHub's 10 pull-request refs still reach 83 old commits (50 with the removed paths); only GitHub
Support or a new repository can drop them. Old SHAs map to new ones in `docs/COMMIT_MAP.csv`.

## 2026-11-08 (Day 33) — P15 diffusion generator (pre-registration v14, A36); S false alarms located in AF

Asked for diffusion models and the best result. P15 adds a class-conditional diffusion model as the third generator family
(built and tested: 4 generator tests incl. a learning check, 4 experiment tests). Separately, on training-side annotations:
89% of Huang's rule's false S calls on the MIT-BIH training records (1,251 of 1,412) sit in AF or flutter, where only 16 of
933 real S beats lie. An RR-irregularity AF detector trained on the MIT-BIH AF database only (afdb annotations, 25 other
subjects), with ectopic filtering and threshold 0.95, raises the rule's S F1 on the MIT-BIH training records from 0.486 to
0.756 (+P 0.349 -> 0.749), leaves INCART (0.316 -> 0.319) and SVDB 865 (0.965) unchanged and costs SVDB a little (0.666 ->
0.650). To be pre-registered as P16 before any DS2 read.

## 2026-11-09 (Day 34) — P15 results, P16 AF veto (pre-registration v15, A37), P17 longer diffusion (v16, A38)

**P15 (diffusion, v14).** The first training kernel failed: Kaggle served an old repository bundle because the client
reused an expired upload token from its cache; the cache record was removed and the bundle uploaded again. Training then
ran on a T4 (`phase6-diffusion-train`): 3 seeds, 20,000 steps each, none diverged, about 18 minutes per seed; 9 grid runs.
The banks resemble more training patients than the cVAE's (effective patients S 34.4 vs 31.6, F 7.0 vs 3.4; real 32.0
and 4.9). Validation chose strength 0.25 (S F1 0.741 ± 0.029; P13: none 0.760, cVAE 0.769). Test (one posthoc read,
logged): rare-class macro-F1 0.280 [0.128, 0.339] (S F1 0.308, F F1 0.252); vs none +0.064 [-0.061, +0.151] and vs the
cVAE -0.076 [-0.148, +0.093], Holm p 0.806 both; secondary vs class weighting -0.100 and random oversampling -0.094
(Holm p 0.442). With Huang et al.'s rule: 0.410. P13's scores reproduced exactly.

**Why the diffusion model fell short (training-side only).** Its loss was still falling at the end of the budget (0.117
over steps 1,001-2,000, 0.098 over 15,001-16,000, 0.096 over 19,001-20,000, all seeds), and its banks' class medians match
real DS1 training beats far less than the cVAE's (`phase6_diffusion_long --part shape`: correlation S 0.10-0.30, V
0.57-0.59, F 0.72-0.73 against 0.58-0.74, 0.94-0.96, 0.95-0.96). An exploratory CPU check on seed 0 (raw weights instead of
their average; guidance 1.0; not part of the repository) did not close the gap. Figure: `figures/phase6_diffusion/samples.png`.

**P16 (AF veto, v15, A37).** The design check (annotations only) now also counts DS1's rhythms (1,251 of the rule's
1,412 false S calls in AF or flutter; 16 of 944 S beats) and chooses the veto's form on validation: undoing only the rule's
own conversions, 0.753 mean validation S F1, against 0.730 for undoing every S call in AF (rule alone 0.727, cnn alone
0.755). Test (one posthoc read, after P15's, logged): the veto raised rare-class macro-F1 in all six conditions: none
0.300 to 0.319, class weighting 0.489 to 0.512, random oversampling 0.501 to 0.525, cVAE 0.523 to 0.557, WGAN-GP 0.356 to
0.375, diffusion 0.410 to 0.436 (every 95% CI above 0; Holm p 0.018, 0.032, then 0.057). It works through precision (cVAE:
S +P 0.415 to 0.496, Se 0.920 to 0.903). Without record 202: cVAE 0.560 to 0.583 (S F1 0.646 to 0.691). The study's
highest score is now cVAE + rule + veto, 0.557 (S F1 0.640, F F1 0.474), post hoc. P13's and P15's rule scores reproduced.

**P17 (v16, A38).** P15's model with 100,000 steps, nothing else changed (the config is checked against P15's), decided
and pre-registered before P15's and P16's test summaries were read (they were ingested and committed unread). Trained on a
Kaggle T4 (`phase6-diffusion-long-train`): 3 seeds, 100,000 steps each, none diverged, about 87 minutes per seed; 9 grid
runs; ingest clean (12 runs, 3 banks, no conflicts, no test access). The loss kept falling slowly (0.117-0.118 over steps
1,001-2,000, 0.096 over 19,001-20,000 as in P15, 0.088 over 59,001-60,000, 0.084-0.085 over 99,001-100,000, all seeds),
but the class medians did not move closer to real DS1 training beats (`--part shape`, median over seeds: correlation S
0.02, V 0.60, F 0.79; P15 0.24, 0.58, 0.72; cVAE 0.64, 0.95, 0.96). More steps alone did not close the gap; a larger
model or another sampler would be the next thing to try. The samples figure shows why: single synthetic beats have sharp
deflections, but not at the R peak as real beats do, so the median flattens. An exploratory count added to `--part shape`
after training and before the test read (`peak_alignment`: share of beats whose largest deflection lies within 10 samples
of the real beats' median position of it) moved with the longer training but not to the real level: V 0.42-0.49 (P15) to
0.61-0.66 (P17), F 0.45-0.50 to 0.63-0.70, S 0.16-0.25 to 0.23-0.25 over seeds; real beats 0.75, 0.75, 0.63; cVAE
0.81-0.84, 0.82-0.83, 0.70-0.72. Figure: `figures/phase6_diffusion_long/check.png` (loss, shape, alignment, spread over
patients; training data only). Effective patients: S 37.8, F 6.4 (real 32.0 and 4.9). Validation
chose strength 0.1 (S F1 0.754 ± 0.002; 0.1 and 0.25 tied on the metric and the tie-break, 0.1 is the simpler).

**P17 test (one posthoc read, logged; CPU kernel `phase6-diffusion-long-test` at f495dbe).** Rare-class macro-F1 0.267
[0.129, 0.341] (S F1 0.307, F F1 0.226); P15 0.280. Confirmatory: vs P15 -0.013 [-0.040, +0.032], vs none +0.050
[-0.071, +0.139], Holm p 0.924 both. Secondary: vs cVAE -0.089, class weighting -0.113, random oversampling -0.108
(Holm p 0.776, 0.737, 0.776). With Huang et al.'s rule 0.404 (S F1 0.581); with the rule and P16's AF veto 0.440 (S F1
0.653, F F1 0.226; the cVAE's 0.557 stays the study's highest). P13's, P15's and P16's scores reproduced exactly. More
steps alone did not close the gap to the cVAE; with the training-side checks above, the next things to try would be a
larger model or another sampler. Ingest kept the local `shape_check.json` (the bundle's copy predates the alignment
count; correlations identical) and the locally drawn samples figure (the test kernel had no P17 banks, so its copy lacks the P17 row).
Figure: `figures/phase6_diffusion_long/results.png`.

Open: PROJECT_GUIDE.md and CODE_GUIDE.md do not yet describe the post hoc analyses (P1-P17); the full test suite on
Kaggle at the final commit; push main.
