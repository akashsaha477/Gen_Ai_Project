# PROGRESS

## 2026-10-08 (Day 2) — Phase 1

Done:
- Kit installed at the repo root (c1bec44). Plan amended after the kickoff review, A1–A11 (39f4f29,
  `docs/DECISIONS.md`); A2 and A7 change the plan and await the whole team.
- Environment: Python 3.11 with `uv.lock` (torch 2.14.1, numpy 2.4.6, scipy 1.17.1,
  scikit-learn 1.9.1, wfdb 4.3.1).
- MIT-BIH 1.0.0 downloaded from PhysioNet. All 144 files (48 records × hea, dat, atr) match
  PhysioNet's published SHA-256 list (`results/data/raw_checksums.json`).
- Beat extraction with the AAMI map and RR intervals; per-record counts for 44 records
  (`results/data/record_counts.csv`). Totals: DS1 N 45,866, S 944, V 3,788, F 415, Q 8; DS2 N 44,259, S 1,837, V 3,221, F 388, Q 7.
  Sanity check passes: the largest deviation from the docs/RULES.md reference is S in DS1,
  944 against 942 (+0.21%)
  (`results/data/sanity_check.json`). DS1 also has 472 flutter-wave
  annotations outside the AAMI map (record 207), excluded like Q.
- Test lock in place: raw reads go through one guarded module, `load_test()` needs `ALLOW_TEST=1`.
  `docs/TEST_ACCESS_LOG.md` holds 1 construction entry (DS2 label counts) and 0 evaluation entries.
- Module 2: metrics, record-level cluster bootstrap, paired differences with bootstrap p-values, Holm,
  and the selection rule with its tie margin, all with hand-computed tests.
- Probe CNN (18,628 parameters; 18,762 with the RR side input), training loop with
  early stopping, run logging.
- Module 1: configs A–D, one brief per variant, and the runner. `--check` on DS1 gives 50,969 usable
  beats for every variant (N 45,824, S 943, V 3,788, F 414; dropped at record edges: N 42, S 1,
  V 0, F 1), an identical beat set across variants, and no non-finite values.
  Shapes: A (50969, 180), B (50969, 360), C (50969, 180), D (50969, 63) (`results/data/m1_representation_check_*.json`).
  Smoke runs on fake data pass for A–D.
- Validation-split options with DS1 per-record counts: `results/data/val_options.md`.
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
- Wall-clock of a Module 1 variant run is not measured yet.
- This laptop has about 11 GB of free disk.
- Commit 2f18fa1 lacks the data package because `.gitignore` matched `src/ecgaug/data/`; fixed in
  410986b. The commits in between do not pass their tests on their own.
- matplotlib's font scan fails on this macOS version; `ecgaug.eval.plotting.import_pyplot()` works
  around it (see README).

Next:
1. Team decides the validation split and approves PREREGISTRATION v1 (with or without A2 and A7).
2. Then: write `configs/validation_split/approved.yaml`, fill §2, set the status to IN FORCE, commit, tag
   `prereg-v1`, push.
3. Draw the Module 1 assignments (`docs/ASSIGNMENTS.md`); each member runs `/variant 1 <letter>`.
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
  diagnosis, then fine-tuning on MIT-BIH; one extra baseline arm in Phase 2. Design is in
  `docs/PLAN.md`. Nothing is built for it yet; it waits for Gate 1 because it needs the winning beat window.
- Variant assignments are still TBD in `docs/ASSIGNMENTS.md`.
- The Kaggle notebook has not been executed on Kaggle. Wall-clock of a variant run is unmeasured.
- Two commits are local only (approved split, dataset decision) until pushed.

Next:
1. Push, draw assignments, run `/variant 1 A..D` (locally or on Kaggle).
2. `/gate 1`: pick the input representation and freeze the dataset.
3. Phase 2: baselines, quality metrics and the PTB-XL pretraining arm.

## 2026-10-13 (Day 7), later — PTB-XL preprocessing

Done:
- R-peak detector (`src/ecgaug/data/rpeaks.py`) for data without beat annotations. Validated on MIT-BIH
  DS1 against the reference annotations: sensitivity 0.9963, positive predictivity
  0.9933 (TP 51,304, FN 189, FP 346); median timing offset 0.0 ms,
  97% of matches within 10 ms (`results/data/rpeak_validation_ds1.json`). Parameters were set
  once and not tuned. DS2 was not read.
- PTB-XL pipeline (`src/ecgaug/data/ptbxl.py`): metadata exclusions, lead II by name, signal checks,
  resampling to 360 Hz, peak detection, QRS-contrast check, RR and usable-beat filters, superclass
  labels per beat, patient-wise folds (1–8 train, 9 validation, 10 unused), hashed output.
- Run on a real sample of 59 PTB-XL recordings (ecg_id 1–64, 5 without a diagnostic superclass):
  59 kept, 587 beats (`results/data/ptbxl_preprocessing_sample.json`).
- Alignment with MIT-BIH on that sample (`results/data/ptbxl_alignment_sample.json`): correlation of the
  mean PTB-XL beat with the mean DS1 normal beat is 0.93 (A), 0.89 (B), 0.93 (D); the largest
  deflection sits at the window centre for 96% of PTB-XL beats against 97% of MIT-BIH beats (A).
- Tests: 133 pass.

Run IDs: none.

Open issues / risks:
- Only 64 of about 21,800 PTB-XL recordings have been processed. The full run should happen on Kaggle
  with a PTB-XL dataset attached; PhysioNet is too slow from this laptop and disk is short.
- The QRS-contrast threshold (4.0) was set from DS1 segments, the 59-recording sample and white noise.
  Check how many recordings it removes on the full data.
- Pretraining and fine-tuning are not built; they follow Gate 1.

Next:
1. Module 1 variants and Gate 1 (unchanged).
2. Full PTB-XL preprocessing on Kaggle, then the pretraining runner in Phase 2.

## 2026-10-13 (Day 7), evening — PTB-XL on Kaggle: preprocessing and pretraining arm

Done:
- PTB-XL preprocessing made ready for the full data: worker processes (same result for any number of
  workers), search for the dataset folder below an attached Kaggle input, optional download of PTB-XL
  1.0.3 from PhysioNet as one zip, comparison of every file read with PhysioNet's checksum list, and
  heart-rate, QRS-contrast and 50 Hz statistics in the report.
- Edge effect measured on MIT-BIH DS1 (`python -m ecgaug.data.checks`,
  `results/data/ptbxl_method_checks_ds1.json`): when a 10-second piece is band-pass filtered on its own,
  beats 0.4-1.0s from an end correlate with the same beats filtered inside the full record at a 5th
  percentile of 0.854 without padding, 0.922 with odd reflection, 0.976 with
  edge-value padding and 0.990 with mirror padding (3,061 beats). Mirror padding of 2 s is now used
  for PTB-XL. It only matters for the band-passed representation (B).
- Mains interference: 60 Hz is clearly present in 55% of DS1 records (median ratio 11.9);
  the PTB-XL sample shows no 50 Hz interference (share above 10: 0%). No notch filter was added.
- Pretraining arm (`src/ecgaug/experiments/ptbxl_pretraining.py`, `configs/ptbxl_pretraining/`): one command
  preprocesses PTB-XL, pretrains once per seed (5 superclasses per beat, binary cross-entropy, early
  stopping on PTB-XL fold 9), fine-tunes on MIT-BIH from the pretrained weights and from scratch with
  the Module 1 budget, applies the selection rule and writes `results/validation/ptbxl_pretraining/<letter>.json`.
- Shared runner code moved to `src/ecgaug/experiments/common.py`; the Module 1 runner uses it and
  gives the same results as before (a test checks that the from-scratch arm equals a Module 1 run).
- Kaggle: notebook parameters and guide for the arm; the bundle and `ingest_runs.py` now carry
  `results/data/ptbxl_preprocessing.json`.
- Verified here: the arm end to end on fake data (A and C); preprocessing and the pretraining stage on
  the real 64-recording sample (59 recordings, 587 beats, median heart rate 73 bpm).
- Tests: 157 pass.

Run IDs: none (no real run of the arm; the sample check was discarded).

Open issues / risks:
- No full PTB-XL run and no real pretraining or fine-tuning run exists yet. Run times on Kaggle are unknown.
- The arm uses the probe CNN and a representation that Gate 1 has not frozen, so its results are
  provisional (DECISIONS A13) and must be rerun after Gate 1 and after Module 3.
- Which PTB-XL copy on Kaggle to attach is not settled; the download option avoids the question.
- The download from PhysioNet inside the runner was tested with a local zip only, not against PhysioNet.

Next:
1. Push. On Kaggle: smoke run of the arm, then the real run for the four letters.
2. Module 1 variants and Gate 1 (still the critical path).

## 2026-10-13 (Day 7), night — clearer file names

Done:
- Config folders and files renamed to say what they hold: `configs/validation_split/`,
  `configs/m1_input_representation/{A_short_window, B_long_window_bandpass, C_short_window_rr_features,
  D_short_window_125hz, shared_budget}.yaml`, `configs/ptbxl_pretraining/` (same four names plus
  `pretraining_settings.yaml`).
- Runner modules renamed: `ecgaug.experiments.input_representation` (was `m1`) and
  `ecgaug.experiments.ptbxl_pretraining` (was `pretrain`). Briefs: `docs/briefs/m1_<letter>_<name>.md`.
- Run labels in the pretraining arm are now `pretrain<letter>`, `scratch<letter>`, `finetune<letter>`.
- Result folders: `results/validation/m1_input_representation/` and `results/validation/ptbxl_pretraining/`.
- README has a table of names. Every reference in code, tests, docs and the Kaggle guide was updated.
- Tests: 159 pass; both runners were smoke-run by their new names.

Run IDs: none.

Open issues / risks:
- Anyone with an older clone or notebook must use the new paths and module names.

Next: unchanged (push, Module 1 variants, Gate 1, Kaggle run of the pretraining arm).
