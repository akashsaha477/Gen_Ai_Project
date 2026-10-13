# Code guide: what each file does

This walks through the code file by file. For what the project is and what is finished, read
`PROJECT_GUIDE.md` first. All package code is in `src/ecgaug/`; everything else calls into it.

## Contents

1. [The big picture: how data flows](#1-the-big-picture-how-data-flows)
2. [`src/ecgaug/data/`: getting from raw files to beats](#2-srcecgaugdata-getting-from-raw-files-to-beats)
3. [`src/ecgaug/data/`: PTB-XL](#3-srcecgaugdata-ptb-xl)
4. [`src/ecgaug/eval/`: scoring and choosing](#4-srcecgaugeval-scoring-and-choosing)
5. [`src/ecgaug/models/` and `train/`: the network and the training loop](#5-srcecgaugmodels-and-train-the-network-and-the-training-loop)
6. [`src/ecgaug/runs.py`: logging every run](#6-srcecgaugrunspy-logging-every-run)
7. [`src/ecgaug/experiments/`: the runners](#7-srcecgaugexperiments-the-runners)
8. [`configs/`: the settings files](#8-configs-the-settings-files)
9. [`kaggle/`: running on Kaggle](#9-kaggle-running-on-kaggle)
10. [`tests/`: what is tested](#10-tests-what-is-tested)
11. [Two walkthroughs: what happens when you run a command](#11-two-walkthroughs-what-happens-when-you-run-a-command)

---

## 1. The big picture: how data flows

```
PhysioNet files                                   PTB-XL files
      |                                                |
   raw.py  (the only code that opens raw files;     raw.py
      |     enforces the test lock)                    |
   beats.py  (labels + RR intervals)               ptbxl.py + rpeaks.py
      |                                            (lead II, 360 Hz, detected beats, labels)
   representation.py  (cut a window per beat) <--------+
      |
   contract.py  (write arrays + hashes)
      |
   loaders.py  (read arrays, verify hashes)
      |
   train/loop.py + models/  (train the network)
      |
   eval/metrics.py  (score on validation)
      |
   runs.py  (save the run)  -->  eval/selection.py (pick the winner at the gate)
```

The runners in `experiments/` string these steps together for one experiment.

`paths.py` is used by everything: it returns the project folders (`data/`, `runs/`, `results/`,
`docs/`, `configs/`). Setting the environment variable `ECGAUG_ROOT` redirects them, which is how the
tests and smoke runs work in a temporary folder without touching real files.

---

## 2. `src/ecgaug/data/`: getting from raw files to beats

### `records.py` — the fixed facts

No logic, only definitions that the rest of the code imports:

- `DS1`, `DS2`, `PACED`: the record numbers of the training half, the test half and the four excluded
  pacemaker records.
- `AAMI` and `SYMBOL_TO_AAMI`: which annotation symbols belong to N, S, V, F and Q.
- `QRS_SYMBOLS`: symbols that count as a heartbeat. `NON_BEAT_SYMBOLS`: rhythm changes, noise markers
  and comments, which are ignored.
- `FS = 360` (sampling rate) and `LEAD = "MLII"`.
- `dataset_of(record)`: returns `"DS1"`, `"DS2"` or `"paced"`.

### `access.py` — the test lock

- `require_test_access(kind)`: raises `LockedTestDataError` unless the access kind is declared
  (`"construction"` or `"evaluation"`) **and** the matching environment variable is set to `1`
  (`ALLOW_TEST_BUILD` or `ALLOW_TEST`).
- `log_test_access(kind, purpose, ...)`: appends one row to `docs/TEST_ACCESS_LOG.md` with the date,
  the kind, the git commit and the command.

### `raw.py` — the only file that opens raw data

Every other module gets signals and annotations through this one, so the lock cannot be skipped.

- `_guard(record, access)`: called before any read. For a DS2 record it calls `require_test_access`;
  for a paced or unknown record it raises an error.
- `download()`: downloads MIT-BIH with the `wfdb` library.
- `verify()`: checks that all 48 records have their three files, hashes every file, compares the
  hashes with PhysioNet's published list (`fetch_reference_checksums`), and checks each header for
  360 Hz and the MLII lead. Returns a report whose `problems` list is empty when all is well.
- `read_annotations(record, access)`: the beat positions and symbols of a record.
- `read_signal(record, access)`: the MLII signal, **selected by name**, because record 114 lists its
  leads in a different order.
- `read_header(record)`: sampling rate and lead names only (no signal, so not locked).
- `download_ptbxl(dest)` and `read_ptbxl_lead(root, filename, lead)`: the same two jobs for PTB-XL.
- `main()`: `python -m ecgaug.data.raw` downloads, verifies and writes
  `results/data/raw_checksums.json`.

### `beats.py` — from annotations to labelled beats

- `rr_intervals(qrs_samples)`: for every beat, the time to the previous beat (`rr_prev`), to the next
  beat (`rr_next`), and `rr_ratio` = `rr_prev` divided by the average of the 10 intervals before it.
  A premature beat has a ratio well below 1.
- `extract_beats(record, access)`: reads the annotations, keeps the heartbeat symbols, checks they are
  in time order, computes the RR features, and maps each symbol to its class. Returns a table with one
  row per beat. RR intervals are measured **before** any beat is dropped, so removing a Q beat does not
  create a fake double-length gap.

### `counts.py` — how many beats of each class

- `count_record(record)`: class counts of one record.
- `record_counts(include_test)`: counts for all DS1 records, and DS2 too if asked (which needs the
  construction permission and is logged).
- `sanity_check(counts)`: compares our totals with the reference numbers in `docs/RULES.md`; passes if
  every class is within 1%.
- `main()`: writes `results/data/record_counts.csv` and `sanity_check.json`.

### `splits.py` — who is in training, validation and test

- `Split`: a small record holding the three lists of record numbers and the selection metric.
- `check_split(split)`: raises an error if a record is listed twice, appears in two parts, is paced,
  or if the test part is not exactly DS2. This is the leakage check.
- `make_split(name, val)`: builds a split from the validation records; training is the rest of DS1.
- `load_approved_split(counts)`: reads `configs/validation_split/approved.yaml`. It refuses if the file
  is missing, if nobody is named as approver, or if the selection metric in the file disagrees with
  what the rule gives for those records.
- `selection_metric_for(counts, split)`: applies the rule "F counts only if validation has at least
  100 F beats from at least 2 records".
- `summarize` and `options_report`: produce `results/data/val_options.md`, the comparison of the three
  candidate validation splits. They read DS1 rows only.

### `representation.py` — cutting a window around each beat

This one file implements all four Module 1 variants; a variant is only a set of parameters.

- `RepresentationSpec`: the parameters: seconds before and after the R peak, optional band-pass
  limits, optional target sampling rate, and whether RR features are used. It computes the window
  size in samples (`window_samples`, `n_samples`) and the number of input values (`n_inputs`).
- `usable_mask(...)`: marks the beats that can be used: the widest window (0.4 s before, 0.6 s after)
  must fit inside the record and both RR intervals must exist. The rule is the same for every
  variant, so all variants get exactly the same beats.
- `bandpass(x, band, order)`: zero-phase Butterworth filter on the whole record.
- `zscore_per_beat(windows)`: subtracts each window's mean and divides by its standard deviation.
- `transform_record(signal, r_idx, spec)`: the full recipe for one record: filter if asked, resample
  if asked, cut one window per beat, z-score. Returns an array `[beats, samples]`.

### `contract.py` — building and checking dataset files

- `build_arrays(records, spec, split_of, access)`: for each record, extracts beats, reads the signal,
  drops unusable, Q and unmapped beats, cuts windows, and collects everything into arrays (`X`, `y`,
  `record`, `split`, `r_idx`, the three RR features, and two fields for later use with synthetic
  beats). Also returns a per-record report of what was kept and dropped.
- `save_npz_deterministic(path, arrays)`: writes the arrays so the file's bytes depend only on the
  data, not on the time it was written. That makes hashes repeatable.
- `content_sha256(arrays)` and `sha256_file(path)`: hashes of the data and of the file.
- `write_dataset(...)`: writes `trainval.npz` (and `test.npz` if given) plus `contract.json`, which
  records how the data was made, the split, the class counts and the file hashes.
- `verify_file(dir, filename)`: recomputes a file's hash and raises `ContractError` if it differs.
- `build_candidate(spec, split)`: builds a DS1-only dataset for a Module 1 variant. No test data.
- `build_frozen(version, spec, split)`: builds the final dataset including `test.npz`. Needs the
  construction permission, logs the access, and refuses to overwrite an existing one. Used at Gate 1.

### `loaders.py` — reading dataset files

- `load_candidate(name)` and `load_trainval(version)`: verify the hash, then load.
- `load_test(version)`: checks the evaluation permission **first**, then verifies and loads, then
  logs the access. This is the only function that can read the test file.

### `fake.py` — fake data for tests and smoke runs

- `write_fake_mitdb` and `write_fake_ptbxl`: write random ECG-like signals in the real file formats
  under real record numbers. They let the whole pipeline run in seconds without real data. They are
  never used for a result.

---

## 3. `src/ecgaug/data/`: PTB-XL

### `rpeaks.py` — finding beats without annotations

- `detect_r_peaks(x, fs)`: band-pass 5–20 Hz, take the slope, square it, smooth over 120 ms; find
  peaks at least 250 ms apart; keep those that are large compared with their neighbours; move each to
  the largest deflection within 80 ms.
- `qrs_contrast(x, peaks)`: how far the detected peaks stand out of the noise. Clean ECG scores above
  10, pure noise about 3; recordings below 4 are dropped.
- `match_peaks(reference, detected, tolerance)`: pairs detected peaks with true ones and counts hits,
  misses and false detections.
- `validate_on_ds1()` / `main()`: runs the detector on every DS1 record and compares with the
  cardiologists' annotations. Writes `results/data/rpeak_validation_ds1.json`.

### `ptbxl.py` — the PTB-XL pipeline

In the order the code runs:

- `find_root(path)`: finds the folder containing `ptbxl_database.csv`, also inside an attached Kaggle
  dataset.
- `load_metadata(root)`: reads the metadata; `superclass_map` and `record_labels` turn each
  recording's diagnostic codes into five yes/no labels. Recordings with no diagnostic label, a
  pacemaker or electrode problems are excluded and counted by reason.
- `signal_problem(x, fs)`: returns the reason a signal is unusable (wrong length, missing values,
  flat, stuck segment, implausible amplitude), or nothing if it is fine.
- `resample_to_fs(x, fs)`: 500 Hz to 360 Hz.
- `process_record(x, fs)`: check, resample, detect peaks, reject if the number of peaks or their
  contrast is implausible; also records heart rate and a mains-interference measure.
- `beat_table(peaks)`: computes RR features and keeps beats with a full window and RR intervals
  between 0.3 and 2.0 s.
- `_process_file(task)`: everything for one recording file; runs in worker processes.
- `build(root, workers, verify)`: runs all recordings, stacks the results, assigns folds 1–8 to
  training, 9 to validation, 10 to unused, checks that no patient is in two parts
  (`check_patient_split`), and assembles a report with every exclusion counted.
- `verify_checksums(root, hashes)`: compares the files read with PhysioNet's checksum list.
- `prepare(root, ...)`: `build`, then save to `data/processed/ptbxl/` with a hash, and write the
  report to `results/data/`.
- `load()` and `is_full_build()`: read the saved arrays (after checking the hash), and tell whether a
  complete build exists.
- `windows_for_record(signal, r_idx, spec)`: mirror-pads the 10-second recording by 2 s, then calls
  the same `transform_record` used for MIT-BIH.
- `beat_arrays(arrays, spec, split)`: the windows, labels and RR features of one split, ready for
  training.
- `alignment_report(ptbxl_x, mitbih_x, spec)`: how similar the PTB-XL beats are to MIT-BIH beats
  (correlation of the average beat; share of beats whose peak is at the window centre).

### `checks.py` — measurements that justify the PTB-XL choices

- `edge_effect(spec)`: cuts 10-second pieces out of DS1 records, filters them alone with four kinds
  of padding, and compares each beat with the same beat filtered inside the full record.
- `powerline_ds1()`: measures 60 Hz mains interference in DS1.
- `main()`: writes `results/data/ptbxl_method_checks_ds1.json`.

---

## 4. `src/ecgaug/eval/`: scoring and choosing

### `metrics.py`

- `confusion_matrix(y_true, y_pred)`: the 4×4 table of true class against predicted class.
- `recall_from_cm`, `precision_from_cm`, `f1_from_cm`: per-class scores. A class with no true beats
  gets "undefined" (NaN), not zero.
- `rare_macro_f1_from_cm`: the project's primary metric, the mean of S F1 and F F1.
- `aami_ppv_v_from_cm`: positive predictivity for V in the older AAMI convention.
- `metrics_from_confusion(cm)`: all of the above as one dictionary.
- `pr_auc_one_vs_rest(probs, y)`: area under the precision-recall curve per class.
- `evaluate(probs, y, records)`: the function the training loop calls: probabilities in, every
  metric out, plus per-record confusion matrices.
- `multilabel_auroc(probs, y, names)`: AUROC per label for the PTB-XL pretraining task.

These functions also accept stacks of confusion matrices, so thousands of bootstrap samples are
scored in one call.

### `bootstrap.py`

- `draw_record_counts(n_records, n, seed)`: for each of `n` resamples, how many times each record is
  drawn.
- `cluster_bootstrap(metric_fn, probs, y, records)`: resamples whole records, recomputes the metric
  each time, and returns the estimate, the 95% interval, and how often a class (or the record holding
  most of a class) was missing from a resample.
- `cluster_bootstrap_confusion(...)`: the same, much faster, for metrics that depend only on the
  confusion matrix.
- `paired_cluster_bootstrap_diff(...)`: compares two methods on the same resampled records and
  returns the difference, its interval and a p-value.
- `holm(pvalues)`: the Holm correction for testing several comparisons at once.

### `selection.py`

- `selection_metric_name(n_f_beats, n_f_records)`: primary metric if validation has enough F beats,
  otherwise the fallback (S F1).
- `selection_value(metrics, name)` and `tie_break_value(...)`: pull the right number out of a
  metrics dictionary.
- `tie_margin(sd_a, sd_b)`: the larger of 0.02 and the pooled seed-to-seed spread.
- `select_winner(candidates)`: applies the rule: take the best mean; keep everything within the tie
  margin; break ties by V F1, then simplicity, then run time. Returns the winner and each step.

### `plotting.py`

- `import_pyplot()`: imports matplotlib safely on macOS versions where its font scan fails.

---

## 5. `src/ecgaug/models/` and `train/`: the network and the training loop

### `models/probe.py`

- `ProbeCNN`: three blocks of convolution, batch normalisation, ReLU and pooling; then an average
  over time and a final linear layer called `head`. If RR features are used, a small side branch
  processes them and is joined before the head. About 19,000 parameters.

### `models/__init__.py`

- `build_model(model_cfg, n_classes, n_side)`: creates the network named in a config.
- `load_encoder(model, state_dict)`: copies pretrained weights into a model, **except the head**, and
  refuses if the two networks differ anywhere else. This is the transfer step of fine-tuning.

### `train/loop.py`

- `set_seed(seed)`: fixes all random number generators and asks PyTorch for deterministic operations.
- `resolve_device(name)`: `"auto"` picks the GPU if there is one.
- `TrainConfig`: learning rate, maximum epochs, patience, batch size, weight decay.
- `_fit(...)`: the shared loop. Each epoch: shuffle, train on batches, evaluate on validation, record
  a line of the training log. It keeps the weights of the best epoch and stops after `patience`
  epochs without improvement.
- `fit_classifier(...)`: `_fit` with cross-entropy loss and the selection metric as the validation
  score. Used for MIT-BIH.
- `fit_multilabel(...)`: `_fit` with binary cross-entropy and macro AUROC. Used for PTB-XL
  pretraining.
- `predict_logits`, `predict_proba`: run the network on data in evaluation mode.

---

## 6. `src/ecgaug/runs.py`: logging every run

- `git_sha()`, `git_dirty()`: the current commit, and whether code, configs or docs have uncommitted
  changes (files under `runs/` and `results/` do not count).
- `make_run_id(...)`: builds names like `p1-m1-A.lr1e-03-s0-<commit>-<time>`: phase, module, variant,
  learning rate, seed, commit, timestamp. `parse_run_id` reverses it.
- `capture_env(device)`: Python and package versions, hardware, lockfile hash.
- `write_run(...)`: creates `runs/<run_id>/` with `config.yaml`, `env.json`, `train_log.csv`,
  `metrics_val.json`, `preds_val.npz` and the checkpoint. It never overwrites an existing run.
- `upsert_runs_index(row)`: adds the run to `results/runs_index.csv`.
- `ingest(sources)`: copies runs and result files from a Kaggle zip into the repo. It refuses any run
  carrying test predictions and never overwrites a differing file.

---

## 7. `src/ecgaug/experiments/`: the runners

### `common.py` — shared by both runners

- `check_preconditions()`: loads the approved split and checks the tree is committed; raises
  `PreconditionError` with the reason otherwise.
- `split_part(arrays, name, with_side)`: picks the training or validation beats out of a dataset.
- `train_grid(...)`: the core loop of every experiment. For each learning rate and each seed: build
  the model, optionally load pretrained weights, train, save the run, record the scores. Returns the
  full grid.
- `best_grid_point(grid)`: the learning rate with the better seed-mean score.
- `temporary_root()`: redirects all folders to a temporary directory for smoke runs.

### `input_representation.py` — Module 1

- `load_config(path)`: merges a variant file with `shared_budget.yaml`, and rejects a variant file
  that tries to change the budget.
- `run_variant(cfg, split)`: builds the DS1 dataset for the variant, calls `train_grid`, and writes
  `results/validation/m1_input_representation/<letter>.json` with per-seed scores, mean, spread and
  run IDs.
- `representation_check(cfg)`: `--check` mode: builds the inputs for all of DS1 and reports shapes and
  counts, without training.
- `smoke(cfg)`: `--smoke` mode: the whole thing on fake data in a temporary folder.
- `main()`: reads the command line and calls one of the three.

### `ptbxl_pretraining.py` — the pretraining arm

- `load_config(path)`: takes the input representation and the training budget from the matching
  Module 1 config, and the pretraining settings from `pretraining_settings.yaml`.
- `ensure_ptbxl(root, workers, verify, download)`: returns the preprocessed PTB-XL data, building it
  first (from an attached copy or by download) if it is not on disk.
- `pretrain(cfg, spec, train, val, device)`: trains on PTB-XL once per seed and saves each run.
- `run(cfg, split, ptbxl_arrays)`: the whole arm: build MIT-BIH data, cut PTB-XL beats, measure
  alignment, pretrain, run `train_grid` twice (from scratch, and with `load_encoder` as the starting
  point), apply `select_winner`, write `results/validation/ptbxl_pretraining/<letter>.json`.
- `smoke(cfg)` and `main()`: as above.

---

## 8. `configs/`: the settings files

| File | Sets |
|---|---|
| `validation_split/approved.yaml` | the validation records, the selection metric, who approved |
| `validation_split/options.yaml` | the three candidate splits that were compared |
| `m1_input_representation/shared_budget.yaml` | seeds, learning rates, epochs, patience, batch size, network size |
| `m1_input_representation/<letter>_<name>.yaml` | one variant's window, filter, sampling rate, RR features |
| `ptbxl_pretraining/pretraining_settings.yaml` | learning rate, epochs and patience for the PTB-XL stage |
| `ptbxl_pretraining/<letter>_<name>.yaml` | which Module 1 variant this pretraining run uses |

---

## 9. `kaggle/`: running on Kaggle

- `module1_input_representation.ipynb`, `ptbxl_pretraining.ipynb`: parameter cell, a cell that clones
  the repo at your branch using the GitHub token from Kaggle Secrets, and a cell that turns your
  variant letters into config files and calls `kaggle_run.py`.
- `kaggle_run.py`: `setup_env` installs the environment; `setup_data` copies or downloads MIT-BIH and
  verifies it; then it runs each config through its runner; `bundle` zips `runs/` and the result
  files.
- `make_raw_dataset.py`: zips the verified MIT-BIH files for upload as a Kaggle dataset.
- `ingest_runs.py`: calls `runs.ingest` on a downloaded zip.

---

## 10. `tests/`: what is tested

| Test file | Checks |
|---|---|
| `test_records.py` | record lists and class mapping match the protocol |
| `test_beats.py` | RR intervals against hand-computed values; symbol handling; lead selected by name |
| `test_test_lock.py` | DS2 cannot be read without permission; accesses are logged; no other file opens raw data |
| `test_splits.py` | leaky or malformed splits are rejected; built datasets have no shared record |
| `test_representation.py` | window sizes of A–D; filtering and centring; identical beats across variants |
| `test_contract.py` | file hashes repeat on rebuild; tampering is detected; the frozen build is guarded |
| `test_metrics.py` | every metric against hand-computed cases |
| `test_bootstrap.py` | whole records are resampled; intervals, p-values and Holm are correct |
| `test_selection.py` | the tie margin and tie-breaks |
| `test_probe_training.py` | network shapes; training is deterministic; early stopping |
| `test_runs.py` | run IDs, run folders, the index, ingest, the Kaggle bundle |
| `test_input_representation.py` | equal budgets; refusals; smoke runs of A–D |
| `test_ptbxl.py` | the detector; every PTB-XL step; patient-wise folds; padding; checksums |
| `test_ptbxl_pretraining.py` | weight transfer; config; the arm end to end on fake data |

`conftest.py` makes every test run in a temporary folder with the test lock closed.

---

## 11. Two walkthroughs: what happens when you run a command

### Module 1, variant A

```
uv run python -m ecgaug.experiments.input_representation --config configs/m1_input_representation/A_short_window.yaml
```

1. `load_config` merges `A_short_window.yaml` with `shared_budget.yaml`.
2. `check_preconditions` loads the approved split and checks the tree is committed.
3. `build_candidate` calls `build_arrays` for the 22 DS1 records: `extract_beats`, `read_signal`,
   `usable_mask`, `transform_record`. The result is saved with its hash.
4. `load_candidate` verifies the hash and loads; `split_part` separates training from validation.
5. `train_grid` runs 6 trainings (2 learning rates × 3 seeds). Each one: `set_seed`, `build_model`,
   `fit_classifier`, `write_run`, `upsert_runs_index`.
6. `best_grid_point` picks the learning rate; the summary is written to
   `results/validation/m1_input_representation/A.json`.

### PTB-XL pretraining arm, variant A

```
uv run python -m ecgaug.experiments.ptbxl_pretraining --config configs/ptbxl_pretraining/A_short_window.yaml --ptbxl-download
```

1. `load_config` reads the representation and budget from Module 1's A config.
2. `check_preconditions`, as above.
3. `ensure_ptbxl`: `download_ptbxl`, then `prepare` (metadata, signal checks, resampling, peak
   detection, beat filtering, labels, folds, checksum comparison), then `load`.
4. `run` builds the MIT-BIH dataset (as in Module 1) and cuts PTB-XL beats with `beat_arrays`.
5. `pretrain`: 3 runs of `fit_multilabel` on PTB-XL, one per seed.
6. `train_grid` from scratch: 6 runs. `train_grid` with `load_encoder`: 6 runs, each starting from the
   pretraining run of the same seed.
7. `select_winner` compares the two; the summary is written to
   `results/validation/ptbxl_pretraining/A.json`.
