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
8. [`configs/`: the settings files](#8-configs-the-settings-files) (7b-7d: workflow, figures, phases 2-6)
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

The runner of each experiment in `experiments/` strings these parts together.

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

Every other file gets signals and annotations through this one, so the lock cannot be skipped.

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
  `results/phase1_data/raw_checksums.json`.

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
- `main()`: writes `results/phase1_data/record_counts.csv` and `sanity_check.json`.

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
- `summarize` and `options_report`: produce `results/phase1_data/val_options.md`, the comparison of the three
  candidate validation splits. They read DS1 rows only.

### `representation.py` — cutting a window around each beat

This one file implements all four phase1_input_representation variants; a variant is only a set of parameters.

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
- `build_candidate(spec, split)`: builds a DS1-only dataset for a phase1_input_representation variant. No test data.
- `build_frozen(version, spec, split)`: builds the final dataset including `test.npz`. Needs the
  construction permission, logs the access, and refuses to overwrite an existing one. Used at the phase 1 gate.

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
  cardiologists' annotations. Writes `results/phase2_ptbxl_data/rpeak_validation_ds1.json`.

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
  report to `results/phase1_data/`.
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
- `main()`: writes `results/phase2_ptbxl_data/method_checks_ds1.json`.

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
- `make_run_id(...)`: builds names like `phase1_input_representation-A.lr1e-03-s0-<commit>-<time>`: experiment, variant,
  learning rate, seed, commit, timestamp. `parse_run_id` reverses it.
- `capture_env(device)`: Python and package versions, hardware, lockfile hash.
- `write_run(..., group)`: creates `runs/<group...>/<run_id>/` with `config.yaml`, `env.json`,
  `train_log.csv`, `metrics_val.json`, `preds_val.npz` and the checkpoint. `group` is the folder path,
  for example `("phase2_ptbxl_pretraining", "D_short_window_125hz", "finetune")`. It never overwrites a run and
  refuses a run ID that already exists in any folder.
- `find_run_dir(run_id)`: finds a run in whichever folder it is.
- `default_group(run_id)`: works out the folder from the run ID, for runs saved before folders existed.
- `upsert_runs_index(row)`: adds the run to `results/runs_index.csv`, including its `folder`.
- `ingest(sources)`: copies runs and result files from a Kaggle zip into the repo, keeping the folder
  layout of the zip. It refuses any run carrying test predictions and never overwrites a differing file.

---

## 7. `src/ecgaug/experiments/`: the runners

### `common.py` — shared by every runner

- `check_preconditions()`: loads the approved split and checks the tree is committed; raises
  `PreconditionError` with the reason otherwise.
- `split_part(arrays, name, with_side)`: picks the training or validation beats out of a dataset.
- `train_grid(...)`: the core loop of every experiment. For each learning rate and each seed: build
  the model, optionally load pretrained weights, train, save the run, record the scores. Returns the
  full grid.
- `best_grid_point(grid)`: the learning rate with the better seed-mean score.
- `temporary_root()`: redirects all folders to a temporary directory for smoke runs.

### `phase1_input_representation.py` — phase1_input_representation

- `load_config(path)`: merges a variant file with `shared_budget.yaml`, and rejects a variant file
  that tries to change the budget.
- `run_variant(cfg, split)`: builds the DS1 dataset for the variant, calls `train_grid`, and writes
  `results/phase1_input_representation/<letter>.json` with per-seed scores, mean, spread and
  run IDs.
- `representation_check(cfg)`: `--check` mode: builds the inputs for all of DS1 and reports shapes and
  counts, without training.
- `smoke(cfg)`: `--smoke` mode: the whole thing on fake data in a temporary folder.
- `main()`: reads the command line and calls one of the three.

### `phase2_ptbxl_pretraining.py` — the phase2_ptbxl_pretraining experiment

- `load_config(path)`: takes the input representation and the training budget from the matching
  phase1_input_representation config, and the pretraining settings from `pretraining_settings.yaml`.
- `ensure_ptbxl(root, workers, verify, download)`: returns the preprocessed PTB-XL data, building it
  first (from an attached copy or by download) if it is not on disk.
- `pretrain(cfg, spec, train, val, device)`: trains on PTB-XL once per seed and saves each run.
- `run(cfg, split, ptbxl_arrays)`: the whole experiment: build MIT-BIH data, cut PTB-XL beats, measure
  alignment, pretrain, run `train_grid` twice (from scratch, and with `load_encoder` as the starting
  point), apply `select_winner`, write `results/phase2_ptbxl_pretraining/<letter>.json`.
- `smoke(cfg)` and `main()`: as above.

---

## 7b. `src/ecgaug/workflow.py`: status and the gates

- `status()`: builds the checklist from what exists on disk (verified data, approved split, the four
  phase1_input_representation result files, the phase 1 gate report, `results/manifest.json`, pretraining results) and returns
  the single next action. `python -m ecgaug.workflow status` prints it.
- `evaluate_phase1_gate()`: refuses unless all four phase1_input_representation variants have results; checks they used the same
  split, selection metric and budget with no reported deviations; then calls `select_winner`.
- `write_phase1_gate_report()`: writes `results/phase1_gate/report.md` (table per variant, each step of the rule,
  the winner) and `report.json`.
- `approve_phase1_gate(approvers)`: re-checks that the results have not changed since the report, builds the
  frozen dataset of the winner (`data/processed/v1`, needs `ALLOW_TEST_BUILD=1` and is logged), writes
  the winner and the file hashes to `results/manifest.json`, and appends the approval to
  `docs/DECISIONS.md`.
- `phase1_winner()`: the approved winner, read from the manifest. The pretraining runner uses it to
  refuse running before the phase 1 gate or for another variant (unless `--preview`).

- The later gates follow the same pattern (evaluate, write the report, approve into the manifest and
  `docs/DECISIONS.md`): `evaluate_phase2_gate` (the `cnn` and `rnn` by `select_winner`, the quality
  composite), `evaluate_phase3a_gate` (`select_generator` per family), `evaluate_phase3_gate` (every
  bank present and matching its hash), `evaluate_phase4a_gate` (the policy under the N-recall limit;
  approval writes the full grid file with `full_grid_yaml`), `evaluate_phase4_gate` (strength per
  condition and best generative per architecture), `evaluate_phase5_gate` (saliency method per
  architecture, note variant, demo smoke). `experiment_results(name)` reads any experiment's result files.

## 7c. `src/ecgaug/figures.py`: before-and-after figures

- `phase1_data_figures()`: raw versus band-passed signal of the training record with the most S beats; the
  same N and S beat as variants A–D present them; beats per class in training and validation; the
  median beat per class and variant (training records only).
- `phase2_ptbxl_data_figures(root)`: one recording before and after resampling, with kept and dropped beats;
  PTB-XL versus MIT-BIH beat shape; recordings kept and excluded; label shares per split.
- `phase2_pretraining_figures(result)`, `phase1_variant_figures(result)`, `phase1_gate_figure()`: training curves,
  confusion matrices, per-class F1 before and after pretraining, and the phase 1 gate comparison.
- `safe_for_result(result)`: called by both runners at the end of every experiment; a plotting error is
  reported but never fails a run.

- Phases 2-6: `phase2_quality_figures`, `phase2_gate_figure`, `phase3_variant_figures` (losses, every
  checkpoint scored), `phase3_sample_figures` (real versus synthetic, class counts before and after,
  nearest real beats), `phase3a_gate_figure`, `phase4_strength_figure`, `final_eval_figures` (forest
  plot, per-record S and F), `phase5_saliency_figure`, `error_analysis_figures`. `safe_call(fn, ...)`
  runs any of them without ever failing a run.

## 7d. Phases 2-6: the packages behind the later experiments

### `experiments/data_source.py` — where a run's beats come from

- `load_data(split, preview, representation)`: after the phase 1 gate, the frozen dataset
  (`data/processed/v1`, or rebuilt from the raw records and checked against the frozen hash); before
  it, only in a preview, the candidate dataset of a representation. Returns a `DataSource` (arrays,
  representation, provenance, and why it is provisional).
- `smoke_data()`: fake records, a split and their dataset in a temporary root; every smoke run uses it.

### `models/baselines.py` — the phase 2 classifiers

`ShallowCNN`, `ResNet1D`, `BiLSTMClassifier`, `CNNBiLSTM`: every one takes the beat and optionally the
RR side features, keeps its last layer in `head`, exposes `embed()` (penultimate features) and, if
convolutional, `last_conv` (for Grad-CAM). `build_model` builds any of them from a config block.

### `quality/` — how good synthetic beats are

- `diagnostics.py`: `prepare_references` (per class: the real beats and the leave-one-out memorization
  threshold), `score_beats` (diversity ratio, memorization fraction, C2ST accuracy per class).
- `metrics.py`: the four metric families registered in `METRIC_FAMILIES` (`dtw`, `spectral`,
  `fid_analog`, `tstr`); `dtw_pairs` is the vectorised banded DTW; each family's `evaluate(context, beats)`
  scores a whole set of synthetic beats against the real training beats in a `QualityContext`.
- `sanity.py`: `score_family` scores a family on real held-out beats and on degraded copies and
  applies the pass rule. `feature_net.py`: the frozen network of the FID-analog metric.
- `composite_ranks` (in `__init__.py`): the equal-weight rank average used to break phase 3 ties.

### `generators/` — the cVAE and the WGAN-GP

- `base.py`: `BeatGenerator` (the interface: `fit` on training beats only, `sample(cls, n, seed)`),
  `assert_train_only`, `ClassBatches` (class-balanced batches) and the step loop with a checkpoint at
  each quarter of the budget and divergence checks.
- `cvae.py` (`CVAE`, conv or residual-FiLM encoder and decoder) and `wgan_gp.py` (`WGANGP`, four kinds of
  conditioning, gradient penalty). `build_generator(family, settings, n_samples, seed)` picks one.
- `bank.py`: `sample_bank` (synthetic beats in the data-contract format, RR features copied from real
  beats), `export_bank` / `load_bank` (never overwritten; hash-checked; audit banks refused).
- `physics.py` (P21): McSharry et al.'s ECG simulator (`simulate`, `euler_distance`, `fit` by least squares) and
  `EulerPrior`, SimGANs' Euler loss for generated patient-relative beats. A generator's optional `prior` and
  `prior_weight` add it to the cVAE's and WGAN-GP's objectives; without them training is unchanged.
  P23 adds `simulation_fit` (the simulator's best beat for given wave positions and widths, heights and baseline
  solved exactly per beat), `class_shape` and `FittedPrior`, the loss of what the simulator cannot explain.

### `experiments/generator_runs.py` — the runner of both phase 3 experiments

`PilotHarness` scores a generator at a checkpoint (diagnostics, quality metrics, pilot classifier);
`train_one` trains one generator, logs its run (losses, `checkpoints.csv`, pilot classifier logs) and,
in a final run, exports the bank; `run_pilot` and `run_final` write the result files; `hard_gates`
applies the phase 3 hard gates. `phase3_cvae.py` and `phase3_wgan_gp.py` only call it.

### `augment.py` and `grid/runner.py` — the conditions and the grid

- `augment.py`: `policy_targets` (how many beats per class), `take_from_bank`, `screened_from_bank`
  (policy D), `smote_beats`, `random_oversample`, `class_weights`, `with_added` (always training beats).
- `grid/runner.py`: `cells(grid)` lists the cells; `run_grid` runs those not done (or only lists them);
  each cell's ledger file is written as soon as it finishes; `summarize` gives the seed means the phase
  4 gate reads.

### `final_eval.py` — the single test evaluation

`plan()` lists checkpoints and comparisons without reading test data; `run(confirm)` checks every
precondition before the first test byte, predicts the test set with every frozen checkpoint, and calls
`statistics()` (per-condition intervals, the two Holm families, the sensitivity analyses);
`write_outputs()` writes the CSV and LaTeX tables, figures and `numbers_final_eval.tex`.

### `eval/rq3.py` and `experiments/phase4_leakage_audit.py` — the two analyses after the grid

- `eval/rq3.py` (RQ3, exploratory): reads every official generator run's `checkpoints.csv`, gives
  Spearman's rho of each quality metric with the pilot score (equal to rho with the gain), with a
  95% interval from resampling generator runs; TSTR apart. `results/phase4_quality_vs_gain/`.
- `experiments/phase4_leakage_audit.py` (optional, after `/final-eval`): `intra_patient_split` pools DS1
  and DS2 and splits each class at random with the inter-patient counts; `train_generator` retrains the
  frozen generator variants on that training part (banks in `data/synthetic/audit/`); `run_classifiers`
  reuses the grid's `run_cell` (runs in `runs/audit/`) and tests each run on the intra-patient test
  part; `report` sets the result beside the final evaluation. DS2 is read only through
  `loaders.load_test_for_audit` or the raw records, with `ALLOW_TEST_AUDIT=1`, and logged.

### `explain/`, `report/`, `demo.py`, `paper.py` — phases 5 and 6

- `explain/methods.py`: `saliency(method, model, x, side, settings)` for Grad-CAM, integrated
  gradients, SmoothGrad and occlusion; `explain/evaluate.py`: `faithfulness`, `stability`, `segment_masses`.
- `report/notes.py`: `note_input` (the note's input JSON), `template_note`, `generate` (variants A-D),
  `check_note` (the fact checker), `LLMClient` / `client_from_env`; `report/error_analysis.py`: `analyse`.
- `demo.py`: `explain_beat` classifies, explains and describes one beat and saves the figure. `write_note` returns the
  note and who wrote it: the note variant chosen at the phase 5 gate (`chosen_notes_variant`, C, Gemini) when a key is
  set, else the template, which also replaces a note whose call failed or that failed the fact checker, with the reason.
- `paper.py`: `build()` writes the paper's tables and numbers from the result files. It then calls `report_folder.write()`, which copies only the files the two versions of the course report (`report/report.tex`, 30 pages, and `report/report_short.tex`, 20 pages) input into `report/generated/` (stale ones are removed, so none of the paper's own texts land there) and writes `report/results.md`: every results table of the full report, converted from the same LaTeX tables to Markdown under the report's section titles, with each caption's first sentence as the title and the files each table was built from.

### `experiments/phase6_*.py` — the post hoc analyses P1-P23

Each file runs one post hoc analysis (`PROJECT_GUIDE.md` section 5.15 says what each asks), listed in `names.POST_HOC`;
most have a config folder `configs/<experiment>/` and a notebook `kaggle/<experiment>.ipynb`. The command line runs one
`--part` at a time, for example `train` or `grid` (on Kaggle), `select` (on validation) and `test`. `--part test` is the
analysis's one read of the test records: it needs `ALLOW_TEST_POSTHOC=1` and is logged in `docs/TEST_ACCESS_LOG.md`.
P2 and P8 read no test records; P4 reads them once under the audit permission (`ALLOW_TEST_AUDIT=1`). Results go to
`results/<experiment>/`: `selection.json` from validation, `summary.json` from the test read. Later analyses reuse
earlier ones' code: P17, for example, calls P15's `train_generators`, `diffusion_grid` and `select` under its own
experiment name, and P18 trains new seeds of P13's grid and averages them with the existing runs (`--part check` writes the run IDs to `members.json` before the test read). P19 reads no test records: `--part build` downloads and builds the Icentia11k patients, `--part evaluate` scores P18's frozen runs on them, and `--part run` does both. P20 builds Icentia11k patients on a CPU session (`--part build`) and predicts them on a GPU session that takes the build session's output as an input (`--part run`); the choice is written to `choice.json` before the evaluation patients are scored. P21 (`phase6_physics_prior`) fits McSharry et al.'s simulator to training beats (`--part check`, with pre-registered stop rules), trains P12's generators with the Euler loss (`--part train --family`), makes fusion banks (`--part fusion`) and reuses P20's choice and scoring code (`--part score`, `--part test`). P22 (`phase6_bigger_classifier`) trains phase 2's ResNet1D in P18's grid and scores it in one GPU session (`--part run`); P23 (`phase6_physics_fitted`) follows P21's parts with `FittedPrior`. Both use `ensemble_compare.py`, which chooses Huang's stage and the veto for new ten-run ensembles on P14's patients, compares them with P18's on P19's patients and reads DS2 once. From P9 on, each test report also recomputes the earlier test scores it compares against and records
the difference (`reproduction_difference` in its summary). `figures.posthoc_figures()` draws every post hoc figure whose
result file exists, and `paper_text.py` writes the paper's post hoc paragraphs from the same files. `report_text.py` writes the course report's short texts from them too (its summary, an example of the system's output, the follow-up analyses told by theme (`posthoc_story(figures=False)` is the 20-page version's, without the three follow-up figures), a discussion that is the paper's in brief form (`paper_text.discussion(brief=True)`) plus what the best pipeline's scores mean in the clinic, and its conclusion), and `figures.report_figures()` draws the report's medical-background figure (training records only) and the figure of how the test score developed.

## 8. `configs/`: the settings files

| File | Sets |
|---|---|
| `validation_split/approved.yaml` | the validation records, the selection metric, who approved |
| `validation_split/options.yaml` | the three candidate splits that were compared |
| `phase1_input_representation/shared_budget.yaml` | seeds, learning rates, epochs, patience, batch size, network size |
| `phase1_input_representation/<letter>_<name>.yaml` | one variant's window, filter, sampling rate, RR features |
| `phase2_ptbxl_pretraining/pretraining_settings.yaml` | learning rate, epochs and patience for the PTB-XL pretraining part |
| `phase2_ptbxl_pretraining/<letter>_<name>.yaml` | which phase1_input_representation variant this pretraining run uses |
| `phase2_baseline_classifiers/` | `shared_budget.yaml` (seeds, learning rates, epochs, `min_epochs`) and one model per variant |
| `phase2_quality_metrics/` | `metric_settings.yaml` (degradations, pass rule, metric settings, feature network) and one family per variant |
| `phase3_cvae/`, `phase3_wgan_gp/` | `generator_settings.yaml` (training, pilot, diagnostics, hard gates, bank size, generator defaults) and each variant's generator settings |
| `phase4_augmentation_policy/` | `pilot_settings.yaml` (classifier, seed, banks, N-recall limit) and each variant's policy and strengths |
| `phase4_full_grid/` | grid files: conditions, strengths, architectures, seeds (`full_grid.yaml` is written by the phase 4a gate) |
| `phase5_explainability/`, `phase5_clinical_notes/` | shared evaluation settings and each variant's method or note generator |

---

## 9. `kaggle/`: running on Kaggle

- `phase1_input_representation.ipynb`, `phase2_ptbxl_pretraining.ipynb`: parameter cell, a cell that clones
  the repo at your branch using the GitHub token from Kaggle Secrets, and a cell that turns your
  variant letters into config files and calls `kaggle_run.py`.
- `kaggle_run.py`: `setup_env` installs the environment; `setup_data` copies or downloads MIT-BIH and
  verifies it; then it runs each config through its runner; `bundle` zips `runs/` and the result
  files.
- `make_raw_dataset.py`: zips the verified MIT-BIH files for upload as a Kaggle dataset.
- `push_results(branch, message, name, email)` in `kaggle_run.py` (only with `PUSH_RESULTS = True`, off by default): after the runs, commits only run
  outputs (run folders without weights, results, figures, runs index) and pushes them to the branch, or
  to a new `<branch>-kaggle-<time>` branch if the branch moved on. Refuses if a run carries test predictions.
- `ingest_runs.py`: the default path; calls `runs.ingest` on a downloaded zip (runs, result files,
  figures, and synthetic banks, whose `bank.json` entries are merged by seed).
- `setup_artifacts(dir)` in `kaggle_run.py`: copies banks and run checkpoints from an attached Kaggle
  dataset (`--artifacts-dir`), for the phase 4 and 5 notebooks.
- One notebook per experiment, plus `phase4_full_grid.ipynb` for the grid runner.

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
| `test_phase1_input_representation.py` | equal budgets; refusals; smoke runs of A–D |
| `test_ptbxl.py` | the detector; every PTB-XL check; patient-wise folds; padding; checksums |
| `test_phase2_ptbxl_pretraining.py` | weight transfer; config; the experiment end to end on fake data |
| `test_workflow.py` | the status checklist and the phase 1 gate |
| `test_figures.py` | the figures are drawn from data and results, never from test data |
| `test_phase2.py` | the four baselines; `min_epochs`; DTW against dtaidistance; every metric family; the sanity harness; the phase 2 gate |
| `test_generators.py` | every generator variant trains, checkpoints and samples deterministically; train-only rule; divergence; banks |
| `test_quality_and_augment.py` | diagnostics (collapse, copies, calibration); composite rank; SMOTE, oversampling, class weights |
| `test_phase3.py` | equal budgets; refusals before the gates; smoke runs; the phase 3a and 3 gates; the grid resumes; banks travel back from Kaggle |
| `test_phase4.py` | policies and the DTW screen; the phase 4a and 4 gates; the statistics; the final evaluation on fake data runs once |
| `test_phase5.py` | saliency finds what a model uses; faithfulness; the fact checker; LLM variants with a fake client; demo; error analysis; the phase 5 gate |
| `test_paper.py` | tables and numbers come from result files; missing results are marked |
| `test_report_folder.py` | `report/` holds exactly what both versions of the report input, each cites only `report/refs.bib` and refers only to labels it has; the Markdown converter; the generator fills the folder |

`conftest.py` makes every test run in a temporary folder with the test lock closed.

---

## 11. Two walkthroughs: what happens when you run a command

### phase1_input_representation, variant A

```
uv run python -m ecgaug.experiments.phase1_input_representation --config configs/phase1_input_representation/A_short_window.yaml
```

1. `load_config` merges `A_short_window.yaml` with `shared_budget.yaml`.
2. `check_preconditions` loads the approved split and checks the tree is committed.
3. `build_candidate` calls `build_arrays` for the 22 DS1 records: `extract_beats`, `read_signal`,
   `usable_mask`, `transform_record`. The result is saved with its hash.
4. `load_candidate` verifies the hash and loads; `split_part` separates training from validation.
5. `train_grid` runs 6 trainings (2 learning rates × 3 seeds). Each one: `set_seed`, `build_model`,
   `fit_classifier`, `write_run` into `runs/phase1_input_representation/A_short_window/`, `upsert_runs_index`.
6. `best_grid_point` picks the learning rate; the summary is written to
   `results/phase1_input_representation/A.json`.

### phase2_ptbxl_pretraining experiment, variant A

```
uv run python -m ecgaug.experiments.phase2_ptbxl_pretraining --config configs/phase2_ptbxl_pretraining/A_short_window.yaml --ptbxl-download
```

1. `load_config` reads the representation and budget from phase1_input_representation's A config.
2. `check_preconditions`, as above.
3. `ensure_ptbxl`: `download_ptbxl`, then `prepare` (metadata, signal checks, resampling, peak
   detection, beat filtering, labels, folds, checksum comparison), then `load`.
4. `run` builds the MIT-BIH dataset (as in phase1_input_representation) and cuts PTB-XL beats with `beat_arrays`.
5. `pretrain`: 3 runs of `fit_multilabel` on PTB-XL, one per seed.
6. `train_grid` from scratch: 6 runs. `train_grid` with `load_encoder`: 6 runs, each starting from the
   pretraining run of the same seed.
7. `select_winner` compares the two; the summary is written to
   `results/phase2_ptbxl_pretraining/A.json`.
