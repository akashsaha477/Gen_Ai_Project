# Project rules and data protocol

The rules every experiment follows, the MIT-BIH data protocol and the data contract. Detailed specs:
`docs/PLAN.md` (phases, experiments, variants, gates) and `docs/PAPER_TRACK.md` (research questions,
statistics, paper artifacts). Live state: `docs/PROGRESS.md`, `docs/DECISIONS.md`. Amended 2026-10-08
after the kickoff review (`docs/DECISIONS.md`, A1–A11). Section and rule numbers are the ones the code,
the briefs and the pre-registration cite.

## 1. Mission

Two goals, in priority order:

1. **Course deliverable in 30 days** (Day 1 = 2026-10-07, Day 30 = 2026-11-05): ECG beat in →
   rare-arrhythmia prediction + saliency map + grounded clinician-readable note.
2. **Publication-grade evidence.** Every experiment is reproducible, leakage-free, selected by
   pre-registered rules and honestly reported, including negative results.

**Research question.** Under a strict inter-patient protocol, does class-conditional generative
augmentation (cVAE vs WGAN-GP) improve detection of rare arrhythmia classes compared with no
augmentation, class-weighted loss and SMOTE? What do synthetic-quality metrics and saliency analysis
reveal about when and why it helps?

## 2. Team model

- Four members, **no fixed owners**. Each experiment has four variants (A–D). Assignments are drawn per
  phase and recorded in `docs/ASSIGNMENTS.md`.
- Shared infrastructure lands on `main` per phase. Members implement variants on their own branches.
  Winners are chosen at each gate by pre-registered rules.
- **The team approves every gate.**

## 3. Non-negotiable rules

1. **Inter-patient only.** Splits are by record (patient): DS1 → train + validation, DS2 → test.
   No record appears in two splits. Generators, oversamplers, normalization statistics and any fitted
   preprocessing see training records only. One known exception to record = patient: records 201 (DS1)
   and 202 (DS2) are the same subject. The de Chazal split is kept for comparability, the overlap is
   disclosed, and test results are also reported without record 202.
2. **Test lock.** Test data lives in a separate file, readable only via `ecgaug.data.load_test()`,
   which raises unless `ALLOW_TEST=1` is set by the final evaluation. All raw-file reads go through
   `ecgaug.data.raw`, which refuses DS2 records without a matching permission. Access has three kinds,
   each appended to `docs/TEST_ACCESS_LOG.md`: *construction* (counting labels and building arrays
   under `ALLOW_TEST_BUILD=1`; no model sees the data), *evaluation* (only the final evaluation) and
   *audit* (the leakage-audit experiment, only after the final evaluation). No decision of any kind
   (model, variant, epoch, threshold, strength, prompt) may use test data.
3. **Selection on validation, by pre-registered rules.** Rules in `docs/PREREGISTRATION.md` are
   committed before the experiments they govern and applied literally. If a rule is ambiguous for the
   actual results, the team decides before anything runs; nothing is improvised post hoc.
4. **Frozen means frozen.** Artifacts frozen at a gate carry SHA-256 hashes in
   `results/manifest.json`. Changing one needs an approved entry in `docs/DECISIONS.md` and a version
   bump; affected downstream results are re-run, not patched.
5. **No invented numbers.** Every number in docs, tables, figures, slides or the paper is produced by
   code from logged runs and traceable to run IDs. Unknown = `TODO`.
6. **Report what happened.** Failed variants, collapsed generators, unsupported hypotheses and bugs
   are recorded plainly. Negative results go in the paper.
7. **Equal budgets.** Variants within an experiment get identical data, seeds, epoch/step caps, tuning
   grid and early-stopping rule. Any deviation is flagged in the gate report.
8. **Seeds.** Default `[0, 1, 2]` (extend to `[0..4]` if approved at the phase 3 gate). Seed Python,
   NumPy and PyTorch; enable deterministic algorithms where available; log known nondeterminism.
9. **Gates are approved by the team.** Each phase ends with a gate; the next phase starts only after
   the team approves it.
10. **Ask the team first** before adding a heavy dependency, changing the plan or a pre-registered
    rule, touching a frozen artifact, launching anything estimated above 2 GPU-hours, deleting run
    outputs, or rewriting git history.
11. **Secrets** come from environment variables only. They are never committed or printed.

## 4. Data protocol (MIT-BIH Arrhythmia Database, inter-patient)

- **Source:** PhysioNet via `wfdb.dl_database('mitdb', 'data/raw/mitdb')`. A Kaggle mirror is
  acceptable only if it keeps record IDs and original annotation symbols.
- **DS1 (train + validation):** 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203,
  205, 207, 208, 209, 215, 220, 223, 230.
- **DS2 (test):** 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222,
  228, 231, 232, 233, 234.
- **Excluded (paced):** 102, 104, 107, 217. (Protocol: de Chazal et al., 2004.)
- **Lead:** MLII, selected by signal name from the header (`sig_name`), never by channel index.
- **Labels:** reference annotations (`atr`) mapped to AAMI EC57 classes:
  N = {N, L, R, e, j}; S = {A, a, J, S}; V = {V, E}; F = {F}; Q = {/, f, Q}.
  Non-beat annotations are ignored. RR intervals are measured between consecutive QRS-type
  annotations (including Q-class beats and flutter waves) before any class is dropped.
- **Q policy:** Q (about 8 beats in DS1, 7 in DS2) is excluded from training and primary metrics;
  counts are reported.
- **Sanity counts** (literature; small deviations fine, large ones mean a bug):
  DS1 ≈ N 45,868 / S 942 / V 3,787 / F 415. DS2 ≈ N 44,258 / S 1,837 / V 3,221 / F 388.
- **Concentration warning:** rare classes sit in very few records; the validation split, the bootstrap
  design and the paper's limitations all depend on the per-record counts.
- **Validation:** a patient-wise subset of DS1 records, approved at the phase 1 gate. Generators never
  see validation records.
- **Primary metric (pre-registered):** rare-class macro-F1 = mean F1 over S and F (validation for
  selection, test for final results). If the validation split holds fewer F beats than the minimum
  support fixed in `docs/PREREGISTRATION.md`, selection uses S F1 with V F1 as tie-break. Decisions are
  argmax in every condition; no threshold is tuned.
- **Secondary metrics:** per-class F1, one-vs-rest PR-AUC, AAMI-style sensitivity (Se) and positive
  predictivity (+P) for S and V, N recall, macro-F1 over N/S/V/F.

## 5. Data contract (frozen at the phase 1 gate)

`data/processed/v{K}/`
- `trainval.npz` — `X` float32 [n, L]; `y` int8 (0 N, 1 S, 2 V, 3 F); `record` int16; `split`
  {'train','val'}; `r_idx` int32 (R-peak sample index); `rr_prev`, `rr_next` float32 seconds;
  `rr_ratio` float32 (`rr_prev` over the mean of the previous 10 RR intervals); `is_synthetic` bool;
  `source` str ('real').
- `test.npz` — same fields for DS2; readable only through the guarded loader.
- `contract.json` — fs, window, filtering, normalization, label map, record lists per split, Q policy,
  RR definition, usable-beat rule, segment windows, SHA-256 of each array file, git SHA at creation.
- The usable-beat rule (window fits in the record, both RR intervals exist) is the same for every
  representation.

`data/synthetic/<generator_id>/seed<s>.npz` — same fields, `is_synthetic=True`, `source=<generator_id>`,
plus `bank.json` (config, checkpoint hash, samples per class, quality report).

**Saliency output (used by the note generator):** per beat — `saliency` float32 [L]; `segments` list of
{name ∈ P|QRS|ST|T, start_ms, end_ms, mass} using fixed R-relative windows; `method`; `model_id`.

**Note-generator input JSON:** `{record_anon_id, predicted_class, probabilities{N,S,V,F},
low_confidence, top_segments[{name, mass}], model_id}`. The low-confidence rule is pre-registered.

## 6. Repository layout

```
configs/               YAML per experiment, e.g. configs/phase1_input_representation/<letter>_<name>.yaml
src/ecgaug/            data, eval, models, train, experiments, generators, quality, grid, explain, report, demo
scripts/               thin CLIs
kaggle/                guide, one notebook per job, kaggle_run.py, ingest_runs.py
runs/<run_id>/         per-run outputs (bulky files not committed)
results/               runs_index.csv, per-experiment summaries, manifest.json
tables/ figures/       generated only by scripts
docs/                  RULES, PLAN, PAPER_TRACK, PROGRESS, DECISIONS, PREREGISTRATION, TEST_ACCESS_LOG, briefs/
tests/                 leakage, test lock, contract hashes, metrics, bootstrap
```

## 7. Engineering conventions

- Python ≥ 3.10, PyTorch, numpy, scipy, scikit-learn, pandas, matplotlib, wfdb, imbalanced-learn,
  captum, a DTW library. Everything pinned in a lockfile.
- Logic lives in `src/ecgaug/`. Scripts are thin CLIs. Kaggle notebooks only call package code.
- Every run is fully specified by config + seed + git SHA.
- Type hints, docstrings on public functions, pytest. Tests are mandatory for split leakage, the test
  lock, contract hashes, metric functions (hand-computed cases) and the bootstrap.
- Small commits with clear messages. `data/` and bulky `runs/` outputs are not committed; `results/`
  summaries are.

## 8. Experiment logging

- `run_id = <experiment>-<variant>[.<tag>]-s<seed>-<gitsha7>-<yyyymmddHHMM>`; the optional tag names a
  point of the tuning grid. A run directory is never overwritten. (`docs/COMMIT_MAP.csv` maps the SHAs
  of commits made before the 2026-11-07 history rewrite to the rewritten ones.)
- `runs/<run_id>/`: `config.yaml`, `env.json` (packages, GPU type, wall-clock), `train_log.csv`,
  `metrics_val.json`, `preds_val.npz`, checkpoint path + hash. `preds_test.npz` exists only after the
  final evaluation.
- `results/runs_index.csv`: one row per run (run_id, phase, experiment, variant, seed, status, primary
  metric, wall-clock). All tables and figures are generated from this index.

## 9. Compute

Kaggle GPU sessions (weekly quota) plus laptops. Targets: classifier run ≤ 15 min, generator pilot
≤ 45 min, full generator training ≤ 2 h per seed. Early stopping on the validation primary metric.
Kaggle flow: a notebook checks out the repo at a given SHA, runs one config, and packages
`runs/<run_id>/`; `kaggle/ingest_runs.py` brings outputs back and updates `results/runs_index.csv`.

## 10. Timeline

| Phase | Days | Dates | Experiments | Gate |
|---|---|---|---|---|
| 1 data | 1–3 | Oct 7–9 | data pipeline, evaluation code, phase1_input_representation | data contract frozen |
| 2 baselines | 4–8 | Oct 10–14 | phase2_baseline_classifiers, phase2_quality_metrics, phase2_ptbxl_pretraining | baselines + metrics frozen |
| 3 generators | 9–14 | Oct 15–20 | phase3_cvae, phase3_wgan_gp, grid runner | synthetic banks frozen (3a on Day 12) |
| Buffer | 15 | Oct 21 | — | — |
| 4 comparison | 16–21 | Oct 22–27 | phase4_augmentation_policy, full grid | grid done → final eval |
| 5 explanations | 22–26 | Oct 28–Nov 1 | phase5_explainability, phase5_clinical_notes, demo | code freeze |
| 6 paper | 27–30 | Nov 2–5 | report, paper draft | submission (29–30 buffer) |

Names follow `docs/NAMING.md`: phase → experiment (`phase<N>_<what>`) → variant (A–D) → run.
