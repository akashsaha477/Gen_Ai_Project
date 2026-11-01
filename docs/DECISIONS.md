# DECISIONS

Entries are append-only. Gate entries follow the template in `docs/PLAN.md`.

---

## Plan amendments after the kickoff review — 2026-10-08 (Day 2) — status: ADOPTED by Akash Saha, PENDING TEAM CONFIRMATION at the phase 1 gate

**Context.** The kickoff review of `docs/RULES.md`, `docs/PLAN.md` and `docs/PAPER_TRACK.md` found ten gaps.
Akash delegated the choice of fixes ("do what you think is best"). All ten were adopted and written into
the three documents in one commit, separate from the kit-install commit, so the team can read the diff
and revert it. Nothing below is in force as a pre-registered rule until `docs/PREREGISTRATION.md` v1 is
approved and committed.

**Two amendments change the plan itself and need all four members' agreement: A2 and A7.**

| ID | Amendment | Why | Files |
|---|---|---|---|
| A1 | The selection metric uses F only if the validation split holds at least the minimum F support fixed in PREREGISTRATION (proposed: 100 beats from 2 records); otherwise S F1 with V F1 as tie-break. The tie margin becomes max(0.02, pooled across-seed SD). | Record 208 holds 373 of DS1's 415 F beats, so validation F is either the bulk of the class or a few dozen beats. The old fallback only covered F being absent. | docs/RULES.md §4, PLAN.md conventions |
| A2 | **Plan change.** Two test families instead of one family of 10: confirmatory (H1: cVAE vs none, WGAN-GP vs none, per architecture = 4 tests) and secondary (the other comparisons). Holm within each family on two-sided bootstrap p-values. Replicates in which S or F has no true beats are excluded and counted. The dominant-record-absent fraction and per-record results for S and F are reported. | Record 213 holds 362 of DS2's 388 F beats and record 232 holds 1,382 of its 1,837 S beats, so intervals will be wide. The old plan did not define the metric for a class-absent replicate or how Holm applies to intervals. | PAPER_TRACK.md statistics |
| A3 | Records 201 (DS1) and 202 (DS2) are the same subject. The de Chazal split is kept, the overlap is disclosed, and test results are also reported without record 202. | MIT-BIH has 48 records from 47 subjects; "record (patient)" in rule 1 was not exact. | docs/RULES.md §3, PAPER_TRACK.md |
| A4 | The leakage-audit experiment runs only after `/final-eval`, in `runs/audit/` and `data/synthetic/audit/`, with each run logged in TEST_ACCESS_LOG. The main grid refuses audit artifacts. | It trains and evaluates on DS2 beats, so before `/final-eval` it would be a second test access. | docs/RULES.md §3, PLAN.md Phase 4 |
| A5 | Every "best" (generative condition, checkpoint, strength, cVAE, WGAN-GP) means highest validation selection metric, recorded in DECISIONS.md before `/final-eval`. | Read off the test table, "best" would be selection on test. | PAPER_TRACK.md, PLAN.md phase5_explainability |
| A6 | Test access has three kinds: construction (labels counted, arrays built, no model involved), evaluation (only `/final-eval`) and audit (A4). Each is logged. Phase 1 gate requires zero evaluation accesses instead of an empty log. All raw-file reads go through one module that enforces the lock. | Phase 1 has to read DS2 to count classes and build `test.npz`, which contradicted "every access is logged" plus "log empty at the phase 1 gate". | docs/RULES.md §3, PLAN.md the phase 1 gate, `/final-eval` |
| A7 | **Plan change.** The grid gains random oversampling of real minority beats at the same three strengths, and class-weighted loss gets three pre-registered weightings instead of one. Runs at 3 seeds: 96 instead of 66. Decisions are argmax with no threshold tuning in any condition. | Without a matched-count control, a gain from generated beats cannot be separated from rebalancing. The baselines had one configuration each while the generative arms had many. | PLAN.md Phase 3–4, PAPER_TRACK.md |
| A8 | Each synthetic bank is exported from the final checkpoint of its step budget. Earlier checkpoints are scored for RQ3 only. | The pilot scores every quarter-checkpoint on validation; leaving the choice open added an unstated selection. | PLAN.md Phase 3 |
| A9 | In RQ3, TSTR is analysed separately from the fidelity and diversity metrics, and intervals come from resampling generator training runs, not checkpoints. | TSTR and the downstream pilot score are both validation F1 of a CNN trained on the same synthetic beats, and checkpoints of one run are dependent. | PAPER_TRACK.md RQ3 |
| A10 | Each PREREGISTRATION version is committed, tagged `prereg-vK` and pushed the same day. H1 is confirmatory; H1b, H1c, H2, H3 and H4 are labelled secondary or exploratory. | A pre-registration is only as credible as its timestamp. | PAPER_TRACK.md, PLAN.md |
| A11 | Specification details: RR intervals are computed over all QRS-type annotations before any class is dropped; the usable-beat rule is identical for every representation; the contract gains `rr_ratio`; run IDs may carry a tuning tag; the layout gains `train/`, `experiments/` and `runs.py`. | Gaps found while specifying Phase 1. | docs/RULES.md §5, §6, §8 |

**Actions only a human can take (A10):** protect `main` against force-pushes in the GitHub settings;
optionally deposit PREREGISTRATION v1 on OSF for an independent timestamp.

**Cost of A7:** 30 more classifier runs in Phase 4 (at the plan's 15-minute cap, at most 7.5 GPU-hours).

---

## Phase 1 gate — status: NOT STARTED

Open human decisions before phase1_input_representation variants may run (evidence in `results/phase1_data/`):

1. Validation split (options in `results/phase1_data/val_options.md`).
2. Q-class handling and the selection-metric fallback (PREREGISTRATION §3).
3. Approval of PREREGISTRATION v1, then commit, tag `prereg-v1` and push.

## 2026-10-13 (Day 7) — decided by Akash Saha

- Validation split: option V1, validation records 115, 118, 201, 205, 223 (`configs/validation_split/approved.yaml`).
  Selection metric: S F1 with V F1 as tie-break, by the minimum-support rule.
- The runner no longer requires `docs/PREREGISTRATION.md` to be in force. A run needs only the approved
  split and a committed tree. The document stays as a description of the rules; it is not a
  time-stamped pre-registration, and the paper should not call it one.

## 2026-10-13 (Day 7) — dataset — decided by Akash Saha

- The project uses the MIT-BIH Arrhythmia Database only. PTB-XL was considered and not adopted: it has
  no beat-level annotations, so it does not fit the beat-level task, the AAMI classes or the generators.
- PTB-XL stays in the paper as related work. External validation, if added after the deadline, uses a
  beat-annotated database (INCART), as listed in `docs/PAPER_TRACK.md`.

## 2026-10-13 (Day 7) — A12: phase2_ptbxl_pretraining experiment — decided by Akash Saha

This replaces "MIT-BIH only" above. MIT-BIH stays the dataset for training, validation, generators and
the test. PTB-XL is added for one purpose: pretraining the classifier before fine-tuning on MIT-BIH.

- **Method:** supervised. Every PTB-XL beat gets its recording's diagnosis as label. Chosen over
  self-supervised pretraining because it is simpler; the labels are noisy at beat level, and that is
  stated as a limitation.
- **Task stays beat-level:** fine-tuning and all evaluation use MIT-BIH beat labels (N, S, V, F).
- **Scope:** one extra condition for the baseline classifier in Phase 2, "pretrained on PTB-XL then
  fine-tuned" against "trained from scratch", same budget and seeds. It enters the Phase 4 grid only if
  it is better on validation by the usual tie rule.
- **Leakage:** PTB-XL never touches DS2 and the test lock is unchanged. PTB-XL patients are not
  MIT-BIH patients.
- **Order:** built in Phase 2, after the phase 1 gate, because the beat window comes from the phase1_input_representation winner.
  Details are in `docs/PLAN.md`, Phase 2.

## 2026-10-13 (Day 7) — A13: phase2_ptbxl_pretraining experiment built and may run before the phase 1 gate — decided by Akash Saha

- The PTB-XL preprocessing for Kaggle and the supervised pretraining plus fine-tuning runner were built
  now, at Akash's instruction, so the team can run both in one Kaggle session.
- This is ahead of the phase order (rule 9: no next phase before the gate). Consequences, recorded so
  they are not forgotten:
  - The input representation is not frozen. Every result file of the experiment carries `provisional: true`
    until `data/processed/v1` exists. The experiment is rerun for the phase 1 gate winner.
  - The architecture is the probe CNN, the only classifier that exists. The experiment is rerun with the
    `cnn` chosen in phase2_baseline_classifiers.
  - Results of the experiment are not an input to the phase 1 gate. Phase 1 gate is decided from the phase1_input_representation variants alone.
- Protocol fixed before any run (details in `docs/PLAN.md`, Phase 2): one pretraining run per seed;
  fine-tuning seed s starts from pretraining seed s; full fine-tuning with a new head; both methods use
  the phase1_input_representation budget; pretraining stops on PTB-XL fold 9 only; from scratch is the simpler option in
  the tie rule.
- Edge treatment for 10-second recordings: mirror padding of 2 s, chosen from a DS1 measurement
  (`results/phase2_ptbxl_data/method_checks_ds1.json`). No mains notch filter is applied to PTB-XL: the
  64-recording sample shows no 50 Hz interference; the full-data figure is in the preprocessing report.

## 2026-10-15 (Day 9) — the phase order is enforced in code — decided by Akash Saha

- `python -m ecgaug.workflow status` shows every checklist item and the next action; `python -m ecgaug.workflow
  gate1` writes the phase 1 gate report, and `--approve` freezes the winner's dataset and records it in
  `results/manifest.json`.
- The PTB-XL pretraining runner now refuses to run before the phase 1 gate is approved, or for a variant that the phase 1 gate
  did not choose, unless `--preview` is given. Preview results are marked provisional. This replaces
  the open-ended exception of A13: the variant D run stays on record as a preview.

## 2026-10-15 (Day 9) — one naming scheme — decided by Akash Saha

- Every name follows `docs/NAMING.md`: phase (1–6) → experiment `phase<N>_<what>` → variant A–D → run.
  The words "stage", "step", "module" and "arm" are no longer used.
- Old → new: Module 1 → `phase1_input_representation`; the PTB-XL pretraining arm →
  `phase2_ptbxl_pretraining`; Module 3 → `phase2_baseline_classifiers`; Module 4 →
  `phase2_quality_metrics`; Module 5 → `phase3_cvae`; Module 6 → `phase3_wgan_gp`; Module 7 →
  `phase4_augmentation_policy`; Module 8 → `phase5_explainability`; Module 9 → `phase5_clinical_notes`;
  Gate N → the phase N gate.
- Each experiment's name is used for its config folder, runner, notebook, run folders, results and
  figures. Run IDs now start with the experiment name; the 15 runs of the variant D preview were renamed
  (for example `p2-m3-scratchD...` → `phase2_ptbxl_pretraining-scratchD...`), with no other change.

## 2026-10-16 (Day 10) — A14: phases 2-6 built ahead of the gates — decided by Akash Saha, choices PENDING TEAM CONFIRMATION

- At Akash's instruction ("do the phase 3 coding", then "finish the project") the code of every remaining
  phase was built on the branch `finish-project` before the phase 1 gate is approved. Nothing official
  has run: every runner refuses an official run until the gates it depends on are approved, and
  `--preview` runs are marked provisional and cannot be approved at a gate. This keeps rule 9 for the
  experiments; only the code is ahead of the plan.
- Gate decisions stay with the team. New gate commands follow the phase 1 gate: `phase2-gate`, `phase3a-gate`
  (generator selection, Day 12), `phase3-gate` (banks frozen), `phase4a-gate` (augmentation policy; its
  approval writes `configs/phase4_full_grid/full_grid.yaml`), `phase4-gate` (strengths and checkpoints for
  the final evaluation), `phase5-gate`. `/final-eval` is `python -m ecgaug.final_eval`.

Choices made where the plan left the detail open (each one needs the team's agreement before official runs):

| ID | Choice | Why | Where |
|---|---|---|---|
| A14.1 | Generators model the waveform only; the RR features of a synthetic beat are copied from a random real training beat of the same class (all three together). | PREREGISTRATION §5 fixed this for the case that C wins. | `generators/bank.py` |
| A14.2 | cVAE B: its two KL weights share the pilot budget (half the steps each); both are reported, the better passing one represents B. WGAN-GP D: the three per-class GANs share the step budget (a third each). | "Both tried inside the pilot budget"; equal compute across variants (rule 7). | `generator_runs.py`, `wgan_gp.py` |
| A14.3 | Memorization threshold: 5th percentile of leave-one-out nearest-neighbour distances among the class's training beats (instead of a 10% hold-out). | With a 10% hold-out the threshold rests on about 39 F beats and an honest generator could fail the 10% ceiling by chance (measured: 0.06-0.18 on same-distribution data); leave-one-out gives about 0.05 as intended. | `quality/diagnostics.py` |
| A14.4 | Hard gates apply to every generated class (S, V, F) and every pilot seed; the phase 3 tie margin is the general one, max(0.02, pooled SD); ties go to the composite quality rank, or V F1 while no composite exists, then fewer parameters. | PLAN Phase 3 with PREREGISTRATION §4 and §9. | `eval/selection.py` |
| A14.5 | DTW is computed with a numpy banded dynamic programme (window 10% of the beat), the 5 nearest by DTW searched among the 20 nearest by Euclidean distance; checked against dtaidistance in the tests. | The compiled dtaidistance library does not load on macOS here; the pure-Python one is far too slow. | `quality/metrics.py` |
| A14.6 | FID-analog features come from a separate frozen network (the probe CNN on the waveform, fixed recipe, about 80 KB, committed with its hash), not from the chosen `cnn`. | Every machine must score with identical weights, and the feature extractor should not be one of the classifiers being compared. | `quality/feature_net.py` |
| A14.7 | Sanity harness: beat-level halves of the training beats (F sits almost entirely in one record), degradations noise / label shuffle / time shift / smoothing at three severities, pass = real beats best + Spearman >= 0.8 per degradation + median CV < 10% over 3 resamples. | PLAN Phase 2 names the degradations but not their levels or how to combine them. | `quality/sanity.py` |
| A14.8 | `min_epochs` training option (default 0 = the pre-registered rule). Setting it to 10 is the early-stopping fix proposed in docs/PHASE1_REPORT.md §6. | The fix needs the team's decision; the option makes it a one-line config change. | `train/loop.py`, phase 2 budget |
| A14.9 | Synthetic banks hold, per class, the training N count minus the class's count (enough for any policy); a policy takes the first k beats in bank order. | One bank serves every strength and policy. | `generators/bank.py` |
| A14.10 | The grid trains its own "none" cells instead of reusing the phase 2 checkpoints. | One code path for all conditions; the checkpoints the final evaluation uses are the grid's. | `grid/runner.py` |
| A14.11 | Augmentation policy: the N-recall limit (0.01) applies to the mean over the two banks; complexity A 1, B 2, C 2, D 3; D's DTW screen (keep the closest half of twice as many bank beats) applies to generator banks only; strengths per policy B and D {0.1, 0.25, 0.5}, C {1, 2, 4}. | PREREGISTRATION v3 items. | `configs/phase4_augmentation_policy/` |
| A14.12 | Phase 4 gate: per condition the strength with the best seed-mean validation metric (ties to the lower strength); best generative per architecture by the same rule. | PLAN Phase 4, A5. | `workflow.py` |
| A14.13 | Explainability: selection on the mean faithfulness gain over the three models of an architecture (none, cVAE, WGAN-GP), ties by stability, then the cheaper method; integrated gradients averages the zero and per-beat-mean baselines; SmoothGrad (not gradient x input) for C; occlusion window 50 ms, stride 25 ms. | PLAN Phase 5 lists options without fixing them. | `explain/`, `configs/phase5_explainability/` |
| A14.14 | Notes: low-confidence rule "top probability below 0.6" (proposed; to pre-register before phase 5 runs); LLM provider through the standard library, model from `ECGAUG_LLM_MODEL` (default `llm-model`), key from `LLM_API_KEY` only; note variant chosen by the hard gate, then coverage, ratings, latency, tokens. | PLAN Phase 5. | `report/notes.py` |
| A14.15 | The paper's `refs.bib` stays empty until each entry is verified from its DOI or arXiv page. | No invented references (rule 5, /paper). | `paper/` |

## 2026-10-26 (Day 20) — A15: early-stopping fix and Kaggle pushes — decided by Akash Saha, PENDING TEAM CONFIRMATION

- **A15.1 `min_epochs: 10`** from phase 2 on (option 1 of docs/PHASE1_REPORT.md §6): patience is not
  allowed to stop a run before epoch 10. Applies to the phase 2 baselines (and through the phase 2 gate
  to phases 3–4), the PTB-XL fine-tuning arms and every preview stand-in classifier. Not applied to
  phase 1 (its decision stands) or to the fixed TSTR and feature-net recipes. Fixed before any official
  phase 2 run; PREREGISTRATION §4 and §11 updated. The phase 3 pilot classifier had dropped
  `min_epochs`; fixed with a test.
- **A15.2 Kaggle notebooks no longer push by default** (`PUSH_RESULTS = False`). Results come back as the
  output zip, are brought in with `kaggle/ingest_runs.py` and committed by the member under their own
  name. `True` still works.

## the phase 1 gate — 2026-10-26 — status: APPROVED by Akash Saha
phase1_input_representation: winner C (C_short_window_rr_features) — evidence: results/phase1_gate/report.md — rule: PREREGISTRATION §4–5
Frozen artifacts (hashes): data/processed/v1, see results/manifest.json

## 2026-10-27 (Day 21) — A16: the leakage audit and the RQ3 analysis built — decided by Akash Saha, PENDING TEAM CONFIRMATION

Choices where PLAN Phase 4 and PAPER_TRACK RQ3 leave the detail open:

| ID | Choice | Why |
|---|---|---|
| A16.1 | Intra-patient split: DS1 and DS2 beats pooled; each class's beats assigned at random (seed 0) to train, validation and test with the inter-patient counts of each part. | Only the patient separation changes; data amount and class balance stay equal. |
| A16.2 | The generator variants frozen at the phase 3 gate are retrained at their full budget on the intra-patient training beats, on the grid seeds (banks in `data/synthetic/audit/`). | "The same pipeline": a generator must not have seen the intra-patient test beats. |
| A16.3 | Medium strength = the middle of the full grid's strengths for each generative condition (the upper of two); the phase 4a gate's policy; the cnn of the phase 2 gate; validation selection as in the approved split. | PLAN: "{none, cVAE, WGAN-GP} at medium strength, `cnn`, 3 seeds". |
| A16.4 | Report: intra-patient test rare-class macro-F1 (mean and SD over seeds) beside the final evaluation's result for the same condition (at the strength the phase 4 gate chose); gain over none on each side; inflation = intra minus inter for none. | The final evaluation is not repeated; strengths are shown on both sides. |
| A16.5 | DS2 for the audit is read with `ALLOW_TEST_AUDIT=1`, only after the final evaluation, from `data/processed/v1` or rebuilt from the raw records and checked against the frozen hash; every read is logged as `audit`. Audit runs and banks are refused by the final evaluation, the grid and the ingest before the final evaluation. | DECISIONS A4, A6. |
| A16.6 | RQ3: Spearman's rho between each metric (the composite's families; diversity, memorization and C2ST averaged over S, V, F) and the pilot score across all checkpoints of the official pilot and final generator runs; 95% percentile interval from 2,000 resamples of runs; TSTR apart. The gain's zero is the phase 2 gate cnn's seed-0 run without synthetic beats (a constant shift, so rho is unchanged). | PREREGISTRATION §9, PAPER_TRACK RQ3. |

## 2026-10-27 (Day 21) — A17: two fixes found by the dress rehearsal — decided by Akash Saha, PENDING TEAM CONFIRMATION

- **A17.1** Selection rule: an undefined (NaN) tie-break value counts as "not given", like a missing one,
  so the tie-break step is skipped and simplicity decides; the gate report prints the skipped step.
  Before, the rule crashed. Seen with Grad-CAM, whose maps can be constant (stability undefined).
- **A17.2** phase4_augmentation_policy run IDs carry the variant letter (`phase4_augmentation_policy-B.cnn.cvae_0.25-s0-...`);
  before, variants B and D produced the same run ID for the same strength and seed.

## 2026-10-27 (Day 21) — A18: the paper's text and references — decided by Akash Saha, PENDING TEAM CONFIRMATION

- **A18.1** Results sentences, the abstract's results sentence and the hypothesis verdicts are generated
  from the result files (`src/ecgaug/paper_text.py`). A difference is called an improvement only when its
  Holm-adjusted p is below 0.05; otherwise "did not detectably change", with the interval. Verdicts:
  H1 supported if any confirmatory comparison is a significant gain; H1c supported if, for one
  architecture, the best generative condition significantly beats all three baselines (partly if some);
  H1b, H2, H3 and H4 are exploratory and described as consistent, mixed or not consistent. No number or
  outcome is written before the result exists.
- **A18.2** `paper/refs.bib`: 29 entries verified on 2026-10-27 by web search of each entry's arXiv page,
  DOI or publisher page, or an institutional repository record (named above each entry), because the
  container cannot reach doi.org or arxiv.org directly (this deviates from A14.15's "from its DOI or
  arXiv page" for 9 entries). Fields no page confirmed (some DOIs, issue and article numbers) are left out.
