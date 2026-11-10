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
| A14.14 | Notes: low-confidence rule "top probability below 0.6" (proposed; to pre-register before phase 5 runs); LLM provider through the standard library (a paid API at first; Gemini's free tier since A21), model from `ECGAUG_LLM_MODEL`, key from the environment only; note variant chosen by the hard gate, then coverage, ratings, latency, tokens. | PLAN Phase 5. | `report/notes.py` |
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

## the phase 2 gate — 2026-11-01 — status: APPROVED by Akash Saha
phase2_baseline_classifiers: cnn = variant A, rnn = variant C; phase2_quality_metrics: composite = dtw, spectral — evidence: results/phase2_gate/report.md — rule: PLAN Phase 2, PREREGISTRATION §4
Frozen artifacts (hashes): checkpoints and feature network, see results/manifest.json

## 2026-11-02 (Day 27) — A19: the policy pilot, changed from the phase 3 pilot results — proposed by Akash Saha, confirmed by Akash Saha on 2026-11-03 (A20)

Evidence: `results/phase3a_gate/report.md`. No phase4_augmentation_policy run existed when these were made, and
no policy result was seen. They belong in PREREGISTRATION v3, which is committed before that pilot.

- **A19.1 Three seeds and the general tie margin.** The pilot runs seeds 0, 1 and 2 (classifier seed s, bank seed
  s) instead of seed 0, and ties use max(0.02, pooled SD) instead of the 0.02 floor. In the phase 3 pilots one setting
  moved by up to 0.154 between two generator seeds, and the variants' SDs were 0.04 to 0.11, so at one seed and a 0.02
  margin the policy would be chosen by noise. The cost is 27 short classifier runs instead of 9.
- **A19.2 The reports show what an average hides; the rules do not change.** The phase 4a report adds N recall
  and V F1 drops per bank and flags a bank past the N recall limit; the limit itself stays on the mean over banks
  (A14.11). The phase 4 report adds V F1 and flags a chosen strength with N recall more than 0.01 below `none`.
  In the pilots cVAE D raised N recall by 0.011 and WGAN-GP A lowered it by 0.015 against the same-seed baseline, so
  an average would look fine; and cVAE D lowered V F1 from 0.679 to 0.542, while selection reads only S F1.
- **A19.3 All four policies run in one Kaggle session.** They share one reference run and the gate refuses reference
  runs that differ, so parallel sessions would make the gate fail. The notebook now defaults to all four letters.

Open for the team: whether the N recall limit should apply per bank (stricter; it may leave no policy eligible).

## the phase 3a gate — 2026-11-02 — status: APPROVED by Akash Saha
phase3_cvae: winner D (setting default) — phase3_wgan_gp: winner A (setting default) — evidence: results/phase3a_gate/report.md — rule: PLAN Phase 3 selection, PREREGISTRATION §9

## the phase 3 gate — 2026-11-03 — status: APPROVED by Akash Saha
Frozen banks: phase3_cvae-D, phase3_wgan_gp-A (hashes in results/manifest.json) — evidence: results/phase3_gate/report.md

## 2026-11-03 (Day 28) — A20: decisions at the phase 3 gate — Akash Saha

Evidence: `results/phase3_gate/report.md`, `figures/phase3_gate/banks.png`. No phase 4 run existed.

- **A20.1 cVAE D bank seed 1 is kept.** Its lowest diversity ratio is 0.694 (class S), under the 0.7 floor; the run's
  own diagnostic sample gave 0.701. The floor is pre-registered for pilot seeds only, so the gate checks pass as
  written. Dropping or rerunning the seed would apply a rule made after seeing the result and leave the families with
  unequal seeds. The value is disclosed in the paper's limitations (PREREGISTRATION §11).
- **A20.2 Three seeds.** The full grid stays at seeds 0, 1, 2. The banks exist for these seeds only, and the deadline
  does not leave quota for 5.
- **A20.3 A19 is confirmed as proposed,** with the N recall limit on the mean over the two banks (A14.11); per-bank drops
  are reported. Committed as PREREGISTRATION v3 before the policy pilot.
- **A20.4 Runs on Akash's Kaggle account.** Notebooks are pushed as private kernels; the code comes from a git bundle of
  `main` uploaded as the private dataset `akashsaha477/ecgaug-repo` (the GitHub repository is private), and the banks
  from a private dataset. The run commit is recorded in every run ID as before.


## the phase 4a gate — 2026-11-04 — status: APPROVED by Akash Saha
phase4_augmentation_policy: winner D (fraction_of_n_screened, strengths [0.1, 0.25, 0.5]) — evidence: results/phase4a_gate/report.md — rule: PLAN Phase 4
Full grid: configs/phase4_full_grid/full_grid.yaml

## 2026-11-04 (Day 29) — A21: Gemini as the default LLM for the note variants — Akash Saha

The note variants B-D (phase5_clinical_notes) call Gemini by default (`ECGAUG_LLM_PROVIDER=gemini`, key `GEMINI_API_KEY`,
free tier), not the paid API first planned (A14.14), which is billed separately. (That provider was removed from the code on
2026-11-07; Gemini is the only one built.) Nothing pre-registered changes: PREREGISTRATION does not name a provider or model;
prompts, the fact checker, `max_tokens` 400, 50 cases and 2 retries are unchanged. The model used is recorded in each
variant's result (`model`). Model `gemini-3.5-flash` with thinking level `minimal` (`ECGAUG_LLM_THINKING`), chosen
from a one-request probe on a made-up beat (no project data): `gemini-2.5-flash` is closed to new keys, the Pro models
have no free quota, `gemini-3.8-flash` answered 503 (high demand) repeatedly, and `gemini-3.5-flash` at its default
thinking level spent 396 of the 400 output tokens thinking and cut the note off; at `minimal` it used 89 tokens and
passed the fact checker. The 400-token limit is not pre-registered and is unchanged. Free-tier inputs may be used by Google; the inputs are only the beat JSON of docs/RULES.md §5
(anonymised record id, class probabilities, segment masses) from the public MIT-BIH database.

## the phase 4 gate — 2026-11-04 — status: APPROVED by Akash Saha
Strengths chosen on validation and best generative condition per architecture: cnn: cvae; rnn: cvae — evidence: results/phase4_gate/report.md — rule: PLAN Phase 4, PAPER_TRACK A5
Frozen artifacts (hashes): the grid checkpoints in results/manifest.json. Next: /final-eval.

## 2026-11-04 (Day 29) — A22: fixes before phase 5 — Akash Saha (asked for a review and fixes before phase 5)

Nothing here retrains, re-selects or re-tunes a phase 1-4 result (docs/RULES.md and the final evaluation's procedure).

- A22.1 Note variant C: `render_c` dropped a disclaimer the model wrote inside a field and then added its own, so
  low-confidence notes carried the disclaimer twice (seen in a one-beat Gemini probe). The disclaimer is now added once.
  Fixed before any official notes run.
- A22.2 Notes cases and the saliency method: the cases are built once, but the method came only from the approved phase 5
  gate, which needs the notes results, so the cases would always have used the default (integrated gradients). The cases
  now use the method that the pre-registered phase 5 rule picks from the finished phase5_explainability results
  (`workflow.saliency_choice`, the same function the gate uses); explainability runs first.
- A22.3 Final-evaluation per-record figure redrawn as a heatmap from `results/final_eval/summary.json` (12 blue bar shades
  could not be told apart); `python -m ecgaug.figures final_eval` redraws it without reading the test set.
- A22.4 Error analysis run on validation and on the final evaluation's saved test predictions (no new test access).
- A22.5 Two lint findings fixed (unused import, unused variable); no behaviour change.

## 2026-11-04 (Day 29) — A23: note model gemini-3.5-flash-lite — Akash Saha (free LLM only, A21)

The first B-D run failed without writing results: B on a 60 s read timeout (not retried), C and D on HTTP 429. Google's
reply: the free tier allows 20 `generateContent` requests per day for `gemini-3.5-flash` (quotaId
GenerateRequestsPerDayPerProjectPerModel-FreeTier, value 20); B-D need about 250. Default changed to
`gemini-3.5-flash-lite` (thinking `minimal`), which answered; on the two made-up probe beats its 6 notes (B, C, D) had zero
unsupported claims and coverage 1.00. All three LLM variants use it, so they stay comparable. The client now retries
timeouts and dropped connections, and stops at once with `LLMQuotaError` when a daily quota is used up. Prompts, fact
checker, `max_tokens`, cases and retries are unchanged.

## the phase 5 gate — 2026-11-04 — status: APPROVED by Akash Saha
Saliency: {'cnn': 'gradcam', 'rnn': 'occlusion'}; notes: variant C — evidence: results/phase5_gate/report.md — rule: PLAN Phase 5
Next: code freeze (git tag v1.0) and phase 6.

## 2026-11-05 (Day 30) — A24: post-hoc analyses for review objections — Akash Saha ("can you fix all these")

After the paper draft, a review of its weaknesses listed: a weak baseline (test rare-class macro-F1 0.088, below
published inter-patient classifiers that use RR features), rare test classes in effectively two patients, 3 seeds, one
database, and a leakage audit with one split and a WGAN-GP strength different from the final evaluation. Akash Saha asked
for all of them to be addressed. Four post-hoc analyses are pre-registered (PREREGISTRATION §10 v4, tag `prereg-v4`)
before any is run: P1 RR reference classifiers, P2 external validation on INCART and SVDB, P3 seeds 3 and 4, P4 audit
check. They break the "nothing retrained after /final-eval" line only in P3 (new seeds of frozen settings) and add DS2
reads (new access kind `posthoc`, logged); every result is labelled post hoc and none replaces the primary results.
Not addressable here: blinded readability ratings of the notes (need people) and diffusion generators (out of scope;
stated as future work).

## 2026-11-05 (Day 30) — A25: follow-up study with a patient-baseline RR ratio (pre-registration v5) — Akash Saha ("Yes, run v5")

**Context.** After v4, the publication review found the absolute scores low (test rare-class macro-F1 0.088 for the
cnn without augmentation; an RR-only discriminant reached S Se 0.068, far below published inter-patient results). A
DS1-only diagnostic showed why S beats are hard for our models: v1's `rr_ratio` divides by the ten preceding RR
intervals, so S beats inside runs (record 209 holds 383 of 643 training S beats) do not look premature; only 51% of
training S beats have a ratio below 0.9, against 87% with a ratio to the median RR of a ±150 s window.

**Decision.** Akash Saha asked for better results and approved the follow-up ("Yes, run v5"). Pre-registration v5
defines a v2 dataset that changes only `rr_ratio`, reruns the unchanged full grid with the unchanged classifiers, banks
(RR values remapped) and selection rule, and evaluates once on DS2 as a post-hoc study. The final evaluation of
2026-11-04 stays the primary result; the v2 study is reported beside it. Compute above 2 GPU-hours approved with the same
answer.


## 2026-11-05 (Day 30) — A26: follow-up study with patient-relative beats (pre-registration v6) — Akash Saha ("do 2 and 3")

The v5 results were weak in absolute terms (test S Se 0.136 for the cnn) and the feature fix did not move them. Asked to
(2) find out why on validation and (3) build a stronger baseline and rerun the comparison. An exploratory look on DS1
(disclosed in v6) showed the networks fit the training patients' beat shapes (S PR-AUC 0.92 on training records, 0.20 on
validation). v6 gives every beat relative to the patient's local median beat, runs the no-augmentation cells first, and
continues to the full grid, one DS2 read and the external databases only if a pre-registered validation rule adopts it.
Everything runs on Kaggle (the laptop crashed under local load): CPU sessions where no training is needed, one T4 session
for the grid. Pushed to GitHub only after the full test suite passes in a Kaggle session at the same commit.

## 2026-11-05 (Day 30) — A27: cnn-only follow-up with patient-relative beats, and a mechanism analysis (pre-registration v7) — Akash Saha ("think deeply once what can be made better and then do it")

v6 was not adopted: its rule averaged a clear validation gain for the cnn (+0.122 S F1) with a clear loss for the rnn
(-0.125). Two things would make the study better. (1) The reviewer's main objection is that the baselines are weak;
the cnn with patient-relative beats is the only baseline that is clearly stronger on validation, so P7 repeats the
comparison for it, disclosed as chosen after v6's result (validation only). (2) The null result needs a mechanism: P8
checks, on DS1 only, whether synthetic beats add patient diversity (nearest-patient attribution) and whether augmentation
widens the gap between training and validation patients. Everything runs on Kaggle; only P7's training uses a GPU.

## 2026-11-05 (Day 30) — A28: patient-transfer augmentation (pre-registration v8) — Akash Saha ("do whatever think of a way how generative augmentation will help improve")

P8 explained why class-conditional generators did not help: they add beats of the same few patients. P9 tests the
direct remedy: keep the arrhythmia-specific deviation (from real beats or from the frozen generators) and place it on
other training patients' normal beats, so the classifier sees each rare class in many patients. Run on dataset v2 (raw
beats), where the networks fit patient shapes; on v3 the transfer would cancel out. Unscreened, because P8 showed the
quality screen narrows patient diversity. GPU for the 60 runs only; test and external parts on CPU.

## 2026-11-05 (Day 30) — A29: patient-adaptive normalization at test time (pre-registration v9) — Akash Saha ("think of a solution to make this better")

Across P1 to P9, every method aimed at class imbalance failed to beat no augmentation, while patient-level methods helped
(patient-relative input: test 0.088 to 0.371; patient transfer above plain generation in 4 of 4 pairs). P10 tests the
cheapest remaining patient-level lever, adaptive batch normalization on the frozen checkpoints: no training, CPU only,
adopted per model only if it clearly improves validation, one DS2 read. If it helps, the study's conclusion becomes that
inter-patient arrhythmia detection is limited by patient variation, which normalization addresses and rebalancing does not.

## 2026-11-06 (Day 31) — A30: bug fix, external bootstraps clustered by patient — Akash Saha ("do what you think would be most beneficial")

Bug: INCART's 75 records come from 32 patients (each header names the patient, "# patient N"), but every external
analysis (P2, P5, P7, P9, P10) resampled records as if they were independent patients, so its INCART intervals were too
narrow and its Holm p-values too small. The pre-registered intent is patient-level resampling (records stand for
patients only in MIT-BIH, where the one exception, 201/202, is disclosed). Fix: `phase6_external_validation.clusters`
maps each beat to its patient from the headers and every external analysis bootstraps over patients; SVDB's headers name
no patient, so its records stay the unit (an assumption, reported). The INCART parts of P2, P5, P7, P9 and P10 are re-run
with nothing else changed; the corrected numbers replace the old ones in the paper, and the old summaries stay in git
history. No MIT-BIH result is affected.

Outcome (re-runs at 153c9ab, summaries replaced at 46dcea9): point estimates identical, INCART intervals wider. Holm
rejections on INCART (generative vs none, all of them "lowered"): P2 4 of 4 -> 1 of 4, P5 3 of 4 -> 0 of 4, P7 0 -> 0,
P9 4 -> 4, P10 unchanged. Details in PROGRESS 2026-11-06.

## 2026-11-06 (Day 31) — A31: training on more patients (pre-registration v10) — Akash Saha ("do what you think would be most beneficial")

Asked whether a better dataset is needed. The evidence says more patients, not more beats. P11 adds the INCART and SVDB
patients to the training data (rare beats all kept, N beats capped per record), keeps DS2 as the test set, and trains
the best configuration (the P7 cnn with patient-relative beats) with fixed strengths. It costs the external test: both
external databases become training and validation data. One T4 session under 2 GPU-hours; the test on CPU.

## 2026-11-06 (Day 31) — A32: generators trained on what the cnn sees (pre-registration v11) — Akash Saha ("think of a solution"; ran P12 at ~2.5 GPU-hours)

P11's test: with 89 more patients the cnn without augmentation stopped predicting F beats (F F1 0.000), class weighting
reached 0.407 and random oversampling 0.383, the highest scores so far, while the cVAE and WGAN-GP conditions stayed at
0.245 and 0.217. Those conditions used banks of the phase 3 generators: trained on ordinary beats of DS1's 17 training
records, their output made patient-relative afterwards by subtracting the template of the beat whose RR values each
synthetic beat carries (`phase6_patient_template.remap_bank`). So the generators learned neither the patients nor the input
the cnn now uses (P8: their S beats sit nearest record 209). P12 trains the same generators, settings unchanged, on P11's
training beats in v3 form, and repeats the comparison. Comparisons were chosen after P11's result: against no augmentation
alone would now mostly measure whether F beats are added at all, so the confirmatory question is new vs old generators and
the matched control (copies of real beats) is secondary. The unscreened policy follows P9 (the screen narrowed patient
diversity in P8) and keeps the cost down. Compute above 2 GPU-hours was approved by Akash Saha.


Outcome (generators and grid at 99451a2, selection committed at fb146a9, one DS2 read at f765ac6): no generator diverged;
the retrained generators' S beats sit nearest 31.6 (cVAE) and 29.2 (WGAN-GP) effective training patients, against 32.0
for the real S beats and 10.9 / 14.3 for P11's banks (F: 3.4 / 3.3 against 4.9 real and 1.3 / 1.3). Chosen on validation:
cVAE 0.1, WGAN-GP 0.25 (validation S F1 0.768 and 0.740; P11's generators 0.778 and 0.770). Test rare-class macro-F1:
cVAE 0.368 (P11 0.245; F F1 0.175 to 0.475, S F1 0.316 to 0.260), WGAN-GP 0.243 (P11 0.217). Confirmatory P12 vs P11
+0.122 [-0.047, +0.177] and +0.026 [-0.044, +0.051], Holm p 1.000 for both; no secondary comparison rejected either
(cVAE vs P11 random oversampling -0.016, Holm p 0.931). The P11 none score was recomputed exactly before the comparison.

## 2026-11-06 (Day 31) — A33: RR intervals over the recording's mean RR (pre-registration v12) — Akash Saha ("search literatures how they did it better ... and then fix this")

A literature search on why published inter-patient methods detect S beats far better than ours (our best S F1 on DS2 is
about 0.33; de Chazal et al. 2004 report S sensitivity 75.9% and +P 38.5%, Huang et al. 2014 91.1% and 42.2%) found one
concrete gap. Published analyses of DS2 (github.com/amitb-gpu/ecg-arrhythmia; medRxiv 10.64898/2026.10.04.26364683)
report that record 232 holds 75% of DS2's S beats, that S beats outnumber N beats there (1,382 vs 397) and come in runs
without a compensatory pause, and that a CNN with RR intervals normalized by local averages finds 9% of them (S F1 0.280
overall, like ours). Methods that find them normalize by the whole recording: Huang et al. (2014) call a non-V beat S when
its RR interval is below 0.8 of the recording's mean RR; Lin and Yang (2014) and Mondejar-Guerra et al. (2019) divide RR
intervals by the patient's average. Both of our ratios are local (v1: mean of the 10 preceding intervals; v2: median
within +/-150 s), and in an S-dominant recording the local reference is set by the S beats themselves. The same pattern is
in P11's own training data, so the change can be justified without DS2: SVDB record 865 (S 1,818 of 2,919; median
post/pre 1.00), where the 0.8 rule finds 2.5% (v1), 1.4% (v2) and 93.6% (recording mean) of the S beats. The
recording-mean ratio is not better everywhere (INCART: rule F1 0.316 vs 0.637 for v2, heart rate drifts within a record),
so it is added beside v2's ratio, not instead of it: `rr_prev` and `rr_next` are divided by the recording's mean RR, which
keeps the cnn, its three side inputs and every code path unchanged. Strengths are carried from P11 and P12 (no new
selection), so the run costs 15 cnn runs. The design check, the comparisons and the descriptive Huang rule are fixed in
v12 before any run. Kaggle's artifact copy flattened nested bank folders (P12's `phase6_relative_generators/phase3_cvae-D`
would have collided with phase 3's `phase3_cvae-D`); it now keeps the path below `synthetic/` (`kaggle_run.bank_name`,
tested), which leaves the existing flat datasets where they were.

Outcome (grid at 3a8561d, runs and selection committed at a26d941, one DS2 read at b971794): the bank check passed; all
five earlier scores reproduced exactly. With `rr_prev` and `rr_next` over the recording's mean, no confirmatory difference
was detectable (none +0.070, class weighting -0.027, random oversampling -0.009, cVAE -0.011, WGAN-GP +0.021; Holm p
1.000), although S F1 rose in three conditions (none 0.294 -> 0.399, class weighting 0.293 -> 0.360, WGAN-GP 0.279 ->
0.345). Huang et al.'s rule on top (descriptive, threshold not tuned): S F1 0.531-0.578 and rare-class macro-F1 up to 0.523
(cVAE), the highest of the study. Reported as descriptive: DS2 has been read many times, and the idea came partly from
published analyses of DS2.

## 2026-11-06 (Day 31) — A34: many more training patients from Icentia11k (pre-registration v13) — Akash Saha ("make it better ... you have to find the solution")

P13 showed the published remedy works as a rule (S F1 0.53-0.58 on DS2) but not when the cnn has to learn it. A design check
on training-side annotations (`phase6_icentia --part check`), with SVDB 865 held out as an unseen S-dominant patient, shows
why: every reference that is local gives better validation S F1 (up to 0.738) but finds 2-8% of the unseen patient's S
beats, the recording mean finds 94-98% at validation S F1 0.627-0.650, and vetoing irregular rhythm (atrial fibrillation)
removes too many real S beats (validation S F1 0.248-0.288). Two refinements of the rule were tried this way and dropped
before any test read: an irregularity veto and a +/-150 s window. Whether to trust the local or the recording reference
cannot be learned from one S-dominant training recording; it can only be learned from many patients. The paper's own
conclusion asks for more training patients, so P14 adds them: 400 Icentia11k patients drawn at random (seed 0), processed
like INCART and SVDB in P11 and in P13's form. Icentia11k (Tan et al., PhysioNet) is open access under CC BY-NC-SA 4.0
(non-commercial; nothing derived is published). WGAN-GP is left out because its saved bank is too small at 0.25 of the
larger N count; the other four conditions keep P13's strengths. The N cap is applied before the templates are computed
(P11 applied it after; the kept beats' templates are the same either way, since templates use all QRS annotations).
Downloads try PhysioNet's AWS mirror first, in parallel, and every file's SHA-256 is recorded.

A34 fix (before any P14 result): the first grid session (kernel `phase6-icentia-grid`, f7ab36a; the full suite passed)
stopped with exit code 1 right after building P13's data: the Icentia11k beats took the frozen representation from
`data/processed/v1/contract.json`, which exists locally but not in the Kaggle session. They now take it from P13's data
source (`p13.spec`), the same representation P11 used for INCART and SVDB. Downloads retry on any error and a patient
whose files never arrive is reported instead of stopping the run; a segment that fails to build is left out and reported
in `data.json`, as INCART's and SVDB's exclusions are. Nothing about the sample, the beats or the comparisons changes.

## 2026-11-07 (Day 32) — A35: repository cleanup and a history rewrite — Akash Saha

Four local-only paths (a settings folder, two local instruction files and the paper draft) are no longer tracked and are
erased from every commit by a history rewrite, at the team's request; they stay on the members' machines. The project rules
and the data protocol they held now live in `docs/RULES.md` (same section and rule numbers, so every "rule 7" or "§4" cited in
the code, briefs and pre-registration still points to the same text). The note generator keeps one LLM provider, Gemini
(the variants ran on it, A21); the other provider and its test were removed. A temporary path in one `TEST_ACCESS_LOG.md`
row is shortened to `<local scratch>/rebuild_v1.py`; its date, kind, commit and purpose are unchanged. The rewrite changes
every commit SHA: run IDs keep the SHA7 they were created with, and `docs/COMMIT_MAP.csv` maps each old SHA to the
rewritten one. No result, rule or pre-registered choice changes. The paper is generated and compiled locally.

A34 outcome (grid at the commit now mapped from 6c8d205, selection committed before the DS2 read, one DS2 read): 388 of the
400 sampled patients were usable (12 had no usable labelled beat); training beats added N 183,189, S 13,412, V 6,722, no F;
40 patients have at least 50 S beats and none is S-dominant. Validation S F1 about 0.80 in every condition (now including
Icentia11k validation patients). Test rare-class macro-F1, P13 -> with the Icentia11k patients: none 0.216 -> 0.121, class
weighting 0.380 -> 0.247, random oversampling 0.374 -> 0.159, cVAE 0.356 -> 0.304; confirmatory differences -0.052 to -0.215,
Holm p 0.543-1.000. F F1 fell most (cVAE 0.474 -> 0.344, random oversampling 0.425 -> 0.066): the added patients bring no F
beats, come from another lead and device, and include no S-dominant recording. The cVAE ranked first among the four (vs none
+0.184, vs random oversampling +0.145, vs class weighting +0.057; Holm p 0.357-0.395), consistent with synthetic F beats
mattering most when real ones are scarce. With Huang's rule on top (descriptive) the cVAE reached 0.460; P13's 0.523 stays
the highest. Reading: more patients help only if they are of the right kind (the same lead, fusion beats, S-dominant rhythms).

## 2026-11-08 (Day 33) — A36: a diffusion model as the third generator family (pre-registration v14) — Akash Saha ("why are we not using diffusion models ... I want the best result")

The study compared a cVAE and a WGAN-GP; diffusion models are the current standard for synthetic ECG and three papers in
the team's reading list use them (SSSD-ECG; Nunez et al. 2024; Kwak and Jung 2024). P15 adds a class-conditional diffusion
model (`ecgaug.generators.diffusion`: DDPM training with a cosine schedule, a small 1D U-Net, classifier-free guidance, an
exponential moving average, DDIM sampling) under the same rules as P12: P13's training beats in the cnn's form, the phase 3
step budget, batch and seeds, banks at the phase 4 strengths chosen on validation, one DS2 read. Its tests include a
learning check: trained briefly on hand-made waveforms, guided samples of each class match that class's waveform. The
guidance scale (1.5) and learning rate (2e-4, the DDPM default) are fixed before any run, not tuned.

## 2026-11-09 (Day 34) — A37: Huang's SVEB stage with an atrial-fibrillation veto (pre-registration v15) — Akash Saha ("our results so far show the main obstacle is how much patients differ ... SO WE CAN IMPROVE THIS FIND A WAY")

The best test scores so far (P13) come from Huang et al.'s SVEB stage on top of the cnn: a beat the cnn calls N becomes S
when its RR interval is below 0.8 of its recording's mean. The rule finds S beats in patients whose S beats come in long
runs without a pause (record 232's pattern), which the cnn misses, but it cannot tell an early beat from an irregular
rhythm. On DS1's annotations, 1,251 of its 1,412 false S calls lie in atrial fibrillation or flutter, where only 16 of 944
S beats lie. P16 (`phase6_af_rule`) adds an AF detector that reads only beat times (the irregularity of the surrounding
intervals after premature beats and their pauses are removed), fitted on the MIT-BIH Atrial Fibrillation Database (other
subjects), and undoes the rule's S calls where it fires. Its threshold (0.95) was chosen on the annotations of DS1, INCART
and SVDB, and its form (only the rule's own conversions) on P13's validation patients, both by rules written into the code
before they were computed. No DS2 data was used. One DS2 read, after P15's test; the diffusion condition is compared too.

## 2026-11-09 (Day 34) — A38: the diffusion model trained five times longer (pre-registration v16) — Akash Saha ("Run whatever you want ... I want industry standard result"; "finish the work")

P15 gave the diffusion model the same step budget as the phase 3 generators (20,000 steps). On training-side evidence
that budget was too small for it: its loss was still falling at the end, and its banks' class medians correlate with
real training beats far less than the cVAE's (P17's shape check). A fair test of the third generator family needs it
trained to a usable state, so P17 (`phase6_diffusion_long`) repeats P15 with 100,000 steps and nothing else changed (the
config is checked against P15's), with one DS2 read that compares it with P15 and with P13's conditions. The decision was
taken before P15's and P16's test results were read. The other generators keep the phase 3 budget; the paper reports the
unequal budget.

## 2026-11-09 (Day 34) — A39: the cnn's probabilities averaged over ten runs per condition (pre-registration v17) — Akash Saha ("can you make the cvae results even better")

The cVAE with Huang et al.'s SVEB stage and P16's AF veto is the study's highest post hoc score (0.557). Each cnn run keeps
one noisy epoch: its validation S F1 moves by a median 0.20 between consecutive epochs. Averaging the class probabilities of
independently trained runs is the standard remedy (a deep ensemble), and on the validation patients the average of the
three existing runs raised S F1 and V F1 in all six of P16's conditions. P18 (`phase6_seed_ensemble`) trains seeds 3-9 of
those conditions with nothing else changed, averages each condition's ten runs, puts the stage and the veto on top, and
reads DS2 once. Every condition gets the same ten seeds, so the comparisons between conditions stay equal-budget. Nothing
is chosen on validation; the validation scores are descriptive. The decision was taken after P17's test result, so P18 is
reported as post hoc.

## 2026-11-10 (Day 35) — A40: P18's frozen pipeline on 1,000 unused Icentia11k patients (pre-registration v18) — Akash Saha ("so improve these"; chose the fresh-patient check)

Against published inter-patient results, P18's best (S F1 0.647, F F1 0.527, V F1 0.936) sits at or near the top of the
published range, but it is the best of 14 post hoc reads of the same 22 test patients, the cnn alone reaches S F1 0.247
there, and record 232 holds three quarters of DS2's S beats. Of the options offered (a check on unused patients, rhythm-
context augmentation, a bigger model), the check on unused patients was chosen: it tests the result without another DS2
read. P19 (`phase6_fresh_patients`) runs P18's sixty runs, P16's veto and Huang's stage unchanged on the first segment of
1,000 Icentia11k patients P14 did not draw, scored by S F1 (Icentia11k has no F labels) with a patient-level bootstrap.
Prediction runs on a Kaggle GPU because the CPU would take about 50 hours; this departs from the habit of using the GPU
only for training and is recorded here.

## 2026-11-10 (Day 35) — A41: Huang's stage and the AF veto chosen on 400 Icentia11k patients (pre-registration v19) — Akash Saha ("do it")

On P19's 977 unused patients the frozen pipeline reached S F1 0.294 with the stage and the veto, while the ten-run
ensembles alone reached 0.259-0.382: the stage doubled S sensitivity but cut S +P from about 0.4 to 0.16. Its 0.8
threshold is Huang et al.'s MIT-BIH value and P16's veto threshold was chosen on MIT-BIH, INCART and SVDB annotations, so
both were set on few patients. Proposed and accepted: choose both thresholds, and whether to use the stage at all, on P14's
400 Icentia11k patients, which none of P18's runs has seen, then score the choice on P19's patients (paired with P19's
setting and with the ensemble alone) and once on DS2. P20 is `phase6_rule_tuning`. The beats are built on a Kaggle CPU
session and only the prediction runs on a T4 (about 0.7 GPU-hours), to keep GPU use low; the CPU session's output (the
built beats) is attached to the GPU session as a kernel input.

## 2026-11-10 (Day 35) — A42: a physics prior for the generators and physiological fusion beats (pre-registration v20) — Akash Saha ("can we do something like the simgans like adding physics"; chose "All three")

SimGANs add McSharry et al.'s ECG simulator to a GAN's generator objective (the Euler loss). P21 (`phase6_physics_prior`)
adds the same loss to P12's cVAE and WGAN-GP, applied to whole beats (each generated patient-relative beat plus a real
training patient's template, since the simulator describes whole beats), with its weight chosen on validation among three
values, and makes a third condition from physiology rather than a learned model: P12's cVAE banks with every synthetic F
beat replaced by a fraction of a real training patient's V deviation, the fraction drawn from the range real training F
beats show. Each condition goes through P18's ten-run pipeline with the stage and the veto chosen on P14's 400 Icentia11k
patients by P20's rule, then is compared with P18's cVAE or WGAN-GP on P19's patients and, once, on DS2. The simulator is
first fitted to real training beats on a CPU session, and pre-registered stop rules end a condition if the simulator (or
the fusion model) explains less than half of the real beats' variance. Offered at about 4 GPU-hours and chosen; the
estimate became about 5 while writing the pre-registration (P12's measured fit times). The physics generators get two
fits more than P12's (the weight choice); the paper says so.
