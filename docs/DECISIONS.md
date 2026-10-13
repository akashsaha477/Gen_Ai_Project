# DECISIONS

Entries are append-only. Gate entries follow the template in `docs/PLAN.md`.

---

## Plan amendments after the kickoff review — 2026-10-08 (Day 2) — status: ADOPTED by Akash Saha, PENDING TEAM CONFIRMATION at Gate 1

**Context.** The kickoff review of `CLAUDE.md`, `docs/PLAN.md` and `docs/PAPER_TRACK.md` found ten gaps.
Akash delegated the choice of fixes ("do what you think is best"). All ten were adopted and written into
the three documents in one commit, separate from the kit-install commit, so the team can read the diff
and revert it. Nothing below is in force as a pre-registered rule until `docs/PREREGISTRATION.md` v1 is
approved and committed.

**Two amendments change the plan itself and need all four members' agreement: A2 and A7.**

| ID | Amendment | Why | Files |
|---|---|---|---|
| A1 | The selection metric uses F only if the validation split holds at least the minimum F support fixed in PREREGISTRATION (proposed: 100 beats from 2 records); otherwise S F1 with V F1 as tie-break. The tie margin becomes max(0.02, pooled across-seed SD). | Record 208 holds 373 of DS1's 415 F beats, so validation F is either the bulk of the class or a few dozen beats. The old fallback only covered F being absent. | CLAUDE.md §4, PLAN.md conventions |
| A2 | **Plan change.** Two test families instead of one family of 10: confirmatory (H1: cVAE vs none, WGAN-GP vs none, per architecture = 4 tests) and secondary (the other comparisons). Holm within each family on two-sided bootstrap p-values. Replicates in which S or F has no true beats are excluded and counted. The dominant-record-absent fraction and per-record results for S and F are reported. | Record 213 holds 362 of DS2's 388 F beats and record 232 holds 1,382 of its 1,837 S beats, so intervals will be wide. The old plan did not define the metric for a class-absent replicate or how Holm applies to intervals. | PAPER_TRACK.md statistics |
| A3 | Records 201 (DS1) and 202 (DS2) are the same subject. The de Chazal split is kept, the overlap is disclosed, and test results are also reported without record 202. | MIT-BIH has 48 records from 47 subjects; "record (patient)" in rule 1 was not exact. | CLAUDE.md §3, PAPER_TRACK.md |
| A4 | The leakage-audit arm runs only after `/final-eval`, in `runs/audit/` and `data/synthetic/audit/`, with each run logged in TEST_ACCESS_LOG. The main grid refuses audit artifacts. | It trains and evaluates on DS2 beats, so before `/final-eval` it would be a second test access. | CLAUDE.md §3, PLAN.md Phase 4 |
| A5 | Every "best" (generative condition, checkpoint, strength, cVAE, WGAN-GP) means highest validation selection metric, recorded in DECISIONS.md before `/final-eval`. | Read off the test table, "best" would be selection on test. | PAPER_TRACK.md, PLAN.md Module 8 |
| A6 | Test access has three kinds: construction (labels counted, arrays built, no model involved), evaluation (only `/final-eval`) and audit (A4). Each is logged. Gate 1 requires zero evaluation accesses instead of an empty log. All raw-file reads go through one module that enforces the lock. | Phase 1 has to read DS2 to count classes and build `test.npz`, which contradicted "every access is logged" plus "log empty at Gate 1". | CLAUDE.md §3, PLAN.md Gate 1, `/final-eval` |
| A7 | **Plan change.** The grid gains random oversampling of real minority beats at the same three strengths, and class-weighted loss gets three pre-registered weightings instead of one. Runs at 3 seeds: 96 instead of 66. Decisions are argmax with no threshold tuning in any condition. | Without a matched-count control, a gain from generated beats cannot be separated from rebalancing. The baselines had one configuration each while the generative arms had many. | PLAN.md Phase 3–4, PAPER_TRACK.md |
| A8 | Each synthetic bank is exported from the final checkpoint of its step budget. Earlier checkpoints are scored for RQ3 only. | The pilot scores every quarter-checkpoint on validation; leaving the choice open added an unstated selection. | PLAN.md Phase 3 |
| A9 | In RQ3, TSTR is analysed separately from the fidelity and diversity metrics, and intervals come from resampling generator training runs, not checkpoints. | TSTR and the downstream pilot score are both validation F1 of a CNN trained on the same synthetic beats, and checkpoints of one run are dependent. | PAPER_TRACK.md RQ3 |
| A10 | Each PREREGISTRATION version is committed, tagged `prereg-vK` and pushed the same day. H1 is confirmatory; H1b, H1c, H2, H3 and H4 are labelled secondary or exploratory. | A pre-registration is only as credible as its timestamp. | PAPER_TRACK.md, PLAN.md |
| A11 | Specification details: RR intervals are computed over all QRS-type annotations before any class is dropped; the usable-beat rule is identical for every representation; the contract gains `rr_ratio`; run IDs may carry a tuning tag; the layout gains `train/`, `experiments/` and `runs.py`. | Gaps found while specifying Phase 1. | CLAUDE.md §5, §6, §8 |

**Actions only a human can take (A10):** protect `main` against force-pushes in the GitHub settings;
optionally deposit PREREGISTRATION v1 on OSF for an independent timestamp.

**Cost of A7:** 30 more classifier runs in Phase 4 (at the plan's 15-minute cap, at most 7.5 GPU-hours).

---

## Gate 1 — status: NOT STARTED

Open human decisions before Module 1 variants may run (evidence in `results/data/`):

1. Validation split (options in `results/data/val_options.md`).
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

## 2026-10-13 (Day 7) — A12: PTB-XL pretraining arm — decided by Akash Saha

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
- **Order:** built in Phase 2, after Gate 1, because the beat window comes from the Module 1 winner.
  Details are in `docs/PLAN.md`, Phase 2.

## 2026-10-13 (Day 7) — A13: pretraining arm built and may run before Gate 1 — decided by Akash Saha

- The PTB-XL preprocessing for Kaggle and the supervised pretraining plus fine-tuning runner were built
  now, at Akash's instruction, so the team can run both in one Kaggle session.
- This is ahead of the phase order (rule 9: no next phase before the gate). Consequences, recorded so
  they are not forgotten:
  - The input representation is not frozen. Every result file of the arm carries `provisional: true`
    until `data/processed/v1` exists. The arm is rerun for the Gate 1 winner.
  - The architecture is the probe CNN, the only classifier that exists. The arm is rerun with the
    `cnn` chosen in Module 3.
  - Results of the arm are not an input to Gate 1. Gate 1 is decided from the Module 1 variants alone.
- Protocol fixed before any run (details in `docs/PLAN.md`, Phase 2): one pretraining run per seed;
  fine-tuning seed s starts from pretraining seed s; full fine-tuning with a new head; both arms use
  the Module 1 budget; pretraining stops on PTB-XL fold 9 only; from scratch is the simpler option in
  the tie rule.
- Edge treatment for 10-second recordings: mirror padding of 2 s, chosen from a DS1 measurement
  (`results/data/ptbxl_method_checks_ds1.json`). No mains notch filter is applied to PTB-XL: the
  64-recording sample shows no 50 Hz interference; the full-data figure is in the preprocessing report.
