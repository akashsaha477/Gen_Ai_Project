# Kickoff — running the project phase by phase with Claude Code

## What's in this kit

| File | Purpose |
|---|---|
| `CLAUDE.md` | Project brain. Claude Code loads it automatically every session: mission, rules, data protocol, contract, repo layout, conventions. |
| `docs/PLAN.md` | Phase-by-phase spec: shared infrastructure, experiments, variants A–D, selection rules, gate checklists. |
| `docs/PAPER_TRACK.md` | Research questions, hypotheses, statistics plan, figures/tables, threats to validity. |
| `.claude/commands/*.md` | Custom commands: `/phase`, `/variant`, `/gate`, `/status`, `/final-eval`, `/paper`. |
| `KICKOFF_PROMPT.md` | This file. |

## One-time setup (one person)

1. Create the repo (e.g. `ecg-genaug`) and copy everything from this kit into its root, **including the
   hidden `.claude/` folder**. Commit and push.
2. Everyone clones the repo and starts Claude Code in the repo root (`claude`).
3. Type `/` and confirm `/phase`, `/variant`, `/gate`, `/status`, `/final-eval` and `/paper` appear.
   (Newer Claude Code versions also offer skills in `.claude/skills/`; files in `.claude/commands/`
   keep working.)

## First message (paste once, on `main`, by one person)

```
You're the engineering and research assistant for our 4-person project "Rare-Arrhythmia Detection
with Generative ECG Augmentation". We want a strong course deliverable in 30 days
(Day 1 = 2026-10-07) and results rigorous enough to publish.

Before doing anything, read CLAUDE.md, docs/PLAN.md and docs/PAPER_TRACK.md completely.

Then, in this order:
1. Critical review (≤ 20 lines): restate the goal in 3 lines; list the non-negotiable rules you'll
   enforce; list gaps, contradictions or risky assumptions in the plan, especially anything that could
   leak test information, bias selection or weaken the paper. Propose fixes; don't apply them silently.
2. Wait for my reply to your review.
3. Then run Phase 1 exactly as /phase 1 describes: scaffold, environment + lockfile, tests, MIT-BIH
   download from PhysioNet, beat extraction with the AAMI mapping, per-record class counts with the
   sanity check, the guarded test loader, the evaluation code eval utilities with tests, the probe CNN, the Kaggle
   runner + ingest script, and a draft docs/PREREGISTRATION.md.
4. Stop at the Phase 1 decision points and show evidence for each:
   (a) validation-split options with per-record class counts,
   (b) Q-class handling and the primary-metric fallback if F is missing from validation,
   (c) phase1_input_representation variants A–D stubbed, briefed and runnable.
   Do not run phase1_input_representation variants until PREREGISTRATION.md v1 is committed. Do not start Phase 2.

Compute: Kaggle GPU sessions with a weekly quota, plus laptops. Keep runs short, measure wall-clock,
and ask before anything over 2 GPU-hours.

Today is Day <N>.
```

## The phase loop (repeat for N = 1 … 6)

1. **Shared build** — one person on `main`: `/phase N`. Review the plan it shows, reply `go`.
2. **Draw assignments** — record who takes which variant in `docs/ASSIGNMENTS.md`.
3. **Variants in parallel** — each member on their own branch: `/variant <experiment> <letter>`.
   Push the result files and open a PR.
4. **Gate** — when all variants are in, merge result files to `main` and run `/gate N`
   (`/gate 3a` on Day 12 for generator selection). Read `results/phase<N>_gate/report.md`.
5. **Decide together** — short meeting, then reply `approve gate N` (or request changes).
   Claude Code freezes artifacts, tags `gate-N` and updates `docs/PROGRESS.md`.
6. `/status`, then `/phase N+1`.

After the phase 4 gate approval: `/final-eval` exactly once. Phase 6: `/paper all`, then section by section.

| Phase | Commands in order |
|---|---|
| 1 | `/phase 1` → `/variant 1 A…D` → `/gate 1` |
| 2 | `/phase 2` → `/variant 3 A…D`, `/variant 4 A…D` → `/gate 2` |
| 3 | `/phase 3` → `/variant 5 A…D`, `/variant 6 A…D` → `/gate 3a` (Day 12) → full training → `/gate 3` |
| 4 | `/phase 4` → `/variant 7 A…D` → grid → `/gate 4` → `/final-eval` |
| 5 | `/phase 5` → `/variant 8 A…D`, `/variant 9 A…D` → `/gate 5` |
| 6 | `/phase 6` → `/paper all` → final checks |

## Daily opener (any member)

```
/status
Today is Day <N>. I'm working on <experiment / variant>. Continue from docs/PROGRESS.md and tell me the
next concrete step before writing code.
```

## When something breaks

```
Stop. Explain what failed with evidence (logs, run IDs). Give 2–3 options with their cost in time and
compute, and recommend one. Don't change any frozen artifact or pre-registered rule.
```

## If Claude Code drifts out of scope

```
Re-read CLAUDE.md rules 1–5. Is what you're doing inside the current phase in docs/PLAN.md? If not,
stop and tell me.
```

## Before writing anything for the paper

```
/paper <section>
Every number from paper/generated/, every citation verified from its DOI or arXiv page, every claim
mapped in paper/claims_vs_evidence.md.
```
