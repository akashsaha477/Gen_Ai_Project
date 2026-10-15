---
description: Run a gate — checks, pre-registered selection, decision draft (usage: /gate <N> or /gate 3a)
---
Gate: $ARGUMENTS

1. Read the checklist for this gate in @docs/PLAN.md and the governing rules in docs/PREREGISTRATION.md.
2. Run every automated check: tests, leakage assertions, train-only assertions for generators, artifact hashes, runs_index consistency. Report pass/fail with evidence.
3. Collect all variant result files for the experiments decided at this gate. Flag missing variants and any budget or seed deviations.
4. Apply the pre-registered selection rule literally. Show the full comparison table (per-seed values and mean ± sd), mark ties within the threshold, apply the stated tie-break. If the rule is ambiguous for the actual results, stop and ask. Don't improvise.
5. Write results/phase$ARGUMENTS_gate/report.md (checks, table, selection, risks, compute used vs plan, days of slack left) and draft the docs/DECISIONS.md entry marked PENDING APPROVAL.
6. Stop. Do not freeze, merge or start the next phase until I reply exactly "approve gate $ARGUMENTS". After approval: write hashes to results/manifest.json, mark the decision APPROVED with the approvers' names I give you, tag git gate-$ARGUMENTS, update docs/PROGRESS.md, and list what the next phase starts with.
