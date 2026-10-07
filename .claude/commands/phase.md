---
description: Build shared infrastructure for a phase and prepare variant stubs (usage: /phase <1-6>)
---
Phase: $ARGUMENTS

1. Read CLAUDE.md, the "Phase $ARGUMENTS" section of @docs/PLAN.md, docs/PROGRESS.md and docs/DECISIONS.md.
2. Check that the previous gate is APPROVED in docs/DECISIONS.md (Phase 1 has no prerequisite). If it isn't, stop and list what's missing.
3. Show a short plan: tasks, files to create or change, tests, compute estimate, and anything that touches a frozen artifact or a pre-registered rule (those need my explicit OK). Wait for "go".
4. Build the shared infrastructure for this phase. Write tests first for anything involving splits, the test lock, hashes, metrics or the bootstrap. Run the full test suite.
5. For each module in this phase, create variant stubs and configs (configs/m<M>/<A-D>.yaml) behind the module's common interface, with identical budgets and seeds. Write one implementer brief per variant at docs/briefs/m<M>-<X>.md: definition, interface, budget, seeds, what to log, the governing selection rule.
6. Update docs/PROGRESS.md. Report: what's built, test results, how to run each variant (local and Kaggle), open risks. Don't run variant experiments here unless the phase plan lists shared runs.
