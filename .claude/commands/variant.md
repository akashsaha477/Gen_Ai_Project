---
description: Implement and evaluate one variant (usage: /variant <experiment> <A|B|C|D>, e.g. /variant phase1_input_representation C)
---
Variant: $ARGUMENTS  (experiment name, then variant letter)

1. Read CLAUDE.md, docs/briefs/<experiment>_<variant config name>.md, the experiment's section in @docs/PLAN.md and the governing rule in docs/PREREGISTRATION.md.
2. Make sure you're on branch variant/<experiment>_<variant>, created from the latest main. Create it if needed.
3. Implement only this variant behind the existing interface. Don't modify shared code. If shared code needs a change, stop and describe the change so it can go through main.
4. Run exactly the configured seeds and budget, on validation only. Never load test data.
5. Sanity-check outputs before reporting (shapes, NaNs, class coverage, learning curves).
6. Write results/<experiment>/<variant>.json (per-seed metrics, mean ± sd, run IDs, wall-clock, any budget deviation) and results/<experiment>/<variant>_notes.md (what worked, what failed, surprises, figure paths).
7. Commit with a clear message and print a PR description: results with uncertainty, caveats, deviations.
