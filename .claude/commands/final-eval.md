---
description: The single test-set evaluation (only after Gate 4 is approved)
---
1. Verify in docs/DECISIONS.md that Gate 4 is APPROVED and that docs/TEST_ACCESS_LOG.md has no prior entry of kind `evaluation` (entries of kind `construction` are expected). If either check fails, stop.
2. List every frozen checkpoint to evaluate (with hashes) and the pre-registered analyses from docs/PREREGISTRATION.md. Ask me to confirm by replying exactly "run final eval".
3. After confirmation, run evaluation in a process with ALLOW_TEST=1 set only for that process. Save runs/<run_id>/preds_test.npz for every checkpoint. Append the access to docs/TEST_ACCESS_LOG.md.
4. Run the pre-registered statistics exactly as written (record-level cluster bootstrap, Holm correction, sensitivity analyses). Generate tables and figures into tables/, figures/ and paper/generated/ (including numbers.tex macros).
5. Report results neutrally, including hypotheses that were not supported.
6. Never retrain, re-select or re-tune anything after this point. If a bug is found later, stop, document it in docs/DECISIONS.md and docs/TEST_ACCESS_LOG.md, and ask before any re-run. Any re-run is disclosed in the paper.
