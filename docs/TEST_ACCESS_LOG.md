# TEST ACCESS LOG

Every read of DS2 (the locked test set) is appended here by `ecgaug.data.access`.
Kinds: `construction` (labels counted or arrays built, no model involved), `evaluation`
(only `/final-eval`), `audit` (leakage-audit experiment, only after `/final-eval`).

| date | kind | git SHA | command | purpose | run IDs | approved by |
|---|---|---|---|---|---|---|
| 2026-10-08 04:19 +0530 | construction | 410986b | `/Users/akashsaha/Desktop/GenAi/Gen_Ai_Project/src/ecgaug/data/counts.py --include-test` | per-record class counts from DS2 annotations (labels only, no signals) |  |  |
| 2026-10-26 17:21 +0000 | construction | db9de5d | `src/ecgaug/workflow.py phase1-gate --approve 'Akash Saha'` | built v1/test.npz for representation C_short_window_rr_features |  |  |
| 2026-11-04 09:31 +0530 | construction | ce45db3 | `<local scratch>/rebuild_v1.py` | built v1/test.npz for representation C_short_window_rr_features |  |  |
| 2026-11-04 09:31 +0530 | evaluation | ce45db3 | `src/ecgaug/final_eval.py --confirm 'run final eval'` | load_test('v1') | 36 grid checkpoints (results/final_eval/summary.json) | Akash Saha |
