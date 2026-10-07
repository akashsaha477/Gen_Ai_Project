# TEST ACCESS LOG

Every read of DS2 (the locked test set) is appended here by `ecgaug.data.access`.
Kinds: `construction` (labels counted or arrays built, no model involved), `evaluation`
(only `/final-eval`), `audit` (leakage-audit arm, only after `/final-eval`).

| date | kind | git SHA | command | purpose | run IDs | approved by |
|---|---|---|---|---|---|---|
| 2026-10-08 04:19 +0530 | construction | 410986b | `/Users/akashsaha/Desktop/GenAi/Gen_Ai_Project/src/ecgaug/data/counts.py --include-test` | per-record class counts from DS2 annotations (labels only, no signals) |  |  |
