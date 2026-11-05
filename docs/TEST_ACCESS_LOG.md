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
| 2026-11-04 15:26 +0000 | audit | e1708f2 | `src/ecgaug/experiments/phase4_leakage_audit.py --config configs/phase4_leakage_audit/audit.yaml --part all --device auto` | phase4_leakage_audit: --part all (rebuilt from the raw DS2 records) |  |  |
| 2026-11-05 00:43 +0530 | posthoc | 9df5c94 | `src/ecgaug/experiments/phase6_rr_reference.py --part test` | phase6_rr_reference: DS2 predictions of the references (data/processed/v1) | phase6_rr_reference-rrlda-s0-1baec70-202611050043, phase6_rr_reference-rrmorphlda-s0-1baec70-202611050043 |  |
| 2026-11-05 01:05 +0530 | audit | 044a3a5 | `src/ecgaug/experiments/phase6_audit_check.py --device cpu` | phase4_leakage_audit: phase6_audit_check: checks of the audit (data/processed/v1) |  |  |
| 2026-11-05 03:04 +0530 | posthoc | 65010c7 | `src/ecgaug/experiments/phase6_extra_seeds.py --config configs/phase6_extra_seeds/extra_seeds.yaml --part test` | phase6_extra_seeds: DS2 predictions of seeds [3, 4] (data/processed/v1) | 24 new checkpoints (results/phase6_extra_seeds/summary.json) |  |
| 2026-11-05 10:45 +0530 | construction | 220f4c6 | `src/ecgaug/experiments/phase6_rr_baseline.py --config configs/phase6_rr_baseline/rr_baseline.yaml --part test-ratios` | phase6_rr_baseline: v2 RR ratios of DS2 from annotation times (no labels, no model) |  |  |
| 2026-11-05 13:54 +0530 | posthoc | 4ba1c4f | `src/ecgaug/experiments/phase6_rr_baseline.py --config configs/phase6_rr_baseline/rr_baseline.yaml --part test` | phase6_rr_baseline: DS2 predictions with the v2 RR ratio (data/processed/v1) | 36 checkpoints (results/phase6_rr_baseline/selection.json) |  |
