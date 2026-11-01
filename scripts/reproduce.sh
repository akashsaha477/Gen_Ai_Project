#!/usr/bin/env bash
# The whole pipeline in order (PLAN Phase 6). Each line is one documented step; the heavy ones run on
# Kaggle with the notebook of the same name (kaggle/). Gates need the team's approval: this script stops
# at each gate and prints what to read. Never run the final evaluation without the phase 4 gate approval.
set -euo pipefail
run() { echo "+ $*"; uv run "$@"; }
gate() { run python -m ecgaug.workflow "$1"; echo "Read the report above; approve with: python -m ecgaug.workflow $1 --approve \"Name\" \"Name\""; exit 0; }

run python -m ecgaug.data.raw                                   # download and verify MIT-BIH
run python -m ecgaug.workflow status                           # where the project stands; NEXT says what to do
# Phase 1: run every variant (kaggle/phase1_input_representation.ipynb), then:
#   python -m ecgaug.workflow phase1-gate; ALLOW_TEST_BUILD=1 python -m ecgaug.workflow phase1-gate --approve ...
# Phase 2: kaggle/phase2_baseline_classifiers.ipynb, kaggle/phase2_quality_metrics.ipynb, kaggle/phase2_ptbxl_pretraining.ipynb;
#   python -m ecgaug.workflow phase2-gate [--approve ...]
# Phase 3: kaggle/phase3_cvae.ipynb and kaggle/phase3_wgan_gp.ipynb (MODE "pilot"); phase3a-gate;
#   the winners with MODE "final" (banks in the zip: python kaggle/ingest_runs.py <zip>); phase3-gate
# Phase 4: kaggle/phase4_augmentation_policy.ipynb; phase4a-gate (writes configs/phase4_full_grid/full_grid.yaml);
#   kaggle/phase4_full_grid.ipynb until every cell is done (ingest the zips: they hold the checkpoints); phase4-gate;
#   python -m ecgaug.final_eval --plan; ALLOW_TEST=1 python -m ecgaug.final_eval --confirm "run final eval"   (once)
#   python -m ecgaug.eval.rq3   (RQ3, from the phase 3 checkpoints)
#   optional, after the final evaluation: kaggle/phase4_leakage_audit.ipynb (ALLOW_TEST_AUDIT=1), then
#   python -m ecgaug.experiments.phase4_leakage_audit --config configs/phase4_leakage_audit/audit.yaml --part report
# Phase 5: python -m ecgaug.experiments.phase5_explainability --config configs/phase5_explainability/<variant>.yaml (A-D)
#   python -m ecgaug.experiments.phase5_clinical_notes --config configs/phase5_clinical_notes/<variant>.yaml (A-D)
#   python -m ecgaug.report.error_analysis [--test]; python -m ecgaug.workflow phase5-gate [--approve ...]; git tag v1.0
# Phase 6:
run python -m ecgaug.figures all
run python -m ecgaug.paper                                     # paper/generated/: tables and numbers for paper/main.tex
