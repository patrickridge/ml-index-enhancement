#!/bin/bash
# run_pipeline.sh — Full diagnostic + feature rebuild pipeline
# Usage: bash run_pipeline.sh
# Runs: 2g → 2h (parallel) → 1g (Cat 16+17 rebuild) → 2a (full re-eval) → 2g → 2h

PY=/opt/anaconda3/bin/python3
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "================================================================="
echo " PIPELINE START: $(date)"
echo "================================================================="

# ── Step 1: 2g + 2h in parallel (current 145-factor panel) ────────────
echo ""
echo "Step 1: Factor decay report (2g) + crowding diagnostic (2h) in parallel..."
$PY 2g_factor_decay_report.py > logs/2g.log 2>&1 &
PID_2G=$!
$PY 2h_factor_crowding.py     > logs/2h.log 2>&1 &
PID_2H=$!
wait $PID_2G && echo "  [2g] Done ✓" || echo "  [2g] FAILED — check logs/2g.log"
wait $PID_2H && echo "  [2h] Done ✓" || echo "  [2h] FAILED — check logs/2h.log"

# ── Step 2: Rebuild panel with Cat 16 + Cat 17 ────────────────────────
echo ""
echo "Step 2: Rebuilding panel with Cat 16 (mined) + Cat 17 (time-signal v2)..."
$PY 1h_feature_engineering.py 2>&1 | tee logs/1g_cat16_17.log
if [ $? -ne 0 ]; then
    echo "  [1g] FAILED — check logs/1g_cat16_17.log"
    exit 1
fi
echo "  [1g] Done ✓"

# ── Step 3: Full factor analysis on new panel ─────────────────────────
echo ""
echo "Step 3: Factor analysis on full Cat 16+17 panel (2a)..."
$PY 2a_factor_analysis.py 2>&1 | tee logs/2a_cat16_17.log
if [ $? -ne 0 ]; then
    echo "  [2a] FAILED — check logs/2a_cat16_17.log"
    exit 1
fi
echo "  [2a] Done ✓"

# ── Step 4: Updated decay + crowding reports ─────────────────────────
echo ""
echo "Step 4: Updated decay (2g) + crowding (2h) on Cat 16+17 panel..."
$PY 2g_factor_decay_report.py > logs/2g_v2.log 2>&1 &
PID_2G2=$!
$PY 2h_factor_crowding.py     > logs/2h_v2.log 2>&1 &
PID_2H2=$!
wait $PID_2G2 && echo "  [2g v2] Done ✓" || echo "  [2g v2] FAILED"
wait $PID_2H2 && echo "  [2h v2] Done ✓" || echo "  [2h v2] FAILED"

echo ""
echo "================================================================="
echo " PIPELINE COMPLETE: $(date)"
echo " Next: review OOS IC results, prune crowded factors, retrain 3c"
echo "================================================================="
