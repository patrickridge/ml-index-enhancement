#!/bin/bash
# run_pipeline.sh — Full data fetch + feature rebuild + analysis pipeline
# Usage: bash run_pipeline.sh [--skip-fetch]
# Runs: 1j-1m (data fetch) → 1h (feature engineering) → 2a (analysis) → 2g+2h (diagnostics)

PY=/opt/anaconda3/bin/python3
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
mkdir -p logs

echo "================================================================="
echo " PIPELINE START: $(date)"
echo "================================================================="

# ── Step 0: Fetch external data (optional, skip with --skip-fetch) ────
if [ "$1" != "--skip-fetch" ]; then
    echo ""
    echo "Step 0a: Fetching external data sources in parallel..."
    echo "  (Use --skip-fetch to skip if data is already downloaded)"
    echo ""

    # 1j: Simfin fundamentals (slowest — 10-20 min via yfinance fallback)
    echo "  Starting 1j (fundamentals)..."
    $PY 1j_fetch_simfin.py > logs/1j.log 2>&1 &
    PID_1J=$!

    # 1m: Prediction markets (~1-2 min)
    echo "  Starting 1m (prediction markets)..."
    $PY 1m_fetch_prediction_markets.py > logs/1m.log 2>&1 &
    PID_1M=$!

    # Wait for fast fetches first
    wait $PID_1M && echo "  [1m] Done ✓" || echo "  [1m] FAILED — check logs/1m.log"

    # 1k: Short interest (~10-20 min, depends on yfinance rate limits)
    echo "  Starting 1k (short interest)..."
    $PY 1k_fetch_short_interest.py > logs/1k.log 2>&1 &
    PID_1K=$!

    # 1l: Institutional ownership (~10-20 min)
    echo "  Starting 1l (institutional ownership)..."
    $PY 1l_fetch_13f.py > logs/1l.log 2>&1 &
    PID_1L=$!

    # Wait for all remaining fetches
    wait $PID_1J && echo "  [1j] Done ✓" || echo "  [1j] FAILED — check logs/1j.log"
    wait $PID_1K && echo "  [1k] Done ✓" || echo "  [1k] FAILED — check logs/1k.log"
    wait $PID_1L && echo "  [1l] Done ✓" || echo "  [1l] FAILED — check logs/1l.log"

    echo ""
    echo "  Data fetch complete. Checking output files:"
    for f in data/fundamental.parquet data/prediction_markets.parquet data/short_interest.parquet data/institutional_ownership.parquet; do
        if [ -f "$f" ]; then
            echo "    ✓ $f"
        else
            echo "    ✗ $f (missing — Cat will be skipped)"
        fi
    done
fi

# ── Step 1: Initial diagnostics on current panel ─────────────────────
echo ""
echo "Step 1: Factor decay (2g) + crowding (2h) on current panel..."
$PY 2g_factor_decay_report.py > logs/2g.log 2>&1 &
PID_2G=$!
$PY 2h_factor_crowding.py     > logs/2h.log 2>&1 &
PID_2H=$!
wait $PID_2G && echo "  [2g] Done ✓" || echo "  [2g] FAILED — check logs/2g.log"
wait $PID_2H && echo "  [2h] Done ✓" || echo "  [2h] FAILED — check logs/2h.log"

# ── Step 2: Rebuild panel with all Cats ───────────────────────────────
echo ""
echo "Step 2: Rebuilding panel (Cats 1-21)..."
$PY 1h_feature_engineering.py 2>&1 | tee logs/1h.log
if [ $? -ne 0 ]; then
    echo "  [1h] FAILED — check logs/1h.log"
    exit 1
fi
echo "  [1h] Done ✓"

# ── Step 3: Full factor analysis ─────────────────────────────────────
echo ""
echo "Step 3: Factor analysis on full panel (2a)..."
$PY 2a_factor_analysis.py 2>&1 | tee logs/2a.log
if [ $? -ne 0 ]; then
    echo "  [2a] FAILED — check logs/2a.log"
    exit 1
fi
echo "  [2a] Done ✓"

# ── Step 4: Updated diagnostics ──────────────────────────────────────
echo ""
echo "Step 4: Updated decay (2g) + crowding (2h) on full panel..."
$PY 2g_factor_decay_report.py > logs/2g_v2.log 2>&1 &
PID_2G2=$!
$PY 2h_factor_crowding.py     > logs/2h_v2.log 2>&1 &
PID_2H2=$!
wait $PID_2G2 && echo "  [2g v2] Done ✓" || echo "  [2g v2] FAILED"
wait $PID_2H2 && echo "  [2h v2] Done ✓" || echo "  [2h v2] FAILED"

echo ""
echo "================================================================="
echo " PIPELINE COMPLETE: $(date)"
echo " Review: logs/2a.log (factor IC), logs/2g_v2.log (decay),"
echo "         logs/2h_v2.log (crowding/correlation)"
echo "================================================================="
