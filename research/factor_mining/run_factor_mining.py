#!/usr/bin/env python3
"""
run_factor_mining.py
====================
Master script for the factor mining exercise.

Orchestrates:
1. Load data and compute candidate features
2. Run triple screen (Lasso + Trees + Autoencoder)
3. Validate survivors (IC, ICIR, persistence, dedup, RAS, BHY)
4. Add entropy regime overlay
5. Produce candidate_factor_catalog.csv
6. Final summary

Usage:
    python research/factor_mining/run_factor_mining.py
"""

import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")

# Add repo root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "research", "factor_mining"))

import numpy as np
import pandas as pd

from candidate_factory import build_all_candidates, get_candidate_columns, get_candidate_metadata
from regime_entropy import add_entropy_regime_signals, validate_entropy_signal, add_regime_interactions
from validation_engine import (
    compute_is_ic, compute_icir, screen_all_candidates,
    TRAIN_END, IC_THRESHOLD, ICIR_THRESHOLD,
)
from screen_lasso import lasso_screen
from screen_trees import tree_screen, lgbm_screen

RESULTS_DIR = os.path.join(REPO_ROOT, "research", "factor_mining", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data():
    """Load prices and panel data."""
    print("=" * 70)
    print("PHASE 1: LOADING DATA")
    print("=" * 70)

    prices_path = os.path.join(REPO_ROOT, "data", "prices.parquet")
    panel_path = os.path.join(REPO_ROOT, "data", "panel_monthly_enriched.parquet")

    print(f"  Loading prices from {prices_path}...")
    prices = pd.read_parquet(prices_path)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Compute ret_d if missing
    if "ret_d" not in prices.columns:
        prices["ret_d"] = prices.groupby("ticker")["close"].pct_change()

    # Add SPX returns if available
    if "spx_ret" not in prices.columns:
        spx = prices[prices["ticker"] == "SPY"][["date", "ret_d"]].rename(columns={"ret_d": "spx_ret"})
        if len(spx) > 0:
            prices = prices.merge(spx[["date", "spx_ret"]], on="date", how="left")

    print(f"  Prices: {prices.shape[0]:,} rows, {prices['ticker'].nunique()} tickers")
    print(f"  Date range: {prices['date'].min()} to {prices['date'].max()}")

    print(f"\n  Loading panel from {panel_path}...")
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    print(f"  Panel: {panel.shape[0]:,} rows × {panel.shape[1]} columns")

    return prices, panel


def compute_candidates(prices):
    """Phase 2: Compute all candidate features on daily prices."""
    print("\n" + "=" * 70)
    print("PHASE 2: COMPUTING CANDIDATE FEATURES")
    print("=" * 70)

    t0 = time.time()
    prices = build_all_candidates(prices)
    cand_cols = get_candidate_columns(prices)
    print(f"\n  Time: {time.time()-t0:.0f}s")

    return prices, cand_cols


def add_entropy(prices):
    """Phase 2b: Add entropy regime signals."""
    print("\n  Adding entropy regime signals...")
    prices = add_entropy_regime_signals(prices, window=21, n_bins=5)
    entropy_cols = [c for c in prices.columns if "entropy" in c]
    print(f"  Entropy columns added: {entropy_cols}")
    return prices


def sample_at_month_end(prices, cand_cols):
    """Sample daily candidate features at month-end for panel merge."""
    print("\n  Sampling candidates at month-end...")
    prices["_ym"] = prices["date"].dt.to_period("M")
    month_end_idx = prices.groupby(["ticker", "_ym"])["date"].idxmax()
    monthly = prices.loc[month_end_idx, ["date", "ticker"] + cand_cols].copy()
    monthly = monthly.drop_duplicates(subset=["date", "ticker"])
    print(f"  Monthly candidate panel: {monthly.shape[0]:,} rows × {len(cand_cols)} candidates")
    return monthly


def merge_candidates_to_panel(panel, monthly_candidates, cand_cols):
    """Merge monthly candidate features into the existing panel."""
    print("\n  Merging candidates into panel...")
    # Drop any existing cand_ columns in panel
    existing_cand = [c for c in panel.columns if c.startswith("cand_")]
    if existing_cand:
        panel = panel.drop(columns=existing_cand)

    panel = panel.merge(
        monthly_candidates[["date", "ticker"] + cand_cols],
        on=["date", "ticker"],
        how="left",
    )
    # Clean inf values across entire panel (some existing features may have extreme values)
    import numpy as np
    numeric_cols = panel.select_dtypes(include=[np.number]).columns
    panel[numeric_cols] = panel[numeric_cols].replace([np.inf, -np.inf], np.nan)
    print(f"  Merged panel: {panel.shape[0]:,} rows × {panel.shape[1]} columns")
    return panel


def run_ml_screen(panel, all_feature_cols):
    """Phase 3: Feed ALL features to penalized ML models - no pre-filtering.

    Design choice: no manual filtering. L1/L2 penalties inside the models
    do the selection work, avoiding human-selection bias.
    """
    print("\n" + "=" * 70)
    print("PHASE 3: ML SCREENING - ALL FEATURES, NO PRE-FILTER (IS-ONLY)")
    print("=" * 70)
    print(f"  Total features fed to models: {len(all_feature_cols)}")

    # Screen 1: Lasso (L1 penalty kills irrelevant features)
    print("\n  --- Lasso Screen (L1 penalty) ---")
    lasso_selected, lasso_imp = lasso_screen(panel, all_feature_cols)

    # Screen 2: Random Forest (natural importance ranking)
    print("\n  --- Tree Screen (RF) ---")
    rf_selected, rf_imp = tree_screen(panel, all_feature_cols)

    # Screen 3: LightGBM (L1+L2 regularization built in)
    print("\n  --- Tree Screen (LightGBM) ---")
    lgbm_selected, lgbm_imp = lgbm_screen(panel, all_feature_cols)

    # Union of survivors from any model
    all_selected = list(set(lasso_selected) | set(rf_selected) | set(lgbm_selected))
    print(f"\n  Union of ML screens: {len(all_selected)} unique features")
    print(f"    Lasso: {len(lasso_selected)}, RF: {len(rf_selected)}, LightGBM: {len(lgbm_selected)}")

    # Save screen results
    if len(lasso_imp) > 0:
        lasso_imp.to_csv(os.path.join(RESULTS_DIR, "lasso_importance.csv"), index=False)
    if len(rf_imp) > 0:
        rf_imp.to_csv(os.path.join(RESULTS_DIR, "rf_importance.csv"), index=False)
    if len(lgbm_imp) > 0:
        lgbm_imp.to_csv(os.path.join(RESULTS_DIR, "lgbm_importance.csv"), index=False)

    return all_selected


def run_validation(panel, selected_cols):
    """Phase 4: Validation - BHY only, no hard IC/ICIR floors.

    Rationale: with weekly rebalancing, noisy factors are acceptable.
    BHY controls false discovery across the whole candidate set.
    IC/ICIR computed for documentation only, not as gates.
    """
    print("\n" + "=" * 70)
    print("PHASE 4: VALIDATION (BHY + RAS + DEDUP - NO HARD IC/ICIR FLOORS)")
    print("=" * 70)

    if not selected_cols:
        print("  No features to validate!")
        return pd.DataFrame()

    # Compute metrics for all - but pass/fail is BHY + dedup only
    catalog = screen_all_candidates(
        panel,
        candidate_cols=selected_cols,
        existing_cols=selected_cols,
        ras_n_sims=200,
    )

    # Override: remove hard IC/ICIR gates, keep only BHY + dedup
    catalog["pass_all_final"] = catalog["pass_dedup"] & catalog["pass_bhy"]

    n_pass = catalog["pass_all_final"].sum()
    print(f"\n  Validation results (BHY + dedup only, no IC/ICIR floor):")
    print(f"    Total screened: {len(catalog)}")
    print(f"    Pass dedup (<0.85 corr): {catalog['pass_dedup'].sum()}")
    print(f"    Pass RAS (p<0.05): {catalog['pass_ras'].sum()}")
    print(f"    Pass BHY (FDR 5%): {catalog['pass_bhy'].sum()}")
    print(f"    PASS ALL (BHY + dedup): {n_pass}")
    print(f"\n  For reference (not used as gates):")
    print(f"    IC > 0.015: {catalog['pass_ic'].sum()}")
    print(f"    ICIR > 0.40: {catalog['pass_icir'].sum()}")

    catalog_path = os.path.join(REPO_ROOT, "research", "factor_mining", "candidate_factor_catalog.csv")
    catalog.to_csv(catalog_path, index=False)
    print(f"\n  Saved: {catalog_path}")

    return catalog


def run_entropy_validation(panel):
    """Phase 5: Validate entropy regime overlay."""
    print("\n" + "=" * 70)
    print("PHASE 5: ENTROPY REGIME VALIDATION")
    print("=" * 70)

    if "cand_low_entropy_regime" not in panel.columns:
        print("  Entropy signal not in panel - skipping")
        return

    result = validate_entropy_signal(panel)
    print(f"  Mean |fwd_ret| in LOW entropy:  {result['mean_abs_ret_low_entropy']:.4f}")
    print(f"  Mean |fwd_ret| in HIGH entropy: {result['mean_abs_ret_high_entropy']:.4f}")
    print(f"  Ratio (low/high):               {result['ratio']:.3f}")
    print(f"  Directional accuracy (low):     {result['directional_accuracy_low']:.3f}")
    print(f"  Directional accuracy (high):    {result['directional_accuracy_high']:.3f}")
    print(f"  N low={result['n_low']:,}, N high={result['n_high']:,}")

    # Add regime interactions for key factors
    print("\n  Adding regime interaction features...")
    panel = add_regime_interactions(panel)
    interaction_cols = [c for c in panel.columns if "_x_entropy" in c]
    print(f"  Interaction features: {interaction_cols}")

    return panel


def write_summary(catalog):
    """Phase 6: Write final summary."""
    print("\n" + "=" * 70)
    print("PHASE 6: FINAL SUMMARY")
    print("=" * 70)

    if len(catalog) == 0:
        print("  No catalog to summarize")
        return

    survivors = catalog[catalog["pass_all_final"]].sort_values("icir", ascending=False)

    summary = f"""# Factor Mining Results Summary

**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
**IS Period:** 2010-01 to 2022-12
**Candidates Generated:** {len(catalog)}
**Survivors (pass all screens):** {len(survivors)}

## Screening Funnel

| Stage | Count |
|-------|-------|
| Total candidates | {len(catalog)} |
| Pass IC > {IC_THRESHOLD} | {catalog['pass_ic'].sum()} |
| Pass ICIR > {ICIR_THRESHOLD} | {catalog['pass_icir'].sum()} |
| Pass persistence | {catalog['pass_persistence'].sum()} |
| Pass dedup (<0.85 corr) | {catalog['pass_dedup'].sum()} |
| Pass RAS (p<0.05) | {catalog['pass_ras'].sum()} |
| Pass BHY (FDR 5%) | {catalog['pass_bhy'].sum()} |
| **Pass ALL** | **{len(survivors)}** |

## Surviving Factors

"""
    if len(survivors) > 0:
        for _, row in survivors.iterrows():
            summary += f"### {row['factor']}\n"
            summary += f"- IC: {row['ic_mean']:.4f}, ICIR: {row['icir']:.3f}\n"
            summary += f"- IC lag1: {row['ic_lag1']:.4f}\n"
            summary += f"- Closest existing: {row['closest_existing']} (corr={row['max_corr_existing']:.3f})\n"
            summary += f"- RAS p-value: {row['ras_pvalue']:.4f}, BHY adjusted: {row['bhy_adjusted_pvalue']:.4f}\n\n"
    else:
        summary += "No candidates survived all screens.\n\n"
        summary += "### Top candidates by ICIR (even if not passing all screens)\n\n"
        top = catalog.sort_values("icir", ascending=False).head(10)
        for _, row in top.iterrows():
            summary += f"- **{row['factor']}**: ICIR={row['icir']:.3f}, IC={row['ic_mean']:.4f}"
            fails = []
            if not row['pass_ic']: fails.append("IC")
            if not row['pass_icir']: fails.append("ICIR")
            if not row['pass_persistence']: fails.append("persist")
            if not row['pass_dedup']: fails.append("dedup")
            if not row['pass_ras']: fails.append("RAS")
            if not row.get('pass_bhy', False): fails.append("BHY")
            summary += f" [FAILED: {', '.join(fails)}]\n"

    summary_path = os.path.join(REPO_ROOT, "research", "factor_mining", "reports", "factor_mining_summary.md")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"  Saved: {summary_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--revalidate-only", action="store_true",
                        help="Skip Phase 2-3 (candidate computation + ML screen). "
                             "Re-run Phase 4 validation using the ML survivors from "
                             "the previous run's candidate_factor_catalog.csv.")
    args = parser.parse_args()

    t_start = time.time()

    # Phase 1: Load data
    prices, panel = load_data()

    merged_panel_path = os.path.join(REPO_ROOT, "research", "factor_mining",
                                      "results", "panel_merged.parquet")

    if args.revalidate_only:
        # Fast path: reload previous ML screen results + saved merged panel
        catalog_path = os.path.join(REPO_ROOT, "research", "factor_mining",
                                     "candidate_factor_catalog.csv")
        prev_catalog = pd.read_csv(catalog_path)
        selected = prev_catalog["factor"].tolist()
        print(f"\n  --revalidate-only: loaded {len(selected)} factors from previous run")

        if os.path.exists(merged_panel_path):
            print(f"  Loading saved merged panel from {merged_panel_path}...")
            panel = pd.read_parquet(merged_panel_path)
            panel["date"] = pd.to_datetime(panel["date"])
            print(f"  Merged panel: {panel.shape[0]:,} rows × {panel.shape[1]} columns")
        else:
            print("  No saved merged panel - recomputing candidates...")
            prices, cand_cols = compute_candidates(prices)
            prices = add_entropy(prices)
            cand_cols = get_candidate_columns(prices)
            monthly = sample_at_month_end(prices, cand_cols)
            panel = merge_candidates_to_panel(panel, monthly, cand_cols)
    else:
        # Full path: compute candidates + ML screen
        # Phase 2: Compute candidates
        prices, cand_cols = compute_candidates(prices)
        prices = add_entropy(prices)

        # Update cand_cols to include entropy
        cand_cols = get_candidate_columns(prices)

        # Sample at month-end and merge
        monthly = sample_at_month_end(prices, cand_cols)
        panel = merge_candidates_to_panel(panel, monthly, cand_cols)

        # Build unified feature list: existing + new candidates, all screened together
        exclude = {"date", "ticker", "fwd_ret_1m", "fwd_ret_3m", "fwd_ret_6m",
                   "fwd_ret_12m", "spx_weights", "volume", "hl_range"}
        all_feature_cols = [c for c in panel.columns
                            if c not in exclude
                            and panel[c].dtype in ("float64", "float32")
                            and panel[c].notna().mean() > 0.3]
        print(f"\n  Unified feature set: {len(all_feature_cols)} features (existing + new)")

        # Phase 3: ML screen - all features, no pre-filter, penalties do the work
        selected = run_ml_screen(panel, all_feature_cols)

        # Save merged panel for fast --revalidate-only reruns
        os.makedirs(os.path.dirname(merged_panel_path), exist_ok=True)
        panel.to_parquet(merged_panel_path, index=False)
        print(f"\n  Saved merged panel: {merged_panel_path}")

    # Phase 4: Validation - dedup within survivors, no existing/new split
    catalog = run_validation(panel, selected)

    # Phase 5: Entropy validation
    run_entropy_validation(panel)

    # Phase 6: Summary
    write_summary(catalog)

    print(f"\n{'=' * 70}")
    print(f"FACTOR MINING COMPLETE - Total time: {time.time()-t_start:.0f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
