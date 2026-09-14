"""
4a_factor_combo_baseline.py - Factor-Combo IE Baseline (No ML)
Uses the optimised factor weights from 2e_ic_optimise.py to create
stock scores directly (weighted sum of z-scored factors) and runs
them through the same index enhancement pipeline as 4b_index_enhancement.py.

This answers: how much does the transformer model ADD on top of a
well-constructed linear factor combination?

If IR(factor-combo) ≈ IR(transformer) → transformer isn't adding much
If IR(factor-combo) << IR(transformer) → transformer is genuinely learning
  non-linear relationships that the linear combo can't capture

Uses the same test period as the ML models (Jan 2023 – Oct 2025).

Outputs:
  data/scores_factor_combo.parquet
  data/bt_ie_factor_combo.csv
  Prints comparison table vs ML models
"""

import time as _time
_t0 = _time.time()

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# Config
from config import clean_universe

DATA_DIR = Path(os.environ.get("ML_DATA_DIR", "data"))

# Use optimised weights if available, else fall back to IC-decay weights
FACTOR_FILE = DATA_DIR / "factor_selected_optimised.csv"
if not FACTOR_FILE.exists():
    FACTOR_FILE = DATA_DIR / "factor_selected.csv"
    print("Note: using IC-decay weights (run 2e_ic_optimise.py for optimised weights)")

WEIGHT_COL = "weight_optimised" if "optimised" in FACTOR_FILE.name else "weight"

# Test period (same as ML model backtests)
TEST_START = "2023-01-01"
TEST_END   = "2025-12-31"

ALPHA_VALUES = [0.002, 0.005, 0.01, 0.02, 0.03, 0.05]


# Helpers
def ie_stats(df: pd.DataFrame, label: str = "") -> dict:
    ar = df["active_ret"].values
    if len(ar) < 3:
        return {}
    ann_alpha  = float(np.mean(ar) * 12)
    track_err  = float(np.std(ar, ddof=1) * np.sqrt(12))
    ir         = ann_alpha / track_err if track_err > 1e-9 else np.nan
    hit_rate   = float(np.mean(ar > 0))
    cum        = np.cumsum(ar)
    max_active_dd = float(np.min(cum - np.maximum.accumulate(cum)))
    sharpe     = float(np.mean(df["port_ret"]) * 12 /
                       (np.std(df["port_ret"], ddof=1) * np.sqrt(12) + 1e-9))
    return dict(model=label, ann_alpha=ann_alpha, track_err=track_err,
                info_ratio=ir, hit_rate=hit_rate,
                max_active_dd=max_active_dd, sharpe=sharpe,
                n_months=len(ar))


# Load data
print("=" * 65)
print("FACTOR-COMBO BASELINE - INDEX ENHANCEMENT")
print("=" * 65)

print("\nLoading factor weights …")
factor_df = pd.read_csv(FACTOR_FILE)
factors   = factor_df["factor"].tolist()
weights   = factor_df[WEIGHT_COL].values
signs     = np.array([1 if s == "+" else -1 for s in factor_df["majority_sign"]])
# Signed weights: positive for positive-IC factors, negative for contrarian
signed_weights = weights * signs
print(f"  {len(factors)} factors  |  weight source: {WEIGHT_COL}")
print(f"  Top 5: {list(zip(factors[:5], weights[:5].round(4)))}")

print("\nLoading panel …")
panel = pq.read_table(DATA_DIR / "panel_monthly_enriched.parquet").to_pandas()
panel["date"] = pd.to_datetime(panel["date"])
test_panel = panel[panel["date"] >= TEST_START].copy()
print(f"  Test period: {test_panel['date'].min().date()} → {test_panel['date'].max().date()}")
print(f"  {test_panel['date'].nunique()} months | {test_panel['ticker'].nunique()} tickers")

print("\nLoading SPX weights …")
spx = pq.read_table(DATA_DIR / "spx_weights.parquet").to_pandas()
spx["date"] = pd.to_datetime(spx["date"])

# Compute factor-combo scores
print("\nComputing factor-combo scores …")

# Check which factors are in the panel
available = [f for f in factors if f in panel.columns]
missing   = [f for f in factors if f not in panel.columns]
if missing:
    print(f"  Warning: {len(missing)} factors not in panel, skipping: {missing[:3]}")
avail_idx = [i for i, f in enumerate(factors) if f in panel.columns]
factors_use    = [factors[i]      for i in avail_idx]
sw_use         = signed_weights[avail_idx]

score_records = []
for dt, grp in test_panel.groupby("date"):
    sub = grp[["ticker", "fwd_ret_1m"] + factors_use].copy()
    sub = sub.dropna(subset=factors_use)
    if len(sub) < 30:
        continue

    X = sub[factors_use].values.astype(float)

    # Cross-sectional z-score each factor
    mu  = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-9
    X_z = (X - mu) / std

    # Weighted combination (signs already baked into sw_use)
    score = X_z @ sw_use   # (n_stocks,)

    sub = sub[["ticker", "fwd_ret_1m"]].copy()
    sub["score"] = score
    sub["date"]  = dt
    score_records.append(sub)

scores = pd.concat(score_records, ignore_index=True)
print(f"  {len(scores):,} stock-month observations")

# Save scores
pq.write_table(
    pa.Table.from_pandas(scores[["date", "ticker", "score", "fwd_ret_1m"]],
                         preserve_index=False),
    DATA_DIR / "scores_factor_combo.parquet"
)
print(f"  Saved → data/scores_factor_combo.parquet")

# Run IE portfolio construction
print("\nRunning IE portfolio construction …")
merged = scores.merge(spx[["date", "ticker", "spx_weight"]], on=["date", "ticker"], how="inner")
merged = merged.dropna(subset=["spx_weight", "score", "fwd_ret_1m"])
merged = clean_universe(merged, label="factor-combo")

results = []
bt_records = {}

for alpha in ALPHA_VALUES:
    monthly = []
    for dt, grp in merged.groupby("date"):
        if len(grp) < 20:
            continue

        # Z-score scores cross-sectionally
        s_mean = grp["score"].mean()
        s_std  = grp["score"].std() + 1e-9
        score_z = (grp["score"] - s_mean) / s_std

        # Tilt off the renormalised benchmark, not the raw column. Tilting off
        # raw and normalising afterwards scales the effective alpha by
        # 1/coverage (~1.23x here), so the swept alpha would not mean what it says.
        coverage = grp["spx_weight"].sum()
        if coverage < 1e-9:
            continue
        bench_w = grp["spx_weight"].values / coverage

        w_active = bench_w + alpha * score_z.values
        w_active = np.clip(w_active, 0, None)
        w_sum    = w_active.sum()
        if w_sum < 1e-9:
            continue
        w_active /= w_sum

        port_ret  = float((w_active * grp["fwd_ret_1m"].values).sum())
        bench_ret = float((bench_w * grp["fwd_ret_1m"].values).sum())
        active_ret = port_ret - bench_ret
        monthly.append(dict(date=dt, port_ret=port_ret, bench_ret=bench_ret,
                             active_ret=active_ret, n_stocks=len(grp)))

    if not monthly:
        continue
    df_m = pd.DataFrame(monthly)
    stats = ie_stats(df_m, label=f"α={alpha}")
    stats["alpha"] = alpha
    results.append(stats)
    bt_records[alpha] = df_m

# Best alpha (highest IR within TE 1–5%)
valid = [r for r in results if 0.01 <= r["track_err"] <= 0.05]
if not valid:
    valid = results
best = max(valid, key=lambda r: r["info_ratio"]) if valid else results[0]
best_alpha = best["alpha"]

# Save best-alpha backtest
bt_best = bt_records[best_alpha]
bt_best.to_csv(DATA_DIR / "bt_ie_factor_combo.csv", index=False)

# Print alpha sweep
print(f"\n{'─'*65}")
print(f"  Factor-Combo Baseline - Alpha Sweep")
print(f"{'─'*65}")
print(f"  {'Alpha':>7} {'Ann α':>7} {'TE':>6} {'IR':>7} {'Hit%':>6} {'Sharpe':>7}")
for r in results:
    marker = " ◄" if r["alpha"] == best_alpha else ""
    print(f"  {r['alpha']:>7.3f} {r['ann_alpha']*100:>6.1f}% "
          f"{r['track_err']*100:>5.1f}% {r['info_ratio']:>7.3f} "
          f"{r['hit_rate']*100:>5.1f}% {r['sharpe']:>7.3f}{marker}")

# Comparison vs ML models
print(f"\n{'─'*65}")
print(f"  Comparison: Factor-Combo vs ML Models (best alpha each)")
print(f"{'─'*65}")
print(f"  {'Model':<28} {'Ann α':>7} {'TE':>6} {'IR':>7} {'Hit%':>6} {'Sharpe':>7}")
print(f"  {'─'*60}")

# Factor combo best
r = best
print(f"  {'Factor-Combo (linear)':<28} {r['ann_alpha']*100:>6.1f}% "
      f"{r['track_err']*100:>5.1f}% {r['info_ratio']:>7.3f} "
      f"{r['hit_rate']*100:>5.1f}% {r['sharpe']:>7.3f}")

# Load ML model results
ml_models = {
    "CS-Transformer": DATA_DIR / "bt_ie_cs_transformer.csv",
    "FT-Transformer": DATA_DIR / "bt_ie_transformer.csv",
    "LGBM":           DATA_DIR / "bt_ie_lgbm.csv",
}
for name, fpath in ml_models.items():
    if not fpath.exists():
        continue
    df_ml = pd.read_csv(fpath)
    df_ml["date"] = pd.to_datetime(df_ml["date"])
    s = ie_stats(df_ml, name)
    print(f"  {name:<28} {s['ann_alpha']*100:>6.1f}% "
          f"{s['track_err']*100:>5.1f}% {s['info_ratio']:>7.3f} "
          f"{s['hit_rate']*100:>5.1f}% {s['sharpe']:>7.3f}")

print(f"{'─'*65}")
print(f"\n  Factor-combo best alpha: {best_alpha}")
print(f"  Saved → data/bt_ie_factor_combo.csv")
print(f"\nDone in {(_time.time() - _t0) / 60:.1f} min")
