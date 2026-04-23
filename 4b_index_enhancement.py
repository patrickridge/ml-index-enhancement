"""
4b_index_enhancement.py — Index Enhancement Portfolio Construction
=================================================================
Builds an index-enhanced portfolio by tilting S&P 500 market-cap weights
using ML model scores (LGBM, FT-Transformer, CS-Transformer).

Strategy:
  w_i = w_SPX_i + α × score_z_i
  where score_z_i is the ML score cross-sectionally z-scored each month.
  Weights are clipped to 0 (long-only) and renormalised to sum to 1.

The script sweeps α (tilt strength) to show the tracking error vs
information ratio tradeoff, then saves the best-performing variant.

Requires (run first):
  python 1c_fetch_market_cap.py  → data/spx_weights.parquet

Outputs:
  data/bt_ie_lgbm.csv             — monthly port_ret and bench_ret (best α)
  data/bt_ie_transformer.csv
  data/bt_ie_cs_transformer.csv
  data/ie_summary.csv             — IR / TE table across all models and α values
"""

import time as _time
_t0 = _time.time()

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR     = Path("data")
WEIGHTS_FILE = DATA_DIR / "spx_weights.parquet"

SCORE_FILES = {
    "LGBM":           DATA_DIR / "scores_lgbm.parquet",
    "FT-Transformer": DATA_DIR / "scores_transformer.parquet",
    "CS-Transformer": DATA_DIR / "scores_cs_transformer.parquet",
}

# ── Tilt strength sweep ────────────────────────────────────────────────────────
# α controls how aggressively to overweight/underweight vs benchmark.
# Higher α → higher potential alpha but higher tracking error.
# Target tracking error: 2–4% annualised.
ALPHA_GRID = [0.002, 0.005, 0.01, 0.02, 0.03, 0.05]


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_enhanced_portfolio(
    scores: pd.DataFrame,
    weights: pd.DataFrame,
    alpha: float,
    top_n: int = 100,
    bottom_n: int = 100,
) -> pd.DataFrame:
    """
    Given ML scores and SPX weights, construct index-enhanced portfolio.

    Overweights top_n stocks and underweights bottom_n stocks vs benchmark.
    Middle stocks held at benchmark weight.

    Parameters
    ----------
    scores   : DataFrame with columns [date, ticker, score, fwd_ret_1m]
    weights  : DataFrame with columns [date, ticker, spx_weight]
    alpha    : tilt strength scalar
    top_n    : number of stocks to overweight
    bottom_n : number of stocks to underweight

    Returns
    -------
    DataFrame with columns [date, port_ret, bench_ret, active_ret]
    """
    df = scores.merge(weights[["date", "ticker", "spx_weight"]],
                      on=["date", "ticker"], how="inner")
    df = df.dropna(subset=["score", "spx_weight", "fwd_ret_1m"])

    # Transaction cost: one-way rate in decimals. 10 bps = 0.0010 per side.
    # Round-trip cost (buy + sell on a new name) = 2 × one_way × |weight change|.
    # Applied to each month's portfolio turnover vs previous month's weights.
    one_way_cost = 0.0010  # 10 bps per side — conservative for S&P 500 names

    results = []
    prev_weights = {}      # ticker → weight last month
    for dt, grp in df.groupby("date"):
        if len(grp) < top_n + bottom_n:
            continue

        grp = grp.copy().sort_values("score", ascending=False).reset_index(drop=True)
        n = len(grp)

        # Assign tilt: +alpha to top_n, -alpha to bottom_n, 0 to middle
        tilt = pd.Series(0.0, index=grp.index)
        tilt.iloc[:top_n] = alpha
        tilt.iloc[n - bottom_n:] = -alpha

        # Active weights = benchmark + tilt, clipped long-only, renormalised
        raw_w = (grp["spx_weight"] + tilt).clip(lower=0.0)
        total = raw_w.sum()
        if total < 1e-8:
            continue
        grp["port_weight"] = raw_w / total

        # Turnover vs previous month → transaction cost
        cur_weights = dict(zip(grp["ticker"], grp["port_weight"]))
        all_tickers = set(cur_weights) | set(prev_weights)
        turnover = sum(abs(cur_weights.get(t, 0.0) - prev_weights.get(t, 0.0))
                       for t in all_tickers)
        txn_cost = one_way_cost * turnover   # one_way × |Δw| on each side

        # Returns (gross then net of costs)
        port_ret_gross = (grp["port_weight"] * grp["fwd_ret_1m"]).sum()
        bench_ret      = (grp["spx_weight"]  * grp["fwd_ret_1m"]).sum()
        port_ret       = port_ret_gross - txn_cost  # net

        results.append({
            "date":          dt,
            "port_ret":      port_ret,           # NET of costs
            "port_ret_gross": port_ret_gross,    # gross (kept for diagnostics)
            "bench_ret":     bench_ret,
            "active_ret":    port_ret - bench_ret,
            "turnover":      turnover,
            "txn_cost":      txn_cost,
            "n_stocks":      len(grp),
        })

        prev_weights = cur_weights

    return pd.DataFrame(results).set_index("date").sort_index()


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE STATS
# ═══════════════════════════════════════════════════════════════════════════════

def ie_stats(bt: pd.DataFrame) -> dict:
    """
    Compute index-enhancement statistics from backtest DataFrame.
    """
    r_port   = bt["port_ret"].dropna()
    r_bench  = bt["bench_ret"].dropna()
    r_active = bt["active_ret"].dropna()

    n = len(r_active)
    if n < 6:
        return {}

    # Benchmark stats
    ann_bench = (1 + r_bench).prod() ** (12 / n) - 1

    # Active return / tracking error / information ratio
    ann_alpha  = (1 + r_active).prod() ** (12 / n) - 1
    track_err  = r_active.std(ddof=1) * np.sqrt(12)
    info_ratio = ann_alpha / track_err if track_err > 0 else np.nan

    # Hit rate — % of months portfolio beats benchmark
    hit_rate = (r_active > 0).mean()

    # Max active drawdown — worst sustained underperformance vs benchmark
    active_nav    = (1 + r_active).cumprod()
    max_active_dd = (active_nav / active_nav.cummax() - 1).min()

    # Portfolio standalone stats
    ann_port = (1 + r_port).prod() ** (12 / n) - 1
    vol_port = r_port.std(ddof=1) * np.sqrt(12)
    sharpe   = ann_port / vol_port if vol_port > 0 else np.nan
    nav      = (1 + r_port).cumprod()
    maxdd    = (nav / nav.cummax() - 1).min()

    return dict(
        months        = n,
        ann_port      = ann_port,
        ann_bench     = ann_bench,
        ann_alpha     = ann_alpha,
        track_err     = track_err,
        info_ratio    = info_ratio,
        hit_rate      = hit_rate,
        max_active_dd = max_active_dd,
        sharpe        = sharpe,
        maxdd         = maxdd,
    )


def print_stats(label: str, s: dict):
    if not s:
        print(f"  {label}: insufficient data")
        return
    print(f"  {label}")
    print(f"    Ann port={s['ann_port']*100:.1f}%  Ann bench={s['ann_bench']*100:.1f}%  "
          f"Ann alpha={s['ann_alpha']*100:.2f}%")
    print(f"    Tracking error={s['track_err']*100:.2f}%  "
          f"IR={s['info_ratio']:.3f}  "
          f"Hit rate={s['hit_rate']*100:.1f}%  "
          f"Max active DD={s['max_active_dd']*100:.1f}%  "
          f"MaxDD={s['maxdd']*100:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Load SPX weights ─────────────────────────────────────────────────────────
    if not WEIGHTS_FILE.exists():
        print(f"ERROR: {WEIGHTS_FILE} not found.")
        print("Run: python 1c_fetch_market_cap.py")
        return

    print("Loading SPX weights …")
    weights = pd.read_parquet(WEIGHTS_FILE)
    weights["date"] = pd.to_datetime(weights["date"])
    print(f"  {len(weights):,} rows | {weights['date'].min().date()} → "
          f"{weights['date'].max().date()} | {weights['ticker'].nunique()} tickers")

    # ── Sweep all models and alpha values ────────────────────────────────────────
    summary_rows = []
    best_bt      = {}   # model → best-α backtest DataFrame

    for model_name, score_path in SCORE_FILES.items():
        if not score_path.exists():
            print(f"\n[SKIP] {model_name}: {score_path.name} not found")
            continue

        scores = pd.read_parquet(score_path)
        scores["date"] = pd.to_datetime(scores["date"])
        print(f"\n{'─'*60}")
        print(f"Model: {model_name}  "
              f"({scores['date'].min().date()} → {scores['date'].max().date()}, "
              f"{scores['date'].nunique()} months)")

        best_ir   = -np.inf
        best_alpha = None

        for alpha in ALPHA_GRID:
            bt = build_enhanced_portfolio(scores, weights, alpha)
            if bt.empty:
                continue
            s = ie_stats(bt)
            if not s:
                continue

            row = {"model": model_name, "alpha": alpha, **s}
            summary_rows.append(row)

            print(f"  α={alpha:.3f}  TE={s['track_err']*100:.2f}%  "
                  f"IR={s['info_ratio']:.3f}  "
                  f"alpha={s['ann_alpha']*100:.2f}%  "
                  f"Sharpe={s['sharpe']:.2f}")

            # Best α = highest IR where TE is within target range 1–5%
            te = s["track_err"]
            if 0.01 <= te <= 0.05 and s["info_ratio"] > best_ir:
                best_ir    = s["info_ratio"]
                best_alpha = alpha
                best_bt[model_name] = bt.copy()

        if best_alpha is not None:
            print(f"  → Best α={best_alpha}  (IR={best_ir:.3f}, target TE 1–5%)")
        else:
            # Fallback: just pick α = 0.01
            bt = build_enhanced_portfolio(scores, weights, alpha=0.01)
            best_bt[model_name] = bt.copy()
            print(f"  → No α hit target TE range — saved α=0.01 as fallback")

    # ── Save best backtests ───────────────────────────────────────────────────────
    file_map = {
        "LGBM":           DATA_DIR / "bt_ie_lgbm.csv",
        "FT-Transformer": DATA_DIR / "bt_ie_transformer.csv",
        "CS-Transformer": DATA_DIR / "bt_ie_cs_transformer.csv",
    }
    for model_name, bt in best_bt.items():
        out = file_map[model_name]
        bt.reset_index().to_csv(out, index=False)
        print(f"\nSaved → {out}")

    # ── Save summary table ────────────────────────────────────────────────────────
    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(DATA_DIR / "ie_summary.csv", index=False)
        print(f"Saved → {DATA_DIR / 'ie_summary.csv'}")

    # ── Final comparison table ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("INDEX ENHANCEMENT — BEST RESULTS PER MODEL (target TE 2–4%)")
    print("=" * 70)
    print(f"{'Model':<22}  {'Ann α':>7}  {'TE':>7}  {'IR':>7}  {'Sharpe':>7}  {'MaxDD':>7}")
    print("-" * 70)
    for model_name, bt in best_bt.items():
        s = ie_stats(bt)
        if s:
            print(f"  {model_name:<20}  "
                  f"{s['ann_alpha']*100:>6.2f}%  "
                  f"{s['track_err']*100:>6.2f}%  "
                  f"{s['info_ratio']:>7.3f}  "
                  f"{s['sharpe']:>7.2f}  "
                  f"{s['maxdd']*100:>6.1f}%")
    print("=" * 70)
    print("\nIR guide: > 0.5 = good, > 1.0 = excellent")
    print("TE guide : 2–4% is typical for enhanced index strategies")
    print("\nRun 4_benchmark_spx.py to compare with standalone strategies.")

    print(f"\nDone in {(_time.time() - _t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
