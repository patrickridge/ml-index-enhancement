"""
5_pitch_validator.py
====================
Investment society pitch tool.

Given a ticker symbol, shows:
  - Current model score and rank vs all ~500 S&P 500 stocks
  - 12-month rank trend (is the model getting more or less bullish?)
  - Top 5 strongest signals (reasons to be long)
  - Top 5 weakest signals (risks / headwinds)

Usage:
  python 5_pitch_validator.py AAPL
  python 5_pitch_validator.py              # will prompt for ticker
  python 5_pitch_validator.py AAPL MSFT NVDA   # compare multiple tickers

Requires:
  - data/scores_lgbm.parquet  (from 2_lgbm_backtest.py)
  - data/panel_monthly_enriched.parquet  (from 1_feature_engineering.py)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

from config import DATA_DIR

SCORES_IN = DATA_DIR / "scores_lgbm.parquet"
PANEL_IN  = DATA_DIR / "panel_monthly_enriched.parquet"

# ── Human-readable feature labels ─────────────────────────────────────────────
FEATURE_LABELS = {
    # Base momentum
    "ret_1m":          "1-month return (short-term)",
    "ret_3m":          "3-month momentum",
    "ret_6m":          "6-month momentum",
    "ret_12m":         "12-month momentum",
    # Base volatility
    "vol_60d":         "60-day volatility",
    "vol_252d":        "252-day volatility",
    # Engineered: volatility ratios
    "vol_ratio_sm":    "Vol ratio: short/medium (rising vol = negative)",
    "vol_ratio_ml":    "Vol ratio: medium/long (vol trend)",
    # Engineered: trend / MA
    "price_to_ma20":   "Price vs 20-day MA (short-term trend)",
    "price_to_ma60":   "Price vs 60-day MA (medium-term trend)",
    "price_to_ma200":  "Price vs 200-day MA (long-term trend)",
    "ma_20_60_cross":  "20d/60d MA cross (short-term vs medium-term)",
    "ma_60_200_cross": "60d/200d MA cross (golden/death cross signal)",
    # Engineered: 52-week high
    "nearness_52w":    "Nearness to 52-week high (breakout signal)",
    # Engineered: liquidity
    "amihud_21d":      "Amihud illiquidity (higher = less liquid)",
    # Engineered: upside/downside
    "up_down_vol":     "Upside/downside vol ratio (asymmetry)",
    # Engineered: tail risk
    "skew_60d":        "60-day return skewness",
    "maxdd_60d":       "60-day max drawdown (lower = worse)",
    # Engineered: momentum extensions
    "mom_2_12":        "Skip-1-month momentum (2-12m, classic factor)",
    "mom_accel":       "Momentum acceleration (recent vs older)",
    "ir_12m":          "12-month information ratio (risk-adj momentum)",
    "ir_3m":           "3-month information ratio (risk-adj momentum)",
    "rev_signal":      "Short-term reversal signal (-ret_1m)",
}


def load_data():
    scores = pd.read_parquet(SCORES_IN)
    scores["date"] = pd.to_datetime(scores["date"])

    panel = pd.read_parquet(PANEL_IN)
    panel["date"] = pd.to_datetime(panel["date"])

    # Add cross-sectional rank column within each month
    scores["rank"] = scores.groupby("date")["score"].rank(ascending=False).astype(int)
    scores["n_stocks"] = scores.groupby("date")["score"].transform("count").astype(int)
    scores["pct_rank"] = (scores["rank"] / scores["n_stocks"] * 100).round(1)

    return scores, panel


def get_feature_cols(panel: pd.DataFrame) -> list:
    exclude = {"date", "ticker", "fwd_ret_1m"}
    return [c for c in panel.columns if c not in exclude]


def validate_ticker(ticker: str, scores: pd.DataFrame) -> bool:
    if ticker not in scores["ticker"].values:
        print(f"  ✗  '{ticker}' not found in model scores. Check the ticker symbol.")
        available = sorted(scores["ticker"].unique())
        close = [t for t in available if ticker[:2].upper() in t][:5]
        if close:
            print(f"     Similar tickers: {', '.join(close)}")
        return False
    return True


def rank_badge(pct: float) -> str:
    """Return a visual badge for a percentile rank."""
    if pct <= 5:   return "⭐ TOP 5%"
    if pct <= 10:  return "🟢 TOP 10%"
    if pct <= 25:  return "🟡 TOP 25%"
    if pct <= 50:  return "🔵 TOP 50%"
    if pct <= 75:  return "🟠 BOT 50%"
    return             "🔴 BOT 25%"


def trend_arrow(ranks: pd.Series) -> str:
    """Return an arrow based on rank trend (lower rank = better)."""
    if len(ranks) < 2:
        return "→"
    # compare last 3 months vs first 3 months (lower rank = higher conviction)
    recent = ranks.iloc[-3:].mean() if len(ranks) >= 3 else ranks.iloc[-1]
    older  = ranks.iloc[:3].mean()  if len(ranks) >= 3 else ranks.iloc[0]
    delta = older - recent   # positive = rank improved (number got smaller)
    if delta > 30:   return "↑↑ Strong improvement"
    if delta > 10:   return "↑  Rising"
    if delta < -30:  return "↓↓ Strong deterioration"
    if delta < -10:  return "↓  Falling"
    return               "→  Stable"


def print_separator(char="─", width=62):
    print(char * width)


def analyse_ticker(ticker: str, scores: pd.DataFrame, panel: pd.DataFrame, feat_cols: list):
    ticker = ticker.upper()
    if not validate_ticker(ticker, scores):
        return

    # ── Score history ─────────────────────────────────────────────────────────
    ts = scores[scores["ticker"] == ticker].sort_values("date")
    latest = ts.iloc[-1]
    latest_month = latest["date"].strftime("%Y-%m-%d")

    print()
    print_separator("═")
    print(f"  PITCH VALIDATOR:  {ticker}")
    print_separator("═")
    print(f"  Most recent month:  {latest_month}")
    print(f"  Model score:        {latest['score']:.4f}")
    print(f"  Rank:               #{int(latest['rank'])} / {int(latest['n_stocks'])} stocks")
    print(f"  Percentile:         Top {latest['pct_rank']:.1f}%  {rank_badge(latest['pct_rank'])}")

    # Trend (last 12 months)
    recent_12 = ts.tail(12)
    trend = trend_arrow(recent_12["rank"])
    print(f"  6-month trend:      {trend}")
    print()

    # ── 12-month rank history ─────────────────────────────────────────────────
    print("  RANK HISTORY (last 12 months):")
    history = ts.tail(12)[["date", "rank", "n_stocks", "pct_rank"]].copy()
    header = f"  {'Month':<12} {'Rank':>8}  {'Pct':>7}  {'Badge'}"
    print(header)
    print_separator()
    for _, row in history.iterrows():
        month = row["date"].strftime("%Y-%m")
        rank  = int(row["rank"])
        n     = int(row["n_stocks"])
        pct   = row["pct_rank"]
        badge = rank_badge(pct)
        print(f"  {month:<12} #{rank:>4}/{n:<4}  {pct:>5.1f}%  {badge}")
    print()

    # ── Feature breakdown ─────────────────────────────────────────────────────
    tp = panel[(panel["ticker"] == ticker)].sort_values("date")
    if tp.empty:
        print("  (No feature data available for this ticker)")
        return

    latest_feats = tp.iloc[-1]
    feat_month = latest_feats["date"].strftime("%Y-%m-%d")

    # Feature values are cross-sectionally ranked to [-0.5, +0.5]
    # Convert to percentile: 0 = bottom, 100 = top
    feat_vals = {c: latest_feats[c] for c in feat_cols if c in latest_feats.index and pd.notna(latest_feats[c])}
    feat_series = pd.Series(feat_vals)

    top5    = feat_series.nlargest(5)
    bottom5 = feat_series.nsmallest(5)

    print(f"  SIGNAL BREAKDOWN  (as of {feat_month})")
    print_separator()
    print(f"  {'Feature':<26} {'Rank':>8}  Description")
    print_separator()

    print("  ── STRONGEST signals (bulls):")
    for feat, val in top5.items():
        pct_pos = (val + 0.5) * 100   # convert [-0.5, 0.5] → [0%, 100%]
        label = FEATURE_LABELS.get(feat, feat)
        bar = "█" * int(pct_pos / 10)
        print(f"  {feat:<26} top {pct_pos:4.0f}%  {label}")

    print()
    print("  ── WEAKEST signals (risks/headwinds):")
    for feat, val in bottom5.items():
        pct_pos = (val + 0.5) * 100
        label = FEATURE_LABELS.get(feat, feat)
        print(f"  {feat:<26} bot {100 - pct_pos:4.0f}%  {label}")

    print()
    print_separator("═")


def compare_tickers(tickers: list, scores: pd.DataFrame):
    """Side-by-side comparison table for multiple tickers."""
    latest_month = scores["date"].max()
    latest = scores[scores["date"] == latest_month].copy()

    rows = []
    for t in tickers:
        t = t.upper()
        row = latest[latest["ticker"] == t]
        if row.empty:
            rows.append({"Ticker": t, "Rank": "N/A", "Top %": "N/A", "Score": "N/A", "Badge": "✗ Not found"})
        else:
            r = row.iloc[0]
            rows.append({
                "Ticker": t,
                "Rank":   f"#{int(r['rank'])}/{int(r['n_stocks'])}",
                "Top %":  f"{r['pct_rank']:.1f}%",
                "Score":  f"{r['score']:.4f}",
                "Badge":  rank_badge(r["pct_rank"]),
            })

    df = pd.DataFrame(rows)
    print()
    print_separator("═")
    print(f"  COMPARISON TABLE  —  {latest_month.strftime('%Y-%m-%d')}")
    print_separator("═")
    print(df.to_string(index=False))
    print_separator("═")
    print()


def main():
    args = [a.upper() for a in sys.argv[1:]]

    print("\nLoading model scores and feature panel...")
    try:
        scores, panel = load_data()
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        print("  Run 2_lgbm_backtest.py first to generate scores_lgbm.parquet")
        sys.exit(1)

    feat_cols = get_feature_cols(panel)
    print(f"  Loaded {scores['ticker'].nunique()} stocks, "
          f"{scores['date'].nunique()} months of scores, "
          f"{len(feat_cols)} features.")

    # No ticker given → prompt
    if not args:
        raw = input("\nEnter ticker(s) to analyse (e.g. AAPL  or  AAPL MSFT NVDA): ").strip()
        args = [t.upper() for t in raw.split()]

    if not args:
        print("No ticker provided. Exiting.")
        sys.exit(0)

    # Multiple tickers → show comparison table first, then detailed view for each
    if len(args) > 1:
        compare_tickers(args, scores)

    for ticker in args:
        analyse_ticker(ticker, scores, panel, feat_cols)


if __name__ == "__main__":
    main()
