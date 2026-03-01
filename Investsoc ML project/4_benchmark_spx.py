"""
4_benchmark_spx.py
==================
Compares all strategy returns against S&P 500 (^GSPC).

Strategies benchmarked:
  - LGBM Long-Only (Top 50 equal weight)
  - LGBM Long-Short (Top/Bottom 10%)
  - FT-Transformer Long-Only
  - FT-Transformer Long-Short
  - LGBM + PCA-RP (experimental — known beta issue)

Outputs:
  - data/benchmark_comparison.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

from config import DATA_DIR

OUT_PATH = DATA_DIR / "benchmark_comparison.csv"

# ── Strategy registry ──────────────────────────────────────────────────────────
# (display label, csv filename, return column, warning note)
STRATEGIES = [
    ("LGBM Long-Only (Top 50)",        "bt_lgbm.csv",             "port_ret",   ""),
    ("LGBM Long-Short (Top/Bot 10%)",  "bt_lgbm_ls.csv",          "ls_ret",     ""),
    ("Transformer Long-Only",          "bt_transformer.csv",       "port_ret",   ""),
    ("Transformer Long-Short",         "bt_transformer_ls.csv",    "ls_ret",     ""),
    ("LGBM + PCA-RP (experimental)",   "bt_monthly_pca_rp.csv",   "port_ret_m", "⚠ beta>2, broken weights"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_spx_monthly(start_date: str) -> pd.Series:
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Run: pip install yfinance")

    spx = yf.download("^GSPC", start=start_date, auto_adjust=True, progress=False)
    if spx.empty:
        raise RuntimeError("Could not download ^GSPC from yfinance.")

    spx["ret_d"] = spx["Close"].pct_change()
    spx_m = spx["ret_d"].resample("M").apply(lambda x: (1 + x).prod() - 1)
    spx_m.index = spx_m.index.to_period("M").to_timestamp("M")
    spx_m.name = "spx_ret"
    return spx_m


def perf_stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 6:
        return dict(months=len(r), ann=np.nan, vol=np.nan, sharpe=np.nan, maxdd=np.nan)
    ann    = (1 + r).prod() ** (12 / len(r)) - 1
    vol    = r.std(ddof=1) * np.sqrt(12)
    sharpe = ann / vol if vol > 0 else np.nan
    nav    = (1 + r).cumprod()
    maxdd  = (nav / nav.cummax() - 1).min()
    return dict(months=len(r), ann=ann, vol=vol, sharpe=sharpe, maxdd=maxdd)


def beta_alpha(port: pd.Series, bm: pd.Series):
    df = pd.concat([port, bm], axis=1).dropna()
    if len(df) < 12 or df.iloc[:, 1].var(ddof=1) == 0:
        return np.nan, np.nan
    beta    = np.cov(df.iloc[:, 0], df.iloc[:, 1], ddof=1)[0, 1] / np.var(df.iloc[:, 1], ddof=1)
    alpha_m = df.iloc[:, 0].mean() - beta * df.iloc[:, 1].mean()
    return beta, (1 + alpha_m) ** 12 - 1


def load_strategy(filename: str, ret_col: str) -> pd.Series | None:
    """Load a backtest CSV and return a monthly return Series, or None if file missing."""
    path = DATA_DIR / filename
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # normalise date column — first column is always date
    df["_date"] = pd.to_datetime(df.iloc[:, 0])
    df["_date"] = df["_date"].dt.to_period("M").dt.to_timestamp("M")
    df = df.sort_values("_date").set_index("_date")
    if ret_col not in df.columns:
        return None
    return df[ret_col].dropna()


def print_comparison(label: str, port: pd.Series, spx: pd.Series) -> dict:
    """Print and return stats dict for one strategy vs SPX."""
    aligned  = pd.concat([port, spx], axis=1).dropna()
    p        = aligned.iloc[:, 0]
    b        = aligned.iloc[:, 1]
    sp       = perf_stats(p)
    sb       = perf_stats(b)
    corr     = p.corr(b)
    beta, alpha = beta_alpha(p, b)

    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"{'=' * 65}")
    print(f"  Months aligned : {len(aligned)}")
    print(f"  {'':20s}  {'Strategy':>10}  {'SPX':>10}")
    print(f"  {'Ann Return':20s}  {sp['ann']*100:>9.2f}%  {sb['ann']*100:>9.2f}%")
    print(f"  {'Volatility':20s}  {sp['vol']*100:>9.2f}%  {sb['vol']*100:>9.2f}%")
    print(f"  {'Sharpe':20s}  {sp['sharpe']:>10.2f}  {sb['sharpe']:>10.2f}")
    print(f"  {'Max Drawdown':20s}  {sp['maxdd']*100:>9.2f}%  {sb['maxdd']*100:>9.2f}%")
    print(f"  {'Correlation':20s}  {corr:>10.2f}")
    print(f"  {'Beta to SPX':20s}  {beta:>10.2f}")
    print(f"  {'Alpha (ann)':20s}  {alpha*100:>9.2f}%")

    return dict(
        label=label,
        months=len(aligned),
        ann_pct=sp["ann"] * 100,
        vol_pct=sp["vol"] * 100,
        sharpe=sp["sharpe"],
        maxdd_pct=sp["maxdd"] * 100,
        corr=corr,
        beta=beta,
        alpha_pct=alpha * 100,
    )


def main():
    # ── Find earliest start date across all available strategies ──────────────
    all_starts = []
    series_map = {}
    for label, filename, ret_col, note in STRATEGIES:
        s = load_strategy(filename, ret_col)
        if s is not None:
            series_map[label] = (s, note)
            all_starts.append(s.index.min())
        else:
            print(f"Skipping '{label}' — {filename} not found")

    if not all_starts:
        print("No strategy files found. Run scripts 2, 2b, 3 first.")
        return

    start = str(min(all_starts).date())
    print(f"Downloading SPX from {start}...")
    spx = get_spx_monthly(start)

    # ── Print individual comparisons ──────────────────────────────────────────
    summary_rows = []
    for label, (port, note) in series_map.items():
        display = label + (f"  [{note}]" if note else "")
        row = print_comparison(display, port, spx)
        row["label"] = label   # clean label for CSV
        summary_rows.append(row)

    # ── Summary ranking table ─────────────────────────────────────────────────
    # Add SPX itself as a row
    spx_stats = perf_stats(spx)
    summary_rows.append(dict(
        label="S&P 500 (benchmark)",
        months=spx_stats["months"],
        ann_pct=spx_stats["ann"] * 100,
        vol_pct=spx_stats["vol"] * 100,
        sharpe=spx_stats["sharpe"],
        maxdd_pct=spx_stats["maxdd"] * 100,
        corr=1.0,
        beta=1.0,
        alpha_pct=0.0,
    ))

    df_summary = (
        pd.DataFrame(summary_rows)
        .set_index("label")
        .sort_values("sharpe", ascending=False)
    )

    print(f"\n{'=' * 75}")
    print("  SUMMARY RANKING  (sorted by Sharpe, highest → lowest)")
    print(f"{'=' * 75}")
    print(f"  {'Strategy':<38} {'Sharpe':>7}  {'Ann Ret':>8}  {'Beta':>6}  {'Alpha':>8}  {'MaxDD':>8}")
    print(f"  {'-' * 73}")
    for lbl, row in df_summary.iterrows():
        sharpe = f"{row['sharpe']:.2f}" if pd.notna(row["sharpe"]) else "  n/a"
        ann    = f"{row['ann_pct']:.1f}%"
        beta   = f"{row['beta']:.2f}"   if pd.notna(row["beta"])   else "  n/a"
        alpha  = f"{row['alpha_pct']:+.1f}%" if pd.notna(row["alpha_pct"]) else "  n/a"
        maxdd  = f"{row['maxdd_pct']:.1f}%"
        print(f"  {lbl:<38} {sharpe:>7}  {ann:>8}  {beta:>6}  {alpha:>8}  {maxdd:>8}")
    print(f"{'=' * 75}")

    # ── Save ──────────────────────────────────────────────────────────────────
    df_summary.to_csv(OUT_PATH)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
