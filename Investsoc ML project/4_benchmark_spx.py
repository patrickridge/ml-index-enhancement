"""
4_benchmark_spx.py
==================
Compares strategy returns against S&P 500 (^GSPC).
Reads:
  - data/bt_monthly_pca_rp.csv   (long-only PCA-RP from script 3)
  - data/bt_lgbm_ls.csv          (long-short from script 2)
Outputs:
  - data/benchmark_comparison.csv
"""

import numpy as np
import pandas as pd

BT_LO_PATH = "data/bt_monthly_pca_rp.csv"   # long-only
BT_LS_PATH = "data/bt_lgbm_ls.csv"           # long-short
OUT_PATH   = "data/benchmark_comparison.csv"


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
    beta     = np.cov(df.iloc[:, 0], df.iloc[:, 1], ddof=1)[0, 1] / np.var(df.iloc[:, 1], ddof=1)
    alpha_m  = df.iloc[:, 0].mean() - beta * df.iloc[:, 1].mean()
    alpha_ann = (1 + alpha_m) ** 12 - 1
    return beta, alpha_ann


def print_comparison(name: str, port: pd.Series, spx: pd.Series):
    df   = pd.concat([port, spx], axis=1).dropna()
    p    = df.iloc[:, 0]
    b    = df.iloc[:, 1]
    sp   = perf_stats(p)
    sb   = perf_stats(b)
    corr = p.corr(b)
    beta, alpha = beta_alpha(p, b)

    print(f"\n{'=' * 65}")
    print(f"  {name}")
    print(f"{'=' * 65}")
    print(f"  Months aligned : {len(df)}")
    print(f"  {'':20s}  {'Strategy':>10}  {'SPX':>10}")
    print(f"  {'Ann Return':20s}  {sp['ann']*100:>9.2f}%  {sb['ann']*100:>9.2f}%")
    print(f"  {'Volatility':20s}  {sp['vol']*100:>9.2f}%  {sb['vol']*100:>9.2f}%")
    print(f"  {'Sharpe':20s}  {sp['sharpe']:>10.2f}  {sb['sharpe']:>10.2f}")
    print(f"  {'Max Drawdown':20s}  {sp['maxdd']*100:>9.2f}%  {sb['maxdd']*100:>9.2f}%")
    print(f"  {'Correlation':20s}  {corr:>10.2f}")
    print(f"  {'Beta to SPX':20s}  {beta:>10.2f}")
    print(f"  {'Alpha (ann)':20s}  {alpha*100:>9.2f}%")
    return df


def main():
    results = {}

    # ── Long-only (PCA-RP) ────────────────────────────────────────────────────
    lo = pd.read_csv(BT_LO_PATH)
    lo["month"] = pd.to_datetime(lo["month"])
    lo = lo.sort_values("month").set_index("month")
    port_lo = lo["port_ret_m"].rename("long_only")

    # ── Long-short ────────────────────────────────────────────────────────────
    ls = pd.read_csv(BT_LS_PATH)
    ls["date"] = pd.to_datetime(ls["date"])
    ls["date"] = ls["date"].dt.to_period("M").dt.to_timestamp("M")
    ls = ls.sort_values("date").set_index("date")
    port_ls = ls["ls_ret"].rename("long_short")

    # ── Download SPX ──────────────────────────────────────────────────────────
    start = str(min(port_lo.index.min(), port_ls.index.min()).date())
    print(f"Downloading SPX from {start}...")
    spx = get_spx_monthly(start)

    # ── Print comparisons ─────────────────────────────────────────────────────
    df_lo = print_comparison("LGBM + PCA Risk Parity (Long-Only Top 50)", port_lo, spx)
    df_ls = print_comparison("LGBM Long-Short (Top/Bottom 10%)", port_ls, spx)

    # ── Save combined ─────────────────────────────────────────────────────────
    out = pd.concat([
        df_lo.rename(columns={df_lo.columns[0]: "long_only_ret", df_lo.columns[1]: "spx_ret_lo"}),
        df_ls.rename(columns={df_ls.columns[0]: "long_short_ret", df_ls.columns[1]: "spx_ret_ls"}),
    ], axis=1)
    out.to_csv(OUT_PATH)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
