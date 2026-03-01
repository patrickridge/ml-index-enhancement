"""
3_pca_rp_backtest.py
====================
Takes LGBM scores (from 2_lgbm_backtest.py), selects top 50 each month,
then applies PCA-based Risk Parity (Equal Risk Contribution) weighting
using daily price history.

Outputs:
  - data/weights_pca_rp.parquet     : monthly weights
  - data/nav_daily_pca_rp.parquet   : daily NAV
  - data/bt_monthly_pca_rp.csv      : monthly returns

This is the same logic as your old 5.1 but cleaner and using
the enriched/updated model scores.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from config import DATA_DIR, TOP_N, VOL_LOOKBACK, MIN_OBS, N_PCA, RIDGE

SCORES_IN    = DATA_DIR / "scores_lgbm.parquet"
PRICES_IN    = DATA_DIR / "prices.parquet"
OUT_WEIGHTS  = DATA_DIR / "weights_pca_rp.parquet"
OUT_NAV      = DATA_DIR / "nav_daily_pca_rp.parquet"
OUT_BT_M     = DATA_DIR / "bt_monthly_pca_rp.csv"


# ── Performance helpers ───────────────────────────────────────────────────────
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


# ── PCA Risk Parity ───────────────────────────────────────────────────────────
def pca_rp_weights(cov: np.ndarray, n_pca: int = 10, ridge: float = 1e-3) -> np.ndarray:
    """
    Approximate PCA risk parity:
      1. Ridge-stabilise covariance
      2. Eigen-decompose, keep top k components
      3. Assign inverse-sqrt-eigenvalue loading in PC space
      4. Map back to asset space, clip negatives, renormalise
    """
    n = cov.shape[0]
    cov = cov.copy() + ridge * np.eye(n)

    vals, vecs = np.linalg.eigh(cov)
    # sort descending
    idx  = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    k  = min(n_pca, n)
    Vk = vecs[:, :k]
    Lk = np.maximum(vals[:k], 1e-12)

    # equal risk per factor: b_j ∝ 1/sqrt(lambda_j)
    b = 1.0 / np.sqrt(Lk)
    w = Vk @ b            # back to asset space

    w = np.maximum(w, 0.0)
    s = w.sum()
    if s <= 0:
        return np.ones(n) / n
    return w / s


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading scores and prices...")
    scores = pd.read_parquet(SCORES_IN)
    scores["date"] = pd.to_datetime(scores["date"])

    prices = pd.read_parquet(PRICES_IN, columns=["date", "ticker", "close"])
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices["ret_d"] = prices.groupby("ticker")["close"].pct_change()
    prices = prices.dropna(subset=["ret_d"])

    test_months = sorted(scores["date"].unique())
    print(f"Test months: {len(test_months)} | {test_months[0].date()} -> {test_months[-1].date()}")

    weights_list  = []
    nav_daily_list = []
    nav = 1.0

    for m in test_months:
        # ── 1. Select top N by LGBM score ────────────────────────────────────
        sub = scores[scores["date"] == m].dropna(subset=["score"])
        if len(sub) < TOP_N:
            continue
        tickers = sub.nlargest(TOP_N, "score")["ticker"].tolist()

        # ── 2. Build covariance from daily returns ────────────────────────────
        end_date   = m
        start_date = end_date - pd.tseries.offsets.BDay(int(VOL_LOOKBACK * 1.5))

        hist = prices[
            (prices["ticker"].isin(tickers)) &
            (prices["date"] > start_date) &
            (prices["date"] <= end_date)
        ][["date", "ticker", "ret_d"]]

        mat = hist.pivot(index="date", columns="ticker", values="ret_d")

        # filter tickers with enough observations
        good     = mat.count(axis=0)
        keep     = good[good >= MIN_OBS].index.tolist()
        if len(keep) < max(10, TOP_N // 3):
            keep = mat.columns.tolist()   # fallback: use all available

        mat = mat[keep].fillna(0.0)
        if mat.shape[0] < MIN_OBS or mat.shape[1] < 5:
            continue

        cov = np.cov(mat.values, rowvar=False)

        # ── 3. PCA-RP weights ─────────────────────────────────────────────────
        w_arr = pca_rp_weights(cov, n_pca=N_PCA, ridge=RIDGE)
        w     = pd.Series(w_arr, index=mat.columns)

        # store weights
        w_df = w.reset_index()
        w_df.columns = ["ticker", "weight"]
        w_df.insert(0, "date", m)
        weights_list.append(w_df)

        # ── 4. Apply weights to NEXT month's daily returns ────────────────────
        next_m = m + pd.offsets.MonthEnd(1)
        # find the actual month label for next month's daily data
        nxt = prices[
            (prices["date"] > m) &
            (prices["date"] <= next_m) &
            (prices["ticker"].isin(w.index))
        ][["date", "ticker", "ret_d"]].copy()

        if nxt.empty:
            continue

        nxt = nxt.merge(w.rename("weight").reset_index().rename(columns={"index": "ticker"}),
                        on="ticker", how="inner")
        nxt["wret"] = nxt["weight"] * nxt["ret_d"]

        port_d = (nxt.groupby("date")["wret"]
                     .sum()
                     .rename("port_ret_d")
                     .reset_index()
                     .sort_values("date"))

        port_d["nav"] = nav * (1 + port_d["port_ret_d"]).cumprod()
        nav = float(port_d["nav"].iloc[-1])
        nav_daily_list.append(port_d)

    if not weights_list or not nav_daily_list:
        raise RuntimeError("No output produced — check date alignment between scores and prices.")

    # ── Save ──────────────────────────────────────────────────────────────────
    weights_all = pd.concat(weights_list, ignore_index=True)
    nav_daily   = pd.concat(nav_daily_list, ignore_index=True)

    weights_all.to_parquet(OUT_WEIGHTS, index=False)
    nav_daily.to_parquet(OUT_NAV, index=False)

    # Monthly returns from NAV
    nav_daily["month"] = nav_daily["date"].dt.to_period("M").dt.to_timestamp("M")
    nav_m = (nav_daily.groupby("month", as_index=False)
                       .agg(nav=("nav", "last")))
    nav_m["port_ret_m"] = nav_m["nav"].pct_change()
    nav_m = nav_m.dropna(subset=["port_ret_m"])
    nav_m.to_csv(OUT_BT_M, index=False)

    s = perf_stats(nav_m["port_ret_m"])
    print(f"\nSaved weights:      {OUT_WEIGHTS}")
    print(f"Saved daily NAV:    {OUT_NAV}  ({len(nav_daily):,} rows)")
    print(f"Saved monthly bt:   {OUT_BT_M}  ({s['months']} months)")
    print("\n" + "=" * 65)
    print("LGBM + PCA Risk Parity (Long-Only Top 50)")
    print("=" * 65)
    print(f"ann={s['ann']*100:.2f}% | vol={s['vol']*100:.2f}% | "
          f"sharpe={s['sharpe']:.2f} | maxdd={s['maxdd']*100:.2f}%")


if __name__ == "__main__":
    main()
