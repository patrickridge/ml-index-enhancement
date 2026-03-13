"""
2_lgbm_backtest.py
==================
Trains LightGBM on enriched panel (from 1_feature_engineering.py).
Uses walk-forward expanding window (retrain every 12 months).
Outputs:
  - data/scores_lgbm.parquet     : per-stock monthly scores for test period
  - data/bt_lgbm.csv             : monthly long-only backtest returns (top 50)
  - data/bt_lgbm_ls.csv          : monthly long-short backtest returns

Key changes vs old scripts:
  - Uses panel_monthly_enriched.parquet (~25 features vs 10)
  - Data starts 2010 (post-GFC, more consistent regime)
  - Train: 2010-2017 (~96 months), Valid: 2018-2019 (24m), Test: 2020+ (~60m)
  - Walk-forward: refit model every 12 months using expanding window
  - Proper early stopping on validation set each fold
"""

import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
from lightgbm import LGBMRegressor

from config import (
    DATA_DIR, START_DATE, TRAIN_END, VALID_END,
    TOP_N, BOTTOM_N, LONG_FRAC, RETRAIN_EVERY, LGBM_PARAMS,
    USE_ORTHOGONALIZED_FEATURES,
)
import time as _time; _t0 = _time.time()

_panel_file = (
    "panel_monthly_orthogonalized.parquet"
    if USE_ORTHOGONALIZED_FEATURES
    else "panel_monthly_enriched.parquet"
)
PANEL_IN  = DATA_DIR / _panel_file
OUT_SCORES = DATA_DIR / "scores_lgbm.parquet"
OUT_BT_LO  = DATA_DIR / "bt_lgbm.csv"        # long-only top 50
OUT_BT_LS  = DATA_DIR / "bt_lgbm_ls.csv"     # long-short


# ── Helpers ───────────────────────────────────────────────────────────────────
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


def print_stats(label: str, r: pd.Series):
    s = perf_stats(r)
    print(f"{label}: months={s['months']} | ann={s['ann']*100:.2f}% | "
          f"vol={s['vol']*100:.2f}% | sharpe={s['sharpe']:.2f} | maxdd={s['maxdd']*100:.2f}%")


def long_only_ret(df_month: pd.DataFrame, top_n: int) -> float:
    """Equal-weight long-only return for top_n stocks in one month."""
    sub = df_month.dropna(subset=["score", "fwd_ret_1m"])
    if len(sub) < top_n:
        return np.nan
    top = sub.nlargest(top_n, "score")
    return top["fwd_ret_1m"].mean()


def long_short_ret(df_month: pd.DataFrame, frac: float) -> float:
    """Long top frac, short bottom frac. Returns net L/S return."""
    sub = df_month.dropna(subset=["score", "fwd_ret_1m"])
    n = len(sub)
    if n < 20:
        return np.nan
    k = max(1, int(np.floor(n * frac)))
    sub_s = sub.sort_values("score", ascending=False)
    long_ret  = sub_s.head(k)["fwd_ret_1m"].mean()
    short_ret = sub_s.tail(k)["fwd_ret_1m"].mean()
    return long_ret - short_ret


def fit_model(X_tr, y_tr, X_va, y_va):
    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="l2",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=500),
        ],
    )
    return model


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading enriched panel...")
    panel = pd.read_parquet(PANEL_IN)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel["date"] >= START_DATE].reset_index(drop=True)

    exclude   = {"date", "ticker", "fwd_ret_1m"}
    feat_cols = [c for c in panel.columns if c not in exclude]
    print(f"Features: {len(feat_cols)} | Rows: {len(panel):,} | Tickers: {panel['ticker'].nunique()}")

    # ── Time splits ──────────────────────────────────────────────────────────
    months = sorted(panel["date"].unique())
    train_end  = pd.Timestamp(TRAIN_END)
    valid_end  = pd.Timestamp(VALID_END)

    train_months = [m for m in months if m <= train_end]
    valid_months = [m for m in months if train_end < m <= valid_end]
    test_months  = [m for m in months if m > valid_end]

    print(f"Train: {train_months[0].date()} -> {train_months[-1].date()} ({len(train_months)} months)")
    print(f"Valid: {valid_months[0].date()} -> {valid_months[-1].date()} ({len(valid_months)} months)")
    print(f"Test:  {test_months[0].date()} -> {test_months[-1].date()} ({len(test_months)} months)")

    tr = panel[panel["date"].isin(train_months)].dropna(subset=feat_cols + ["fwd_ret_1m"])
    va = panel[panel["date"].isin(valid_months)].dropna(subset=feat_cols + ["fwd_ret_1m"])

    # ── Walk-forward prediction ───────────────────────────────────────────────
    # For first fold: fit on train, validated on valid
    # Every RETRAIN_EVERY months: expand train to include seen test data, refit
    print("\nFitting initial model...")
    model = fit_model(
        tr[feat_cols].values, tr["fwd_ret_1m"].values,
        va[feat_cols].values, va["fwd_ret_1m"].values,
    )
    print(f"  Best iteration: {model.best_iteration_}")

    all_scores = []

    for i, m in enumerate(test_months):
        # Retrain every RETRAIN_EVERY months using expanding window
        if i > 0 and i % RETRAIN_EVERY == 0:
            # expand training to include all months up to (not including) current
            seen_months = train_months + valid_months + test_months[:i]
            tr_exp = panel[panel["date"].isin(seen_months)].dropna(subset=feat_cols + ["fwd_ret_1m"])
            # use last RETRAIN_EVERY months as expanding valid proxy
            va_new = panel[panel["date"].isin(test_months[max(0,i-RETRAIN_EVERY):i])].dropna(subset=feat_cols + ["fwd_ret_1m"])
            if len(va_new) > 0:
                print(f"  Retraining at month {i} ({m.date()})...")
                model = fit_model(
                    tr_exp[feat_cols].values, tr_exp["fwd_ret_1m"].values,
                    va_new[feat_cols].values,  va_new["fwd_ret_1m"].values,
                )

        sub = panel[panel["date"] == m].copy()
        if len(sub) == 0:
            continue

        sub["score"] = model.predict(sub[feat_cols].fillna(0).values,
                                     num_iteration=model.best_iteration_)
        all_scores.append(sub[["date", "ticker", "score", "fwd_ret_1m"]])

    scores = pd.concat(all_scores, ignore_index=True)
    scores.to_parquet(OUT_SCORES, index=False)
    print(f"\nSaved scores: {OUT_SCORES} | rows={len(scores):,}")

    # ── Backtest: long-only top 50 ────────────────────────────────────────────
    lo_rets = scores.groupby("date").apply(long_only_ret, top_n=TOP_N).rename("port_ret")
    lo_rets = lo_rets.dropna().reset_index()
    lo_rets.to_csv(OUT_BT_LO, index=False)

    # ── Backtest: long-short ──────────────────────────────────────────────────
    ls_rets = scores.groupby("date").apply(long_short_ret, frac=LONG_FRAC).rename("ls_ret")
    ls_rets = ls_rets.dropna().reset_index()
    ls_rets.to_csv(OUT_BT_LS, index=False)

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("TEST PERIOD RESULTS")
    print("=" * 65)
    print_stats(f"Long-only Top{TOP_N} (equal weight)", lo_rets["port_ret"])
    print_stats(f"Long-Short top/bot {int(LONG_FRAC*100)}%", ls_rets["ls_ret"])

    # Feature importance
    fi = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print("\nTop 15 feature importances:")
    print(fi.head(15).to_string())
    print(f"\nDone in {(_time.time() - _t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
