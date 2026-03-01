# 7_diagnostics_lgbm_pca_rp.py
# Full diagnostics for: LGBM score + PCA Risk Parity
# Outputs:
#   - data/diag_summary_lgbm_pca_rp.txt
#   - data/diag_turnover_lgbm_pca_rp.csv
#   - data/diag_rank_decay_lgbm_pca_rp.csv
#   - data/diag_pca_cov_lgbm_pca_rp.csv
#   - data/diag_strategy_compare_lgbm_pca_rp.csv

import os
import warnings
import numpy as np
import pandas as pd

from typing import Dict, Tuple, List

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")


# -----------------------------
# Config
# -----------------------------
DATA_DIR = "data"
PANEL_PATH = os.path.join(DATA_DIR, "panel_monthly_enriched.parquet")
PRICES_PATH = os.path.join(DATA_DIR, "prices.parquet")

OUT_SUMMARY_TXT = os.path.join(DATA_DIR, "diag_summary_lgbm_pca_rp.txt")
OUT_TURNOVER = os.path.join(DATA_DIR, "diag_turnover_lgbm_pca_rp.csv")
OUT_RANK_DECAY = os.path.join(DATA_DIR, "diag_rank_decay_lgbm_pca_rp.csv")
OUT_PCA_COV = os.path.join(DATA_DIR, "diag_pca_cov_lgbm_pca_rp.csv")
OUT_COMPARE = os.path.join(DATA_DIR, "diag_strategy_compare_lgbm_pca_rp.csv")

# LGBM training split (from your earlier: total=239 months)
N_TRAIN = 144
N_VALID = 36

# Portfolio construction
TOP_N_DEFAULT = 50
TOP_N_ALT = 10
BOTTOM_N_ALT = 50
BOTTOM_N_ALT2 = 10

# PCA covariance diagnostics
DO_PCA_COV_DIAGNOSTICS = True     # set False if too slow
COV_LOOKBACK_DAYS = 252           # ~1Y
COV_MIN_OBS_PER_ASSET = 95        # like you said "more serious"
PCA_VAR_THRESHOLD = 0.95          # keep components up to 95% explained variance
COV_MONTH_STEP = 1                # 1 = every month; 2 = every 2 months, etc. to speed up


# -----------------------------
# Helpers: performance metrics
# -----------------------------
def perf_stats(returns_m: pd.Series) -> Dict[str, float]:
    """Monthly returns series -> ann, vol, sharpe, maxDD."""
    r = returns_m.dropna().astype(float)
    if len(r) == 0:
        return {"ann": np.nan, "vol": np.nan, "sharpe": np.nan, "maxDD": np.nan}

    ann = (1 + r).prod() ** (12 / len(r)) - 1
    vol = r.std(ddof=1) * np.sqrt(12)

    sharpe = np.nan
    if vol and vol > 0:
        sharpe = ann / vol

    nav = (1 + r).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1.0
    maxdd = dd.min()

    return {"ann": float(ann), "vol": float(vol), "sharpe": float(sharpe), "maxDD": float(maxdd)}


def format_stats(name: str, s: Dict[str, float]) -> str:
    return (
        f"{name}: ann={s['ann']*100:6.2f}% | vol={s['vol']*100:6.2f}% | "
        f"sharpe={s['sharpe']:5.2f} | maxDD={s['maxDD']*100:6.2f}%"
    )


# -----------------------------
# Helpers: PCA denoise covariance
# -----------------------------
def pca_denoise_cov(cov: np.ndarray, var_threshold: float = 0.95) -> Tuple[np.ndarray, int, float]:
    """
    Denoise a covariance matrix by keeping top PCs up to var_threshold (based on eigenvalues).
    Returns: (cov_denoised, k, explained_ratio_k)
    """
    # Symmetrize
    cov = 0.5 * (cov + cov.T)

    # Eigen-decomp (cov is PSD-ish; use eigh)
    vals, vecs = np.linalg.eigh(cov)
    # sort descending
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    total = np.sum(vals)
    if total <= 0 or not np.isfinite(total):
        return cov.copy(), 0, np.nan

    cum = np.cumsum(vals) / total
    k = int(np.searchsorted(cum, var_threshold) + 1)
    k = max(1, min(k, cov.shape[0]))

    vals_k = vals[:k]
    vecs_k = vecs[:, :k]

    cov_k = vecs_k @ np.diag(vals_k) @ vecs_k.T

    # keep original diagonal to preserve individual vol scale (optional but common)
    cov_dn = cov_k.copy()
    np.fill_diagonal(cov_dn, np.diag(cov))

    explained = float(cum[k - 1])
    return cov_dn, k, explained


def cov_condition_number(cov: np.ndarray) -> float:
    cov = 0.5 * (cov + cov.T)
    vals = np.linalg.eigvalsh(cov)
    vals = np.sort(vals)
    # avoid 0 / negative
    eps = 1e-12
    vmax = max(vals[-1], eps)
    vmin = max(vals[0], eps)
    return float(vmax / vmin)


def fro_norm(a: np.ndarray) -> float:
    return float(np.linalg.norm(a, ord="fro"))


# -----------------------------
# Load data
# -----------------------------
def load_panel() -> pd.DataFrame:
    df = pd.read_parquet(PANEL_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Identify features (exclude target + identifiers)
    must_have = {"date", "ticker", "fwd_ret_1m"}
    missing = must_have - set(df.columns)
    if missing:
        raise ValueError(f"panel_monthly missing columns: {missing}")

    return df


def infer_feature_cols(panel: pd.DataFrame) -> List[str]:
    exclude = {"date", "ticker", "fwd_ret_1m"}
    # If you have extra leakage columns, add here.
    # e.g. exclude |= {"some_future_feature"}
    feats = [c for c in panel.columns if c not in exclude]
    return feats


# -----------------------------
# Train LGBM & make predictions
# -----------------------------
def train_lgbm(panel: pd.DataFrame, feature_cols: List[str]) -> Tuple[LGBMRegressor, pd.DataFrame, Dict]:
    months = pd.Index(sorted(panel["date"].unique()))
    if len(months) < (N_TRAIN + N_VALID + 1):
        raise ValueError(f"Not enough months. got={len(months)} need>={N_TRAIN+N_VALID+1}")

    train_months = months[:N_TRAIN]
    valid_months = months[N_TRAIN:N_TRAIN + N_VALID]
    test_months = months[N_TRAIN + N_VALID:]

    tr = panel[panel["date"].isin(train_months)].copy()
    va = panel[panel["date"].isin(valid_months)].copy()
    te = panel[panel["date"].isin(test_months)].copy()

    X_tr, y_tr = tr[feature_cols], tr["fwd_ret_1m"]
    X_va, y_va = va[feature_cols], va["fwd_ret_1m"]

    model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    # Use callbacks for early stopping (avoid 'verbose' arg issue)
    import lightgbm as lgb
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="l2",
        callbacks=[
            lgb.early_stopping(stopping_rounds=80, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    # Predictions for VALID/TEST too (for diagnostics)
    for part, d in [("train", tr), ("valid", va), ("test", te)]:
        d["score"] = model.predict(d[feature_cols])

    # Quick fit sanity metrics (MSE)
    mse_tr = mean_squared_error(tr["fwd_ret_1m"], tr["score"])
    mse_va = mean_squared_error(va["fwd_ret_1m"], va["score"])
    mse_te = mean_squared_error(te["fwd_ret_1m"], te["score"])

    split_info = {
        "train_months": train_months,
        "valid_months": valid_months,
        "test_months": test_months,
        "mse_train": float(mse_tr),
        "mse_valid": float(mse_va),
        "mse_test": float(mse_te),
    }
    scored = pd.concat([tr, va, te], ignore_index=True)
    return model, scored, split_info


# -----------------------------
# Portfolio construction (monthly using fwd_ret_1m)
# -----------------------------
def select_top_bottom(df_m: pd.DataFrame, top_n: int, bottom_n: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_m = df_m.sort_values("score", ascending=False)
    top = df_m.head(top_n).copy()
    bottom = df_m.tail(bottom_n).copy() if bottom_n > 0 else df_m.iloc[0:0].copy()
    return top, bottom


def build_pca_rp_weights(prices_ret_window: pd.DataFrame) -> pd.Series:
    """
    prices_ret_window: DataFrame indexed by date, columns=tickers, values=daily returns
    Output: weights indexed by tickers, sum=1 (long-only RP proxy on denoised cov)
    We use inverse-vol as initial and solve equal-risk approximately by iterative scaling.
    (Fast, stable, good enough for diagnostics.)
    """
    R = prices_ret_window.dropna(axis=1, how="all").copy()
    # require enough obs per asset
    ok = (R.notna().sum(axis=0) >= COV_MIN_OBS_PER_ASSET)
    R = R.loc[:, ok]
    if R.shape[1] < 2:
        # fallback equal weight
        cols = prices_ret_window.columns
        w = pd.Series(1.0 / len(cols), index=cols)
        return w

    # fill remaining NaNs with 0 (assume missing return = 0 that day) for covariance
    R = R.fillna(0.0)

    cov_raw = np.cov(R.values, rowvar=False)
    cov_dn, _, _ = pca_denoise_cov(cov_raw, PCA_VAR_THRESHOLD)

    # inverse-vol init
    vol = np.sqrt(np.clip(np.diag(cov_dn), 1e-12, None))
    w = 1.0 / vol
    w = w / w.sum()

    # iterative risk-budgeting (simple heuristic)
    # risk contribution RC_i = w_i * (Sigma w)_i / (w' Sigma w)
    Sigma = cov_dn
    for _ in range(200):
        Sw = Sigma @ w
        port_var = float(w.T @ Sw)
        if port_var <= 0:
            break
        rc = w * Sw / port_var
        target = 1.0 / len(w)
        # multiplicative update
        adj = target / np.clip(rc, 1e-12, None)
        w = w * adj
        w = np.clip(w, 1e-12, None)
        w = w / w.sum()

    return pd.Series(w, index=R.columns)


def make_monthly_returns_from_weights(
    scored_panel: pd.DataFrame,
    weights_by_month: Dict[pd.Timestamp, pd.Series],
) -> pd.Series:
    """
    Use fwd_ret_1m to compute portfolio monthly return:
      ret_m = sum_i w_i * fwd_ret_1m_i
    """
    rets = {}
    for m, w in weights_by_month.items():
        sub = scored_panel[scored_panel["date"] == m].set_index("ticker")
        common = sub.index.intersection(w.index)
        if len(common) == 0:
            continue
        r = float(np.sum(w.loc[common].values * sub.loc[common, "fwd_ret_1m"].values))
        rets[m] = r
    return pd.Series(rets).sort_index()


# -----------------------------
# Turnover diagnostics
# -----------------------------
def turnover_table(weights_by_month: Dict[pd.Timestamp, pd.Series], top_n: int) -> pd.DataFrame:
    months = sorted(weights_by_month.keys())
    rows = []
    prev_w = None
    prev_names = None

    for m in months:
        w = weights_by_month[m].copy()
        w = w / w.sum()
        names = set(w.index.tolist())

        row = {"date": m, "n": len(w), "max_w": float(w.max()), "hhi": float((w**2).sum())}

        if prev_w is None:
            row.update({"name_overlap": np.nan, "name_turnover": np.nan, "weight_turnover": np.nan, "half_turnover": np.nan})
        else:
            overlap = len(names.intersection(prev_names)) / max(1, len(names))
            name_turn = 1 - overlap

            # align weights
            all_names = sorted(list(names.union(prev_names)))
            w_now = w.reindex(all_names).fillna(0.0)
            w_prev = prev_w.reindex(all_names).fillna(0.0)

            wt_turn = float(np.sum(np.abs(w_now - w_prev)))
            half_turn = 0.5 * wt_turn

            row.update({
                "name_overlap": float(overlap),
                "name_turnover": float(name_turn),
                "weight_turnover": wt_turn,
                "half_turnover": half_turn,
            })

        rows.append(row)
        prev_w = w
        prev_names = names

    df = pd.DataFrame(rows)
    df["top_n"] = top_n
    return df


# -----------------------------
# Rank decay diagnostics
# -----------------------------
def rank_decay_table(scored_panel: pd.DataFrame, test_months: pd.Index, top_n: int = 50) -> pd.DataFrame:
    """
    For each formation month t:
      - take top_n by score at t
      - compute average forward return at t+1, t+2, t+3 (using fwd_ret_1m on those future months, same names)
    """
    months = pd.Index(sorted(test_months))
    m2pos = {m: i for i, m in enumerate(months)}
    rows = []

    for m in months:
        pos = m2pos[m]
        sub = scored_panel[scored_panel["date"] == m].copy()
        sub = sub.sort_values("score", ascending=False)
        top = sub.head(top_n).copy()
        names = top["ticker"].tolist()

        def avg_future(k: int) -> float:
            if pos + k >= len(months):
                return np.nan
            m_f = months[pos + k]
            fut = scored_panel[(scored_panel["date"] == m_f) & (scored_panel["ticker"].isin(names))]
            if len(fut) == 0:
                return np.nan
            return float(fut["fwd_ret_1m"].mean())

        rows.append({
            "date": m,
            "top_n": top_n,
            "avg_fwd_ret_t1": avg_future(1),
            "avg_fwd_ret_t2": avg_future(2),
            "avg_fwd_ret_t3": avg_future(3),
        })

    return pd.DataFrame(rows).sort_values("date")


# -----------------------------
# PCA covariance diagnostics (raw vs denoised)
# -----------------------------
def load_prices_close_for_tickers(start_date: pd.Timestamp, end_date: pd.Timestamp, tickers: List[str]) -> pd.DataFrame:
    """
    Load close prices from prices.parquet (big). We only read columns we need.
    """
    prices = pd.read_parquet(PRICES_PATH, columns=["date", "ticker", "close"])
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[(prices["date"] >= start_date) & (prices["date"] <= end_date)]
    prices = prices[prices["ticker"].isin(tickers)]
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    return prices


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_values(["ticker", "date"])
    prices["ret_d"] = prices.groupby("ticker")["close"].pct_change()
    return prices.dropna(subset=["ret_d"])


def month_end(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts).to_period("M").to_timestamp("M")


def pca_cov_diagnostics(
    prices_parquet_path: str,
    weights_by_month: Dict[pd.Timestamp, pd.Series],
) -> pd.DataFrame:
    """
    For each month:
      - build daily return matrix (lookback 252d) for that month's portfolio names
      - compute raw cov and PCA-denoised cov
      - metrics: condition number, stability vs prev month, explained var, k PCs
    """
    months = sorted(weights_by_month.keys())
    if COV_MONTH_STEP > 1:
        months = months[::COV_MONTH_STEP]

    # gather tickers used in months for efficient read
    all_tickers = sorted(list({t for m in months for t in weights_by_month[m].index}))
    if len(all_tickers) == 0:
        return pd.DataFrame()

    # date range needed
    start = min(months) - pd.Timedelta(days=int(COV_LOOKBACK_DAYS * 2.2))
    end = max(months)

    prices = pd.read_parquet(prices_parquet_path, columns=["date", "ticker", "close"])
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[(prices["date"] >= start) & (prices["date"] <= end)]
    prices = prices[prices["ticker"].isin(all_tickers)]
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    prices = compute_daily_returns(prices)
    prices["month"] = prices["date"].dt.to_period("M").dt.to_timestamp("M")

    rows = []
    prev_cov_raw = None
    prev_cov_dn = None

    for m in months:
        names = list(weights_by_month[m].index)
        # returns up to month-end m
        sub = prices[(prices["date"] <= m) & (prices["ticker"].isin(names))].copy()
        if len(sub) == 0:
            rows.append({"date": m, "n": len(names), "raw_cond": np.nan, "dn_cond": np.nan, "k": np.nan, "explained": np.nan,
                         "raw_stability": np.nan, "dn_stability": np.nan})
            prev_cov_raw, prev_cov_dn = None, None
            continue

        # Use last lookback window by date
        last_date = sub["date"].max()
        start_w = last_date - pd.Timedelta(days=int(COV_LOOKBACK_DAYS * 1.8))
        sub = sub[sub["date"] >= start_w]

        # pivot date x ticker
        R = sub.pivot(index="date", columns="ticker", values="ret_d").sort_index()
        # enforce min obs
        ok = (R.notna().sum(axis=0) >= COV_MIN_OBS_PER_ASSET)
        R = R.loc[:, ok]

        if R.shape[1] < 2:
            rows.append({"date": m, "n": int(R.shape[1]), "raw_cond": np.nan, "dn_cond": np.nan, "k": np.nan, "explained": np.nan,
                         "raw_stability": np.nan, "dn_stability": np.nan})
            prev_cov_raw, prev_cov_dn = None, None
            continue

        R = R.fillna(0.0)
        cov_raw = np.cov(R.values, rowvar=False)
        cov_dn, k, explained = pca_denoise_cov(cov_raw, PCA_VAR_THRESHOLD)

        raw_cond = cov_condition_number(cov_raw)
        dn_cond = cov_condition_number(cov_dn)

        raw_stab = np.nan
        dn_stab = np.nan
        if prev_cov_raw is not None and prev_cov_raw.shape == cov_raw.shape:
            raw_stab = fro_norm(cov_raw - prev_cov_raw) / (fro_norm(prev_cov_raw) + 1e-12)
        if prev_cov_dn is not None and prev_cov_dn.shape == cov_dn.shape:
            dn_stab = fro_norm(cov_dn - prev_cov_dn) / (fro_norm(prev_cov_dn) + 1e-12)

        rows.append({
            "date": m,
            "n": int(R.shape[1]),
            "raw_cond": float(raw_cond),
            "dn_cond": float(dn_cond),
            "k": int(k),
            "explained": float(explained),
            "raw_stability": float(raw_stab) if np.isfinite(raw_stab) else np.nan,
            "dn_stability": float(dn_stab) if np.isfinite(dn_stab) else np.nan,
        })

        prev_cov_raw = cov_raw
        prev_cov_dn = cov_dn

    return pd.DataFrame(rows).sort_values("date")


# -----------------------------
# Build weights for strategies (score-based)
# -----------------------------
def build_weights_for_months(
    scored_panel: pd.DataFrame,
    months: pd.Index,
    top_n: int,
    use_pca_rp: bool,
    prices_daily_ret: pd.DataFrame = None,
) -> Dict[pd.Timestamp, pd.Series]:
    """
    Build long-only weights for each month.
    If use_pca_rp=True, we compute PCA-RP weights from daily return window of selected names.
    Else equal-weight.
    """
    weights = {}

    if use_pca_rp and prices_daily_ret is None:
        raise ValueError("use_pca_rp=True requires prices_daily_ret DataFrame (daily returns).")

    for m in months:
        sub = scored_panel[scored_panel["date"] == m].copy()
        if len(sub) == 0:
            continue
        top, _ = select_top_bottom(sub, top_n=top_n, bottom_n=0)
        names = top["ticker"].tolist()

        if not use_pca_rp:
            w = pd.Series(1.0 / len(names), index=names)
            weights[m] = w
            continue

        # daily returns window
        subd = prices_daily_ret[(prices_daily_ret["date"] <= m) & (prices_daily_ret["ticker"].isin(names))].copy()
        if len(subd) == 0:
            w = pd.Series(1.0 / len(names), index=names)
            weights[m] = w
            continue

        last_date = subd["date"].max()
        start_w = last_date - pd.Timedelta(days=int(COV_LOOKBACK_DAYS * 1.8))
        subd = subd[subd["date"] >= start_w]
        R = subd.pivot(index="date", columns="ticker", values="ret_d").sort_index()

        w = build_pca_rp_weights(R)
        # ensure only selected names
        w = w.reindex(names).fillna(0.0)
        if w.sum() <= 0:
            w = pd.Series(1.0 / len(names), index=names)
        else:
            w = w / w.sum()
        weights[m] = w

    return weights


def build_ls_equal_weights(
    scored_panel: pd.DataFrame,
    months: pd.Index,
    top_n: int,
    bottom_n: int,
) -> Dict[pd.Timestamp, pd.Series]:
    """
    Long-short equal-weight:
      +1/top_n on top_n
      -1/bottom_n on bottom_n
    Sum absolute exposure = 2, net exposure = 0
    """
    weights = {}
    for m in months:
        sub = scored_panel[scored_panel["date"] == m].copy()
        if len(sub) == 0:
            continue
        top, bottom = select_top_bottom(sub, top_n=top_n, bottom_n=bottom_n)
        if len(top) == 0 or len(bottom) == 0:
            continue
        w_long = pd.Series(1.0 / len(top), index=top["ticker"].tolist())
        w_short = pd.Series(-1.0 / len(bottom), index=bottom["ticker"].tolist())
        w = pd.concat([w_long, w_short]).groupby(level=0).sum()
        weights[m] = w
    return weights


# -----------------------------
# MAIN
# -----------------------------
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    panel = load_panel()
    feature_cols = infer_feature_cols(panel)

    model, scored, split = train_lgbm(panel, feature_cols)
    test_months = split["test_months"]

    # --- Load prices for PCA-RP weight building (only for TEST window)
    # We need daily returns for all tickers that appear in test months (top ranks),
    # but simplest: load all tickers in dataset for test date range.
    prices = pd.read_parquet(PRICES_PATH, columns=["date", "ticker", "close"])
    prices["date"] = pd.to_datetime(prices["date"])
    # focus on test period + lookback buffer
    start = min(test_months) - pd.Timedelta(days=int(COV_LOOKBACK_DAYS * 2.2))
    end = max(test_months)
    prices = prices[(prices["date"] >= start) & (prices["date"] <= end)].copy()
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices["ret_d"] = prices.groupby("ticker")["close"].pct_change()
    prices = prices.dropna(subset=["ret_d"])[["date", "ticker", "ret_d"]].copy()

    # --- Strategy A: LGBM score + PCA-RP (Top50 long-only)
    w_top50_rp = build_weights_for_months(scored, test_months, top_n=TOP_N_DEFAULT, use_pca_rp=True, prices_daily_ret=prices)
    ret_top50_rp = make_monthly_returns_from_weights(scored, w_top50_rp)
    stat_top50_rp = perf_stats(ret_top50_rp)

    # --- Strategy B: Top10 long-only + PCA-RP
    w_top10_rp = build_weights_for_months(scored, test_months, top_n=TOP_N_ALT, use_pca_rp=True, prices_daily_ret=prices)
    ret_top10_rp = make_monthly_returns_from_weights(scored, w_top10_rp)
    stat_top10_rp = perf_stats(ret_top10_rp)

    # --- Strategy C: Long Top10 / Short Bottom50 (equal weight legs)
    w_ls_10_50 = build_ls_equal_weights(scored, test_months, top_n=TOP_N_ALT, bottom_n=BOTTOM_N_ALT)
    ret_ls_10_50 = make_monthly_returns_from_weights(scored, w_ls_10_50)
    stat_ls_10_50 = perf_stats(ret_ls_10_50)

    # --- Strategy D: Long Top10 / Short Bottom10 (equal weight legs)
    w_ls_10_10 = build_ls_equal_weights(scored, test_months, top_n=TOP_N_ALT, bottom_n=BOTTOM_N_ALT2)
    ret_ls_10_10 = make_monthly_returns_from_weights(scored, w_ls_10_10)
    stat_ls_10_10 = perf_stats(ret_ls_10_10)

    # --- Turnover for PCA-RP strategies
    turn_top50 = turnover_table(w_top50_rp, TOP_N_DEFAULT)
    turn_top10 = turnover_table(w_top10_rp, TOP_N_ALT)
    turn_df = pd.concat([turn_top50, turn_top10], ignore_index=True)
    turn_df.to_csv(OUT_TURNOVER, index=False)

    # --- Rank decay
    decay_df = rank_decay_table(scored, test_months, top_n=TOP_N_DEFAULT)
    decay_df.to_csv(OUT_RANK_DECAY, index=False)

    # --- PCA covariance diagnostics (raw vs denoised)
    if DO_PCA_COV_DIAGNOSTICS:
        cov_df = pca_cov_diagnostics(PRICES_PATH, w_top50_rp)
        cov_df.to_csv(OUT_PCA_COV, index=False)
    else:
        cov_df = pd.DataFrame()

    # --- Strategy compare table
    compare = pd.DataFrame([
        {"strategy": "Top50 Long-only (LGBM + PCA-RP)", **stat_top50_rp, "months": int(ret_top50_rp.dropna().shape[0])},
        {"strategy": "Top10 Long-only (LGBM + PCA-RP)", **stat_top10_rp, "months": int(ret_top10_rp.dropna().shape[0])},
        {"strategy": "L/S Top10 - Bottom50 (EqualW)", **stat_ls_10_50, "months": int(ret_ls_10_50.dropna().shape[0])},
        {"strategy": "L/S Top10 - Bottom10 (EqualW)", **stat_ls_10_10, "months": int(ret_ls_10_10.dropna().shape[0])},
    ])
    compare.to_csv(OUT_COMPARE, index=False)

    # --- Summary report
    lines = []
    lines.append("=== LGBM score + PCA Risk Parity | Full Diagnostics ===\n")
    lines.append(f"Panel: {PANEL_PATH}\n")
    lines.append(f"Prices: {PRICES_PATH}\n\n")

    lines.append("== Split ==\n")
    lines.append(f"Total months: {len(sorted(panel['date'].unique()))}\n")
    lines.append(f"Train={N_TRAIN}, Valid={N_VALID}, Test={len(split['test_months'])}\n")
    lines.append(f"MSE train={split['mse_train']:.6f} | valid={split['mse_valid']:.6f} | test={split['mse_test']:.6f}\n\n")

    lines.append("== Strategy Performance (TEST) ==\n")
    lines.append(format_stats("Top50 Long-only (LGBM + PCA-RP)", stat_top50_rp) + "\n")
    lines.append(format_stats("Top10 Long-only (LGBM + PCA-RP)", stat_top10_rp) + "\n")
    lines.append(format_stats("L/S Top10 - Bottom50 (EqualW)", stat_ls_10_50) + "\n")
    lines.append(format_stats("L/S Top10 - Bottom10 (EqualW)", stat_ls_10_10) + "\n\n")

    # turnover summary
    def turn_summary(df: pd.DataFrame) -> str:
        x = df.dropna(subset=["half_turnover"])
        if len(x) == 0:
            return "No turnover data"
        return (
            f"avg_half_turnover={x['half_turnover'].mean():.3f} | "
            f"avg_name_overlap={x['name_overlap'].mean():.3f} | "
            f"avg_max_w={x['max_w'].mean():.3f} | "
            f"avg_hhi={x['hhi'].mean():.3f}"
        )

    lines.append("== Turnover (TEST) ==\n")
    lines.append("Top50: " + turn_summary(turn_top50) + "\n")
    lines.append("Top10: " + turn_summary(turn_top10) + "\n\n")

    # decay summary
    lines.append("== Rank Decay (Top50; avg fwd ret of selected names) ==\n")
    d = decay_df[["avg_fwd_ret_t1", "avg_fwd_ret_t2", "avg_fwd_ret_t3"]].mean()
    lines.append(f"Mean t+1={d['avg_fwd_ret_t1']:.4f} | t+2={d['avg_fwd_ret_t2']:.4f} | t+3={d['avg_fwd_ret_t3']:.4f}\n\n")

    if DO_PCA_COV_DIAGNOSTICS and len(cov_df) > 0:
        lines.append("== PCA Covariance Diagnostics (Top50 universe) ==\n")
        lines.append(f"Avg raw condition number: {cov_df['raw_cond'].mean():.1f}\n")
        lines.append(f"Avg denoised condition number: {cov_df['dn_cond'].mean():.1f}\n")
        lines.append(f"Avg k PCs kept: {cov_df['k'].mean():.1f}\n")
        lines.append(f"Avg explained variance: {cov_df['explained'].mean():.3f}\n")
        lines.append(f"Avg cov stability raw: {cov_df['raw_stability'].mean():.4f}\n")
        lines.append(f"Avg cov stability denoised: {cov_df['dn_stability'].mean():.4f}\n\n")

    lines.append("== Output files ==\n")
    lines.append(f"- {OUT_SUMMARY_TXT}\n")
    lines.append(f"- {OUT_TURNOVER}\n")
    lines.append(f"- {OUT_RANK_DECAY}\n")
    lines.append(f"- {OUT_PCA_COV}\n")
    lines.append(f"- {OUT_COMPARE}\n")

    with open(OUT_SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print("Saved:")
    print(" -", OUT_SUMMARY_TXT)
    print(" -", OUT_TURNOVER)
    print(" -", OUT_RANK_DECAY)
    if DO_PCA_COV_DIAGNOSTICS:
        print(" -", OUT_PCA_COV)
    print(" -", OUT_COMPARE)

    print("\nKey TEST stats:")
    print(format_stats("Top50 Long-only (LGBM + PCA-RP)", stat_top50_rp))
    print(format_stats("Top10 Long-only (LGBM + PCA-RP)", stat_top10_rp))
    print(format_stats("L/S Top10 - Bottom50 (EqualW)", stat_ls_10_50))
    print(format_stats("L/S Top10 - Bottom10 (EqualW)", stat_ls_10_10))


if __name__ == "__main__":
    main()
