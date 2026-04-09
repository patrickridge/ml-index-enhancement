"""
validation_engine.py
====================
IS-only factor screening and validation utilities for the factor mining exercise.

All functions enforce temporal constraints: discovery uses only IS data (≤ TRAIN_END).
OOS data is never touched during screening.

Usage:
    from research.factor_mining.validation_engine import (
        compute_is_ic, compute_icir, ic_persistence,
        ras_test, bhy_correction, dedup_check, screen_candidate,
    )
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Tuple, List, Dict


# ── Configuration ────────────────────────────────────────────────────────────

TRAIN_END = "2022-12-31"
IC_THRESHOLD = 0.015        # minimum |mean IC|
ICIR_THRESHOLD = 0.40       # minimum |ICIR|
PERSISTENCE_RATIO = 0.50    # IC_lag1 must be > this fraction of IC_lag0
DEDUP_THRESHOLD = 0.85      # max |correlation| with existing features
RAS_N_SIMS = 1000           # number of random alpha simulations
RAS_PERCENTILE = 95         # percentile threshold for RAS test
BHY_FDR = 0.05              # false discovery rate for BHY correction


# ── Core IC Computation ──────────────────────────────────────────────────────

def compute_is_ic(
    panel: pd.DataFrame,
    factor_col: str,
    target: str = "fwd_ret_1m",
    end_date: str = TRAIN_END,
) -> pd.Series:
    """
    Compute monthly Spearman rank-IC for a factor within IS period.

    Returns:
        pd.Series indexed by date with monthly IC values.
    """
    df = panel[panel["date"] <= end_date].copy()
    df = df.dropna(subset=[factor_col, target])

    def _spearman_ic(grp):
        if len(grp) < 20:
            return np.nan
        return grp[factor_col].rank().corr(grp[target].rank(), method="pearson")

    ic_series = df.groupby("date").apply(_spearman_ic)
    ic_series.name = factor_col
    return ic_series.dropna()


def compute_icir(ic_series: pd.Series) -> float:
    """ICIR = mean(IC) / std(IC). Returns 0.0 if std is near zero."""
    if len(ic_series) < 6:
        return 0.0
    std = ic_series.std()
    if std < 1e-9:
        return 0.0
    return ic_series.mean() / std


# ── IC Persistence ───────────────────────────────────────────────────────────

def ic_persistence(
    panel: pd.DataFrame,
    factor_col: str,
    target: str = "fwd_ret_1m",
    lags: List[int] = [1, 2, 3, 6],
    end_date: str = TRAIN_END,
) -> Dict[int, float]:
    """
    Compute IC at multiple forward lags.

    For lag k, we correlate factor[t] with fwd_ret[t+k].
    This measures how quickly the factor's predictive power decays.

    Returns:
        dict mapping lag -> mean IC at that lag.
    """
    df = panel[panel["date"] <= end_date].copy()
    df = df.dropna(subset=[factor_col, target])
    dates = sorted(df["date"].unique())

    results = {}
    for lag in lags:
        # For each month, correlate factor with target from lag months ahead
        ics = []
        for i, dt in enumerate(dates):
            if i + lag >= len(dates):
                break
            future_dt = dates[i + lag]
            current = df[df["date"] == dt][["ticker", factor_col]].set_index("ticker")
            future = df[df["date"] == future_dt][["ticker", target]].set_index("ticker")
            merged = current.join(future, how="inner")
            if len(merged) < 20:
                continue
            ic = merged[factor_col].rank().corr(merged[target].rank())
            if np.isfinite(ic):
                ics.append(ic)
        results[lag] = np.mean(ics) if ics else 0.0

    return results


# ── Deduplication ────────────────────────────────────────────────────────────

def dedup_check(
    panel: pd.DataFrame,
    new_col: str,
    existing_cols: List[str],
    threshold: float = DEDUP_THRESHOLD,
    end_date: str = TRAIN_END,
) -> Tuple[bool, str, float]:
    """
    Check if a new factor is too correlated with existing features.

    Returns:
        (is_duplicate, closest_existing_col, max_abs_corr)
    """
    df = panel[panel["date"] <= end_date].copy()

    max_corr = 0.0
    closest = ""

    for ecol in existing_cols:
        if ecol == new_col:
            continue  # don't dedup against self
        if ecol not in df.columns or new_col not in df.columns:
            continue

        # Compute mean cross-sectional correlation across months
        corrs = []
        for dt, grp in df.groupby("date"):
            sub = grp[[new_col, ecol]].dropna()
            if len(sub) < 20:
                continue
            c = sub[new_col].rank().corr(sub[ecol].rank())
            if np.isfinite(c):
                corrs.append(c)

        if corrs:
            avg_corr = abs(np.mean(corrs))
            if avg_corr > max_corr:
                max_corr = avg_corr
                closest = ecol

    is_dup = max_corr > threshold
    return is_dup, closest, max_corr


# ── RAS Test (Random Alpha Simulation) ───────────────────────────────────────

def ras_test(
    panel: pd.DataFrame,
    factor_col: str,
    target: str = "fwd_ret_1m",
    n_sims: int = RAS_N_SIMS,
    end_date: str = TRAIN_END,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Random Alpha Simulation test.

    Shuffles the factor values within each cross-section n_sims times,
    computes ICIR for each shuffle, and returns the p-value.

    Returns:
        (p_value, actual_icir)
        p_value = fraction of random ICIRs with |ICIR| >= |actual_ICIR|
    """
    rng = np.random.RandomState(seed)
    df = panel[panel["date"] <= end_date].copy()
    df = df.dropna(subset=[factor_col, target])

    # Actual ICIR
    actual_ic = compute_is_ic(df, factor_col, target, end_date="2099-12-31")
    actual_icir = compute_icir(actual_ic)

    # Random simulations
    random_icirs = np.zeros(n_sims)
    dates = sorted(df["date"].unique())

    for sim in range(n_sims):
        # Shuffle factor within each month (preserving cross-sectional structure)
        df_shuf = df.copy()
        for dt in dates:
            mask = df_shuf["date"] == dt
            vals = df_shuf.loc[mask, factor_col].values.copy()
            rng.shuffle(vals)
            df_shuf.loc[mask, factor_col] = vals

        sim_ic = compute_is_ic(df_shuf, factor_col, target, end_date="2099-12-31")
        random_icirs[sim] = compute_icir(sim_ic)

    # p-value: fraction of random |ICIR| >= actual |ICIR|
    p_value = (np.abs(random_icirs) >= abs(actual_icir)).mean()
    return p_value, actual_icir


def ras_test_fast(
    panel: pd.DataFrame,
    factor_col: str,
    target: str = "fwd_ret_1m",
    n_sims: int = RAS_N_SIMS,
    end_date: str = TRAIN_END,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Fast RAS test using vectorized operations.

    Instead of recomputing IC series for each shuffle, we precompute
    the target ranks and use numpy operations.

    Returns:
        (p_value, actual_icir)
    """
    rng = np.random.RandomState(seed)
    df = panel[panel["date"] <= end_date].copy()
    df = df.dropna(subset=[factor_col, target])

    dates = sorted(df["date"].unique())
    monthly_data = {}
    actual_ics = []

    # Precompute per-month data
    for dt in dates:
        grp = df[df["date"] == dt]
        if len(grp) < 20:
            continue
        factor_vals = grp[factor_col].values
        target_vals = grp[target].values
        # Rank both
        f_rank = stats.rankdata(factor_vals)
        t_rank = stats.rankdata(target_vals)
        monthly_data[dt] = (f_rank, t_rank, len(grp))
        # Actual IC
        ic = np.corrcoef(f_rank, t_rank)[0, 1]
        if np.isfinite(ic):
            actual_ics.append(ic)

    actual_ic_arr = np.array(actual_ics)
    if len(actual_ic_arr) < 6:
        return 1.0, 0.0

    actual_icir = actual_ic_arr.mean() / (actual_ic_arr.std() + 1e-9)

    # Random simulations
    random_icirs = np.zeros(n_sims)
    for sim in range(n_sims):
        sim_ics = []
        for dt, (f_rank, t_rank, n) in monthly_data.items():
            shuffled = f_rank.copy()
            rng.shuffle(shuffled)
            ic = np.corrcoef(shuffled, t_rank)[0, 1]
            if np.isfinite(ic):
                sim_ics.append(ic)
        sim_arr = np.array(sim_ics)
        if len(sim_arr) >= 6:
            random_icirs[sim] = sim_arr.mean() / (sim_arr.std() + 1e-9)

    p_value = (np.abs(random_icirs) >= abs(actual_icir)).mean()
    return p_value, actual_icir


# ── BHY Multiple Testing Correction ─────────────────────────────────────────

def bhy_correction(
    p_values: np.ndarray,
    fdr: float = BHY_FDR,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg-Yekutieli (BHY) procedure for FDR control
    under arbitrary dependence.

    Parameters:
        p_values: array of raw p-values
        fdr: target false discovery rate

    Returns:
        (adjusted_pvalues, reject_mask)
        reject_mask: boolean array, True = reject null (factor is significant)
    """
    m = len(p_values)
    if m == 0:
        return np.array([]), np.array([], dtype=bool)

    # BHY correction factor: c(m) = sum(1/k for k=1..m)
    c_m = np.sum(1.0 / np.arange(1, m + 1))

    # Sort p-values
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # BHY critical values: (k / (m * c_m)) * fdr
    k = np.arange(1, m + 1)
    critical = (k / (m * c_m)) * fdr

    # Find largest k where p_(k) <= critical value
    reject_sorted = sorted_p <= critical

    # Adjusted p-values (step-up)
    adjusted = np.zeros(m)
    adjusted[sorted_idx[-1]] = min(sorted_p[-1] * m * c_m / m, 1.0)
    for i in range(m - 2, -1, -1):
        raw_adj = sorted_p[i] * m * c_m / (i + 1)
        adjusted[sorted_idx[i]] = min(raw_adj, adjusted[sorted_idx[i + 1]])

    adjusted = np.clip(adjusted, 0, 1)

    # Reject mask: all indices up to the largest significant one
    if reject_sorted.any():
        max_reject = np.where(reject_sorted)[0][-1]
        final_reject = np.zeros(m, dtype=bool)
        final_reject[sorted_idx[:max_reject + 1]] = True
    else:
        final_reject = np.zeros(m, dtype=bool)

    return adjusted, final_reject


# ── Unified Screening Function ───────────────────────────────────────────────

def screen_candidate(
    panel: pd.DataFrame,
    factor_col: str,
    existing_cols: List[str],
    target: str = "fwd_ret_1m",
    end_date: str = TRAIN_END,
    run_ras: bool = True,
    ras_n_sims: int = 200,   # reduced for speed; full 1000 in final pass
) -> Dict:
    """
    Run a candidate factor through the full IS screening pipeline.

    Returns a dict with all metrics and pass/fail flags.
    """
    result = {
        "factor": factor_col,
        "ic_mean": np.nan,
        "ic_std": np.nan,
        "icir": np.nan,
        "ic_lag1": np.nan,
        "ic_lag3": np.nan,
        "persistence_pass": False,
        "is_duplicate": False,
        "closest_existing": "",
        "max_corr_existing": 0.0,
        "ras_pvalue": np.nan,
        "pass_ic": False,
        "pass_icir": False,
        "pass_persistence": False,
        "pass_dedup": False,
        "pass_ras": False,
        "pass_all": False,
    }

    # Stage 1-2: IC and ICIR
    ic_series = compute_is_ic(panel, factor_col, target, end_date)
    if len(ic_series) < 12:
        return result

    result["ic_mean"] = ic_series.mean()
    result["ic_std"] = ic_series.std()
    result["icir"] = compute_icir(ic_series)
    result["pass_ic"] = abs(result["ic_mean"]) > IC_THRESHOLD
    result["pass_icir"] = abs(result["icir"]) > ICIR_THRESHOLD

    # Stage 3: Persistence
    persistence = ic_persistence(panel, factor_col, target, lags=[1, 3], end_date=end_date)
    result["ic_lag1"] = persistence.get(1, 0.0)
    result["ic_lag3"] = persistence.get(3, 0.0)
    ic_lag0 = abs(result["ic_mean"])
    result["persistence_pass"] = abs(persistence.get(1, 0.0)) > PERSISTENCE_RATIO * ic_lag0
    result["pass_persistence"] = result["persistence_pass"]

    # Stage 4: Deduplication
    is_dup, closest, max_corr = dedup_check(panel, factor_col, existing_cols, end_date=end_date)
    result["is_duplicate"] = is_dup
    result["closest_existing"] = closest
    result["max_corr_existing"] = max_corr
    result["pass_dedup"] = not is_dup

    # Stage 5: RAS test — run for all non-duplicate factors (no IC/ICIR gate)
    # IC/ICIR are diagnostic, not gates. BHY on RAS is the real filter.
    if run_ras and result["pass_dedup"]:
        p_val, _ = ras_test_fast(panel, factor_col, target, n_sims=ras_n_sims, end_date=end_date)
        result["ras_pvalue"] = p_val
        result["pass_ras"] = p_val < 0.05
    else:
        result["pass_ras"] = False

    # Overall (legacy — pass_all_final in screen_all_candidates uses BHY + dedup only)
    result["pass_all"] = all([
        result["pass_dedup"],
        result["pass_ras"],
    ])

    return result


def screen_all_candidates(
    panel: pd.DataFrame,
    candidate_cols: List[str],
    existing_cols: List[str],
    target: str = "fwd_ret_1m",
    end_date: str = TRAIN_END,
    ras_n_sims: int = 200,
) -> pd.DataFrame:
    """
    Screen all candidates and return a catalog DataFrame.
    BHY correction is applied after all candidates are evaluated.
    """
    results = []
    for i, col in enumerate(candidate_cols):
        print(f"  Screening {i+1}/{len(candidate_cols)}: {col}")
        r = screen_candidate(panel, col, existing_cols, target, end_date, ras_n_sims=ras_n_sims)
        results.append(r)

    catalog = pd.DataFrame(results)

    # Apply BHY correction to RAS p-values
    ras_pvals = catalog["ras_pvalue"].fillna(1.0).values
    adjusted_pvals, reject_mask = bhy_correction(ras_pvals, fdr=BHY_FDR)
    catalog["bhy_adjusted_pvalue"] = adjusted_pvals
    catalog["pass_bhy"] = reject_mask
    catalog["pass_all_final"] = catalog["pass_all"] & catalog["pass_bhy"]

    return catalog
