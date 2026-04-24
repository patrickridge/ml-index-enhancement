"""
regime_entropy.py
=================
Singha-inspired volatility regime detector using return distribution entropy.

Concept: Low entropy in the return distribution over a rolling window signals
ordered/concentrated trading → predicts larger absolute price moves (not direction).

This is a REGIME OVERLAY, not a standalone alpha factor. Use it to:
1. Gate existing factors (turn signals on/off by regime)
2. Condition model behavior (interact with CS-Transformer features)
3. Size positions differently in low-entropy vs high-entropy periods

Reference: Singha (2025), "Hidden Order in Trades Predicts the Size of Price Moves",
           arXiv:2512.15720
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


def compute_return_entropy(
    returns: np.ndarray,
    n_bins: int = 5,
) -> float:
    """
    Compute Shannon entropy of the return distribution using histogram binning.

    Lower entropy = more concentrated distribution = more predictable magnitude.

    Parameters:
        returns: array of daily returns (no NaNs)
        n_bins: number of histogram bins

    Returns:
        Entropy value (nats). Max = ln(n_bins) for uniform distribution.
    """
    if len(returns) < 10:
        return np.nan

    # Use fixed bin edges based on percentiles to be robust to outliers
    counts, _ = np.histogram(returns, bins=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # avoid log(0)
    return -np.sum(probs * np.log(probs))


def add_entropy_regime_signals(
    prices: pd.DataFrame,
    window: int = 21,
    n_bins: int = 5,
    threshold_percentile: float = 25,
    is_end_date: str = "2022-12-31",
) -> pd.DataFrame:
    """
    Add entropy-based volatility regime signals to daily prices.

    NEW COLUMNS:
      cand_return_entropy_21d    - Rolling Shannon entropy of daily returns (21d)
      cand_low_entropy_regime    - Binary: 1 if entropy < IS threshold (low entropy = stressed)
      cand_entropy_transition    - Binary: 1 if entropy dropped below threshold this period
      cand_entropy_zscore        - Z-scored entropy (per-stock, rolling 126d)

    Parameters:
        prices: DataFrame with columns [date, ticker, ret_d]
        window: Rolling window for entropy computation
        n_bins: Histogram bins for entropy
        threshold_percentile: Percentile of IS entropy distribution for threshold
                              (25th = bottom quartile = "low entropy")
        is_end_date: End of in-sample period for threshold calibration
    """
    prices = prices.copy()

    for tk, grp in prices.groupby("ticker", sort=False):
        idx = grp.index
        ret_d = grp["ret_d"].values if "ret_d" in grp.columns else np.full(len(grp), np.nan)
        dates = grp["date"].values
        n = len(grp)

        # Compute rolling entropy
        entropy_vals = np.full(n, np.nan)
        for i in range(window - 1, n):
            rets = ret_d[max(0, i - window + 1):i + 1]
            rets = rets[~np.isnan(rets)]
            if len(rets) >= 10:
                entropy_vals[i] = compute_return_entropy(rets, n_bins)

        prices.loc[idx, "cand_return_entropy_21d"] = entropy_vals

        # Calibrate threshold on IS data only
        is_mask = pd.to_datetime(dates) <= pd.Timestamp(is_end_date)
        is_entropy = entropy_vals[is_mask]
        is_entropy = is_entropy[~np.isnan(is_entropy)]

        if len(is_entropy) < 50:
            prices.loc[idx, "cand_low_entropy_regime"] = 0.0
            prices.loc[idx, "cand_entropy_transition"] = 0.0
            prices.loc[idx, "cand_entropy_zscore"] = 0.0
            continue

        threshold = np.percentile(is_entropy, threshold_percentile)

        # Binary regime signal
        low_entropy = (entropy_vals < threshold).astype(np.float32)
        prices.loc[idx, "cand_low_entropy_regime"] = low_entropy

        # Transition signal: entered low-entropy regime this period
        prev_low = np.concatenate([[0], low_entropy[:-1]])
        transition = ((low_entropy == 1) & (prev_low == 0)).astype(np.float32)
        prices.loc[idx, "cand_entropy_transition"] = transition

        # Z-scored entropy (per-stock rolling normalization)
        ent_s = pd.Series(entropy_vals, index=idx)
        mu = ent_s.rolling(126, min_periods=30).mean()
        sigma = ent_s.rolling(126, min_periods=30).std()
        prices.loc[idx, "cand_entropy_zscore"] = ((ent_s - mu) / (sigma + 1e-9)).values

    return prices


def validate_entropy_signal(
    panel: pd.DataFrame,
    entropy_col: str = "cand_low_entropy_regime",
    target: str = "fwd_ret_1m",
    end_date: str = "2022-12-31",
) -> Dict:
    """
    Validate whether low-entropy periods predict larger absolute moves (IS only).

    Returns dict with:
        - mean_abs_ret_low_entropy: mean |fwd_ret| when low_entropy=1
        - mean_abs_ret_high_entropy: mean |fwd_ret| when low_entropy=0
        - ratio: low/high (should be > 1 if signal works)
        - directional_accuracy: fraction correct direction (should be ~0.5)
        - n_low, n_high: sample counts
    """
    df = panel[panel["date"] <= end_date].dropna(subset=[entropy_col, target])

    low = df[df[entropy_col] == 1][target]
    high = df[df[entropy_col] == 0][target]

    result = {
        "mean_abs_ret_low_entropy": low.abs().mean() if len(low) > 0 else np.nan,
        "mean_abs_ret_high_entropy": high.abs().mean() if len(high) > 0 else np.nan,
        "ratio": (low.abs().mean() / (high.abs().mean() + 1e-9)) if len(low) > 0 and len(high) > 0 else np.nan,
        "directional_accuracy_low": (np.sign(low) == 1).mean() if len(low) > 0 else np.nan,
        "directional_accuracy_high": (np.sign(high) == 1).mean() if len(high) > 0 else np.nan,
        "n_low": len(low),
        "n_high": len(high),
    }
    return result


def add_regime_interactions(
    panel: pd.DataFrame,
    regime_col: str = "cand_low_entropy_regime",
    factor_cols: list = None,
) -> pd.DataFrame:
    """
    Create interaction features: factor × regime_signal.

    For each factor in factor_cols, create:
      {factor}_x_entropy_regime = factor × regime_col

    These test whether factors work differently in low-entropy periods.
    """
    if factor_cols is None:
        factor_cols = ["ret_12m", "ret_6m", "ret_1m", "vol_21d", "rsi_14"]

    panel = panel.copy()
    for f in factor_cols:
        if f in panel.columns and regime_col in panel.columns:
            panel[f"cand_{f}_x_entropy"] = panel[f] * panel[regime_col]

    return panel
