"""
screen_lasso.py
===============
Sparse linear factor screening using Lasso / Elastic Net.

Fits penalized regression on IS data to identify candidates with nonzero coefficients.
Uses purged time-series CV to avoid lookahead within IS period.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple


def purged_ts_cv_splits(dates: np.ndarray, n_splits: int = 5, purge_months: int = 1):
    """
    Generate purged time-series cross-validation splits.
    Each fold trains on earlier data, tests on later data, with a gap.
    """
    unique_dates = np.sort(np.unique(dates))
    n = len(unique_dates)
    fold_size = n // (n_splits + 1)

    for i in range(n_splits):
        train_end_idx = fold_size * (i + 1)
        test_start_idx = train_end_idx + purge_months
        test_end_idx = min(test_start_idx + fold_size, n)

        if test_start_idx >= n:
            break

        train_dates = set(unique_dates[:train_end_idx])
        test_dates = set(unique_dates[test_start_idx:test_end_idx])

        train_mask = np.isin(dates, list(train_dates))
        test_mask = np.isin(dates, list(test_dates))

        yield np.where(train_mask)[0], np.where(test_mask)[0]


def lasso_screen(
    panel: pd.DataFrame,
    candidate_cols: List[str],
    target: str = "fwd_ret_1m",
    end_date: str = "2022-12-31",
    n_splits: int = 5,
    model_type: str = "lasso",
) -> Tuple[List[str], pd.DataFrame]:
    """
    Screen candidate factors using Lasso or Elastic Net.

    Returns:
        (selected_cols, importance_df)
        selected_cols: columns with nonzero coefficients
        importance_df: DataFrame with factor, coefficient, abs_coef
    """
    df = panel[panel["date"] <= end_date].copy()
    valid_cols = [c for c in candidate_cols if c in df.columns]
    df = df.dropna(subset=valid_cols + [target])

    if len(df) < 100 or len(valid_cols) < 2:
        return [], pd.DataFrame()

    X = df[valid_cols].values.astype(np.float64)
    y = df[target].values.astype(np.float64)
    dates = df["date"].values

    # Clean inf/NaN BEFORE scaling (StandardScaler rejects inf)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Purged CV splits
    cv_splits = list(purged_ts_cv_splits(dates, n_splits=n_splits))

    if model_type == "elastic_net":
        model = ElasticNetCV(
            l1_ratio=[0.5, 0.7, 0.9, 0.95],
            cv=cv_splits,
            max_iter=5000,
            random_state=42,
            n_jobs=-1,
        )
    else:
        model = LassoCV(
            cv=cv_splits,
            max_iter=5000,
            random_state=42,
            n_jobs=-1,
        )

    model.fit(X_scaled, y)
    coefs = model.coef_

    importance = pd.DataFrame({
        "factor": valid_cols,
        "coefficient": coefs,
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False)

    selected = importance[importance["abs_coef"] > 1e-6]["factor"].tolist()

    print(f"  Lasso screen: {len(selected)}/{len(valid_cols)} factors with nonzero coefficients")
    if hasattr(model, 'alpha_'):
        print(f"  Best alpha: {model.alpha_:.6f}")

    return selected, importance
