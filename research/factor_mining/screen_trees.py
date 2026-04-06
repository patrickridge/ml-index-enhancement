"""
screen_trees.py
===============
Tree-based factor screening using Random Forest and LightGBM.

Uses permutation importance (not impurity-based) to avoid bias toward
high-cardinality / continuous features.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from typing import List, Tuple


def tree_screen(
    panel: pd.DataFrame,
    candidate_cols: List[str],
    target: str = "fwd_ret_1m",
    end_date: str = "2022-12-31",
    top_k: int = 30,
    n_repeats: int = 5,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Screen candidates using Random Forest permutation importance.

    Returns:
        (selected_cols, importance_df)
    """
    df = panel[panel["date"] <= end_date].copy()
    valid_cols = [c for c in candidate_cols if c in df.columns]
    df = df.dropna(subset=valid_cols + [target])

    if len(df) < 100 or len(valid_cols) < 2:
        return [], pd.DataFrame()

    X = df[valid_cols].values
    y = df[target].values

    # Replace NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Use last 20% as held-out for permutation importance
    n_train = int(len(X) * 0.8)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=60,
        max_features=0.5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Permutation importance on held-out set
    perm_imp = permutation_importance(
        rf, X_test, y_test,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
    )

    importance = pd.DataFrame({
        "factor": valid_cols,
        "perm_importance_mean": perm_imp.importances_mean,
        "perm_importance_std": perm_imp.importances_std,
    }).sort_values("perm_importance_mean", ascending=False)

    # Select top_k or those with positive importance
    positive = importance[importance["perm_importance_mean"] > 0]
    selected = positive.head(top_k)["factor"].tolist()

    print(f"  Tree screen: {len(selected)}/{len(valid_cols)} factors selected "
          f"(top {top_k} with positive permutation importance)")

    return selected, importance


def lgbm_screen(
    panel: pd.DataFrame,
    candidate_cols: List[str],
    target: str = "fwd_ret_1m",
    end_date: str = "2022-12-31",
    top_k: int = 30,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Screen candidates using LightGBM split-based importance.
    Falls back to RF if LightGBM not installed.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        print("  LightGBM not available, falling back to RF")
        return tree_screen(panel, candidate_cols, target, end_date, top_k)

    df = panel[panel["date"] <= end_date].copy()
    valid_cols = [c for c in candidate_cols if c in df.columns]
    df = df.dropna(subset=valid_cols + [target])

    if len(df) < 100 or len(valid_cols) < 2:
        return [], pd.DataFrame()

    X = df[valid_cols].values
    y = df[target].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n_train = int(len(X) * 0.8)

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=5,
        min_child_samples=60,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        X[:n_train], y[:n_train],
        eval_set=[(X[n_train:], y[n_train:])],
    )

    importance = pd.DataFrame({
        "factor": valid_cols,
        "lgbm_importance": model.feature_importances_,
    }).sort_values("lgbm_importance", ascending=False)

    selected = importance.head(top_k)["factor"].tolist()
    print(f"  LightGBM screen: {len(selected)}/{len(valid_cols)} top factors selected")

    return selected, importance
