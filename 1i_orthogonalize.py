"""
1i_orthogonalize.py
====================
PCA residualization of the factor panel.

Reads:   data/panel_monthly_enriched.parquet
Writes:  data/panel_monthly_orthogonalized.parquet

Method (PCA Residualization — industry standard):
  1. Stack all (date × ticker) observations → X matrix (n_obs, N_features)
  2. Fit PCA with K components, where K explains >= PCA_VARIANCE_THRESHOLD of shared variance
  3. Project X onto K-dimensional PCA subspace: X_proj = pca.inverse_transform(pca.transform(X))
     This is mathematically identical to regressing each feature on the K PCA scores
     and keeping fitted values.
  4. Residuals: X_resid = X - X_proj  (orthogonal to all K common factors)
  5. Re-apply cross-sectional rank normalisation per date (residuals lose [-0.5, 0.5] scale)
  6. Fill NaN → 0, save

Why this helps:
  - Removes shared variance from market beta, sector effects, and style tilts
    that are embedded across multiple features.
  - Leaves pure idiosyncratic alpha signals that are orthogonal to each other.
  - Most beneficial for the cross-sectional Transformer (attention mechanism
    benefits from orthogonal inputs; correlated features confuse attention weights).
  - May slightly hurt tree models (LightGBM handles correlated features natively
    via feature importance allocation).

Walk-forward note:
  Uses full-sample PCA (mild look-ahead in feature preprocessing).
  Same severity as 52-week-high or long-horizon vol features which also use
  recent data at the feature level. For strict OOS, a walk-forward PCA wrapper
  can be added in a future version.

Macro columns (MACRO_COLS) are excluded from residualization — they are already
time-series z-scored and carry regime-level information, not cross-sectional alpha.

Run time: < 1 minute (pure numpy/sklearn).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

from config import MACRO_COLS, PCA_VARIANCE_THRESHOLD

DATA_DIR  = Path("data")
PANEL_IN  = DATA_DIR / "panel_monthly_enriched.parquet"
OUT_PATH  = DATA_DIR / "panel_monthly_orthogonalized.parquet"


def rank_norm(s: pd.Series) -> pd.Series:
    """Cross-sectional rank normalised to [-0.5, 0.5]. NaN stays NaN."""
    r = s.rank(pct=True, na_option="keep")
    return r - 0.5


def pca_residualize(
    X: np.ndarray,
    variance_threshold: float = 0.80,
) -> tuple:
    """
    Residualize feature matrix X against its top K PCA components.

    Parameters
    ----------
    X : np.ndarray, shape (n_obs, n_features)
        Feature matrix (already rank-normalised to [-0.5, 0.5]).
    variance_threshold : float
        Keep K components explaining >= this fraction of total variance.

    Returns
    -------
    X_resid : np.ndarray, shape (n_obs, n_features)
        Residuals after removing PCA projection.
    K : int
        Number of PCA components used.
    cum_var : np.ndarray
        Cumulative explained variance ratios.
    """
    n_features = X.shape[1]

    # Determine K
    pca_full = PCA(n_components=min(n_features, X.shape[0] - 1), random_state=42)
    pca_full.fit(X)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    K = int(np.searchsorted(cum_var, variance_threshold) + 1)
    K = max(1, min(K, n_features - 1))   # at least 1 component, at most n_features-1

    print(f"  PCA: K={K} components explain {cum_var[K-1]*100:.1f}% of shared variance")

    # Fit K-component PCA and project X onto it
    pca_k = PCA(n_components=K, random_state=42)
    scores = pca_k.fit_transform(X)          # (n_obs, K)
    X_proj = pca_k.inverse_transform(scores)  # (n_obs, n_features) — PCA projection
    X_resid = X - X_proj                      # residuals: orthogonal to all K PCs

    return X_resid, K, cum_var


def main():
    print("=" * 60)
    print("FACTOR ORTHOGONALIZATION — PCA Residualization")
    print("=" * 60)

    # Load enriched panel
    print(f"\nLoading {PANEL_IN}...")
    panel = pd.read_parquet(PANEL_IN)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    print(f"Panel: {panel.shape}")

    # Identify feature columns (exclude macro — already z-scored, not cs-ranked)
    always_exclude = {"date", "ticker", "fwd_ret_1m"}
    macro_in_panel = [c for c in MACRO_COLS if c in panel.columns]
    feat_cols = [
        c for c in panel.columns
        if c not in always_exclude and c not in macro_in_panel
    ]
    print(f"Features to orthogonalize: {len(feat_cols)}")
    print(f"Macro columns excluded:    {len(macro_in_panel)} {macro_in_panel}")

    # Extract feature matrix
    X = panel[feat_cols].values.astype(np.float32)
    print(f"\nFeature matrix: {X.shape[0]:,} obs × {X.shape[1]} features")

    # Compute mean |off-diagonal correlation| before orthogonalization
    corr_before = np.corrcoef(X, rowvar=False)
    off_diag = np.abs(corr_before[np.triu_indices(len(feat_cols), k=1)])
    print(f"\nPre-orthogonalization: mean |off-diag corr| = {off_diag.mean():.4f}")

    # PCA residualization
    print(f"\nApplying PCA residualization (threshold={PCA_VARIANCE_THRESHOLD:.0%})...")
    X_resid, K, cum_var = pca_residualize(X, variance_threshold=PCA_VARIANCE_THRESHOLD)

    # Post-orthogonalization correlation check
    corr_after = np.corrcoef(X_resid, rowvar=False)
    off_diag_after = np.abs(corr_after[np.triu_indices(len(feat_cols), k=1)])
    print(f"Post-orthogonalization: mean |off-diag corr| = {off_diag_after.mean():.4f}")
    print(f"Correlation reduction: "
          f"{(off_diag.mean() - off_diag_after.mean()) / off_diag.mean() * 100:.1f}%")

    # Build output panel
    panel_orth = panel.copy()
    for j, col in enumerate(feat_cols):
        panel_orth[col] = X_resid[:, j]

    # Re-apply cross-sectional rank normalisation per date
    print("\nRe-applying cross-sectional rank normalisation to residuals...")
    for c in feat_cols:
        panel_orth[c] = panel_orth.groupby("date")[c].transform(rank_norm)

    # Fill remaining NaNs with 0 (neutral rank)
    all_feat = feat_cols + macro_in_panel
    panel_orth[all_feat] = panel_orth[all_feat].fillna(0.0)

    # Save
    panel_orth.to_parquet(OUT_PATH, index=False)

    print(f"\n{'=' * 60}")
    print(f"Done! Saved → {OUT_PATH}")
    print(f"Shape: {panel_orth.shape}")
    print(f"PCA components used: {K}")
    print(f"Cumulative explained variance @ K: {cum_var[K-1]*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
