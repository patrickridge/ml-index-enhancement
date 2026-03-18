"""
7_factor_diagnostics.py
========================
READ-ONLY diagnostic script — does NOT modify any model inputs or pipeline data.

Computes three complementary redundancy diagnostics on the enriched factor panel:

  Part A — IC Correlation Matrix (Spearman)
    Pairwise Spearman rank correlations between all features.
    Flags pairs with |corr| > 0.5 as potentially redundant.
    Saves correlation matrix CSV + heatmap PNG.

  Part B — RMT Eigenvalue Analysis
    Compares the empirical eigenvalue spectrum of the feature correlation matrix
    against the Marchenko-Pastur theoretical bounds.
    Shows how many eigenvalues carry genuine signal vs noise.
    Saves eigenvalue CSV + RMT-denoised correlation heatmap.

  Part C — Variance Inflation Factor (VIF)
    VIF_i = 1 / (1 - R²_i) where R²_i is R² from regressing feature i on all others.
    VIF > 10 → highly redundant (multicollinear); flags these features for review.
    Saves VIF table CSV.

  Part D — Pre/Post-Orthogonalization Comparison (optional)
    If panel_monthly_orthogonalized.parquet exists, computes and prints
    the mean |off-diagonal correlation| before and after PCA residualization.

Outputs (all to data/ directory):
  diag_ic_corr_matrix.csv
  diag_ic_corr_heatmap.png
  diag_redundant_pairs.csv
  diag_rmt_eigenvalues.csv
  diag_rmt_corr_heatmap.png
  diag_vif.csv

Run AFTER 1_feature_engineering.py.
"""

import matplotlib
matplotlib.use("Agg")   # headless — no display needed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

from utils_rmt import rmt_denoise, marchenko_pastur_upper
from config import MACRO_COLS

DATA_DIR = Path("data")
FIG_DIR  = DATA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PANEL_IN = DATA_DIR / "panel_monthly_enriched.parquet"
ORTH_IN  = DATA_DIR / "panel_monthly_orthogonalized.parquet"

IC_CORR_FLAG      = 0.50    # flag pairs with |corr| above this
VIF_FLAG          = 10.0    # flag features with VIF above this
VIF_SAMPLE_ROWS   = 20_000  # sample size for VIF computation (speed)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_corr_heatmap(
    corr_df:   pd.DataFrame,
    title:     str,
    out_path:  Path,
    flag_val:  float = 0.5,
    figsize_k: float = 0.45,
):
    """Save a correlation heatmap. Annotates cells with |corr| > flag_val."""
    n   = len(corr_df)
    fig_size = max(12, n * figsize_k)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size * 0.9))

    data = corr_df.values
    im   = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    for i in range(n):
        for j in range(n):
            if i != j and abs(data[i, j]) >= flag_val:
                ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center",
                        fontsize=max(4, 8 - n // 20), color="black", fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr_df.columns, rotation=90, fontsize=max(5, 9 - n // 20))
    ax.set_yticklabels(corr_df.index,   fontsize=max(5, 9 - n // 20))
    ax.set_title(title, fontsize=12, pad=10)
    plt.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def compute_vif(X: np.ndarray, feature_names: list) -> pd.DataFrame:
    """
    Compute VIF for each feature.
    VIF_i = 1 / (1 - R²_i)  where R²_i = R² from regressing feature i on all others.
    Larger = more redundant.
    """
    n = X.shape[1]
    records = []
    for i in range(n):
        X_others = np.delete(X, i, axis=1)
        y_i      = X[:, i]
        reg      = LinearRegression(fit_intercept=True)
        reg.fit(X_others, y_i)
        y_pred   = reg.predict(X_others)
        ss_res   = np.sum((y_i - y_pred) ** 2)
        ss_tot   = np.sum((y_i - y_i.mean()) ** 2)
        r2       = float(np.clip(1.0 - ss_res / (ss_tot + 1e-14), 0.0, 1.0 - 1e-9))
        vif      = 1.0 / (1.0 - r2)
        records.append({
            "feature":       feature_names[i],
            "VIF":           round(vif, 3),
            "R2_vs_others":  round(r2, 4),
            "flag_redundant": vif > VIF_FLAG,
        })
    return (pd.DataFrame(records)
            .sort_values("VIF", ascending=False)
            .reset_index(drop=True))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("FACTOR DIAGNOSTICS")
    print("=" * 65)

    # Load panel
    print(f"\nLoading {PANEL_IN}...")
    panel = pd.read_parquet(PANEL_IN)
    panel["date"] = pd.to_datetime(panel["date"])

    always_exclude = {"date", "ticker", "fwd_ret_1m"}
    macro_in_panel = [c for c in MACRO_COLS if c in panel.columns]
    feat_cols      = [c for c in panel.columns
                      if c not in always_exclude and c not in macro_in_panel]
    N  = len(feat_cols)
    X  = panel[feat_cols].values.astype(np.float64)
    T  = X.shape[0]

    print(f"Features: {N} | Observations: {T:,}")
    if macro_in_panel:
        print(f"Macro cols excluded from correlation diagnostics: {macro_in_panel}")

    # ─────────────────────────────────────────────────────────────────────────
    # PART A — IC CORRELATION MATRIX (SPEARMAN)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("PART A: Spearman IC Correlation Matrix")
    print(f"{'─'*50}")

    print("  Computing Spearman correlations (may take ~30s for 100+ features)...")
    corr_matrix, _ = spearmanr(X)
    if N == 2:
        corr_matrix = np.array([[1.0, corr_matrix], [corr_matrix, 1.0]])

    ic_df = pd.DataFrame(corr_matrix, index=feat_cols, columns=feat_cols)
    ic_df.to_csv(DATA_DIR / "diag_ic_corr_matrix.csv")

    # Flag redundant pairs
    rows = []
    for i in range(N):
        for j in range(i + 1, N):
            c = float(corr_matrix[i, j])
            if abs(c) >= IC_CORR_FLAG:
                rows.append({
                    "feature_A":    feat_cols[i],
                    "feature_B":    feat_cols[j],
                    "spearman_corr": round(c, 4),
                    "abs_corr":      round(abs(c), 4),
                })
    redundant = (pd.DataFrame(rows)
                 .sort_values("abs_corr", ascending=False)
                 .reset_index(drop=True))
    redundant.to_csv(DATA_DIR / "diag_redundant_pairs.csv", index=False)

    off_diag_vals = corr_matrix[np.triu_indices(N, k=1)]
    print(f"  Mean |off-diag corr|: {np.abs(off_diag_vals).mean():.4f}")
    print(f"  Max |off-diag corr|:  {np.abs(off_diag_vals).max():.4f}")
    print(f"  Flagged pairs (|corr|>{IC_CORR_FLAG}): {len(redundant)}")
    if len(redundant) > 0:
        print(f"  Top 10 redundant pairs:")
        print(redundant.head(10).to_string(index=False))

    plot_corr_heatmap(
        ic_df,
        title=f"Spearman IC Correlation Matrix  ({N} features, {T:,} obs)",
        out_path=FIG_DIR / "diag_ic_corr_heatmap.png",
        flag_val=IC_CORR_FLAG,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PART B — RMT EIGENVALUE ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("PART B: RMT Eigenvalue Analysis")
    print(f"{'─'*50}")

    q = N / T
    print(f"  q = N/T = {N}/{T:,} = {q:.6f}")

    lambda_plus  = marchenko_pastur_upper(q)
    lambda_minus = max(0.0, (1.0 - np.sqrt(q)) ** 2)

    # Eigenvalues of empirical correlation matrix
    eigenvalues = np.sort(np.linalg.eigvalsh(corr_matrix))[::-1]
    n_signal    = int((eigenvalues >= lambda_plus).sum())
    n_noise     = N - n_signal

    print(f"  MP upper bound λ+:  {lambda_plus:.4f}")
    print(f"  MP lower bound λ-:  {lambda_minus:.4f}")
    print(f"  Signal eigenvalues: {n_signal}/{N}  "
          f"(explain {eigenvalues[:n_signal].sum()/eigenvalues.sum()*100:.1f}% of variance)")
    print(f"  Noise eigenvalues:  {n_noise}/{N}")
    print(f"  Top 5 eigenvalues:  {eigenvalues[:5].round(4)}")

    eigen_df = pd.DataFrame({
        "rank":        np.arange(1, N + 1),
        "eigenvalue":  eigenvalues,
        "lambda_plus": lambda_plus,
        "lambda_minus": lambda_minus,
        "is_signal":   eigenvalues >= lambda_plus,
    })
    eigen_df.to_csv(DATA_DIR / "diag_rmt_eigenvalues.csv", index=False)

    # RMT-denoised correlation matrix heatmap
    from utils_rmt import rmt_denoise_feature_corr
    corr_clean, rmt_info = rmt_denoise_feature_corr(X, verbose=True)
    plot_corr_heatmap(
        pd.DataFrame(corr_clean, index=feat_cols, columns=feat_cols),
        title=f"RMT-Denoised Correlation Matrix  (signal={rmt_info['n_signal']}/{N})",
        out_path=FIG_DIR / "diag_rmt_corr_heatmap.png",
        flag_val=IC_CORR_FLAG,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PART C — VARIANCE INFLATION FACTOR
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("PART C: Variance Inflation Factor (VIF)")
    print(f"{'─'*50}")

    X_vif = X
    if T > VIF_SAMPLE_ROWS:
        rng   = np.random.default_rng(42)
        idx   = rng.choice(T, size=VIF_SAMPLE_ROWS, replace=False)
        X_vif = X[idx]
        print(f"  Sampling {VIF_SAMPLE_ROWS:,} rows for VIF (faster)...")

    print(f"  Computing VIF for {N} features...")
    vif_df = compute_vif(X_vif, feat_cols)
    vif_df.to_csv(DATA_DIR / "diag_vif.csv", index=False)

    flagged_vif = vif_df[vif_df["flag_redundant"]]
    print(f"\n  VIF results (sorted highest first):")
    print(vif_df.head(20).to_string(index=False))
    if len(flagged_vif) > 0:
        print(f"\n  WARNING: {len(flagged_vif)} features with VIF > {VIF_FLAG}:")
        print(flagged_vif[["feature", "VIF"]].to_string(index=False))
    else:
        print(f"\n  All features have VIF <= {VIF_FLAG}  (acceptable)")

    # ─────────────────────────────────────────────────────────────────────────
    # PART D — PRE/POST ORTHOGONALIZATION COMPARISON (optional)
    # ─────────────────────────────────────────────────────────────────────────
    if ORTH_IN.exists():
        print(f"\n{'─'*50}")
        print("PART D: Pre/Post-Orthogonalization Comparison")
        print(f"{'─'*50}")

        panel_orth = pd.read_parquet(ORTH_IN)
        feat_orth  = [c for c in panel_orth.columns
                      if c not in always_exclude and c not in macro_in_panel]
        X_orth     = panel_orth[feat_orth].values.astype(np.float64)

        corr_orth, _      = spearmanr(X_orth)
        if N == 2:
            corr_orth = np.array([[1.0, corr_orth], [corr_orth, 1.0]])

        N_orth   = corr_orth.shape[0]
        off_raw  = np.abs(corr_matrix[np.triu_indices(N, k=1)]).mean()
        off_orth = np.abs(corr_orth[np.triu_indices(N_orth, k=1)]).mean()
        print(f"  Mean |off-diag corr|: raw={off_raw:.4f} → orth={off_orth:.4f}")
        print(f"  Reduction: {(off_raw - off_orth) / off_raw * 100:.1f}%")

        plot_corr_heatmap(
            pd.DataFrame(corr_orth, index=feat_orth, columns=feat_orth),
            title=f"IC Correlation Matrix (ORTHOGONALIZED, {N} features)",
            out_path=FIG_DIR / "diag_ic_corr_orth_heatmap.png",
            flag_val=IC_CORR_FLAG,
        )
    else:
        print(f"\nPart D skipped — {ORTH_IN.name} not found.")
        print("  Run 1b_orthogonalize.py first to enable comparison.")

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("DIAGNOSTIC SUMMARY")
    print("=" * 65)
    print(f"  Features analysed:          {N}")
    print(f"  Total observations:         {T:,}")
    print(f"  Mean |off-diag corr|:       {np.abs(off_diag_vals).mean():.4f}")
    print(f"  Redundant pairs (>0.50):    {len(redundant)}")
    print(f"  RMT signal eigenvalues:     {n_signal}/{N}")
    print(f"  VIF > {VIF_FLAG} (flagged):         {len(flagged_vif)}")
    print(f"\n  Output files saved to: {DATA_DIR}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
