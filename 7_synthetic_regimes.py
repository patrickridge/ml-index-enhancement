"""
7_synthetic_regimes.py - Diffusion-based regime stress test for the CS-T IE strategy.

The real test period (2023-2025) has only 7 risk-off months, giving an IR
confidence interval around ±0.6. This script trains a tiny conditional DDPM on
the full 2010-2025 macro history (spx_ret_1m, spx_vol_63d, market_trend_spx),
generates 1000 synthetic bear-market months, then pushes them through a linear
"feature → active return" bridge fitted on the real CS-T test window. The
output is a stress-tested IR distribution across the synthetic scenarios.

Writes figures/diffusion_stress_test.png and data/synthetic_stress_results.csv.

Run:  python 7_synthetic_regimes.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Paths
DATA_DIR    = Path("data")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

PANEL_PATH  = DATA_DIR / "panel_monthly_enriched.parquet"
CST_PATH    = DATA_DIR / "bt_ie_cs_transformer.csv"
OUT_FIG     = FIGURES_DIR / "diffusion_stress_test.png"
OUT_CSV     = DATA_DIR / "synthetic_stress_results.csv"

MACRO_COLS  = ["spx_ret_1m", "spx_vol_63d", "market_trend_spx"]
T_STEPS     = 100       # diffusion steps
N_EPOCHS    = 2000      # training epochs
N_SAMPLES   = 1000      # synthetic bear months to generate
BEAR_PCTILE = 33        # bottom N% SPX return months = bear regime
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
if TORCH_AVAILABLE:
    torch.manual_seed(RANDOM_SEED)


# 1.  Load macro features

def load_macro_features() -> pd.DataFrame:
    """Extract one macro row per month from the enriched panel."""
    print("Loading macro features from panel ...")
    needed = ["date", "ticker"] + MACRO_COLS
    panel  = pd.read_parquet(PANEL_PATH, columns=needed)

    # One row per month - macro cols are the same across all tickers in a month
    macro = (panel.groupby("date")[MACRO_COLS]
                  .first()
                  .reset_index()
                  .sort_values("date"))
    macro = macro.dropna(subset=MACRO_COLS)
    macro = macro[macro["date"] >= "2010-01-01"].reset_index(drop=True)

    # Bear regime: bottom BEAR_PCTILE of SPX return months
    threshold = np.percentile(macro["spx_ret_1m"], BEAR_PCTILE)
    macro["regime"] = (macro["spx_ret_1m"] < threshold).astype(int)  # 1=bear, 0=bull

    n_bear = macro["regime"].sum()
    n_bull = len(macro) - n_bear
    print(f"  Macro months: {len(macro)} | Bear: {n_bear} | Bull: {n_bull}")
    return macro


# 2.  DDPM components

def linear_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    betas     = torch.linspace(beta_start, beta_end, T)
    alphas    = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar


def sinusoidal_embedding(t: torch.Tensor, dim: int = 16) -> torch.Tensor:
    half  = dim // 2
    freqs = torch.exp(-torch.arange(half, dtype=torch.float32) *
                      np.log(10000.0) / half)
    args  = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class Denoiser(nn.Module):
    """
    Conditioned MLP denoiser.
    Inputs: noisy x_t (3-dim) + regime embedding (8-dim) + time embedding (16-dim)
    Output: predicted noise epsilon (3-dim)
    """
    def __init__(self, x_dim: int = 3, regime_dim: int = 8,
                 time_dim: int = 16, hidden: int = 128):
        super().__init__()
        self.regime_emb = nn.Embedding(2, regime_dim)
        in_dim = x_dim + regime_dim + time_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, x_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                regime: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_embedding(t)
        r_emb = self.regime_emb(regime)
        return self.net(torch.cat([x_t, t_emb, r_emb], dim=1))


# 3.  Training

def train_ddpm(X_norm: np.ndarray, regimes: np.ndarray,
               T: int = T_STEPS, n_epochs: int = N_EPOCHS,
               lr: float = 1e-3) -> tuple:
    """Train DDPM and return (model, schedule tensors, normalisation stats)."""
    betas, alphas, alpha_bar = linear_schedule(T)

    model     = Denoiser(x_dim=X_norm.shape[1])
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=n_epochs)

    X_t = torch.tensor(X_norm, dtype=torch.float32)
    R_t = torch.tensor(regimes, dtype=torch.long)
    n   = len(X_t)

    print(f"\nTraining DDPM on {n} months ({T} diffusion steps, {n_epochs} epochs) ...")
    for epoch in range(n_epochs):
        # Random diffusion timesteps
        t_batch  = torch.randint(1, T, (n,))
        noise    = torch.randn_like(X_t)
        ab       = alpha_bar[t_batch].unsqueeze(1)
        x_noisy  = torch.sqrt(ab) * X_t + torch.sqrt(1.0 - ab) * noise

        noise_pred = model(x_noisy, t_batch, R_t)
        loss       = F.mse_loss(noise_pred, noise)

        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step()

        if epoch % 500 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:5d} | Loss {loss.item():.5f}")

    return model, betas, alphas, alpha_bar


# 4.  Sampling

@torch.no_grad()
def sample_regime(model: nn.Module, betas: torch.Tensor, alphas: torch.Tensor,
                  alpha_bar: torch.Tensor, regime_label: int,
                  n_samples: int = N_SAMPLES, x_dim: int = 3,
                  T: int = T_STEPS) -> np.ndarray:
    """Generate synthetic feature vectors for a given regime via DDPM reverse process."""
    x      = torch.randn(n_samples, x_dim)
    regime = torch.full((n_samples,), regime_label, dtype=torch.long)

    for t in reversed(range(1, T)):
        t_tensor = torch.full((n_samples,), t, dtype=torch.long)
        eps_pred = model(x, t_tensor, regime)

        beta_t  = betas[t]
        alpha_t = alphas[t]
        ab_t    = alpha_bar[t]
        ab_prev = alpha_bar[t - 1] if t > 1 else torch.tensor(1.0)

        # DDPM posterior mean
        x0_pred = (x - torch.sqrt(1.0 - ab_t) * eps_pred) / (torch.sqrt(ab_t) + 1e-8)
        x0_pred = torch.clamp(x0_pred, -4.0, 4.0)

        coef1 = torch.sqrt(ab_prev) * beta_t / (1.0 - ab_t + 1e-8)
        coef2 = torch.sqrt(alpha_t) * (1.0 - ab_prev) / (1.0 - ab_t + 1e-8)
        mu    = coef1 * x0_pred + coef2 * x

        if t > 1:
            posterior_var = beta_t * (1.0 - ab_prev) / (1.0 - ab_t + 1e-8)
            x = mu + torch.sqrt(posterior_var.clamp(min=0)) * torch.randn_like(x)
        else:
            x = mu

    return x.numpy()


# 5.  Linear bridge: synthetic features → active returns

def fit_linear_bridge(macro: pd.DataFrame) -> tuple:
    """
    Fit OLS: CS-T active_ret ~ spx_ret + spx_vol + market_trend + const
    on the real test period (2023–2025).
    Returns (coefficients, intercept, r2).
    """
    cst = pd.read_csv(CST_PATH, parse_dates=["date"])
    merged = cst.merge(macro[["date"] + MACRO_COLS], on="date", how="inner")
    if len(merged) < 10:
        print("  Warning: insufficient overlap for linear bridge - using fallback.")
        return np.zeros(len(MACRO_COLS)), 0.0, 0.0

    X = merged[MACRO_COLS].values
    y = merged["active_ret"].values
    slope, intercept, *_ = stats.linregress(X[:, 0], y)  # primary: SPX return
    # Multi-variate via numpy lstsq
    X_aug = np.column_stack([np.ones(len(X)), X])
    coefs, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    intercept_mv = coefs[0]
    coefs_mv     = coefs[1:]
    y_pred       = X_aug @ coefs
    residuals    = y - y_pred
    residual_std = float(np.std(residuals))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    print(f"  Linear bridge R² = {r2:.3f}  residual σ = {residual_std*100:.3f}%  (n={len(merged)} months)")
    return coefs_mv, intercept_mv, r2, residual_std


def synthetic_to_active_returns(X_synth_norm: np.ndarray, X_mean: np.ndarray,
                                 X_std: np.ndarray, coefs: np.ndarray,
                                 intercept: float,
                                 residual_std: float = 0.0) -> np.ndarray:
    """
    Denormalise synthetic features, apply linear bridge, then add residual noise.

    The OLS bridge explains only part of the variance in active returns (low R²
    is typical with 3 macro features). Without residual noise, all 1,000 synthetic
    predictions cluster tightly around the regression mean → std ≈ 0 → IR blows up.
    Adding N(0, residual_std) restores realistic return dispersion.
    """
    X_real = X_synth_norm * X_std + X_mean
    y_hat  = intercept + X_real @ coefs
    noise  = np.random.normal(0.0, residual_std, size=len(y_hat)) if residual_std > 0 else 0.0
    return y_hat + noise


# 6.  Stress test statistics

def stress_ir(active_returns: np.ndarray, n_boot: int = 5000) -> dict:
    """Compute IR statistics from a vector of monthly active returns."""
    r = active_returns[np.isfinite(active_returns)]
    ann_alpha = r.mean() * 12
    te        = r.std() * np.sqrt(12) + 1e-8
    ir        = ann_alpha / te

    # Bootstrap CI
    boot_irs = []
    for _ in range(n_boot):
        s = np.random.choice(r, size=len(r), replace=True)
        a = s.mean() * 12; v = s.std() * np.sqrt(12) + 1e-8
        boot_irs.append(a / v)
    boot_irs = np.array(boot_irs)

    return dict(
        ann_alpha   = ann_alpha * 100,
        te          = te * 100,
        ir          = ir,
        ir_p5       = np.nanpercentile(boot_irs, 5),
        ir_p50      = np.nanpercentile(boot_irs, 50),
        ir_p95      = np.nanpercentile(boot_irs, 95),
        prob_pos    = (r > 0).mean() * 100,
        n           = len(r),
    )


# 7.  Plotting

ACADEMIC_STYLE = {
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "grid.linestyle":     "--",
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "font.family":        "sans-serif",
    "figure.dpi":         150,
}

def make_figure(macro: pd.DataFrame, X_norm: np.ndarray,
                X_synth_bear: np.ndarray, synth_active: np.ndarray,
                real_stats: dict, synth_stats: dict,
                X_mean: np.ndarray, X_std: np.ndarray):
    """3-row academic figure: feature distributions, active return dist, IR comparison."""
    with plt.rc_context(ACADEMIC_STYLE):
        fig = plt.figure(figsize=(14, 11))
        gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

        feature_labels = ["SPX Monthly Return", "SPX 63-day Vol", "Market Trend (SPX)"]
        bear_mask = macro["regime"] == 1
        bull_mask = macro["regime"] == 0

        # Row 1: Feature distributions (real bear vs bull vs synthetic bear)
        for col_idx, (feat_norm, feat_label) in enumerate(
                zip(MACRO_COLS, feature_labels)):
            ax = fig.add_subplot(gs[0, col_idx])
            real_bear = X_norm[bear_mask.values, col_idx]
            real_bull = X_norm[bull_mask.values, col_idx]
            synth_col = X_synth_bear[:, col_idx]

            ax.hist(real_bull,  bins=20, density=True, alpha=0.55,
                    color="#1565C0", label="Real bull",  zorder=2)
            ax.hist(real_bear,  bins=15, density=True, alpha=0.65,
                    color="#B71C1C", label="Real bear",  zorder=3)
            ax.hist(synth_col,  bins=30, density=True, alpha=0.40,
                    color="#E65100", label="Synthetic bear", zorder=4,
                    histtype="stepfilled")

            ax.set_title(feat_label, fontsize=10, fontweight="semibold")
            ax.set_ylabel("Density" if col_idx == 0 else "")
            if col_idx == 0:
                ax.legend(fontsize=8, framealpha=0.7)

        fig.text(0.02, 0.94, "A.  Feature Distributions: Real vs Synthetic",
                 fontsize=10, fontweight="bold", va="top")

        # Row 2: Synthetic active return distribution
        ax2 = fig.add_subplot(gs[1, :2])
        cst  = pd.read_csv(CST_PATH, parse_dates=["date"])
        real_bear_months = macro[bear_mask]["date"]
        real_bear_active = cst[cst["date"].isin(real_bear_months)]["active_ret"].values * 100

        ax2.hist(synth_active * 100, bins=50, density=True, alpha=0.65,
                 color="#E65100", label=f"Synthetic bear (n={len(synth_active):,})")
        if len(real_bear_active) > 3:
            ax2.hist(real_bear_active, bins=12, density=True, alpha=0.75,
                     color="#B71C1C", label=f"Real bear (n={len(real_bear_active)})")

        ax2.axvline(synth_active.mean() * 100, color="#E65100", lw=2, ls="--",
                    label=f"Synthetic mean = {synth_active.mean()*100:.3f}%")
        ax2.axvline(0, color="black", lw=1.0)
        ax2.set_xlabel("Monthly Active Return (%)")
        ax2.set_ylabel("Density")
        ax2.set_title("B.  Active Return Distribution - Bear Regime",
                      fontsize=10, fontweight="semibold")
        ax2.legend(fontsize=8, framealpha=0.7)

        # Row 2: Stress-tested IR bootstrap
        ax3 = fig.add_subplot(gs[1, 2])
        n_boot_plot = 3000
        boot_irs = []
        rng = np.random.default_rng(RANDOM_SEED)
        for _ in range(n_boot_plot):
            s = rng.choice(synth_active, size=len(synth_active), replace=True)
            a = s.mean() * 12; v = s.std() * np.sqrt(12) + 1e-8
            boot_irs.append(a / v)
        boot_irs = np.array(boot_irs)

        ax3.hist(boot_irs, bins=40, density=True, color="#4A148C", alpha=0.7)
        ax3.axvline(synth_stats["ir"], color="#4A148C", lw=2, ls="--",
                    label=f"IR = {synth_stats['ir']:.3f}")
        ax3.axvline(0, color="black", lw=1)
        ax3.set_xlabel("Information Ratio")
        ax3.set_title("Bootstrap IR Distribution\n(Synthetic Bear)",
                      fontsize=9, fontweight="semibold")
        ax3.legend(fontsize=8)

        # Row 3: Comparison bar chart
        ax4 = fig.add_subplot(gs[2, :])
        labels  = ["CS-T\nFull Period\n(real)", "CS-T\nRisk-Off\n(real)",
                   "Synthetic\nBear (mean)", "Synthetic\nBear (p5)",
                   "Synthetic\nBear (p95)"]
        values  = [real_stats["full_ir"], real_stats["riskoff_ir"],
                   synth_stats["ir"], synth_stats["ir_p5"], synth_stats["ir_p95"]]
        colors  = ["#1565C0", "#1565C0", "#E65100", "#E65100", "#E65100"]
        alphas  = [0.85, 0.65, 0.85, 0.45, 0.45]

        bars = ax4.bar(labels, values, color=colors,
                       alpha=0.8, width=0.55, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width() / 2, v + 0.03,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="semibold")

        ax4.axhline(0,   color="black",  lw=1.0)
        ax4.axhline(0.5, color="#2E7D32", lw=1.2, ls="--", alpha=0.7, label="IR = 0.5 (good)")
        ax4.axhline(1.0, color="#F57F17", lw=1.2, ls="--", alpha=0.7, label="IR = 1.0 (excellent)")
        ax4.set_ylabel("Information Ratio")
        ax4.set_title("C.  IR Comparison: Real vs Synthetic Bear Scenarios",
                      fontsize=10, fontweight="semibold")
        ax4.legend(fontsize=9, loc="upper right")

        fig.suptitle(
            "Regime-Conditional Diffusion Model - Stress Testing the CS-Transformer IE Strategy\n"
            f"DDPM: T={T_STEPS} steps, {N_EPOCHS} epochs | {N_SAMPLES:,} synthetic bear months generated",
            fontsize=11, fontweight="bold", y=0.98,
        )

        plt.savefig(OUT_FIG, bbox_inches="tight", dpi=150)
        print(f"\nFigure saved: {OUT_FIG}")
        plt.close()


# 8.  Main

def main():
    print("=" * 65)
    print(" 7_synthetic_regimes.py - Regime-Conditional DDPM Stress Test")
    print("=" * 65)

    if not TORCH_AVAILABLE:
        print("\nERROR: PyTorch not available. Install with: pip install torch")
        return

    # Load data
    macro = load_macro_features()
    X_raw = macro[MACRO_COLS].values.astype(np.float64)

    # Normalise
    X_mean = X_raw.mean(axis=0)
    X_std  = X_raw.std(axis=0) + 1e-8
    X_norm = (X_raw - X_mean) / X_std

    regimes = macro["regime"].values

    # Train DDPM
    model, betas, alphas, alpha_bar = train_ddpm(
        X_norm.astype(np.float32), regimes,
        T=T_STEPS, n_epochs=N_EPOCHS,
    )

    # Generate synthetic bear months
    print(f"\nGenerating {N_SAMPLES:,} synthetic bear market months ...")
    X_synth_bear = sample_regime(
        model, betas, alphas, alpha_bar,
        regime_label=1, n_samples=N_SAMPLES, x_dim=len(MACRO_COLS), T=T_STEPS,
    )
    print(f"  Synthetic bear features - mean: {X_synth_bear.mean(axis=0).round(3)}")

    # Linear bridge: features → active returns
    print("\nFitting linear bridge (active_ret ~ macro features) ...")
    coefs, intercept, r2, residual_std = fit_linear_bridge(macro)

    synth_active = synthetic_to_active_returns(
        X_synth_bear, X_mean, X_std, coefs, intercept, residual_std
    )

    # Statistics
    synth_stats = stress_ir(synth_active)

    cst     = pd.read_csv(CST_PATH, parse_dates=["date"])
    s_full  = cst["active_ret"]
    bear_dt = macro[macro["regime"] == 1]["date"]
    s_bear  = cst[cst["date"].isin(bear_dt)]["active_ret"]

    from numpy import sqrt
    def quick_ir(series):
        r = series.dropna().values
        if len(r) < 3:
            return np.nan
        return (r.mean() * 12) / (r.std() * sqrt(12) + 1e-8)

    real_stats = {
        "full_ir":    quick_ir(s_full),
        "riskoff_ir": quick_ir(s_bear),
    }

    # Print results
    print("\n" + "=" * 65)
    print(" STRESS TEST RESULTS")
    print("=" * 65)
    print(f"  Real CS-T full period IR:            {real_stats['full_ir']:>7.3f}")
    print(f"  Real CS-T risk-off IR (n={len(s_bear):2d} months):  {real_stats['riskoff_ir']:>7.3f}")
    print(f"  Synthetic bear IR (mean):             {synth_stats['ir']:>7.3f}")
    print(f"  Synthetic bear IR - 5th percentile:  {synth_stats['ir_p5']:>7.3f}")
    print(f"  Synthetic bear IR - 95th percentile: {synth_stats['ir_p95']:>7.3f}")
    print(f"  Probability of positive monthly α:   {synth_stats['prob_pos']:>6.1f}%")
    print(f"  Synthetic ann. alpha:                {synth_stats['ann_alpha']:>7.2f}%")
    print(f"  Synthetic tracking error:            {synth_stats['te']:>7.2f}%")
    print(f"  Linear bridge R²:                    {r2:>7.3f}")
    print("-" * 65)
    pct_above_half = (synth_active > 0).mean()
    n_above_half = int(pct_above_half * N_SAMPLES)
    print(f"  {n_above_half:,}/{N_SAMPLES:,} synthetic bear months show positive alpha "
          f"({pct_above_half*100:.1f}%)")
    print("=" * 65)

    # Save CSV
    results_df = pd.DataFrame({
        "metric":    ["full_period_ir", "real_riskoff_ir", "synthetic_bear_ir",
                      "synthetic_ir_p5", "synthetic_ir_p95", "synthetic_ann_alpha_pct",
                      "synthetic_te_pct", "prob_positive_pct", "bridge_r2"],
        "value":     [real_stats["full_ir"], real_stats["riskoff_ir"],
                      synth_stats["ir"], synth_stats["ir_p5"], synth_stats["ir_p95"],
                      synth_stats["ann_alpha"], synth_stats["te"],
                      synth_stats["prob_pos"], r2],
    })
    results_df.to_csv(OUT_CSV, index=False)
    print(f"Results saved: {OUT_CSV}")

    # Plot
    make_figure(macro, X_norm, X_synth_bear, synth_active,
                real_stats, synth_stats, X_mean, X_std)

    print("\nDone.")


if __name__ == "__main__":
    main()
