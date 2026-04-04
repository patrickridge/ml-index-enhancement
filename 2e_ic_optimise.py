"""
2e_ic_optimise.py — Differentiable IC Optimisation of Factor Weights
=====================================================================
Finds the optimal linear combination of the 43 selected factors that
maximises mean cross-sectional IC on the training set (2010–2020).

This replaces the static IC-decay weights (mean |IC| over 12 months)
with gradient-optimised weights that directly maximise predictive power.

Method:
  - Weight vector w ∈ R^43, softmax-normalised so weights sum to 1
  - Combined score per stock per month: score = Σ w_j × zscore(factor_j)
  - Contrarian factors (majority_sign == '-') are sign-flipped before combining
  - IC per month: Pearson correlation between score and fwd_ret_1m
    (differentiable proxy for Spearman IC — valid when both are z-scored)
  - Loss: -mean(IC) across all training months
  - Optimiser: Adam lr=0.01, 500 epochs

Train / Val split:
  - Train : 2010–2020 (same as factor analysis training period)
  - Val   : 2021–2022 (held out from both training and model test period)
  - Test  : 2023–     (model test period — not touched here)

Outputs:
  data/factor_selected_optimised.csv   — factor_selected.csv + new weight column
  figures/factor_weights_optimised.png — bar chart: current vs optimised weights

Usage:
  python 2e_ic_optimise.py
"""

import time as _time
_t0 = _time.time()

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
FIG_DIR    = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START = "2010-01-01"
TRAIN_END   = "2020-12-31"
VAL_END     = "2022-12-31"

EPOCHS   = 500
LR       = 0.01
MIN_STOCKS = 30   # minimum stocks per month to include that month


# ── Load data ──────────────────────────────────────────────────────────────────
print("=" * 65)
print("DIFFERENTIABLE IC OPTIMISATION")
print("=" * 65)

print("\nLoading factor_selected.csv …")
factor_df = pd.read_csv(DATA_DIR / "factor_selected.csv")
factors      = factor_df["factor"].tolist()
current_weights = factor_df["weight"].values.copy()

# Sign: +1 for positive IC factors, -1 for contrarian (flip so all point same direction)
signs = np.array([1 if s == "+" else -1 for s in factor_df["majority_sign"]])

print(f"  {len(factors)} selected factors")
print(f"  Positive IC: {(signs == 1).sum()}  |  Contrarian (flipped): {(signs == -1).sum()}")

print("\nLoading panel_monthly_enriched.parquet …")
panel = pq.read_table(DATA_DIR / "panel_monthly_enriched.parquet").to_pandas()
panel["date"] = pd.to_datetime(panel["date"])

# Split
train = panel[(panel["date"] >= TRAIN_START) & (panel["date"] <= TRAIN_END)].copy()
val   = panel[(panel["date"] >  TRAIN_END)   & (panel["date"] <= VAL_END)].copy()

print(f"  Train: {train['date'].nunique()} months  ({TRAIN_START[:4]}–{TRAIN_END[:4]})")
print(f"  Val:   {val['date'].nunique()}  months  ({TRAIN_END[:4]}–{VAL_END[:4]})")

# Check all factors present
missing_facs = [f for f in factors if f not in panel.columns]
if missing_facs:
    print(f"  WARNING: {len(missing_facs)} factors not in panel — {missing_facs[:5]}")
    factors    = [f for f in factors if f in panel.columns]
    idx_keep   = [i for i, f in enumerate(factor_df["factor"]) if f in panel.columns]
    factor_df  = factor_df.iloc[idx_keep].reset_index(drop=True)
    signs      = signs[idx_keep]
    current_weights = current_weights[idx_keep]
    print(f"  Using {len(factors)} factors after filtering")

n_factors = len(factors)


# ── Build monthly cross-sections ──────────────────────────────────────────────
def build_monthly_data(df: pd.DataFrame) -> list[tuple]:
    """
    Returns list of (factor_matrix, fwd_ret) tuples per month.
    factor_matrix: (n_stocks, n_factors) — cross-sectionally z-scored, sign-applied
    fwd_ret:       (n_stocks,) — forward 1m return
    """
    months = []
    for dt, grp in df.groupby("date"):
        sub = grp[factors + ["fwd_ret_1m"]].dropna()
        if len(sub) < MIN_STOCKS:
            continue

        X = sub[factors].values.astype(np.float32)   # (n_stocks, n_factors)
        y = sub["fwd_ret_1m"].values.astype(np.float32)

        # Cross-sectional z-score each factor
        mu  = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-9
        X   = (X - mu) / std

        # Apply sign flip for contrarian factors
        X = X * signs[np.newaxis, :]   # broadcast: (n_stocks, n_factors)

        # z-score forward returns too (for Pearson IC = differentiable Spearman proxy)
        y = (y - y.mean()) / (y.std() + 1e-9)

        months.append((X, y))
    return months


print("\nBuilding monthly cross-sections …")
train_months = build_monthly_data(train)
val_months   = build_monthly_data(val)
print(f"  Train: {len(train_months)} valid months")
print(f"  Val:   {len(val_months)} valid months")


# ── IC computation (numpy, differentiable via manual grad) ──────────────────
def compute_ic(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Pearson IC between combined score and forward returns."""
    score = X @ w                           # (n_stocks,)
    score = (score - score.mean()) / (score.std() + 1e-9)
    ic    = float(np.mean(score * y))       # Pearson correlation (y already z-scored)
    return ic


def mean_ic(months: list, w: np.ndarray) -> float:
    """Mean IC across all months."""
    return float(np.mean([compute_ic(X, y, w) for X, y in months]))


# ── Gradient of mean IC w.r.t. log-weights (softmax parameterisation) ────────
def grad_mean_ic(months: list, log_w: np.ndarray) -> np.ndarray:
    """
    Compute gradient of mean IC w.r.t. log_w using finite differences.
    Fast enough for 43 factors × 500 epochs.
    """
    eps   = 1e-4
    grad  = np.zeros_like(log_w)
    w0    = softmax(log_w)
    ic0   = mean_ic(months, w0)

    for i in range(len(log_w)):
        lw_p = log_w.copy(); lw_p[i] += eps
        ic_p = mean_ic(months, softmax(lw_p))
        grad[i] = (ic_p - ic0) / eps
    return grad


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ── Try PyTorch first (faster + exact gradients) ─────────────────────────────
USE_TORCH = False
try:
    import torch
    USE_TORCH = True
    print("\nUsing PyTorch for gradient computation (faster)")
except ImportError:
    print("\nPyTorch not available — using finite-difference gradients (slower)")


def optimise_torch(train_months: list) -> np.ndarray:
    """Adam optimisation using PyTorch autograd."""
    import torch

    # Convert to tensors
    Xs = [torch.tensor(X, dtype=torch.float32) for X, _ in train_months]
    ys = [torch.tensor(y, dtype=torch.float32) for _, y in train_months]

    log_w = torch.zeros(n_factors, requires_grad=True)
    opt   = torch.optim.Adam([log_w], lr=LR)

    best_loss = float("inf")
    best_log_w = log_w.detach().clone()

    for epoch in range(EPOCHS):
        opt.zero_grad()
        w = torch.softmax(log_w, dim=0)
        ics = []
        for X, y in zip(Xs, ys):
            score = X @ w
            score = (score - score.mean()) / (score.std() + 1e-9)
            ic    = (score * y).mean()
            ics.append(ic)
        loss = -torch.stack(ics).mean()
        loss.backward()
        opt.step()

        if float(loss) < best_loss:
            best_loss  = float(loss)
            best_log_w = log_w.detach().clone()

        if (epoch + 1) % 100 == 0:
            w_np   = torch.softmax(log_w, dim=0).detach().numpy()
            val_ic = mean_ic(val_months, w_np)
            print(f"  Epoch {epoch+1:4d} | train IC: {-float(loss):.4f} | val IC: {val_ic:.4f}")

    return torch.softmax(best_log_w, dim=0).detach().numpy()


def optimise_numpy(train_months: list) -> np.ndarray:
    """Adam optimisation using numpy (finite-difference gradients)."""
    log_w  = np.zeros(n_factors)
    m, v   = np.zeros(n_factors), np.zeros(n_factors)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_ic = -np.inf
    best_w  = softmax(log_w).copy()

    for epoch in range(EPOCHS):
        g   = -grad_mean_ic(train_months, log_w)   # negative because we maximise
        m   = b1 * m + (1 - b1) * g
        v   = b2 * v + (1 - b2) * g ** 2
        t   = epoch + 1
        m_h = m / (1 - b1 ** t)
        v_h = v / (1 - b2 ** t)
        log_w -= LR * m_h / (np.sqrt(v_h) + eps)

        if (epoch + 1) % 100 == 0:
            w_np   = softmax(log_w)
            tr_ic  = mean_ic(train_months, w_np)
            val_ic = mean_ic(val_months, w_np)
            print(f"  Epoch {epoch+1:4d} | train IC: {tr_ic:.4f} | val IC: {val_ic:.4f}")
            if tr_ic > best_ic:
                best_ic = tr_ic
                best_w  = w_np.copy()

    return best_w


# ── Baseline IC using current IC-decay weights ────────────────────────────────
print("\nBaseline (current IC-decay weights):")
baseline_w    = current_weights / current_weights.sum()   # normalise just in case
baseline_tr   = mean_ic(train_months, baseline_w * signs)  # signs already in X, undo
# Actually signs are already baked into X matrices. Weights should not re-apply signs.
# Re-check: X has signs applied, so baseline_w just needs to be the weight vector.
baseline_tr   = mean_ic(train_months, baseline_w)
baseline_val  = mean_ic(val_months,   baseline_w)
print(f"  Train IC: {baseline_tr:.4f}  |  Val IC: {baseline_val:.4f}")

# ── Optimise ──────────────────────────────────────────────────────────────────
print(f"\nOptimising weights ({EPOCHS} epochs, lr={LR}) …")
if USE_TORCH:
    opt_weights = optimise_torch(train_months)
else:
    opt_weights = optimise_numpy(train_months)

opt_tr  = mean_ic(train_months, opt_weights)
opt_val = mean_ic(val_months,   opt_weights)
print(f"\nOptimised weights:")
print(f"  Train IC: {opt_tr:.4f}  (vs baseline {baseline_tr:.4f}, Δ={opt_tr-baseline_tr:+.4f})")
print(f"  Val IC:   {opt_val:.4f}  (vs baseline {baseline_val:.4f}, Δ={opt_val-baseline_val:+.4f})")


# ── Comparison table ──────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"{'Factor':<28} {'Current Wt':>10} {'Opt Wt':>10} {'Change':>10}")
print(f"{'─'*65}")

factor_df["weight_optimised"] = opt_weights
factor_df["weight_change"]    = opt_weights - baseline_w
sorted_idx = np.argsort(-opt_weights)

for i in sorted_idx[:20]:
    fac    = factors[i]
    cur_w  = baseline_w[i]
    opt_w  = opt_weights[i]
    delta  = opt_w - cur_w
    sign   = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else " ")
    print(f"  {fac:<26} {cur_w:>9.4f}  {opt_w:>9.4f}  {delta:>+9.4f} {sign}")

if len(factors) > 20:
    print(f"  … ({len(factors) - 20} more factors)")
print(f"{'─'*65}")


# ── Save updated factor_selected ──────────────────────────────────────────────
out_csv = DATA_DIR / "factor_selected_optimised.csv"
factor_df.to_csv(out_csv, index=False)
print(f"\nSaved → {out_csv}")


# ── Plot: current vs optimised weights ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sort by optimised weight descending
order = np.argsort(-opt_weights)
fac_labels  = [factors[i][:20] for i in order]   # truncate long names
cur_ordered = baseline_w[order]
opt_ordered = opt_weights[order]
x = np.arange(len(factors))
w = 0.38

ax = axes[0]
ax.barh(x + w/2, cur_ordered[::-1], w, label="IC-decay (current)", color="#1565C0", alpha=0.75)
ax.barh(x - w/2, opt_ordered[::-1], w, label="Optimised",          color="#E65100", alpha=0.75)
ax.set_yticks(x)
ax.set_yticklabels(fac_labels[::-1], fontsize=7)
ax.set_xlabel("Weight")
ax.set_title("Factor Weights: Current vs Optimised")
ax.legend(fontsize=9)
ax.grid(axis="x", alpha=0.3)

# Right: IC improvement bar
ax2 = axes[1]
metrics = pd.DataFrame({
    "IC": [baseline_tr, opt_tr, baseline_val, opt_val],
    "Period": ["Train", "Train", "Val", "Val"],
    "Method": ["Current", "Optimised", "Current", "Optimised"],
})
for i, (method, color) in enumerate([("Current", "#1565C0"), ("Optimised", "#E65100")]):
    sub = metrics[metrics["Method"] == method]
    ax2.bar(np.array([0, 1]) + i * 0.35, sub["IC"].values, 0.35,
            label=method, color=color, alpha=0.8, edgecolor="white")

ax2.set_xticks([0.175, 1.175])
ax2.set_xticklabels(["Train (2010–2020)", "Val (2021–2022)"], fontsize=10)
ax2.set_ylabel("Mean IC")
ax2.set_title("IC Comparison: Current vs Optimised Weights")
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)
# Annotate bars
for bars in ax2.containers:
    ax2.bar_label(bars, fmt="%.4f", padding=2, fontsize=9)

plt.tight_layout()
out_png = FIG_DIR / "factor_weights_optimised.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out_png}")

print(f"\nNext step: use 'weight_optimised' column from {out_csv}")
print("  in place of 'weight' column when combining factor scores.")
print(f"\nDone in {(_time.time() - _t0) / 60:.1f} min")
