"""
5e_dapo_agent.py — DAPO: Dynamic Sampling Policy Optimisation
=============================================================
Implements DAPO for portfolio alpha tilt, extending GRPO (5d) with three
key innovations from the ByteDance/Seed 2025 paper:

  1. Clip-Higher (asymmetric clipping)
       Standard PPO/GRPO clips both directions symmetrically at ε.
       DAPO uses ε_low (negative advantage) and ε_high (positive advantage):
         ratio_clipped = clamp(ratio, 1-ε_low, 1+ε_high)   ε_high > ε_low
       → allows the policy to move aggressively toward rewarding alphas
         while still capping downside to prevent large negative updates.

  2. Dynamic Sampling (adaptive G per state)
       Standard GRPO samples a fixed G candidates per state.
       DAPO estimates reward variance from an initial G_INIT sample:
         if var(rewards) > VARIANCE_THRESH → sample G_MAX total
         else                              → keep G_MIN total
       → hard states (volatile months where the right alpha is unclear)
         get more simulation budget; easy states get fewer.

  3. No KL penalty
       DAPO removes the β·KL regularisation that DeepSeek-R1's GRPO uses.
       The clip-higher mechanism alone provides sufficient stability.
       (Our GRPO in 5d has KL; DAPO here intentionally omits it.)

Walk-forward structure: identical 5-fold expanding window as 5c and 5d.
Factor weights recomputed from training data only per fold.

Run:
  python 5e_dapo_agent.py

Outputs:
  data/dapo_comparison.csv       — per-fold: DAPO vs GRPO vs Fixed
  figures/dapo_comparison.png    — cumulative alpha + per-fold IR chart
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from collections import deque
import random
import copy
from scipy.stats import pearsonr

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not installed — rule-based fallback only.")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
FIG_DIR  = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

PANEL_FILE   = DATA_DIR / "panel_monthly_enriched.parquet"
FACTORS_FILE = DATA_DIR / "factor_selected.csv"
WEIGHTS_FILE = DATA_DIR / "spx_weights.parquet"

# ── Walk-forward folds (same as 5c / 5d) ──────────────────────────────────────
FOLDS = [
    ("2013-12-31", "2014-01-01", "2015-12-31", "2014–2015"),
    ("2015-12-31", "2016-01-01", "2017-12-31", "2016–2017"),
    ("2017-12-31", "2018-01-01", "2019-12-31", "2018–2019"),
    ("2019-12-31", "2020-01-01", "2021-12-31", "2020–2021"),
    ("2021-12-31", "2022-01-01", "2025-12-31", "2022–2025"),
]
TRAIN_START_GLOBAL = "2010-01-01"

# ── Shared hyperparameters ─────────────────────────────────────────────────────
ALPHA_MIN   = 0.002
ALPHA_MAX   = 0.050
FIXED_ALPHA = 0.010
HIDDEN_DIM  = 64
SEED        = 42

# GRPO baseline (same as 5d for fair comparison)
GRPO_EPOCHS  = 400
GRPO_LR      = 3e-4
GRPO_G       = 4
GRPO_KL_BETA = 0.01

# DAPO-specific
DAPO_EPOCHS        = 400
DAPO_LR            = 3e-4
DAPO_EPS_LOW       = 0.20    # clip lower bound (negative advantage)
DAPO_EPS_HIGH      = 0.28    # clip upper bound (positive advantage) — asymmetric
DAPO_G_INIT        = 2       # initial candidates to estimate reward variance
DAPO_G_MIN         = 2       # minimum total group size
DAPO_G_MAX         = 8       # maximum total group size (hard states)
DAPO_VAR_THRESH    = 0.05    # reward variance threshold for dynamic sampling

STATE_COLS = [
    "signal_strength",
    "signal_dispersion",
    "bench_vol",
    "recent_active_ret",
    "regime",
    "rolling_te",
]

np.random.seed(SEED)
random.seed(SEED)
if HAS_TORCH:
    torch.manual_seed(SEED)


# =============================================================================
# FACTOR COMBO SCORES
# =============================================================================

def compute_fold_weights(panel_train, factor_names, signs):
    ics = {}
    for f in factor_names:
        if f not in panel_train.columns:
            ics[f] = 0.0
            continue
        ic_vals = []
        for dt, grp in panel_train.groupby("date"):
            sub = grp[[f, "fwd_ret_1m"]].dropna()
            if len(sub) < 20:
                continue
            try:
                r, _ = pearsonr(sub[f], sub["fwd_ret_1m"])
                if np.isfinite(r):
                    ic_vals.append(r)
            except Exception:
                pass
        ics[f] = float(np.mean(np.abs(ic_vals))) if ic_vals else 0.0

    total = sum(ics.values())
    w = ({f: 1.0 / len(factor_names) for f in factor_names} if total < 1e-9
         else {f: v / total for f, v in ics.items()})
    return np.array([w[f] * signs.get(f, 1.0) for f in factor_names], dtype=np.float64)


def build_scores(panel_slice, factor_names, signed_w):
    available = [f for f in factor_names if f in panel_slice.columns]
    idx = [factor_names.index(f) for f in available]
    sw  = signed_w[idx]
    rows = []
    for dt, grp in panel_slice.groupby("date"):
        X  = grp[available].values.astype(np.float64)
        mu = np.nanmean(X, axis=0)
        sg = np.nanstd(X,  axis=0) + 1e-8
        Xz = np.where(np.isnan(X), 0.0, (X - mu) / sg)
        sc = Xz @ sw
        for i, (_, row) in enumerate(grp.iterrows()):
            rows.append({"date": dt, "ticker": row["ticker"],
                         "score": float(sc[i]),
                         "fwd_ret_1m": row.get("fwd_ret_1m", np.nan)})
    return pd.DataFrame(rows)


# =============================================================================
# PORTFOLIO SIMULATION
# =============================================================================

def simulate_month(scores_month, weights_month, alpha, top_n=100, bottom_n=100):
    df = scores_month.merge(
        weights_month[["ticker", "spx_weight"]], on="ticker", how="inner"
    ).dropna(subset=["score", "spx_weight", "fwd_ret_1m"])
    if len(df) < 50:
        return None
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    n  = len(df)
    tilt = pd.Series(0.0, index=df.index)
    tilt.iloc[:top_n]        = +alpha
    tilt.iloc[n - bottom_n:] = -alpha
    raw_w = (df["spx_weight"] + tilt).clip(lower=0.0)
    total = raw_w.sum()
    if total < 1e-8:
        return None
    port_w    = raw_w / total
    port_ret  = (port_w          * df["fwd_ret_1m"]).sum()
    bench_ret = (df["spx_weight"] * df["fwd_ret_1m"]).sum()
    return {"port_ret": port_ret, "bench_ret": bench_ret,
            "active_ret": port_ret - bench_ret}


# =============================================================================
# EPISODE TABLE
# =============================================================================

def build_episodes(scores, weights, ref_alpha=0.01, norm_params=None, fit_norm=False):
    scores  = scores.copy()
    weights = weights.copy()
    scores["date"]  = pd.to_datetime(scores["date"])
    weights["date"] = pd.to_datetime(weights["date"])

    rows = []
    for dt, sc_m in scores.groupby("date"):
        w_m = weights[weights["date"] == dt][["ticker", "spx_weight"]].copy()
        if len(sc_m) < 50:
            continue
        s_vals = sc_m["score"].values
        z      = (s_vals - s_vals.mean()) / (s_vals.std() + 1e-8)
        ref    = simulate_month(sc_m, w_m, alpha=ref_alpha)
        if ref is None:
            continue
        rows.append({"date": dt,
                     "signal_strength":   float(np.abs(z).mean()),
                     "signal_dispersion": float(z.std()),
                     "bench_ret":         ref["bench_ret"],
                     "active_ret_ref":    ref["active_ret"],
                     "_scores":           sc_m,
                     "_weights":          w_m})

    if not rows:
        return pd.DataFrame(), norm_params

    df = pd.DataFrame(rows).set_index("date").sort_index()
    df["bench_vol"]         = (df["bench_ret"]
                               .rolling(3, min_periods=2).std()
                               .fillna(df["bench_ret"].expanding().std())
                               .fillna(0.0)) * np.sqrt(12)
    df["rolling_te"]        = (df["active_ret_ref"]
                               .rolling(3, min_periods=2).std()
                               .fillna(df["active_ret_ref"].expanding().std())
                               .fillna(0.0)) * np.sqrt(12)
    df["recent_active_ret"] = df["active_ret_ref"].rolling(3, min_periods=1).mean().fillna(0.0)

    if fit_norm:
        vol_med = df["bench_vol"].expanding(min_periods=6).median()
        df["regime"] = (df["bench_vol"] > vol_med).astype(float).fillna(0.0)
        norm_params = {}
        for col in [c for c in STATE_COLS if c != "regime"]:
            mu, sg = df[col].mean(), df[col].std() + 1e-8
            norm_params[col] = (mu, sg)
        norm_params["_vol_median"] = float(
            df["bench_vol"].expanding(min_periods=6).median().iloc[-1])
    else:
        vol_thresh = (norm_params.get("_vol_median", df["bench_vol"].median())
                      if norm_params else df["bench_vol"].median())
        df["regime"] = (df["bench_vol"] > vol_thresh).astype(float).fillna(0.0)

    if norm_params:
        for col in [c for c in STATE_COLS if c != "regime"]:
            mu, sg = norm_params[col]
            df[col] = ((df[col] - mu) / sg).fillna(0.0).clip(-5.0, 5.0)

    return df, norm_params


# =============================================================================
# NEURAL NETWORK COMPONENTS
# =============================================================================

if HAS_TORCH:
    LOG_STD_MIN, LOG_STD_MAX = -10, 2

    class MLP(nn.Module):
        def __init__(self, in_dim, out_dim, hidden=HIDDEN_DIM):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, out_dim),
            )
        def forward(self, x): return self.net(x)

    class GaussianActor(nn.Module):
        """Squashed Gaussian policy over [ALPHA_MIN, ALPHA_MAX]."""
        def __init__(self, state_dim, hidden=HIDDEN_DIM):
            super().__init__()
            self.mean_net    = MLP(state_dim, 1, hidden)
            self.log_std_net = MLP(state_dim, 1, hidden)

        def forward(self, state):
            mean    = self.mean_net(state)
            log_std = self.log_std_net(state).clamp(LOG_STD_MIN, LOG_STD_MAX)
            return mean, log_std.exp()

        def sample(self, state):
            mean, std = self(state)
            dist  = torch.distributions.Normal(mean, std)
            x_t   = dist.rsample()
            y_t   = torch.tanh(x_t)
            alpha = ALPHA_MIN + (y_t + 1.0) / 2.0 * (ALPHA_MAX - ALPHA_MIN)
            log_p = dist.log_prob(x_t) - torch.log(1.0 - y_t.pow(2) + 1e-6)
            return alpha, log_p.sum(-1, keepdim=True), mean


def _safe_state(row):
    return np.clip(
        np.nan_to_num(row[STATE_COLS].values.astype(np.float32),
                      nan=0.0, posinf=0.0, neginf=0.0),
        -5.0, 5.0)


def compute_reward(active_ret):
    return active_ret * 12.0


# =============================================================================
# GRPO AGENT (baseline — same as 5d, with KL penalty)
# =============================================================================

class GRPOAgent:
    """GRPO with fixed G=4 and KL penalty. Baseline to compare against DAPO."""
    name = "GRPO"

    def __init__(self, state_dim):
        self.actor     = GaussianActor(state_dim)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=GRPO_LR)
        self.ref_actor = copy.deepcopy(self.actor)
        for p in self.ref_actor.parameters():
            p.requires_grad_(False)

    def select_action(self, state, deterministic=False):
        s = torch.nan_to_num(torch.FloatTensor(state).unsqueeze(0), nan=0.0)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(s)
                y = torch.tanh(mean)
                a = ALPHA_MIN + (y + 1.0) / 2.0 * (ALPHA_MAX - ALPHA_MIN)
            else:
                a, _, _ = self.actor.sample(s)
        return float(a.squeeze())

    def train(self, train_rows, verbose=False):
        avg_r = 0.0
        for epoch in range(GRPO_EPOCHS):
            ep_rewards = []
            for dt, row in train_rows:
                state = _safe_state(row)
                s     = torch.nan_to_num(torch.FloatTensor(state).unsqueeze(0), nan=0.0)

                group_log_probs, group_rewards = [], []
                for _ in range(GRPO_G):
                    alpha_t, log_p, _ = self.actor.sample(s)
                    a   = float(alpha_t.squeeze())
                    res = simulate_month(row["_scores"], row["_weights"], alpha=a)
                    if res is None:
                        continue
                    group_log_probs.append(log_p)
                    group_rewards.append(compute_reward(res["active_ret"]))

                if len(group_rewards) < 2:
                    continue

                ep_rewards.extend(group_rewards)
                r_arr = np.array(group_rewards, dtype=np.float32)
                adv   = torch.FloatTensor(
                    (r_arr - r_arr.mean()) / (r_arr.std() + 1e-8))

                log_probs = torch.cat(group_log_probs, dim=0).squeeze(-1)
                pg_loss   = -(log_probs * adv).mean()

                mu1, s1 = self.actor(s)
                with torch.no_grad():
                    mu2, s2 = self.ref_actor(s)
                kl   = (torch.log(s2 / (s1 + 1e-8))
                        + (s1.pow(2) + (mu1 - mu2).pow(2)) / (2 * s2.pow(2) + 1e-8)
                        - 0.5).mean()
                loss = pg_loss + GRPO_KL_BETA * kl

                self.actor_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_opt.step()

            avg_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            if verbose and epoch % 100 == 0:
                print(f"    [GRPO] Epoch {epoch:4d}  avg_r={avg_r:+.4f}")
        return avg_r


# =============================================================================
# DAPO AGENT
# =============================================================================

class DAPOAgent:
    """
    DAPO: Dynamic Sampling Policy Optimisation.

    Three improvements over GRPO:
    1. Clip-higher: asymmetric clipping eps_low=0.20 / eps_high=0.28
    2. Dynamic G: sample more candidates for high-variance (hard) states
    3. No KL penalty: clip-higher provides sufficient regularisation

    For each market state:
      a) Sample G_INIT=2 candidate alphas → estimate reward variance
      b) If var(rewards) > VARIANCE_THRESH → extend to G_MAX=8 total samples
         else keep G_MIN=2 (no extra simulation budget needed)
      c) Compute group-relative advantage A_i = (r_i - mean) / std
      d) Clip-higher policy gradient update
    """
    name = "DAPO"

    def __init__(self, state_dim):
        self.actor     = GaussianActor(state_dim)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=DAPO_LR)

    def select_action(self, state, deterministic=False):
        s = torch.nan_to_num(torch.FloatTensor(state).unsqueeze(0), nan=0.0)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(s)
                y = torch.tanh(mean)
                a = ALPHA_MIN + (y + 1.0) / 2.0 * (ALPHA_MAX - ALPHA_MIN)
            else:
                a, _, _ = self.actor.sample(s)
        return float(a.squeeze())

    def _sample_group(self, s, row, n_samples):
        """Sample n_samples candidate alphas, simulate each, return (alphas, log_probs, rewards)."""
        alphas, log_probs, rewards = [], [], []
        for _ in range(n_samples):
            alpha_t, log_p, _ = self.actor.sample(s)
            a   = float(alpha_t.squeeze())
            res = simulate_month(row["_scores"], row["_weights"], alpha=a)
            if res is None:
                continue
            alphas.append(a)
            log_probs.append(log_p)
            rewards.append(compute_reward(res["active_ret"]))
        return alphas, log_probs, rewards

    def train(self, train_rows, verbose=False):
        avg_r = 0.0
        for epoch in range(DAPO_EPOCHS):
            ep_rewards = []
            for dt, row in train_rows:
                state = _safe_state(row)
                s     = torch.nan_to_num(torch.FloatTensor(state).unsqueeze(0), nan=0.0)

                # ── Phase 1: initial sample to estimate state difficulty ──────
                alphas, log_probs, rewards = self._sample_group(s, row, DAPO_G_INIT)

                if len(rewards) >= 2:
                    # Dynamic sampling: hard states get more budget
                    r_var = float(np.var(rewards))
                    if r_var > DAPO_VAR_THRESH and len(rewards) < DAPO_G_MAX:
                        extra = DAPO_G_MAX - len(rewards)
                        a2, lp2, r2 = self._sample_group(s, row, extra)
                        alphas    += a2
                        log_probs += lp2
                        rewards   += r2

                if len(rewards) < 2:
                    continue

                ep_rewards.extend(rewards)

                # ── Group-relative advantage ──────────────────────────────────
                r_arr = np.array(rewards, dtype=np.float32)
                adv   = torch.FloatTensor(
                    (r_arr - r_arr.mean()) / (r_arr.std() + 1e-8))

                # ── Clip-higher policy gradient ───────────────────────────────
                # Re-compute log probs under current policy (needed for ratio)
                log_probs_old = torch.cat(log_probs, dim=0).squeeze(-1).detach()
                alphas_t      = torch.FloatTensor(alphas).unsqueeze(-1)

                # Get current log probs via reparameterisation
                mean_c, std_c = self.actor(s.expand(len(alphas), -1))
                y_t = ((alphas_t - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN) * 2.0 - 1.0
                       ).clamp(-0.9999, 0.9999)
                x_t = torch.atanh(y_t)
                dist = torch.distributions.Normal(mean_c, std_c)
                log_probs_new = (dist.log_prob(x_t)
                                 - torch.log(1.0 - y_t.pow(2) + 1e-6)).squeeze(-1)

                ratio = (log_probs_new - log_probs_old).exp()

                # Asymmetric clip: positive advantage → higher upper bound
                pos_mask = adv > 0
                clip_high = torch.where(pos_mask,
                                        torch.full_like(ratio, 1.0 + DAPO_EPS_HIGH),
                                        torch.full_like(ratio, 1.0 + DAPO_EPS_LOW))
                clip_low  = torch.full_like(ratio, 1.0 - DAPO_EPS_LOW)
                ratio_clipped = torch.max(torch.min(ratio, clip_high), clip_low)

                # Conservative objective: min(unclipped, clipped) × advantage
                obj  = torch.min(ratio * adv, ratio_clipped * adv)
                loss = -obj.mean()

                self.actor_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_opt.step()

            avg_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            if verbose and epoch % 100 == 0:
                n_ep = len(train_rows)
                print(f"    [DAPO] Epoch {epoch:4d}  avg_r={avg_r:+.4f}  "
                      f"(eps_low={DAPO_EPS_LOW}, eps_high={DAPO_EPS_HIGH})")
        return avg_r


# =============================================================================
# RULE-BASED FALLBACK
# =============================================================================

class RuleBasedAgent:
    name = "Rule"
    def __init__(self, state_dim): pass
    def select_action(self, state, deterministic=False):
        regime = state[4] if len(state) > 4 else 0.0
        return (ALPHA_MIN + ALPHA_MAX) / 2.0 * (1.0 - 0.4 * regime)
    def train(self, train_rows, verbose=False): return 0.0


# =============================================================================
# EVALUATE + STATS
# =============================================================================

def evaluate(agent, ep_test):
    rl_rows, fixed_rows = [], []
    for dt, row in ep_test.iterrows():
        state    = _safe_state(row)
        alpha_rl = agent.select_action(state, deterministic=True)
        r_rl     = simulate_month(row["_scores"], row["_weights"], alpha=alpha_rl)
        if r_rl:
            rl_rows.append({"date": dt, "alpha_used": alpha_rl,
                            "regime": row["regime"], **r_rl})
        r_fx = simulate_month(row["_scores"], row["_weights"], alpha=FIXED_ALPHA)
        if r_fx:
            fixed_rows.append({"date": dt, "alpha_used": FIXED_ALPHA,
                                "regime": row["regime"], **r_fx})
    rl_bt    = (pd.DataFrame(rl_rows).set_index("date")
                if rl_rows else pd.DataFrame())
    fixed_bt = (pd.DataFrame(fixed_rows).set_index("date")
                if fixed_rows else pd.DataFrame())
    return rl_bt, fixed_bt


def ie_stats(bt):
    if bt.empty or "active_ret" not in bt.columns:
        return {}
    r = bt["active_ret"].dropna()
    if len(r) < 4:
        return {}
    n         = len(r)
    ann_alpha = (1 + r).prod() ** (12 / n) - 1
    track_err = r.std(ddof=1) * np.sqrt(12)
    ir        = ann_alpha / track_err if track_err > 0 else float("nan")
    nav       = (1 + r).cumprod()
    return dict(ann_alpha=ann_alpha, track_err=track_err, info_ratio=ir,
                hit_rate=float((r > 0).mean()),
                max_active_dd=float((nav / nav.cummax() - 1).min()),
                n_months=n)


# =============================================================================
# PLOTTING
# =============================================================================

COLORS = {"DAPO": "#FF6B9D", "GRPO": "#66BB6A", "Fixed": "#AAAAAA"}


def plot_comparison(all_bt, fold_summary, algo_names):
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    fig.suptitle(
        "DAPO vs GRPO — Walk-Forward Out-of-Sample\n"
        "DAPO: clip-higher (ε↑=0.28) + dynamic sampling (G=2–8)",
        fontsize=13, fontweight="bold")

    ax = axes[0]
    for name, bt in all_bt.items():
        if bt.empty:
            continue
        nav = (1 + bt["active_ret"]).cumprod() - 1
        ax.plot(nav.index, nav * 100, label=name,
                color=COLORS.get(name, "#999"),
                lw=2, linestyle="--" if name == "Fixed" else "-")
    ax.axhline(0, color="black", lw=0.8, linestyle=":")
    ax.set_ylabel("Cumulative Active Return (%)")
    ax.set_title("Cumulative Active Return (All Folds Concatenated)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax    = axes[1]
    lbls  = fold_summary["label"].tolist()
    x     = np.arange(len(lbls))
    order = algo_names + ["Fixed"]
    n_a   = len(order)
    w     = 0.20
    for i, name in enumerate(order):
        key  = f"{name.lower()}_ir"
        if key not in fold_summary.columns:
            continue
        vals   = fold_summary[key].tolist()
        offset = (i - (n_a - 1) / 2) * (w + 0.02)
        ax.bar(x + offset, vals, w, label=name,
               color=COLORS.get(name, "#999"), alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(lbls, fontsize=9)
    ax.set_ylabel("Information Ratio")
    ax.set_title("Per-Fold IR: DAPO vs GRPO vs Fixed Alpha")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = FIG_DIR / "dapo_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("5e_dapo_agent.py — DAPO vs GRPO Walk-Forward Comparison")
    print("DAPO: clip-higher (ε↑=0.28) + dynamic G (2–8) + no KL")
    print("=" * 70)

    if not HAS_TORCH:
        print("WARNING: PyTorch not available — rule-based fallback.\n")

    print("\nLoading data ...")
    panel   = pd.read_parquet(PANEL_FILE)
    panel["date"] = pd.to_datetime(panel["date"])
    factors = pd.read_csv(FACTORS_FILE)
    weights = pd.read_parquet(WEIGHTS_FILE)
    weights["date"] = pd.to_datetime(weights["date"])

    factor_names = list(factors["factor"])
    signs        = (factors.set_index("factor")["majority_sign"]
                    .map({"+": 1.0, "-": -1.0}).fillna(1.0).to_dict())

    print(f"  Panel:   {panel['date'].min().date()} -> {panel['date'].max().date()}")
    print(f"  Factors: {len(factor_names)}")

    state_dim  = len(STATE_COLS)
    algo_names = ["DAPO", "GRPO"]

    all_bt        = {n: [] for n in algo_names + ["Fixed"]}
    fold_summary  = []

    for fold_idx, (train_end, test_start, test_end, label) in enumerate(FOLDS):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx+1}/5 — {label}  |  train up to {train_end}  |"
              f"  test {test_start} → {test_end}")
        print("="*70)

        # ── Data slices ───────────────────────────────────────────────────────
        panel_train = panel[(panel["date"] >= TRAIN_START_GLOBAL) &
                            (panel["date"] <= train_end)].copy()
        panel_test  = panel[(panel["date"] >= test_start) &
                            (panel["date"] <= test_end)].copy()

        if len(panel_train) < 500 or len(panel_test) < 50:
            print(f"  Skipping fold — insufficient data.")
            continue

        # ── Factor weights from training data only ────────────────────────────
        print("  Computing fold factor weights (training data only) ...")
        signed_w = compute_fold_weights(panel_train, factor_names, signs)

        # ── Build scores ──────────────────────────────────────────────────────
        print("  Building scores ...")
        sc_train = build_scores(panel_train, factor_names, signed_w)
        sc_test  = build_scores(panel_test,  factor_names, signed_w)

        # ── Build episode tables ──────────────────────────────────────────────
        ep_train, norm_p = build_episodes(sc_train, weights, fit_norm=True)
        ep_test,  _      = build_episodes(sc_test,  weights, norm_params=norm_p)

        if ep_train.empty or ep_test.empty:
            print("  Skipping fold — empty episodes.")
            continue

        train_rows = list(ep_train.iterrows())
        print(f"  Train: {len(train_rows)} months  |  Test: {len(ep_test)} months")

        # ── Train each algorithm ──────────────────────────────────────────────
        fold_bt   = {}
        fold_rows = {"label": label, "n_months": len(ep_test)}

        AgentClasses = [DAPOAgent, GRPOAgent] if HAS_TORCH else [RuleBasedAgent] * 2

        for name, AgentCls in zip(algo_names, AgentClasses):
            print(f"\n  [{name}] Training {DAPO_EPOCHS} epochs ...")
            agent   = AgentCls(state_dim)
            avg_r   = agent.train(train_rows, verbose=True)
            bt, fbt = evaluate(agent, ep_test)
            s       = ie_stats(bt)
            fold_bt[name] = bt
            print(f"  [{name}] IR={s.get('info_ratio', float('nan')):.3f}  "
                  f"α={s.get('ann_alpha', 0)*100:.2f}%  "
                  f"TE={s.get('track_err', 0)*100:.2f}%")
            fold_rows[f"{name.lower()}_ir"]    = s.get("info_ratio", float("nan"))
            fold_rows[f"{name.lower()}_alpha"] = s.get("ann_alpha", float("nan"))
            fold_rows[f"{name.lower()}_te"]    = s.get("track_err", float("nan"))
            fold_rows[f"{name.lower()}_maxdd"] = s.get("max_active_dd", float("nan"))

        # Fixed baseline
        if not fbt.empty:
            fs = ie_stats(fbt)
            fold_bt["Fixed"] = fbt
            fold_rows["fixed_ir"]    = fs.get("info_ratio",   float("nan"))
            fold_rows["fixed_alpha"] = fs.get("ann_alpha",    float("nan"))
            fold_rows["fixed_te"]    = fs.get("track_err",    float("nan"))
            fold_rows["fixed_maxdd"] = fs.get("max_active_dd",float("nan"))

        fold_summary.append(fold_rows)

        for name in algo_names + ["Fixed"]:
            if name in fold_bt and not fold_bt[name].empty:
                all_bt[name].append(fold_bt[name])

    # ── Concatenate all folds ─────────────────────────────────────────────────
    all_bt_concat = {}
    for name in algo_names + ["Fixed"]:
        if all_bt[name]:
            all_bt_concat[name] = pd.concat(all_bt[name]).sort_index()

    fold_df = pd.DataFrame(fold_summary)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("WALK-FORWARD RESULTS SUMMARY")
    print("="*70)
    if not fold_df.empty:
        for name in algo_names + ["Fixed"]:
            col = f"{name.lower()}_ir"
            if col in fold_df.columns:
                irs   = fold_df[col].dropna()
                avg   = irs.mean()
                wins  = int((fold_df.get(f"{name.lower()}_ir", pd.Series())
                             > fold_df.get("fixed_ir", pd.Series())).sum()) if name != "Fixed" else 0
                print(f"  {name:<6} — avg IR: {avg:.3f}  (folds: "
                      + "  ".join(f"{v:.3f}" for v in fold_df[col].tolist()) + ")"
                      + (f"  beats Fixed: {wins}/5" if name != "Fixed" else ""))

    # ── Save outputs ──────────────────────────────────────────────────────────
    if not fold_df.empty:
        fold_df.to_csv(DATA_DIR / "dapo_comparison.csv", index=False)
        print(f"\nSaved -> data/dapo_comparison.csv")

    if fold_df.shape[0] > 0 and all_bt_concat:
        plot_comparison(all_bt_concat, fold_df, algo_names)

    print("\nDone.")


if __name__ == "__main__":
    main()
