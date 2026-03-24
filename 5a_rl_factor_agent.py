"""
5a_rl_factor_agent.py — Layer 1 SAC: Adaptive Factor Weighting
===============================================================
Replaces fixed IC-optimised factor weights with a dynamic RL policy that
adapts which factors to trust each month based on recent IC history and
market regime.

State  : rolling 6m IC per factor (43) + rolling 12m IC per factor (43)
         + IC momentum per factor (43) + macro state (4) = 133-dim
Action : 43-dim weight vector via softmax over Gaussian samples
Reward : IC of combined factor signal this month (direct IC maximisation)
Policy : MLP only — no recurrence, no memory (Markov)

Baselines compared:
  - Equal weights (1/43 per factor)
  - IC-decay weights        (2d_factor_weights.py → 'weight')
  - IC-optimised weights    (2e_ic_optimise.py   → 'weight_optimised')
  - RL adaptive weights     (this script)

Train : 2010-2020  (~120 months, 6m warmup → ~114 actionable)
Val   : 2021-2022  (24 months)
Test  : 2023+

Usage:
  python 5a_rl_factor_agent.py
"""

import warnings, os, sys
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

# ── PyTorch ────────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    from torch.distributions import Normal
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not found — will use numpy fallback.")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_START  = "2010-01-01"
TRAIN_END    = "2020-12-31"
VAL_END      = "2022-12-31"
TEST_START   = "2023-01-01"

IC_LOOKBACK_SHORT  = 6    # months for short rolling IC
IC_LOOKBACK_LONG   = 12   # months for long rolling IC

MACRO_COLS   = ["vix_level", "vix_change_21d", "market_trend_spx", "spx_ret_3m"]
N_FACTORS    = 43

# SAC hyperparameters
HIDDEN_DIM   = 128
LR_ACTOR     = 3e-4
LR_CRITIC    = 3e-4
LR_ALPHA     = 3e-4
GAMMA        = 0.99
TAU          = 0.005
BUFFER_SIZE  = 50_000
BATCH_SIZE   = 64
N_EPOCHS     = 800
LOG_STD_MIN  = -4
LOG_STD_MAX  = 1
SEED         = 42

STATE_DIM    = N_FACTORS * 3 + len(MACRO_COLS)   # 133

if HAS_TORCH:
    torch.manual_seed(SEED)
np.random.seed(SEED)

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    panel   = pd.read_parquet(DATA_DIR / "panel_monthly_enriched.parquet")
    factors = pd.read_csv(DATA_DIR / "factor_selected_optimised.csv")
    panel["date"] = pd.to_datetime(panel["date"])
    return panel, factors


def compute_ic_matrix(panel, factor_names, signs):
    """
    Compute monthly cross-sectional IC for each factor.
    IC is sign-adjusted (positive IC = factor predicts returns in its natural direction).
    Returns DataFrame (months × factors).
    """
    print("  Computing monthly IC matrix ...")
    records = []
    months  = sorted(panel["date"].unique())
    for dt in months:
        grp = panel[panel["date"] == dt][factor_names + ["fwd_ret_1m"]].dropna()
        if len(grp) < 50:
            continue
        y  = grp["fwd_ret_1m"].values
        y  = (y - y.mean()) / (y.std() + 1e-8)
        row = {"date": dt}
        for f in factor_names:
            x  = grp[f].values
            x  = (x - x.mean()) / (x.std() + 1e-8)
            ic = float(np.corrcoef(x * signs[f], y)[0, 1])
            row[f] = 0.0 if np.isnan(ic) else ic
        records.append(row)

    ic_df = pd.DataFrame(records).set_index("date")
    print(f"  IC matrix: {ic_df.shape[0]} months × {ic_df.shape[1]} factors")
    return ic_df


def compute_combined_ic(panel, weights_vec, factor_names, signs, months):
    """
    Given a weight vector, compute the IC of the combined factor signal
    for each month. Used as reward signal.
    Returns dict {date -> combined IC}.
    """
    result = {}
    for dt in months:
        grp = panel[panel["date"] == dt][factor_names + ["fwd_ret_1m"]].dropna()
        if len(grp) < 50:
            result[dt] = 0.0
            continue
        X  = grp[factor_names].values.astype(np.float64)
        mu = np.nanmean(X, axis=0)
        sg = np.nanstd(X,  axis=0) + 1e-8
        Xz = np.where(np.isnan(X), 0.0, (X - mu) / sg)
        sw = weights_vec * signs.values
        sc = Xz @ sw
        y  = grp["fwd_ret_1m"].values
        if sc.std() < 1e-8 or y.std() < 1e-8:
            result[dt] = 0.0
        else:
            ic, _ = pearsonr(sc, y)
            result[dt] = 0.0 if np.isnan(ic) else float(ic)
    return result


# =============================================================================
# EPISODE / STATE BUILDER
# =============================================================================

def build_state_matrix(ic_df, macro_df, short=IC_LOOKBACK_SHORT, long=IC_LOOKBACK_LONG):
    """
    Build state vectors for all months.
    State = [rolling_ic_short (43), rolling_ic_long (43), ic_momentum (43), macro (4)]
    """
    months = ic_df.index.tolist()
    records = []

    for i, dt in enumerate(months):
        # Rolling IC lookbacks (need enough history)
        short_start = max(0, i - short)
        long_start  = max(0, i - long)

        ic_short = ic_df.iloc[short_start:i].mean().values if i > 0 else np.zeros(N_FACTORS)
        ic_long  = ic_df.iloc[long_start:i].mean().values  if i > 0 else np.zeros(N_FACTORS)
        ic_mom   = ic_short - ic_long   # momentum: recent vs long-term IC

        # Macro state
        if dt in macro_df.index:
            macro = macro_df.loc[dt].values
        elif len(macro_df[macro_df.index <= dt]) > 0:
            macro = macro_df[macro_df.index <= dt].iloc[-1].values
        else:
            macro = np.zeros(len(MACRO_COLS))

        macro = np.nan_to_num(macro.astype(np.float32), nan=0.0)
        state = np.concatenate([ic_short, ic_long, ic_mom, macro]).astype(np.float32)
        records.append({"date": dt, "state": state})

    return records


# =============================================================================
# PYTORCH SAC COMPONENTS
# =============================================================================

if HAS_TORCH:
    def mlp(in_dim, out_dim, hidden):
        return nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    class FactorActor(nn.Module):
        """
        Outputs a 43-dim weight vector via softmax.
        Each weight controls how much to trust that factor this month.
        No memory — each call is fully independent (Markov).
        """
        def __init__(self, state_dim=STATE_DIM, n_factors=N_FACTORS, hidden=HIDDEN_DIM):
            super().__init__()
            self.mean_net    = mlp(state_dim, n_factors, hidden)
            self.log_std_net = mlp(state_dim, n_factors, hidden)
            self.n_factors   = n_factors

        def forward(self, state):
            mean    = self.mean_net(state)
            log_std = self.log_std_net(state).clamp(LOG_STD_MIN, LOG_STD_MAX)
            return mean, log_std.exp()

        def sample(self, state):
            mean, std = self(state)
            dist      = Normal(mean, std)
            raw       = dist.rsample()                          # [B, 43]
            weights   = F.softmax(raw, dim=-1)                  # project to simplex
            log_prob  = dist.log_prob(raw).sum(-1, keepdim=True)
            return weights, log_prob, F.softmax(mean, dim=-1)   # weights, logp, mean_weights

        def get_weights(self, state):
            """Deterministic weights for evaluation."""
            with torch.no_grad():
                mean, _ = self(state)
                return F.softmax(mean, dim=-1)

    class TwinCritic(nn.Module):
        """Twin Q-networks. Input: (state, factor_weights) → Q-value."""
        def __init__(self, state_dim=STATE_DIM, n_factors=N_FACTORS, hidden=HIDDEN_DIM):
            super().__init__()
            self.q1 = mlp(state_dim + n_factors, 1, hidden)
            self.q2 = mlp(state_dim + n_factors, 1, hidden)

        def forward(self, state, action):
            sa = torch.cat([state, action], dim=-1)
            return self.q1(sa), self.q2(sa)

    import random, collections

    class ReplayBuffer:
        def __init__(self, capacity=BUFFER_SIZE):
            self.buf = collections.deque(maxlen=capacity)

        def push(self, s, a, r, ns, done):
            self.buf.append((
                np.array(s,  dtype=np.float32),
                np.array(a,  dtype=np.float32),
                float(r),
                np.array(ns, dtype=np.float32),
                float(done),
            ))

        def sample(self, n):
            batch = random.sample(self.buf, n)
            s, a, r, ns, d = zip(*batch)
            return (np.stack(s), np.stack(a),
                    np.array(r, dtype=np.float32).reshape(-1, 1),
                    np.stack(ns),
                    np.array(d, dtype=np.float32).reshape(-1, 1))

        def __len__(self):
            return len(self.buf)

    class SACFactorAgent:
        """
        SAC agent for adaptive factor weighting.
        Action = 43-dim weight vector (softmax normalised).
        Reward = IC of combined signal this month.
        No memory — Markov policy only.
        """
        def __init__(self):
            self.actor         = FactorActor()
            self.critic        = TwinCritic()
            self.critic_target = TwinCritic()
            self.critic_target.load_state_dict(self.critic.state_dict())

            self.actor_opt  = Adam(self.actor.parameters(),  lr=LR_ACTOR)
            self.critic_opt = Adam(self.critic.parameters(), lr=LR_CRITIC)

            # Auto-tuned temperature
            self.log_alpha  = torch.zeros(1, requires_grad=True)
            self.alpha_opt  = Adam([self.log_alpha], lr=LR_ALPHA)
            self.target_ent = -float(N_FACTORS)   # target entropy

            self.buffer = ReplayBuffer()

        @property
        def alpha(self):
            return self.log_alpha.exp()

        def select_weights(self, state, deterministic=False):
            s = torch.FloatTensor(state).unsqueeze(0)
            if deterministic:
                return self.actor.get_weights(s).squeeze(0).numpy()
            else:
                with torch.no_grad():
                    w, _, _ = self.actor.sample(s)
                return w.squeeze(0).numpy()

        def update(self):
            if len(self.buffer) < BATCH_SIZE:
                return

            S, A, R, NS, D = self.buffer.sample(BATCH_SIZE)
            S  = torch.FloatTensor(S)
            A  = torch.FloatTensor(A)
            R  = torch.FloatTensor(R)
            NS = torch.FloatTensor(NS)
            D  = torch.FloatTensor(D)

            # ── Critic update ──────────────────────────────────────────────
            with torch.no_grad():
                na, lp, _ = self.actor.sample(NS)
                q1_t, q2_t = self.critic_target(NS, na)
                q_target   = R + GAMMA * (1 - D) * (
                    torch.min(q1_t, q2_t) - self.alpha * lp
                )

            q1, q2 = self.critic(S, A)
            critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            # ── Actor update ───────────────────────────────────────────────
            wa, lpa, _ = self.actor.sample(S)
            q1_a, q2_a = self.critic(S, wa)
            actor_loss  = (self.alpha * lpa - torch.min(q1_a, q2_a)).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # ── Temperature update ─────────────────────────────────────────
            alpha_loss = -(self.log_alpha * (lpa + self.target_ent).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

            # ── Soft target update ─────────────────────────────────────────
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.data.copy_(TAU * p.data + (1 - TAU) * pt.data)


# =============================================================================
# TRAINING
# =============================================================================

def precompute_month_data(panel, factor_names, signs_vec, months):
    """
    Precompute z-scored factor matrices and returns for all months.
    Called ONCE before training — avoids rescanning the panel 91,200 times.
    Returns dict {date: (Xz_signed, y_norm)} ready for fast dot product.
    """
    print("  Precomputing factor matrices for all months ...")
    cache = {}
    for dt in months:
        grp = panel[panel["date"] == dt][factor_names + ["fwd_ret_1m"]].dropna()
        if len(grp) < 50:
            continue
        X  = grp[factor_names].values.astype(np.float64)
        mu = np.nanmean(X, axis=0)
        sg = np.nanstd(X,  axis=0) + 1e-8
        Xz = np.where(np.isnan(X), 0.0, (X - mu) / sg)
        Xz_signed = Xz * signs_vec          # apply sign flip once
        y  = grp["fwd_ret_1m"].values
        ys = y.std()
        if ys < 1e-8:
            continue
        y_norm = (y - y.mean()) / ys
        cache[dt] = (Xz_signed, y_norm)
    print(f"  Cached {len(cache)} months.")
    return cache


def ic_from_cache(cache, weights_vec, dt):
    """Fast IC computation using precomputed matrices — just a dot product."""
    if dt not in cache:
        return 0.0
    Xz_signed, y_norm = cache[dt]
    sc = Xz_signed @ weights_vec
    ss = sc.std()
    if ss < 1e-8:
        return 0.0
    ic = float(np.dot(sc - sc.mean(), y_norm) / (ss * len(sc)))
    return 0.0 if np.isnan(ic) else ic


def train_agent(agent, state_records, panel, factor_names, signs_vec, train_months):
    """Run SAC training loop with precomputed IC cache."""
    # Precompute once — avoids 91,200 full panel scans
    cache     = precompute_month_data(panel, factor_names, signs_vec, train_months)
    train_set = [r for r in state_records if r["date"] in cache]

    print(f"  Training on {len(train_set)} months ...")
    reward_hist = []

    for epoch in range(N_EPOCHS):
        epoch_rewards = []
        np.random.shuffle(train_set)

        for i, rec in enumerate(train_set):
            dt    = rec["date"]
            state = np.nan_to_num(rec["state"], nan=0.0)

            # Agent selects weights
            weights = agent.select_weights(state)

            # Reward = IC of combined signal (fast, uses cache)
            reward  = ic_from_cache(cache, weights, dt)
            epoch_rewards.append(reward)

            # Next state (use index directly — no linear search)
            if i + 1 < len(train_set):
                next_state = np.nan_to_num(train_set[i + 1]["state"], nan=0.0)
                done = 0.0
            else:
                next_state = np.zeros(STATE_DIM, dtype=np.float32)
                done = 1.0

            agent.buffer.push(state, weights, reward, next_state, done)
            agent.update()

        avg_r = float(np.mean(epoch_rewards))
        reward_hist.append(avg_r)

        if epoch % 100 == 0:
            temp = float(agent.alpha.item()) if HAS_TORCH else 0.0
            print(f"    Epoch {epoch:4d}  avg_IC={avg_r:+.4f}  temp={temp:.3f}  "
                  f"buffer={len(agent.buffer)}")

    return reward_hist


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(agent, state_records, panel, factor_names, signs_vec,
             static_weights_dict, eval_months):
    """
    Evaluate RL weights vs static baselines on eval_months.
    Returns DataFrame with monthly IC per method.
    """
    cache     = precompute_month_data(panel, factor_names, signs_vec, eval_months)
    state_map = {r["date"]: r["state"] for r in state_records}
    results   = []

    for dt in eval_months:
        if dt not in state_map or dt not in cache:
            continue
        state = np.nan_to_num(state_map[dt], nan=0.0)
        rl_w  = agent.select_weights(state, deterministic=True)

        row = {"date": dt, "rl": ic_from_cache(cache, rl_w, dt)}
        for name, w in static_weights_dict.items():
            row[name] = ic_from_cache(cache, w, dt)
        results.append(row)

    return pd.DataFrame(results).set_index("date")


def print_results(ic_df, label):
    print(f"\n{'─'*65}")
    print(f"  {label}")
    print(f"{'─'*65}")
    print(f"  {'Method':<25}  {'Mean IC':>8}  {'ICIR':>7}  {'Hit%':>6}")
    print(f"  {'─'*25}  {'─'*8}  {'─'*7}  {'─'*6}")
    for col in ic_df.columns:
        s    = ic_df[col].dropna()
        mean = s.mean()
        icir = mean / (s.std() + 1e-8)
        hit  = (s > 0).mean() * 100
        flag = " ◄" if col == "rl" else ""
        print(f"  {col:<25}  {mean:>8.4f}  {icir:>7.3f}  {hit:>5.1f}%{flag}")


# =============================================================================
# FIGURES
# =============================================================================

def plot_results(ic_train, ic_val, ic_test, reward_hist, rl_weights_over_time, factor_names):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Layer 1 RL: Adaptive Factor Weighting", fontsize=14, fontweight="bold")

    # 1. Training reward curve
    ax = axes[0, 0]
    ax.plot(reward_hist, color="#2196F3", linewidth=1.2, alpha=0.8)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title("Training Reward (avg monthly IC)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IC")
    ax.grid(True, alpha=0.3)

    # 2. IC comparison — test period
    ax = axes[0, 1]
    methods = list(ic_test.columns)
    colors  = {"rl": "#2196F3", "equal": "#9E9E9E", "ic_decay": "#FF9800", "ic_optimised": "#4CAF50"}
    for m in methods:
        s = ic_test[m].dropna()
        c = colors.get(m, "#607D8B")
        ax.plot(s.index, s.cumsum(), label=m, color=c, linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title("Cumulative IC — Test Period (2023+)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative IC")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. RL vs IC-optimised weights comparison
    ax = axes[1, 0]
    if len(rl_weights_over_time) > 0:
        rl_w_mean = np.mean(rl_weights_over_time, axis=0)
        x = np.arange(len(factor_names))
        ax.bar(x, rl_w_mean, color="#2196F3", alpha=0.7, label="RL (mean test)")
        ax.set_xticks(x)
        ax.set_xticklabels(factor_names, rotation=90, fontsize=6)
        ax.set_title("RL Mean Factor Weights (Test Period)")
        ax.set_ylabel("Weight")
        ax.grid(True, alpha=0.3, axis="y")

    # 4. Monthly IC bar — test period RL vs IC-optimised
    ax = axes[1, 1]
    if "rl" in ic_test.columns and "ic_optimised" in ic_test.columns:
        months = ic_test.index
        x      = np.arange(len(months))
        w      = 0.35
        ax.bar(x - w/2, ic_test["rl"].values,           w, label="RL Agent",     color="#2196F3", alpha=0.8)
        ax.bar(x + w/2, ic_test["ic_optimised"].values, w, label="IC-Optimised", color="#4CAF50", alpha=0.8)
        ax.axhline(0, color="gray", linewidth=0.5)
        step = max(1, len(months) // 8)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([str(m.date())[:7] for m in months[::step]], rotation=45, fontsize=7)
        ax.set_title("Monthly IC: RL Agent vs IC-Optimised")
        ax.set_ylabel("IC")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "rl_factor_agent.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {out}")


# =============================================================================
# NUMPY FALLBACK
# =============================================================================

def numpy_baseline(state_records, panel, factor_names, signs_vec,
                   static_weights_dict, ic_df, train_months, eval_months):
    """Simple gradient-free optimisation using IC matrix directly."""
    print("  Using numpy fallback (no PyTorch).")
    print("  Finding best static weights via mean IC on training set ...")

    train_ic = ic_df[ic_df.index.isin(set(train_months))]
    mean_ic  = train_ic.mean().values.clip(0)
    if mean_ic.sum() > 0:
        rl_w = mean_ic / mean_ic.sum()
    else:
        rl_w = np.ones(N_FACTORS) / N_FACTORS

    class FakeAgent:
        def __init__(self, w): self.w = w
        def select_weights(self, *a, **kw): return self.w

    return FakeAgent(rl_w), []


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("5a_rl_factor_agent.py — Layer 1 SAC: Adaptive Factor Weighting")
    print("=" * 65)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading data ...")
    panel, factors_df = load_data()
    factor_names = factors_df["factor"].tolist()
    signs        = factors_df.set_index("factor")["majority_sign"].map(
                       {"+": 1.0, "-": -1.0}).fillna(1.0)
    signs_vec    = signs[factor_names].values.astype(np.float64)

    # Static baseline weights
    w_equal     = np.ones(N_FACTORS) / N_FACTORS
    w_ic_decay  = factors_df.set_index("factor")["weight"][factor_names].values.astype(np.float64)
    w_ic_opt    = factors_df.set_index("factor")["weight_optimised"][factor_names].values.astype(np.float64)

    static_weights = {
        "equal":        w_equal,
        "ic_decay":     w_ic_decay,
        "ic_optimised": w_ic_opt,
    }

    print(f"  Panel:   {panel['date'].min().date()} → {panel['date'].max().date()}")
    print(f"  Factors: {N_FACTORS}")

    # ── IC matrix ─────────────────────────────────────────────────────────────
    panel_scoped = panel[panel["date"] >= TRAIN_START].copy()
    ic_df        = compute_ic_matrix(panel_scoped, factor_names, signs)

    # ── Macro state ───────────────────────────────────────────────────────────
    macro_df = panel.groupby("date")[MACRO_COLS].first().sort_index()
    # Normalise macro (rolling z-score)
    macro_df = macro_df.apply(lambda s: (s - s.rolling(36, min_periods=6).mean()) /
                                         (s.rolling(36, min_periods=6).std() + 1e-8))
    macro_df = macro_df.fillna(0.0)

    # ── State vectors ─────────────────────────────────────────────────────────
    print("  Building state vectors ...")
    state_records = build_state_matrix(ic_df, macro_df)

    all_months   = ic_df.index.tolist()
    train_months = [m for m in all_months if pd.Timestamp(TRAIN_START) <= m <= pd.Timestamp(TRAIN_END)]
    val_months   = [m for m in all_months if pd.Timestamp(TRAIN_END)   <  m <= pd.Timestamp(VAL_END)]
    test_months  = [m for m in all_months if m > pd.Timestamp(VAL_END)]

    print(f"  Train: {len(train_months)} months  |  Val: {len(val_months)} months  |  Test: {len(test_months)} months")
    print(f"  State dim: {STATE_DIM}  (43×IC_short + 43×IC_long + 43×IC_mom + {len(MACRO_COLS)}×macro)")

    # ── Train agent ───────────────────────────────────────────────────────────
    print(f"\n{'-'*65}")
    print("Training SAC agent ...")

    if HAS_TORCH:
        agent       = SACFactorAgent()
        reward_hist = train_agent(agent, state_records, panel_scoped,
                                  factor_names, signs_vec, train_months)
        print(f"  Training complete. Final avg IC: {reward_hist[-1]:+.4f}")
    else:
        agent, reward_hist = numpy_baseline(
            state_records, panel_scoped, factor_names, signs_vec,
            static_weights, ic_df, train_months, test_months
        )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"\n{'-'*65}")
    print("Evaluating ...")

    ic_train = evaluate(agent, state_records, panel_scoped, factor_names,
                        signs_vec, static_weights, train_months)
    ic_val   = evaluate(agent, state_records, panel_scoped, factor_names,
                        signs_vec, static_weights, val_months)
    ic_test  = evaluate(agent, state_records, panel_scoped, factor_names,
                        signs_vec, static_weights, test_months)

    print_results(ic_train, "TRAIN PERIOD (2010–2020)")
    print_results(ic_val,   "VALIDATION PERIOD (2021–2022)")
    print_results(ic_test,  "TEST PERIOD (2023+)")

    # ── RL weight dynamics ────────────────────────────────────────────────────
    state_map = {r["date"]: r["state"] for r in state_records}
    rl_weights_over_time = []
    for dt in test_months:
        if dt in state_map:
            state = np.nan_to_num(state_map[dt], nan=0.0)
            w     = agent.select_weights(state, deterministic=True)
            rl_weights_over_time.append(w)

    if len(rl_weights_over_time) > 0:
        rl_w_mean = np.mean(rl_weights_over_time, axis=0)
        rl_w_std  = np.std(rl_weights_over_time, axis=0)
        print(f"\n  Top 5 factors by mean RL weight (test period):")
        top_idx = np.argsort(rl_w_mean)[::-1][:5]
        for idx in top_idx:
            print(f"    {factor_names[idx]:<30}  {rl_w_mean[idx]:.4f} ± {rl_w_std[idx]:.4f}")

        print(f"\n  Bottom 5 (most downweighted):")
        bot_idx = np.argsort(rl_w_mean)[:5]
        for idx in bot_idx:
            print(f"    {factor_names[idx]:<30}  {rl_w_mean[idx]:.4f} ± {rl_w_std[idx]:.4f}")

    # ── Compare val IC improvement ─────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("SUMMARY — Mean IC by period")
    print(f"{'='*65}")
    print(f"  {'Method':<25}  {'Train':>8}  {'Val':>8}  {'Test':>8}")
    print(f"  {'─'*25}  {'─'*8}  {'─'*8}  {'─'*8}")
    for col in ic_train.columns:
        tr = ic_train[col].mean()
        va = ic_val[col].mean()
        te = ic_test[col].mean() if col in ic_test.columns else float("nan")
        flag = " ◄" if col == "rl" else ""
        print(f"  {col:<25}  {tr:>8.4f}  {va:>8.4f}  {te:>8.4f}{flag}")
    print(f"{'='*65}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    ic_test.to_csv(DATA_DIR / "rl_factor_ic_test.csv")
    print(f"\nSaved → data/rl_factor_ic_test.csv")

    # Full-period RL IC file for L1→L2 pipeline connection.
    # 5b_rl_portfolio_agent.py loads this to add L1 IC as a state feature,
    # so Layer 2 knows whether Layer 1's signal is currently reliable.
    ic_full = (pd.concat([ic_train[["rl"]], ic_val[["rl"]], ic_test[["rl"]]])
               .rename(columns={"rl": "l1_ic"})
               .sort_index())
    ic_full.to_csv(DATA_DIR / "l1_rl_ic_full.csv")
    print(f"Saved → data/l1_rl_ic_full.csv  ({len(ic_full)} months, train+val+test combined)")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_results(ic_train, ic_val, ic_test, reward_hist,
                 rl_weights_over_time, factor_names)

    print("\nDone.")


if __name__ == "__main__":
    main()
