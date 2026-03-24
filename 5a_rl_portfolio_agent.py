"""
5a_rl_portfolio_agent.py — Reinforcement Learning Portfolio Tilt Agent
======================================================================
Trains a Soft Actor-Critic (SAC) agent to dynamically adapt the alpha
(tilt strength) used in index enhancement each month.

Instead of a fixed alpha, the agent observes the current market state
and learns the optimal tilt for that environment.

Key design constraint: NO memory / no recurrence.
Each month's decision is fully self-contained (Markov).
Policy is a plain MLP — no LSTM, no GRU, no hidden state.

Architecture:
  State  (6 features): signal strength, signal dispersion, benchmark vol,
                        recent active return, regime, rolling tracking error
  Action : alpha in [0.002, 0.05]  (continuous — how much to tilt this month)
  Reward : Sharpe of active return = IR proxy
           active_ret × 12 − TE_penalty − bench_vol_penalty
           (total-portfolio Sharpe doesn't work: benchmark dominates
           numerator+denominator, agent gets no learning signal)
  Policy : MLP(6 -> 64 -> 64 -> 1)  — NO memory

Two-layer RL design — objectives are complementary, not conflicting:
  Layer 1 (5b_rl_factor_agent.py) : maximises IC / ICIR of the combined
    factor signal — optimises WHAT signal to generate.
  Layer 2 (this file)             : maximises portfolio Sharpe ratio —
    optimises HOW AGGRESSIVELY to act on that signal each month.
  No conflict: L1 never sees portfolio vol; L2 takes signal quality as given
  and adjusts tilt to maximise risk-adjusted total return.

SAC (Soft Actor-Critic):
  - Off-policy, model-free RL for continuous action spaces
  - Entropy regularisation prevents collapsing to a single alpha
  - Twin Q-networks reduce overestimation bias
  - Auto-tuned temperature parameter

Training: builds factor-combo signal from panel_monthly_enriched.parquet
  for 2010-2022 (12 years of history) — sufficient for RL convergence.
Evaluation: uses CS-Transformer scores on test period 2023-2025.

Run:
  python 5a_rl_portfolio_agent.py

Outputs:
  data/bt_ie_rl_agent.csv         — monthly backtest (port/bench/active ret)
  data/rl_agent_actions.csv       — alpha decisions per month
  figures/rl_agent_comparison.png — RL vs fixed-alpha cumulative return
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

# ── PyTorch (optional — falls back to rule-based if not available) ─────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR     = Path("data")
FIG_DIR      = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

PANEL_FILE   = DATA_DIR / "panel_monthly_enriched.parquet"
FACTORS_FILE = DATA_DIR / "factor_selected_optimised.csv"
SCORES_FILE  = DATA_DIR / "scores_cs_transformer.parquet"
WEIGHTS_FILE = DATA_DIR / "spx_weights.parquet"

# ── Hyperparameters ────────────────────────────────────────────────────────────
ALPHA_MIN   = 0.002
ALPHA_MAX   = 0.050
TE_TARGET   = 0.030          # 3% annualised tracking error (used in state feature only)
TE_PENALTY  = 5.0            # legacy — no longer used in reward (kept for reference)
RF_ANNUAL   = 0.042          # approximate risk-free rate (4.2% — US 3m T-bill 2024 avg)
RF_MONTHLY  = RF_ANNUAL / 12 # monthly risk-free rate for Sharpe calculation

TRAIN_START = "2010-01-01"
TRAIN_END   = "2022-12-31"
TEST_START  = "2023-01-01"

TRAIN_EPOCHS = 400
BATCH_SIZE   = 32
LR           = 3e-4
GAMMA        = 0.95
TAU          = 0.005
HIDDEN_DIM   = 64
BUFFER_SIZE  = 10_000
SEED         = 42

np.random.seed(SEED)
random.seed(SEED)
if HAS_TORCH:
    torch.manual_seed(SEED)

STATE_COLS = [
    "signal_strength",    # mean abs z-score of scores
    "signal_dispersion",  # cross-sectional std of z-scores
    "bench_vol",          # rolling 3m annualised benchmark vol
    "recent_active_ret",  # rolling 3m mean active return
    "regime",             # 0 = risk-on, 1 = risk-off
    "rolling_te",         # rolling 3m annualised tracking error
]


# =============================================================================
# PORTFOLIO SIMULATION
# =============================================================================

def simulate_month(scores_month, weights_month, alpha,
                   top_n=100, bottom_n=100):
    """Apply alpha tilt for one month. Returns dict or None."""
    df = scores_month.merge(
        weights_month[["ticker", "spx_weight"]], on="ticker", how="inner"
    ).dropna(subset=["score", "spx_weight", "fwd_ret_1m"])

    if len(df) < top_n + bottom_n:
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

    return {
        "port_ret":   port_ret,
        "bench_ret":  bench_ret,
        "active_ret": port_ret - bench_ret,
    }


# =============================================================================
# BUILD TRAINING SCORES FROM PANEL
# =============================================================================

def build_factor_combo_scores(panel, factors_df):
    """
    Build factor-combo signal for full panel history using optimised weights.
    Gives 12+ years of training data for the RL agent.
    """
    factor_names = list(factors_df["factor"])
    signs        = factors_df.set_index("factor")["majority_sign"].map(
                       {"+": 1.0, "-": -1.0}).fillna(1.0)
    weights_opt  = factors_df.set_index("factor")["weight_optimised"]
    signed_w     = (weights_opt * signs).values.astype(np.float64)

    available = [f for f in factor_names if f in panel.columns]
    if len(available) < len(factor_names):
        idx      = [factor_names.index(f) for f in available]
        signed_w = signed_w[idx]
        factor_names = available

    rows = []
    for dt, grp in panel.groupby("date"):
        X  = grp[factor_names].values.astype(np.float64)
        mu = np.nanmean(X, axis=0)
        sg = np.nanstd(X,  axis=0) + 1e-8
        Xz = np.where(np.isnan(X), 0.0, (X - mu) / sg)
        sc = Xz @ signed_w
        for i, (_, row) in enumerate(grp.iterrows()):
            rows.append({
                "date":       dt,
                "ticker":     row["ticker"],
                "score":      float(sc[i]),
                "fwd_ret_1m": row.get("fwd_ret_1m", np.nan),
            })

    return pd.DataFrame(rows)


# =============================================================================
# BUILD EPISODE TABLE
# =============================================================================

def build_episodes(scores, weights, ref_alpha=0.01):
    """
    Build monthly state vectors for the RL environment.
    Each row = one month's state + raw scores/weights for simulation.
    """
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

        ref = simulate_month(sc_m, w_m, alpha=ref_alpha)
        if ref is None:
            continue

        rows.append({
            "date":              dt,
            "signal_strength":   float(np.abs(z).mean()),
            "signal_dispersion": float(z.std()),
            "bench_ret":         ref["bench_ret"],
            "active_ret_ref":    ref["active_ret"],
            "_scores":           sc_m,
            "_weights":          w_m,
        })

    df = pd.DataFrame(rows).set_index("date").sort_index()

    # rolling(3, min_periods=2) avoids std=NaN on single-observation windows
    df["bench_vol"]         = (df["bench_ret"]
                               .rolling(3, min_periods=2).std()
                               .fillna(df["bench_ret"].expanding().std())
                               .fillna(0.0)) * np.sqrt(12)
    df["rolling_te"]        = (df["active_ret_ref"]
                               .rolling(3, min_periods=2).std()
                               .fillna(df["active_ret_ref"].expanding().std())
                               .fillna(0.0)) * np.sqrt(12)
    df["recent_active_ret"] = df["active_ret_ref"].rolling(3, min_periods=1).mean().fillna(0.0)

    vol_med      = df["bench_vol"].expanding(min_periods=6).median()
    df["regime"] = (df["bench_vol"] > vol_med).astype(float).fillna(0.0)

    for col in STATE_COLS:
        if col == "regime":
            continue
        mu, sg  = df[col].mean(), df[col].std() + 1e-8
        df[col] = (df[col] - mu) / sg
        # Replace any residual NaN / inf after normalisation and clip to ±5σ
        df[col] = df[col].fillna(0.0).clip(-5.0, 5.0)

    return df


# =============================================================================
# SAC COMPONENTS — MLP only, no memory
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
        def forward(self, x):
            return self.net(x)

    class GaussianActor(nn.Module):
        """MLP policy — no recurrence. Each call is independent."""
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
            log_p = log_p.sum(-1, keepdim=True)
            return alpha, log_p, mean

    class TwinCritic(nn.Module):
        def __init__(self, state_dim, hidden=HIDDEN_DIM):
            super().__init__()
            self.q1 = MLP(state_dim + 1, 1, hidden)
            self.q2 = MLP(state_dim + 1, 1, hidden)

        def forward(self, state, action):
            sa = torch.cat([state, action], dim=-1)
            return self.q1(sa), self.q2(sa)


class ReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self.buf.append((
            np.array(s,   dtype=np.float32),
            np.array([a], dtype=np.float32),
            float(r),
            np.array(ns,  dtype=np.float32),
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


# =============================================================================
# SAC AGENT
# =============================================================================

class SACAgent:
    """SAC with auto-tuned temperature. Stateless MLP — no memory."""

    def __init__(self, state_dim):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for SACAgent.")

        self.actor         = GaussianActor(state_dim)
        self.critic        = TwinCritic(state_dim)
        self.critic_target = TwinCritic(state_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=LR)

        self.target_entropy = -1.0
        self.log_alpha_temp = torch.zeros(1, requires_grad=True)
        self.temp_opt       = optim.Adam([self.log_alpha_temp], lr=LR)

        self.buffer = ReplayBuffer()

    @property
    def temperature(self):
        return float(self.log_alpha_temp.exp())

    def _norm_action(self, alpha):
        return (alpha - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN) * 2.0 - 1.0

    def select_action(self, state, deterministic=False):
        s = torch.FloatTensor(state).unsqueeze(0)
        s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(s)
                y       = torch.tanh(mean)
                alpha   = ALPHA_MIN + (y + 1.0) / 2.0 * (ALPHA_MAX - ALPHA_MIN)
            else:
                alpha, _, _ = self.actor.sample(s)
        return float(alpha.squeeze())

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return {}

        s, a, r, ns, d = self.buffer.sample(BATCH_SIZE)
        S  = torch.FloatTensor(s)
        A  = torch.FloatTensor(a)
        R  = torch.FloatTensor(r)
        NS = torch.FloatTensor(ns)
        D  = torch.FloatTensor(d)

        with torch.no_grad():
            na, lp, _ = self.actor.sample(NS)
            na_n      = self._norm_action(na)
            q1t, q2t  = self.critic_target(NS, na_n)
            q_tgt     = R + GAMMA * (1 - D) * (torch.min(q1t, q2t) - self.temperature * lp)

        a_n    = self._norm_action(A)
        q1, q2 = self.critic(S, a_n)
        c_loss = F.mse_loss(q1, q_tgt) + F.mse_loss(q2, q_tgt)
        self.critic_opt.zero_grad(); c_loss.backward(); self.critic_opt.step()

        na, lp, _ = self.actor.sample(S)
        na_n      = self._norm_action(na)
        q1, q2    = self.critic(S, na_n)
        a_loss    = (self.temperature * lp - torch.min(q1, q2)).mean()
        self.actor_opt.zero_grad(); a_loss.backward(); self.actor_opt.step()

        t_loss = -(self.log_alpha_temp * (lp + self.target_entropy).detach()).mean()
        self.temp_opt.zero_grad(); t_loss.backward(); self.temp_opt.step()

        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(TAU * p.data + (1.0 - TAU) * tp.data)

        return {"critic_loss": float(c_loss), "actor_loss": float(a_loss),
                "temperature": self.temperature}


# =============================================================================
# REWARD
# =============================================================================

def compute_reward(active_ret: float, bench_vol_ann: float, rolling_te_ann: float) -> float:
    """
    Layer 2 reward: risk-adjusted active return (Sharpe of the active component).

    For index enhancement, the agent only controls the ACTIVE component
    (how much to tilt away from the benchmark). The appropriate reward is
    therefore the Sharpe of that active component, which equals the IR:

        reward = active_ret × 12  −  risk_penalty

    where the risk penalty incorporates both tracking error AND total
    portfolio vol (bench_vol + te), so the agent accounts for the full
    risk environment when choosing how aggressively to tilt.

    Note: using total portfolio Sharpe = (port_ret - RF) / port_vol does NOT
    work here — the benchmark dominates both numerator and denominator and
    the agent's alpha barely moves the ratio, giving near-zero learning signal.

    Two-layer design (non-conflicting):
      Layer 1 (5b): maximises IC/ICIR of the combined factor signal.
      Layer 2 (this): maximises Sharpe of the active return = IR.
      L1 asks "which factors to trust?", L2 asks "how hard to act on them?".
    """
    annualised  = active_ret * 12.0
    te_breach   = max(0.0, rolling_te_ann - TE_TARGET)
    # Additional penalty when total market vol is high (risk-off regime)
    vol_penalty = 0.5 * max(0.0, bench_vol_ann - 0.15)  # penalise excess bench vol
    return annualised - TE_PENALTY * te_breach ** 2 - vol_penalty


# =============================================================================
# TRAINING
# =============================================================================

def train(agent, episodes):
    train_rows = [
        (dt, row) for dt, row in episodes.iterrows()
        if TRAIN_START <= str(dt.date()) <= TRAIN_END
    ]
    print(f"  Training on {len(train_rows)} months ({TRAIN_START[:4]}-{TRAIN_END[:4]}) ...")

    reward_hist = []
    for epoch in range(TRAIN_EPOCHS):
        random.shuffle(train_rows)
        ep_rewards = []

        for i, (dt, row) in enumerate(train_rows):
            state = np.clip(
                np.nan_to_num(row[STATE_COLS].values.astype(np.float32),
                              nan=0.0, posinf=0.0, neginf=0.0),
                -5.0, 5.0)
            alpha = agent.select_action(state)

            result = simulate_month(row["_scores"], row["_weights"], alpha=alpha)
            if result is None:
                continue

            reward    = compute_reward(
                result["active_ret"],
                float(row["bench_vol"]),
                float(row["rolling_te"]),
            )
            ep_rewards.append(reward)

            done = (i == len(train_rows) - 1)
            if done or i + 1 >= len(train_rows):
                next_state = np.zeros(len(STATE_COLS), dtype=np.float32)
                done_flag  = 1.0
            else:
                next_state = np.nan_to_num(train_rows[i + 1][1][STATE_COLS].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                done_flag  = 0.0

            agent.buffer.push(state, alpha, reward, next_state, done_flag)
            agent.update()

        avg_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        reward_hist.append(avg_r)

        if epoch % 100 == 0:
            print(f"    Epoch {epoch:4d}  avg_reward={avg_r:+.4f}"
                  f"  temp={agent.temperature:.3f}  buffer={len(agent.buffer)}")

    return reward_hist


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(agent, episodes, fixed_alpha=0.01):
    rl_rows, fixed_rows = [], []

    for dt, row in episodes.iterrows():
        if str(dt.date()) < TEST_START:
            continue

        state = np.clip(
            np.nan_to_num(row[STATE_COLS].values.astype(np.float32),
                          nan=0.0, posinf=0.0, neginf=0.0),
            -5.0, 5.0)

        if agent is not None:
            alpha_rl = agent.select_action(state, deterministic=True)
        else:
            # Fallback: reduce alpha in high-vol (risk-off) regimes
            alpha_rl = (ALPHA_MIN + ALPHA_MAX) / 2.0 * (1.0 - 0.4 * float(row["regime"]))

        r_rl = simulate_month(row["_scores"], row["_weights"], alpha=alpha_rl)
        if r_rl:
            rl_rows.append({"date": dt, "alpha_used": alpha_rl,
                            "regime": row["regime"], **r_rl})

        r_fx = simulate_month(row["_scores"], row["_weights"], alpha=fixed_alpha)
        if r_fx:
            fixed_rows.append({"date": dt, "alpha_used": fixed_alpha,
                                "regime": row["regime"], **r_fx})

    rl_bt    = pd.DataFrame(rl_rows).set_index("date")    if rl_rows    else pd.DataFrame()
    fixed_bt = pd.DataFrame(fixed_rows).set_index("date") if fixed_rows else pd.DataFrame()
    return rl_bt, fixed_bt


# =============================================================================
# STATS
# =============================================================================

def ie_stats(bt):
    if bt.empty or "active_ret" not in bt.columns:
        return {}
    r = bt["active_ret"].dropna()
    n = len(r)
    if n < 4:
        return {}
    ann_alpha  = (1 + r).prod() ** (12 / n) - 1
    track_err  = r.std(ddof=1) * np.sqrt(12)
    info_ratio = ann_alpha / track_err if track_err > 0 else float("nan")
    hit_rate   = (r > 0).mean()
    nav        = (1 + r).cumprod()
    max_dd     = float((nav / nav.cummax() - 1).min())
    return dict(ann_alpha=ann_alpha, track_err=track_err, info_ratio=info_ratio,
                hit_rate=hit_rate, max_active_dd=max_dd, n_months=n)


# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(rl_bt, fixed_bt):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle("RL Portfolio Agent vs Fixed-Alpha Baseline\n"
                 "(CS-Transformer scores, test period 2023-2025)",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    if not rl_bt.empty:
        rl_nav = (1 + rl_bt["active_ret"]).cumprod() - 1
        ax.plot(rl_nav.index, rl_nav * 100, label="RL Agent", color="#2196F3", lw=2)
    if not fixed_bt.empty:
        fx_nav = (1 + fixed_bt["active_ret"]).cumprod() - 1
        ax.plot(fx_nav.index, fx_nav * 100, label="Fixed a=0.01",
                color="#FF9800", lw=2, linestyle="--")
    ax.axhline(0, color="black", lw=0.8, linestyle=":")
    ax.set_ylabel("Cumulative Active Return (%)")
    ax.set_title("Cumulative Active Return vs Benchmark")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ax = axes[1]
    if not rl_bt.empty:
        ax.bar(rl_bt.index, rl_bt["alpha_used"] * 100, width=20,
               color="#2196F3", alpha=0.75, label="RL alpha")
    ax.axhline(1.0, color="#FF9800", lw=1.5, linestyle="--", label="Fixed a=0.01")
    ax.set_ylabel("Alpha Used (%)")
    ax.set_title("RL Agent: Monthly Alpha Decisions")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ax = axes[2]
    if not rl_bt.empty and not fixed_bt.empty:
        months   = rl_bt.index
        x        = np.arange(len(months))
        w        = 0.38
        rl_ar    = rl_bt["active_ret"].values * 100
        fixed_ar = fixed_bt.reindex(months)["active_ret"].fillna(0).values * 100
        ax.bar(x - w / 2, rl_ar,    w, label="RL Agent",     color="#2196F3", alpha=0.8)
        ax.bar(x + w / 2, fixed_ar, w, label="Fixed a=0.01", color="#FF9800", alpha=0.8)
        step = max(1, len(months) // 8)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([m.strftime("%Y-%m") for m in months[::step]],
                           rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Active Return (%)")
    ax.set_title("Monthly Active Return")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = FIG_DIR / "rl_agent_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("5a_rl_portfolio_agent.py — SAC Portfolio Tilt Agent")
    print("No memory: MLP policy, each month independent (Markov)")
    print("=" * 65)

    if not HAS_TORCH:
        print("\nWARNING: PyTorch not installed — using fallback rule-based agent.")
        print("Install: pip install torch\n")

    print("\nLoading data ...")
    panel   = pd.read_parquet(PANEL_FILE)
    panel["date"] = pd.to_datetime(panel["date"])
    factors = pd.read_csv(FACTORS_FILE)
    weights = pd.read_parquet(WEIGHTS_FILE)
    weights["date"] = pd.to_datetime(weights["date"])
    scores_test = pd.read_parquet(SCORES_FILE)
    scores_test["date"] = pd.to_datetime(scores_test["date"])

    print(f"  Panel:       {panel['date'].min().date()} -> {panel['date'].max().date()}"
          f"  ({panel['date'].nunique()} months)")
    print(f"  CS-T scores: {scores_test['date'].min().date()} -> "
          f"{scores_test['date'].max().date()}"
          f"  ({scores_test['date'].nunique()} months)")

    # Build factor-combo scores for training (2010-2022)
    print(f"\nBuilding training scores ({TRAIN_START[:4]}-{TRAIN_END[:4]}) ...")
    panel_train  = panel[(panel["date"] >= TRAIN_START) &
                         (panel["date"] <= TRAIN_END)].copy()
    scores_train = build_factor_combo_scores(panel_train, factors)
    print(f"  Built {len(scores_train):,} rows across "
          f"{scores_train['date'].nunique()} months")

    print("\nBuilding episode tables ...")
    ep_train = build_episodes(scores_train, weights)
    ep_test  = build_episodes(scores_test,  weights)
    print(f"  Train: {len(ep_train)} months  |  Test: {len(ep_test)} months")
    print(f"  State: {len(STATE_COLS)} features: {STATE_COLS}")

    # Train SAC (or fallback)
    agent = None
    if HAS_TORCH:
        print("\n" + "-" * 65)
        print("Training SAC agent ...")
        agent = SACAgent(state_dim=len(STATE_COLS))
        reward_hist = train(agent, ep_train)
        print(f"Training complete. Final avg reward: {reward_hist[-1]:+.4f}")
    else:
        print("\nSkipping SAC training (PyTorch not available). Using fallback.")

    # Evaluate
    print("\n" + "-" * 65)
    print(f"Evaluating on test period ({TEST_START[:4]}-present) ...")
    rl_bt, fixed_bt = evaluate(agent, ep_test, fixed_alpha=0.01)

    # Results
    rl_s = ie_stats(rl_bt)
    fx_s = ie_stats(fixed_bt)

    print("\n" + "=" * 65)
    print("TEST PERIOD RESULTS")
    print("=" * 65)
    print(f"{'Metric':<28}  {'RL Agent':>10}  {'Fixed a=0.01':>12}")
    print("-" * 55)

    def _v(d, k, pct=True, fmt=".2f"):
        if not d or k not in d:
            return "n/a"
        v = d[k] * (100 if pct else 1)
        return f"{v:{fmt}}" + ("%" if pct else "")

    print(f"  {'Ann Alpha':<26}  {_v(rl_s,'ann_alpha'):>10}  {_v(fx_s,'ann_alpha'):>12}")
    print(f"  {'Tracking Error':<26}  {_v(rl_s,'track_err'):>10}  {_v(fx_s,'track_err'):>12}")
    print(f"  {'Information Ratio':<26}  {_v(rl_s,'info_ratio',False,'.3f'):>10}  "
          f"{_v(fx_s,'info_ratio',False,'.3f'):>12}")
    print(f"  {'Hit Rate':<26}  {_v(rl_s,'hit_rate',True,'.1f'):>10}  "
          f"{_v(fx_s,'hit_rate',True,'.1f'):>12}")
    print(f"  {'Max Active DD':<26}  {_v(rl_s,'max_active_dd'):>10}  "
          f"{_v(fx_s,'max_active_dd'):>12}")

    if not rl_bt.empty:
        print(f"\n  RL alpha: mean={rl_bt['alpha_used'].mean()*100:.2f}%  "
              f"min={rl_bt['alpha_used'].min()*100:.2f}%  "
              f"max={rl_bt['alpha_used'].max()*100:.2f}%")
        print("\n  By regime:")
        for r_val, label in [(0.0, "Risk-On"), (1.0, "Risk-Off")]:
            mask = rl_bt["regime"] == r_val
            if mask.any():
                sub  = rl_bt[mask]
                s    = ie_stats(sub)
                a    = sub["alpha_used"].mean() * 100
                ir   = s.get("info_ratio", float("nan"))
                print(f"    {label} ({mask.sum()} months): avg a={a:.2f}%  IR={ir:.3f}")

    print("=" * 65)

    # Save
    if not rl_bt.empty:
        rl_bt.reset_index().to_csv(DATA_DIR / "bt_ie_rl_agent.csv",    index=False)
        rl_bt[["alpha_used", "regime"]].reset_index().to_csv(
            DATA_DIR / "rl_agent_actions.csv", index=False)
        print(f"\nSaved -> data/bt_ie_rl_agent.csv")
        print(f"Saved -> data/rl_agent_actions.csv")

    plot_results(rl_bt, fixed_bt)
    print("\nDone.")


if __name__ == "__main__":
    main()
