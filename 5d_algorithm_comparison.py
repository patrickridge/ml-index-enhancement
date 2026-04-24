"""
5d_algorithm_comparison.py - SAC vs PPO vs GRPO vs DAPO on the 5c walk-forward.

Same 5-fold expanding-window setup as 5c, run with four RL algorithms so we
can see which one actually handles this regime (continuous-action portfolio
tilt, ~100-140 training months per fold):

  SAC   Off-policy, replay buffer, twin Q-critics, auto-tuned entropy. Our
        primary algorithm; sample-efficient enough for the small fold sizes.
  PPO   On-policy with clipped surrogate and a critic baseline. Canonical
        deep RL, but on-policy discards past experience - with ~100 months
        per fold it barely converges. Included as the honest baseline.
  GRPO  No critic at all. Samples G candidate alphas per state, simulates
        each, uses the group-relative reward as the advantage. From
        DeepSeek-R1 (2025), originally for LLM fine-tuning. KL penalty
        (β=0.01) against a frozen reference policy stops collapse.
  DAPO  GRPO + three ByteDance/Seed (2025) tweaks: asymmetric clipping
        (ε_low=0.20, ε_high=0.28), dynamic G (G_max=8 for noisy states,
        G_min=2 for easy ones), and no KL penalty (clip-higher does the
        regularisation).

Run:     python 5d_algorithm_comparison.py
Writes:  data/algo_comparison.csv     per-fold × per-algorithm metrics
         figures/algo_comparison.png  cumulative alpha + per-fold IR bars
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
from scipy.stats import pearsonr

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not installed - using rule-based fallback for all agents.")

# Paths
DATA_DIR = Path("data")
FIG_DIR  = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

PANEL_FILE   = DATA_DIR / "panel_monthly_enriched.parquet"
FACTORS_FILE = DATA_DIR / "factor_selected.csv"
WEIGHTS_FILE = DATA_DIR / "spx_weights.parquet"

# Walk-forward folds (same as 5c)
FOLDS = [
    ("2013-12-31", "2014-01-01", "2015-12-31", "2014–2015"),
    ("2015-12-31", "2016-01-01", "2017-12-31", "2016–2017"),
    ("2017-12-31", "2018-01-01", "2019-12-31", "2018–2019"),
    ("2019-12-31", "2020-01-01", "2021-12-31", "2020–2021"),
    ("2021-12-31", "2022-01-01", "2025-12-31", "2022–2025"),
]
TRAIN_START_GLOBAL = "2010-01-01"

# Shared hyperparameters
ALPHA_MIN   = 0.002
ALPHA_MAX   = 0.050
FIXED_ALPHA = 0.010
HIDDEN_DIM  = 64
BATCH_SIZE  = 32
BUFFER_SIZE = 10_000
SEED        = 42

# SAC
SAC_EPOCHS = 400
SAC_LR     = 3e-4
SAC_GAMMA  = 0.95
SAC_TAU    = 0.005

# PPO
PPO_EPOCHS    = 400
PPO_LR        = 3e-4
PPO_CLIP_EPS  = 0.20
PPO_VAL_COEF  = 0.50
PPO_ENT_COEF  = 0.01
PPO_K_UPDATES = 4

# GRPO
GRPO_EPOCHS  = 400
GRPO_LR      = 3e-4
GRPO_G       = 4      # candidate alphas sampled per state
GRPO_KL_BETA = 0.01   # KL penalty weight - keeps policy close to reference

# DAPO (ByteDance/Seed 2025)
DAPO_EPOCHS     = 400
DAPO_LR         = 3e-4
DAPO_EPS_LOW    = 0.20   # clip lower bound (negative advantage)
DAPO_EPS_HIGH   = 0.28   # clip upper bound (positive advantage) - asymmetric
DAPO_G_INIT     = 2      # initial candidates to estimate reward variance
DAPO_G_MIN      = 2      # minimum group size (easy states)
DAPO_G_MAX      = 8      # maximum group size (hard/volatile states)
DAPO_VAR_THRESH = 0.05   # reward variance threshold for dynamic sampling

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


# FACTOR COMBO SCORES - recomputed per fold from training data only

def compute_fold_weights(panel_train, factor_names, signs):
    """Mean-|IC| weights from training data only. No forward-looking information."""
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


# PORTFOLIO SIMULATION

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


# EPISODE TABLE - training-period normalisation only

def build_episodes(scores, weights, ref_alpha=0.01, norm_params=None, fit_norm=False):
    """
    Always returns (episodes_df, norm_params_dict).
    fit_norm=True: compute norm_params from this data (use for training set).
    norm_params provided: apply those params (use for test set).
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


# SHARED NEURAL NETWORK COMPONENTS

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
        """Shared actor for SAC, PPO, and GRPO - squashed Gaussian over [ALPHA_MIN, ALPHA_MAX]."""
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

        def log_prob_of(self, state, alpha_tensor):
            """Log π(alpha|state) for PPO ratio computation."""
            mean, std = self(state)
            y_t = ((alpha_tensor - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN) * 2.0 - 1.0
                   ).clamp(-0.9999, 0.9999)
            x_t   = torch.atanh(y_t)
            dist  = torch.distributions.Normal(mean, std)
            log_p = dist.log_prob(x_t) - torch.log(1.0 - y_t.pow(2) + 1e-6)
            return log_p.sum(-1, keepdim=True)

    class TwinCritic(nn.Module):
        """SAC twin Q-critics."""
        def __init__(self, state_dim, hidden=HIDDEN_DIM):
            super().__init__()
            self.q1 = MLP(state_dim + 1, 1, hidden)
            self.q2 = MLP(state_dim + 1, 1, hidden)
        def forward(self, state, action):
            sa = torch.cat([state, action], dim=-1)
            return self.q1(sa), self.q2(sa)

    class ValueNet(nn.Module):
        """PPO state-value critic."""
        def __init__(self, state_dim, hidden=HIDDEN_DIM):
            super().__init__()
            self.net = MLP(state_dim, 1, hidden)
        def forward(self, state): return self.net(state)


class ReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self.buf.append((np.array(s, dtype=np.float32),
                         np.array([a], dtype=np.float32),
                         float(r),
                         np.array(ns, dtype=np.float32),
                         float(done)))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, ns, d = zip(*batch)
        return (np.stack(s), np.stack(a),
                np.array(r, dtype=np.float32).reshape(-1, 1),
                np.stack(ns),
                np.array(d, dtype=np.float32).reshape(-1, 1))

    def __len__(self): return len(self.buf)


def compute_reward(active_ret): return active_ret * 12.0

def _safe_state(row):
    return np.clip(
        np.nan_to_num(row[STATE_COLS].values.astype(np.float32),
                      nan=0.0, posinf=0.0, neginf=0.0),
        -5.0, 5.0)


# SAC AGENT - off-policy, replay buffer, twin critics

class SACAgent:
    name = "SAC"

    def __init__(self, state_dim):
        self.actor         = GaussianActor(state_dim)
        self.critic        = TwinCritic(state_dim)
        self.critic_target = TwinCritic(state_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=SAC_LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=SAC_LR)
        self.log_temp   = torch.zeros(1, requires_grad=True)
        self.temp_opt   = optim.Adam([self.log_temp], lr=SAC_LR)
        self.buffer     = ReplayBuffer()
        self._target_entropy = -1.0

    @property
    def temperature(self): return float(self.log_temp.exp())

    def _norm_a(self, alpha):
        return (alpha - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN) * 2.0 - 1.0

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

    def _update(self):
        if len(self.buffer) < BATCH_SIZE:
            return
        s, a, r, ns, d = self.buffer.sample(BATCH_SIZE)
        S  = torch.FloatTensor(s);  A  = torch.FloatTensor(a)
        R  = torch.FloatTensor(r);  NS = torch.FloatTensor(ns)
        D  = torch.FloatTensor(d)

        with torch.no_grad():
            na, lp, _ = self.actor.sample(NS)
            q1t, q2t  = self.critic_target(NS, self._norm_a(na))
            q_tgt     = R + SAC_GAMMA * (1 - D) * (torch.min(q1t, q2t) - self.temperature * lp)

        q1, q2 = self.critic(S, self._norm_a(A))
        c_loss = F.mse_loss(q1, q_tgt) + F.mse_loss(q2, q_tgt)
        self.critic_opt.zero_grad(); c_loss.backward(); self.critic_opt.step()

        na, lp, _ = self.actor.sample(S)
        q1, q2    = self.critic(S, self._norm_a(na))
        a_loss    = (self.temperature * lp - torch.min(q1, q2)).mean()
        self.actor_opt.zero_grad(); a_loss.backward(); self.actor_opt.step()

        t_loss = -(self.log_temp * (lp + self._target_entropy).detach()).mean()
        self.temp_opt.zero_grad(); t_loss.backward(); self.temp_opt.step()

        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(SAC_TAU * p.data + (1.0 - SAC_TAU) * tp.data)

    def train(self, train_rows, verbose=False):
        avg_r = 0.0
        for epoch in range(SAC_EPOCHS):
            random.shuffle(train_rows)
            ep_rewards = []
            for i, (dt, row) in enumerate(train_rows):
                state = _safe_state(row)
                alpha = self.select_action(state)
                res   = simulate_month(row["_scores"], row["_weights"], alpha=alpha)
                if res is None:
                    continue
                r    = compute_reward(res["active_ret"])
                done = float(i == len(train_rows) - 1)
                ns   = (np.zeros(len(STATE_COLS), dtype=np.float32) if done
                        else _safe_state(train_rows[i + 1][1]))
                self.buffer.push(state, alpha, r, ns, done)
                self._update()
                ep_rewards.append(r)
            avg_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            if verbose and epoch % 100 == 0:
                print(f"    [SAC]  Epoch {epoch:4d}  avg_r={avg_r:+.4f}  "
                      f"temp={self.temperature:.3f}  buf={len(self.buffer)}")
        return avg_r


# PPO AGENT - on-policy, clipped surrogate, critic baseline

class PPOAgent:
    name = "PPO"

    def __init__(self, state_dim):
        self.actor  = GaussianActor(state_dim)
        self.critic = ValueNet(state_dim)
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=PPO_LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=PPO_LR)

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
        for epoch in range(PPO_EPOCHS):
            # Collect on-policy rollout
            rollout    = []
            ep_rewards = []
            for dt, row in train_rows:
                state = _safe_state(row)
                s     = torch.nan_to_num(torch.FloatTensor(state).unsqueeze(0), nan=0.0)
                with torch.no_grad():
                    alpha_t, log_p, _ = self.actor.sample(s)
                alpha = float(alpha_t.squeeze())
                res   = simulate_month(row["_scores"], row["_weights"], alpha=alpha)
                if res is None:
                    continue
                r = compute_reward(res["active_ret"])
                ep_rewards.append(r)
                rollout.append((state, alpha, r, float(log_p.squeeze())))

            if len(rollout) < 4:
                continue

            states    = torch.FloatTensor(np.stack([x[0] for x in rollout]))
            alphas    = torch.FloatTensor([[x[1]] for x in rollout])
            rewards   = torch.FloatTensor([[x[2]] for x in rollout])
            log_p_old = torch.FloatTensor([[x[3]] for x in rollout])

            # Advantage: A = r - V(s), single-step episodes
            with torch.no_grad():
                values = self.critic(states)
            adv = rewards - values
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # K PPO update steps on the collected rollout
            for _ in range(PPO_K_UPDATES):
                log_p_new = self.actor.log_prob_of(states, alphas)
                ratio     = (log_p_new - log_p_old.detach()).exp()
                surr      = torch.min(
                    ratio * adv.detach(),
                    ratio.clamp(1 - PPO_CLIP_EPS, 1 + PPO_CLIP_EPS) * adv.detach()
                )
                _, std   = self.actor(states)
                entropy  = 0.5 * torch.log(2 * np.pi * np.e * std.pow(2)).mean()
                a_loss   = -surr.mean() - PPO_ENT_COEF * entropy
                self.actor_opt.zero_grad()
                a_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_opt.step()

                v_loss = F.mse_loss(self.critic(states), rewards)
                self.critic_opt.zero_grad()
                v_loss.backward()
                self.critic_opt.step()

            avg_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            if verbose and epoch % 100 == 0:
                print(f"    [PPO]  Epoch {epoch:4d}  avg_r={avg_r:+.4f}")
        return avg_r


# GRPO AGENT - Group Relative Policy Optimisation (DeepSeek-R1 method)

class GRPOAgent:
    """
    For each market state, sample G candidate alpha values from the current
    policy. Simulate each candidate. Compute group-relative advantage:

        A_i = (r_i - mean(r_group)) / (std(r_group) + ε)

    Update the actor via policy gradient weighted by A_i. No critic, no
    value function - eliminates bootstrapping error entirely.

    KL penalty (DeepSeek-R1): β * KL(π_θ || π_ref) added to the loss to
    prevent the policy from collapsing. Reference policy π_ref is the
    initial actor - frozen after initialisation.

    The group-relative baseline is unbiased (it's the empirical mean of the
    group) and has lower variance than a single-sample REINFORCE baseline.
    Trade-off: G× more simulations per state per epoch vs SAC/PPO.
    """
    name = "GRPO"

    def __init__(self, state_dim):
        self.actor     = GaussianActor(state_dim)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=GRPO_LR)
        # Reference policy - frozen copy of initial actor weights
        # KL penalty against ref prevents policy from collapsing to degenerate solutions
        import copy
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

                # Sample G candidate alphas from current policy
                group_log_probs = []
                group_rewards   = []
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

                # Group-relative advantage - no critic needed
                r_arr = np.array(group_rewards, dtype=np.float32)
                adv   = torch.FloatTensor(
                    (r_arr - r_arr.mean()) / (r_arr.std() + 1e-8))

                log_probs = torch.cat(group_log_probs, dim=0).squeeze(-1)
                pg_loss = -(log_probs * adv).mean()

                # KL penalty: keep current policy close to reference
                # KL(N(mu1,s1) || N(mu2,s2)) = log(s2/s1) + (s1^2 + (mu1-mu2)^2)/(2*s2^2) - 0.5
                mu1, s1 = self.actor(s)
                with torch.no_grad():
                    mu2, s2 = self.ref_actor(s)
                kl = (torch.log(s2 / (s1 + 1e-8))
                      + (s1.pow(2) + (mu1 - mu2).pow(2)) / (2 * s2.pow(2) + 1e-8)
                      - 0.5).mean()
                loss = pg_loss + GRPO_KL_BETA * kl

                self.actor_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_opt.step()

            avg_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            if verbose and epoch % 100 == 0:
                print(f"    [GRPO] Epoch {epoch:4d}  avg_r={avg_r:+.4f}  "
                      f"(G={GRPO_G} samples/state)")
        return avg_r


# DAPO AGENT (ByteDance/Seed 2025)

class DAPOAgent:
    """
    DAPO: Dynamic Sampling Policy Optimisation.

    Three improvements over GRPO:
    1. Clip-higher: asymmetric clipping eps_low=0.20 / eps_high=0.28
    2. Dynamic G: sample more candidates for high-variance (hard) states
    3. No KL penalty: clip-higher provides sufficient regularisation
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

                # Phase 1: initial sample to estimate state difficulty
                alphas, log_probs, rewards = self._sample_group(s, row, DAPO_G_INIT)

                if len(rewards) >= 2:
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

                r_arr = np.array(rewards, dtype=np.float32)
                adv   = torch.FloatTensor(
                    (r_arr - r_arr.mean()) / (r_arr.std() + 1e-8))

                log_probs_old = torch.cat(log_probs, dim=0).squeeze(-1).detach()
                alphas_t      = torch.FloatTensor(alphas).unsqueeze(-1)

                mean_c, std_c = self.actor(s.expand(len(alphas), -1))
                y_t = ((alphas_t - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN) * 2.0 - 1.0
                       ).clamp(-0.9999, 0.9999)
                x_t = torch.atanh(y_t)
                dist = torch.distributions.Normal(mean_c, std_c)
                log_probs_new = (dist.log_prob(x_t)
                                 - torch.log(1.0 - y_t.pow(2) + 1e-6)).squeeze(-1)

                ratio = (log_probs_new - log_probs_old).exp()

                # Asymmetric clip: positive advantage → higher upper bound
                pos_mask  = adv > 0
                clip_high = torch.where(pos_mask,
                                        torch.full_like(ratio, 1.0 + DAPO_EPS_HIGH),
                                        torch.full_like(ratio, 1.0 + DAPO_EPS_LOW))
                clip_low  = torch.full_like(ratio, 1.0 - DAPO_EPS_LOW)
                ratio_clipped = torch.max(torch.min(ratio, clip_high), clip_low)

                obj  = torch.min(ratio * adv, ratio_clipped * adv)
                loss = -obj.mean()

                self.actor_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_opt.step()

            avg_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            if verbose and epoch % 100 == 0:
                print(f"    [DAPO] Epoch {epoch:4d}  avg_r={avg_r:+.4f}  "
                      f"(eps_low={DAPO_EPS_LOW}, eps_high={DAPO_EPS_HIGH})")
        return avg_r


# RULE-BASED FALLBACK (no PyTorch)

class RuleBasedAgent:
    name = "Rule"
    def __init__(self, state_dim): pass
    def select_action(self, state, deterministic=False):
        regime = state[4] if len(state) > 4 else 0.0
        return (ALPHA_MIN + ALPHA_MAX) / 2.0 * (1.0 - 0.4 * regime)
    def train(self, train_rows, verbose=False): return 0.0


# EVALUATE + STATS

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
    n          = len(r)
    ann_alpha  = (1 + r).prod() ** (12 / n) - 1
    track_err  = r.std(ddof=1) * np.sqrt(12)
    info_ratio = ann_alpha / track_err if track_err > 0 else float("nan")
    nav        = (1 + r).cumprod()
    return dict(ann_alpha=ann_alpha, track_err=track_err, info_ratio=info_ratio,
                hit_rate=float((r > 0).mean()),
                max_active_dd=float((nav / nav.cummax() - 1).min()),
                n_months=n)


# PLOTTING

COLORS = {"SAC": "#4A9EE0", "PPO": "#FF9800", "GRPO": "#66BB6A",
          "DAPO": "#FF6B9D", "Fixed": "#AAAAAA", "Rule": "#CC88CC"}

def plot_comparison(all_bt, fold_summary, algo_names):
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    fig.suptitle(
        "RL Algorithm Comparison - Walk-Forward Out-of-Sample\n"
        "SAC vs PPO vs GRPO vs DAPO vs Fixed Alpha",
        fontsize=13, fontweight="bold")

    # Cumulative active return
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

    # Per-fold IR
    ax    = axes[1]
    lbls  = fold_summary["label"].tolist()
    x     = np.arange(len(lbls))
    order = algo_names + ["Fixed"]
    n_a   = len(order)
    w     = 0.16
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
    ax.set_title("Per-Fold IR: SAC vs PPO vs GRPO vs DAPO vs Fixed Alpha")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = FIG_DIR / "algo_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


# MAIN

def main():
    print("=" * 70)
    print("5d_algorithm_comparison.py - SAC vs PPO vs GRPO vs DAPO")
    print("Walk-Forward Backtest  |  5 Expanding Folds  |  No Data Leakage")
    print("=" * 70)

    if not HAS_TORCH:
        print("WARNING: PyTorch not available - rule-based fallback only.\n")

    # Load data
    print("\nLoading data ...")
    panel   = pd.read_parquet(PANEL_FILE)
    panel["date"] = pd.to_datetime(panel["date"])
    weights = pd.read_parquet(WEIGHTS_FILE)
    weights["date"] = pd.to_datetime(weights["date"])
    factors_df   = pd.read_csv(FACTORS_FILE)
    factor_names = list(factors_df["factor"])
    signs_raw    = factors_df.set_index("factor").get(
        "majority_sign", pd.Series("+", index=factors_df["factor"]))
    signs = {f: (1.0 if signs_raw.get(f, "+") == "+" else -1.0)
             for f in factor_names}

    print(f"  Panel:   {panel['date'].min().date()} → {panel['date'].max().date()}"
          f"  ({panel['date'].nunique()} months)")
    print(f"  Factors: {len(factor_names)}")

    # Agent classes to compare
    AgentClasses = ([SACAgent, PPOAgent, GRPOAgent, DAPOAgent] if HAS_TORCH
                    else [RuleBasedAgent])
    algo_names   = [cls.name for cls in AgentClasses]
    print(f"  Algorithms: {', '.join(algo_names)}")

    all_bt     = {n: [] for n in algo_names + ["Fixed"]}
    fold_rows  = []

    # Fold loop
    for fold_idx, (train_end, test_start, test_end, label) in enumerate(FOLDS):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx+1}/{len(FOLDS)} - Test: {label}")
        print(f"  Train: {TRAIN_START_GLOBAL[:4]}–{train_end[:4]}  |  "
              f"Test: {test_start} → {test_end}")
        print("=" * 70)

        panel_train = panel[(panel["date"] >= TRAIN_START_GLOBAL) &
                            (panel["date"] <= train_end)].copy()
        panel_test  = panel[(panel["date"] >= test_start) &
                            (panel["date"] <= test_end)].copy()

        print(f"  Panel: {panel_train['date'].nunique()} train months, "
              f"{panel_test['date'].nunique()} test months")

        print("  Computing factor IC weights from training data ...")
        signed_w     = compute_fold_weights(panel_train, factor_names, signs)

        print("  Building scores ...")
        scores_train = build_scores(panel_train, factor_names, signed_w)
        scores_test  = build_scores(panel_test,  factor_names, signed_w)

        ep_train, norm_params = build_episodes(scores_train, weights, fit_norm=True)
        ep_test,  _           = build_episodes(scores_test,  weights,
                                               norm_params=norm_params)

        print(f"  Episodes: {len(ep_train)} train, {len(ep_test)} test")
        train_rows = list(ep_train.iterrows())

        fold_row        = {"label": label, "n_months": len(ep_test)}
        first_fixed_bt  = None

        for AgentClass in AgentClasses:
            state_dim = len(STATE_COLS)
            agent     = AgentClass(state_dim)
            print(f"\n  Training {agent.name} ...")
            agent.train(train_rows, verbose=True)
            rl_bt, fixed_bt = evaluate(agent, ep_test)
            s    = ie_stats(rl_bt)
            n    = agent.name.lower()
            fold_row[f"{n}_ir"]    = s.get("info_ratio",    float("nan"))
            fold_row[f"{n}_alpha"] = s.get("ann_alpha",     float("nan"))
            fold_row[f"{n}_te"]    = s.get("track_err",     float("nan"))
            fold_row[f"{n}_maxdd"] = s.get("max_active_dd", float("nan"))
            all_bt[agent.name].append(rl_bt)
            if first_fixed_bt is None and not fixed_bt.empty:
                first_fixed_bt = fixed_bt

        if first_fixed_bt is not None:
            s = ie_stats(first_fixed_bt)
            fold_row["fixed_ir"]    = s.get("info_ratio",    float("nan"))
            fold_row["fixed_alpha"] = s.get("ann_alpha",     float("nan"))
            fold_row["fixed_te"]    = s.get("track_err",     float("nan"))
            fold_row["fixed_maxdd"] = s.get("max_active_dd", float("nan"))
            all_bt["Fixed"].append(first_fixed_bt)

        fold_rows.append(fold_row)

        # Per-fold table
        print(f"\n  Results ({label}):")
        col_w   = 11
        header  = f"  {'Metric':<22}"
        for n in algo_names:
            header += f"  {n:>{col_w}}"
        header += f"  {'Fixed':>{col_w}}"
        print(header)
        print("  " + "-" * (22 + (col_w + 2) * (len(algo_names) + 1)))

        for metric, key, pct in [("Ann Alpha",    "alpha", True),
                                  ("Track Err",    "te",    True),
                                  ("Info Ratio",   "ir",    False),
                                  ("Max Active DD","maxdd", True)]:
            line = f"  {metric:<22}"
            for name in algo_names:
                v = fold_row.get(f"{name.lower()}_{key}", float("nan"))
                line += f"  {(v*100 if pct else v):>{col_w}.{'2f' if pct else '3f'}}{'%' if pct else ' '}"
            v = fold_row.get(f"fixed_{key}", float("nan"))
            line += f"  {(v*100 if pct else v):>{col_w}.{'2f' if pct else '3f'}}{'%' if pct else ' '}"
            print(line)

    # Aggregate
    fold_summary = pd.DataFrame(fold_rows)
    for name in algo_names + ["Fixed"]:
        all_bt[name] = (pd.concat(all_bt[name]).sort_index()
                        if all_bt[name] else pd.DataFrame())

    print(f"\n{'='*70}")
    print("OVERALL WALK-FORWARD RESULTS (all folds concatenated)")
    print("=" * 70)
    col_w  = 11
    header = f"{'Metric':<24}"
    for n in algo_names:
        header += f"  {n:>{col_w}}"
    header += f"  {'Fixed':>{col_w}}"
    print(header)
    print("-" * (24 + (col_w + 2) * (len(algo_names) + 1)))

    overall = {n: ie_stats(all_bt[n]) for n in algo_names + ["Fixed"]}
    for metric, key, pct in [
        ("Ann Alpha",         "ann_alpha",    True),
        ("Tracking Error",    "track_err",    True),
        ("Information Ratio", "info_ratio",   False),
        ("Hit Rate",          "hit_rate",     True),
        ("Max Active DD",     "max_active_dd",True),
    ]:
        line = f"{metric:<24}"
        for name in algo_names:
            v = overall[name].get(key, float("nan"))
            line += f"  {(v*100 if pct else v):>{col_w}.{'2f' if pct else '3f'}}{'%' if pct else ' '}"
        v = overall["Fixed"].get(key, float("nan"))
        line += f"  {(v*100 if pct else v):>{col_w}.{'2f' if pct else '3f'}}{'%' if pct else ' '}"
        print(line)

    print(f"\n  Per-fold IR:")
    col_w  = 9
    header = f"  {'Fold':<14}"
    for n in algo_names:
        header += f"  {n:>{col_w}}"
    header += f"  {'Fixed':>{col_w}}"
    print(header)
    print("  " + "-" * (14 + (col_w + 2) * (len(algo_names) + 1)))
    for _, fr in fold_summary.iterrows():
        fixed_ir = fr.get("fixed_ir", float("nan"))
        line = f"  {fr['label']:<14}"
        for name in algo_names:
            ir = fr.get(f"{name.lower()}_ir", float("nan"))
            tick = "✓" if ir > fixed_ir else " "
            line += f"  {ir:>{col_w}.3f}{tick}"
        line += f"  {fixed_ir:>{col_w}.3f}"
        print(line)

    print(f"\n  Algorithm wins: ", end="")
    for name in algo_names:
        wins = sum(
            fr.get(f"{name.lower()}_ir", float("-inf")) > fr.get("fixed_ir", float("-inf"))
            for _, fr in fold_summary.iterrows()
        )
        print(f"{name} {wins}/{len(FOLDS)}", end="  ")
    print()

    # Save
    fold_summary.to_csv(DATA_DIR / "algo_comparison.csv", index=False)
    print(f"\nSaved -> data/algo_comparison.csv")
    plot_comparison(all_bt, fold_summary, algo_names)

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("  SAC:  Off-policy replay buffer. Each of ~100 monthly obs reused")
    print("        many times. Twin critics reduce Q overestimation. Best fit")
    print("        for small-data continuous-action RL.")
    print("  PPO:  On-policy - throws away rollout after each update. With")
    print("        only ~100 training months, barely converges. Expected to")
    print("        underperform SAC at this sample size.")
    print("  GRPO: No critic. Group-relative reward replaces value baseline.")
    print("        Eliminates bootstrapping error. G=4 simulations per state.")
    print("        KL penalty (β=0.01) anchors policy to prevent collapse.")
    print("  DAPO: GRPO + asymmetric clip-higher (ε↓=0.20, ε↑=0.28) +")
    print("        dynamic G (2–8 based on reward variance) + no KL penalty.")
    print("        Best at limiting downside in volatile/uncertain regimes.")
    print("Done.")


if __name__ == "__main__":
    main()
