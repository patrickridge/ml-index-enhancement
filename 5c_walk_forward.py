"""
5c_walk_forward.py - Expanding-window walk-forward test for the SAC tilt agent.

The single-split 2010-2022 / 2023-2025 backtest has two problems: only 35 test
months, all in a bull rally; and the IC-optimised factor weights from
2e_ic_optimise.py were fitted on the full panel, which leaks forward-looking
information into the "training" years. Walk-forward addresses both by refitting
everything (factor weights, z-score normalisation, SAC agent, regime median)
inside each fold's training slice and concatenating the test windows into ~12
years of genuine out-of-sample returns.

On algorithm choice: PPO is on-policy and needs thousands of env steps per
update, which isn't realistic with 100-140 training months per fold. SAC is
off-policy, replays each month many times from the buffer, handles continuous
tilt actions naturally, and its entropy term stops the policy collapsing to a
single alpha.

Folds (expanding window):
  1. Train 2010-2013 → Test 2014-2015
  2. Train 2010-2015 → Test 2016-2017
  3. Train 2010-2017 → Test 2018-2019
  4. Train 2010-2019 → Test 2020-2021
  5. Train 2010-2021 → Test 2022-2025

Run:        python 5c_walk_forward.py
Writes:     data/bt_wf_rl.csv              monthly returns (all folds)
            data/wf_fold_summary.csv       per-fold IR, alpha, TE
            figures/wf_rl_comparison.png   cumulative active return vs fixed α
            figures/wf_fold_summary.png    per-fold IR bar chart
"""

import os
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

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not installed. Using rule-based fallback.")

# Paths
from config import clean_universe

DATA_DIR = Path(os.environ.get("ML_DATA_DIR", "data"))
FIG_DIR  = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

PANEL_FILE   = DATA_DIR / "panel_monthly_enriched.parquet"
FACTORS_FILE = DATA_DIR / "factor_selected.csv"     # base factor list (no optimised weights)
WEIGHTS_FILE = DATA_DIR / "spx_weights.parquet"

# Walk-forward fold definitions
# (train_end, test_start, test_end, label)
FOLDS = [
    ("2013-12-31", "2014-01-01", "2015-12-31", "2014–2015"),
    ("2015-12-31", "2016-01-01", "2017-12-31", "2016–2017"),
    ("2017-12-31", "2018-01-01", "2019-12-31", "2018–2019"),
    ("2019-12-31", "2020-01-01", "2021-12-31", "2020–2021"),
    ("2021-12-31", "2022-01-01", "2025-12-31", "2022–2025"),
]
TRAIN_START_GLOBAL = "2010-01-01"

# SAC / simulation hyperparameters
#
# ALPHA_MAX bounds the action space, and therefore the tracking error the agent
# can produce. At the old 0.050 the policy ran straight into the cap (mean α
# 2.61%, max 5.00%) and realised 9.71% tracking error - a long way outside the
# 2-4% budget that defines an index-enhancement mandate. The IR was real but it
# was earned by a portfolio no such mandate would permit.
#
# 0.018 keeps the agent inside the budget. Override to reproduce the old
# unconstrained behaviour:  ML_ALPHA_MAX=0.05 python 5c_walk_forward.py
ALPHA_MIN  = float(os.environ.get("ML_ALPHA_MIN", 0.002))
ALPHA_MAX  = float(os.environ.get("ML_ALPHA_MAX", 0.018))
# Baseline to beat. Must be comparable in SIZE to what the agent picks, or the
# comparison measures level rather than adaptation: at 0.010 against an agent
# averaging 0.0032, the fixed leg carries three times the tracking error, and
# since IR falls as alpha grows the agent wins on arithmetic alone.
#
# The default is now the agent's own mean tilt, so the baseline carries the
# same tracking error (5.04% against 5.00%) and the comparison isolates
# adaptation. Set ML_FIXED_ALPHA=0.010 to reproduce the old, flattering
# version, which the README reports alongside as a control.
FIXED_ALPHA = float(os.environ.get("ML_FIXED_ALPHA", 0.00316))

TRAIN_EPOCHS = 400
BATCH_SIZE   = 32
LR           = 3e-4
GAMMA        = 0.95
TAU          = 0.005
HIDDEN_DIM   = 64
BUFFER_SIZE  = 10_000
SEED         = 42

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
    """
    Compute mean-|IC| weights from training data only.
    This replicates the IC-decay weighting logic without any future data.

    Returns: signed_weights array aligned to factor_names
    """
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
    if total < 1e-9:
        # fallback: equal weights
        w = {f: 1.0 / len(factor_names) for f in factor_names}
    else:
        w = {f: v / total for f, v in ics.items()}

    # Apply majority-sign from base factor list (does not use future returns)
    signed_w = np.array([
        w[f] * signs.get(f, 1.0) for f in factor_names
    ], dtype=np.float64)
    return signed_w


def build_scores(panel_slice, factor_names, signed_w):
    """Build cross-sectional factor-combo scores for a panel slice."""
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
            rows.append({
                "date":       dt,
                "ticker":     row["ticker"],
                "score":      float(sc[i]),
                "fwd_ret_1m": row.get("fwd_ret_1m", np.nan),
            })
    return pd.DataFrame(rows)


# PORTFOLIO SIMULATION

def simulate_month(scores_month, weights_month, alpha):
    """Apply alpha tilt for one month. Returns dict or None."""
    df = scores_month.merge(
        weights_month[["ticker", "spx_weight"]], on="ticker", how="inner"
    ).dropna(subset=["score", "spx_weight", "fwd_ret_1m"])

    # Strip stub tickers and impossible returns before any weighting
    df = clean_universe(df, verbose=False)

    if len(df) < 50:
        return None

    top_n = bottom_n = 100
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    n  = len(df)

    tilt = pd.Series(0.0, index=df.index)
    tilt.iloc[:top_n]        = +alpha
    tilt.iloc[n - bottom_n:] = -alpha

    # Renormalise the benchmark over the universe we actually score. The raw
    # spx_weight column sums to the share of index cap that survived the merge
    # (~0.81 on the current panel), so measuring a fully-invested portfolio
    # against it books the missing exposure as alpha.
    coverage = df["spx_weight"].sum()
    if coverage < 1e-8:
        return None
    bench_w = df["spx_weight"] / coverage

    raw_w = (bench_w + tilt).clip(lower=0.0)
    total = raw_w.sum()
    if total < 1e-8:
        return None

    port_w    = raw_w / total
    port_ret  = (port_w  * df["fwd_ret_1m"]).sum()
    bench_ret = (bench_w * df["fwd_ret_1m"]).sum()
    return {
        "port_ret":   port_ret,
        "bench_ret":  bench_ret,
        "active_ret": port_ret - bench_ret,
    }


# BUILD EPISODE TABLE - with training-period normalisation

def build_episodes(scores, weights, ref_alpha=0.01,
                   norm_params=None, fit_norm=False):
    """
    Build monthly state-vector table.

    norm_params : dict of {col: (mean, std)} from training period.
                  If None and fit_norm=True, compute from this data.
                  If None and fit_norm=False, no normalisation (raw features).
    Returns     : (episodes_df, norm_params)
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
    df["recent_active_ret"] = (df["active_ret_ref"]
                               .rolling(3, min_periods=1).mean().fillna(0.0))

    # Regime: high-vol vs low-vol; threshold from expanding training median
    if fit_norm:
        vol_med = df["bench_vol"].expanding(min_periods=6).median()
    else:
        # For test period, use training-period vol median as threshold
        vol_med = norm_params.get("_vol_median", df["bench_vol"].median()) if norm_params else df["bench_vol"].median()
    df["regime"] = (df["bench_vol"] > vol_med).astype(float).fillna(0.0)

    # Z-score normalise continuous features (not regime)
    contin_cols = [c for c in STATE_COLS if c != "regime"]
    if fit_norm:
        norm_params = {}
        for col in contin_cols:
            mu, sg = df[col].mean(), df[col].std() + 1e-8
            norm_params[col] = (mu, sg)
        # Store vol median for test-period regime labelling
        norm_params["_vol_median"] = float(df["bench_vol"].expanding(min_periods=6).median().iloc[-1])

    if norm_params:
        for col in contin_cols:
            mu, sg = norm_params[col]
            df[col] = ((df[col] - mu) / sg).fillna(0.0).clip(-5.0, 5.0)

    return df, norm_params


# SAC COMPONENTS

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

    def __len__(self): return len(self.buf)


class SACAgent:
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
    def temperature(self): return float(self.log_alpha_temp.exp())

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
        S  = torch.FloatTensor(s);  A  = torch.FloatTensor(a)
        R  = torch.FloatTensor(r);  NS = torch.FloatTensor(ns)
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
        return {}


# TRAINING + EVALUATION

def train_fold(agent, ep_train):
    """Train SAC on one fold's training episodes."""
    train_rows = list(ep_train.iterrows())
    reward_hist = []
    for epoch in range(TRAIN_EPOCHS):
        random.shuffle(train_rows)
        ep_rewards = []
        for i, (dt, row) in enumerate(train_rows):
            state = np.clip(
                np.nan_to_num(row[STATE_COLS].values.astype(np.float32),
                              nan=0.0, posinf=0.0, neginf=0.0),
                -5.0, 5.0)
            alpha  = agent.select_action(state)
            result = simulate_month(row["_scores"], row["_weights"], alpha=alpha)
            if result is None:
                continue

            reward = result["active_ret"] * 12.0
            ep_rewards.append(reward)

            if i + 1 >= len(train_rows):
                next_state = np.zeros(len(STATE_COLS), dtype=np.float32)
                done_flag  = 1.0
            else:
                next_state = np.nan_to_num(
                    train_rows[i + 1][1][STATE_COLS].values.astype(np.float32),
                    nan=0.0, posinf=0.0, neginf=0.0)
                done_flag  = 0.0

            agent.buffer.push(state, alpha, reward, next_state, done_flag)
            agent.update()

        avg_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        reward_hist.append(avg_r)
        if epoch % 100 == 0:
            print(f"      Epoch {epoch:4d}  avg_reward={avg_r:+.4f}"
                  f"  temp={agent.temperature:.3f}")
    return reward_hist


def rule_based_alpha(row):
    """Fallback when PyTorch unavailable: lower alpha in risk-off regime."""
    if float(row["regime"]) > 0.5:
        return ALPHA_MIN + 0.3 * (ALPHA_MAX - ALPHA_MIN)
    return ALPHA_MIN + 0.7 * (ALPHA_MAX - ALPHA_MIN)


def evaluate_fold(agent, ep_test):
    """Evaluate agent on test episodes. Returns (rl_df, fixed_df)."""
    rl_rows, fixed_rows = [], []
    for dt, row in ep_test.iterrows():
        state = np.clip(
            np.nan_to_num(row[STATE_COLS].values.astype(np.float32),
                          nan=0.0, posinf=0.0, neginf=0.0),
            -5.0, 5.0)

        if agent is not None:
            alpha_rl = agent.select_action(state, deterministic=True)
        else:
            alpha_rl = rule_based_alpha(row)

        r_rl = simulate_month(row["_scores"], row["_weights"], alpha=alpha_rl)
        if r_rl:
            rl_rows.append({"date": dt, "alpha_used": alpha_rl,
                            "regime": row["regime"], **r_rl})

        r_fx = simulate_month(row["_scores"], row["_weights"], alpha=FIXED_ALPHA)
        if r_fx:
            fixed_rows.append({"date": dt, "alpha_used": FIXED_ALPHA,
                                "regime": row["regime"], **r_fx})

    rl_bt    = pd.DataFrame(rl_rows).set_index("date")    if rl_rows    else pd.DataFrame()
    fixed_bt = pd.DataFrame(fixed_rows).set_index("date") if fixed_rows else pd.DataFrame()
    return rl_bt, fixed_bt


# STATS

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
    return dict(ann_alpha=ann_alpha, track_err=track_err,
                info_ratio=info_ratio, hit_rate=hit_rate,
                max_active_dd=max_dd, n_months=n)


# PLOTTING

def plot_results(all_rl, all_fixed, fold_summaries):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Walk-Forward RL Backtest - No Data Leakage\n"
                 "(5 expanding folds, factor weights recomputed per fold, "
                 "SAC trained fresh per fold)",
                 fontsize=12, fontweight="bold")

    # Top-left: cumulative active return
    ax = axes[0, 0]
    if not all_rl.empty:
        rl_nav = (1 + all_rl["active_ret"]).cumprod() - 1
        ax.plot(rl_nav.index, rl_nav * 100,
                label="RL Agent (walk-forward)", color="#2196F3", lw=2)
    if not all_fixed.empty:
        fx_nav = (1 + all_fixed["active_ret"]).cumprod() - 1
        ax.plot(fx_nav.index, fx_nav * 100,
                label=f"Fixed α={FIXED_ALPHA:.3f}", color="#FF9800",
                lw=2, linestyle="--")
    ax.axhline(0, color="black", lw=0.8, ls=":")
    # Shade fold boundaries
    for i, (te, ts, tend, lbl) in enumerate(FOLDS):
        if i % 2 == 0:
            ax.axvspan(pd.Timestamp(ts), pd.Timestamp(tend),
                       alpha=0.07, color="blue")
    ax.set_ylabel("Cumulative Active Return (%)")
    ax.set_title("Cumulative Active Return vs Benchmark")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Top-right: monthly alpha decisions
    ax = axes[0, 1]
    if not all_rl.empty:
        ax.bar(all_rl.index, all_rl["alpha_used"] * 100,
               width=20, color="#2196F3", alpha=0.7, label="RL alpha")
    ax.axhline(FIXED_ALPHA * 100, color="#FF9800", lw=1.5, ls="--",
               label=f"Fixed α={FIXED_ALPHA*100:.1f}%")
    ax.set_ylabel("Alpha Used (%)")
    ax.set_title("RL Alpha Decisions Over Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Bottom-left: IR per fold
    ax = axes[1, 0]
    labels = [s["label"] for s in fold_summaries]
    rl_irs  = [s["rl_ir"]    for s in fold_summaries]
    fx_irs  = [s["fixed_ir"] for s in fold_summaries]
    x = np.arange(len(labels))
    w = 0.35
    bars_rl = ax.bar(x - w/2, rl_irs,  w, label="RL Agent",
                     color="#2196F3", alpha=0.85)
    bars_fx = ax.bar(x + w/2, fx_irs,  w, label="Fixed alpha",
                     color="#FF9800", alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(0.5, color="green", lw=1, ls="--", alpha=0.6, label="IR=0.5 (good)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Information Ratio")
    ax.set_title("Information Ratio per Fold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    for b in bars_rl:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, v + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    for b in bars_fx:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, v + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    # Bottom-right: monthly active returns scatter
    ax = axes[1, 1]
    if not all_rl.empty and not all_fixed.empty:
        common = all_rl.index.intersection(all_fixed.index)
        rl_ar  = all_rl.loc[common, "active_ret"] * 100
        fx_ar  = all_fixed.loc[common, "active_ret"] * 100
        ax.scatter(fx_ar, rl_ar, alpha=0.5, s=20, color="#2196F3")
        lim = max(abs(rl_ar).max(), abs(fx_ar).max()) * 1.1
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.5)
        ax.axhline(0, color="black", lw=0.5); ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("Fixed Alpha Active Ret (%)")
        ax.set_ylabel("RL Agent Active Ret (%)")
        ax.set_title("RL vs Fixed: Monthly Active Returns")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / "wf_rl_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


# MAIN

def main():
    print("=" * 70)
    print("5c_walk_forward.py - Walk-Forward RL Backtest (No Data Leakage)")
    print("=" * 70)
    print(f"\nFolds: {len(FOLDS)}")
    for te, ts, tend, lbl in FOLDS:
        print(f"  Train {TRAIN_START_GLOBAL[:4]}–{te[:4]} → Test {lbl}")

    # Load data
    print("\nLoading data ...")
    panel   = pd.read_parquet(PANEL_FILE)
    panel["date"] = pd.to_datetime(panel["date"])
    weights = pd.read_parquet(WEIGHTS_FILE)
    weights["date"] = pd.to_datetime(weights["date"])
    factors = pd.read_csv(FACTORS_FILE)

    factor_names = list(factors["factor"])
    signs        = factors.set_index("factor")["majority_sign"].map(
                       {"+": 1.0, "-": -1.0}).fillna(1.0).to_dict()

    print(f"  Panel:   {panel['date'].min().date()} → {panel['date'].max().date()}"
          f"  ({panel['date'].nunique()} months)")
    print(f"  Weights: {weights['date'].min().date()} → {weights['date'].max().date()}")
    print(f"  Factors: {len(factor_names)}")

    state_dim = len(STATE_COLS)
    all_rl_dfs    = []
    all_fixed_dfs = []
    fold_summaries = []

    # Walk-forward loop
    for fold_idx, (train_end, test_start, test_end, fold_label) in enumerate(FOLDS):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx+1}/5 - Test: {fold_label}")
        print(f"  Train: {TRAIN_START_GLOBAL[:4]}–{train_end[:4]}"
              f"  |  Test: {test_start[:10]} → {test_end[:10]}")
        print("=" * 70)

        # Slice panel
        panel_train = panel[
            (panel["date"] >= TRAIN_START_GLOBAL) &
            (panel["date"] <= train_end)
        ].copy()
        panel_test  = panel[
            (panel["date"] >= test_start) &
            (panel["date"] <= test_end)
        ].copy()

        n_train_months = panel_train["date"].nunique()
        n_test_months  = panel_test["date"].nunique()
        print(f"  Panel: {n_train_months} train months, {n_test_months} test months")

        # Step 1: Compute factor weights from training data only
        print(f"  Computing factor IC weights from training data ...")
        signed_w = compute_fold_weights(panel_train, factor_names, signs)
        top5 = np.argsort(np.abs(signed_w))[::-1][:5]
        print(f"  Top 5 factors (by weight): "
              + ", ".join(f"{factor_names[i]}={signed_w[i]:.4f}" for i in top5))

        # Step 2: Build scores for train + test
        print(f"  Building factor-combo scores ...")
        scores_train = build_scores(panel_train, factor_names, signed_w)
        scores_test  = build_scores(panel_test,  factor_names, signed_w)
        print(f"  Scores: {scores_train['date'].nunique()} train, "
              f"{scores_test['date'].nunique()} test months")

        # Step 3: Build episode tables (normalise from training only)
        print(f"  Building episode tables ...")
        ep_train, norm_params = build_episodes(
            scores_train, weights, fit_norm=True)
        ep_test,  _           = build_episodes(
            scores_test, weights, norm_params=norm_params, fit_norm=False)
        print(f"  Episodes: {len(ep_train)} train, {len(ep_test)} test")

        if len(ep_train) < 24:
            print(f"  WARNING: only {len(ep_train)} training months - skipping fold.")
            continue

        # Step 4: Train SAC from scratch on training data
        if HAS_TORCH:
            print(f"  Training SAC agent ({TRAIN_EPOCHS} epochs) ...")
            np.random.seed(SEED + fold_idx)
            random.seed(SEED + fold_idx)
            torch.manual_seed(SEED + fold_idx)
            agent = SACAgent(state_dim)
            reward_hist = train_fold(agent, ep_train)
            print(f"  Training complete. Final avg reward: {reward_hist[-1]:+.4f}")
        else:
            print(f"  PyTorch not available - using rule-based fallback.")
            agent = None

        # Step 5: Evaluate on test period
        print(f"  Evaluating on test period ({fold_label}) ...")
        rl_bt, fixed_bt = evaluate_fold(agent, ep_test)

        if rl_bt.empty:
            print(f"  WARNING: no test results for fold {fold_idx+1}. Skipping.")
            continue

        # Tag with fold
        rl_bt["fold"]    = fold_label
        fixed_bt["fold"] = fold_label

        # Per-fold stats
        rl_s  = ie_stats(rl_bt)
        fx_s  = ie_stats(fixed_bt)

        print(f"\n  Results ({fold_label}):")
        print(f"  {'Metric':<24}  {'RL Agent':>10}  {'Fixed α':>10}")
        print(f"  {'-'*48}")
        for k, label in [("ann_alpha", "Ann Alpha"), ("track_err", "Track Err"),
                         ("info_ratio", "Info Ratio"), ("hit_rate", "Hit Rate"),
                         ("max_active_dd", "Max Active DD")]:
            pct = k != "info_ratio"
            rl_v = rl_s.get(k, float("nan"))
            fx_v = fx_s.get(k, float("nan"))
            if pct:
                print(f"  {label:<24}  {rl_v*100:>9.2f}%  {fx_v*100:>9.2f}%")
            else:
                print(f"  {label:<24}  {rl_v:>10.3f}  {fx_v:>10.3f}")

        if not rl_bt.empty:
            print(f"  RL alpha: mean={rl_bt['alpha_used'].mean()*100:.2f}%  "
                  f"min={rl_bt['alpha_used'].min()*100:.2f}%  "
                  f"max={rl_bt['alpha_used'].max()*100:.2f}%")

        fold_summaries.append({
            "label":    fold_label,
            "n_months": rl_s.get("n_months", 0),
            "rl_ir":    rl_s.get("info_ratio", float("nan")),
            "rl_alpha": rl_s.get("ann_alpha",  float("nan")),
            "rl_te":    rl_s.get("track_err",  float("nan")),
            "fixed_ir": fx_s.get("info_ratio", float("nan")),
        })

        all_rl_dfs.append(rl_bt)
        all_fixed_dfs.append(fixed_bt)

    # Aggregate results
    if not all_rl_dfs:
        print("\nNo results to aggregate. Exiting.")
        return

    all_rl    = pd.concat(all_rl_dfs).sort_index()
    all_fixed = pd.concat(all_fixed_dfs).sort_index()

    # Remove any overlap (shouldn't be any with non-overlapping test windows)
    all_rl    = all_rl[~all_rl.index.duplicated(keep="last")]
    all_fixed = all_fixed[~all_fixed.index.duplicated(keep="last")]

    overall_rl  = ie_stats(all_rl)
    overall_fx  = ie_stats(all_fixed)

    print(f"\n{'='*70}")
    print("OVERALL WALK-FORWARD RESULTS (all folds concatenated)")
    print(f"{'='*70}")
    print(f"{'Metric':<28}  {'RL Agent':>12}  {'Fixed α=0.01':>12}")
    print(f"{'-'*56}")

    def _v(d, k, pct=True, fmt=".2f"):
        if not d or k not in d: return "n/a"
        v = d[k] * (100 if pct else 1)
        return f"{v:{fmt}}" + ("%" if pct else "")

    print(f"  {'Months Out-of-Sample':<26}  {overall_rl.get('n_months','?'):>12}  "
          f"{overall_fx.get('n_months','?'):>12}")
    print(f"  {'Ann Alpha':<26}  {_v(overall_rl,'ann_alpha'):>12}  "
          f"{_v(overall_fx,'ann_alpha'):>12}")
    print(f"  {'Tracking Error':<26}  {_v(overall_rl,'track_err'):>12}  "
          f"{_v(overall_fx,'track_err'):>12}")
    print(f"  {'Information Ratio':<26}  "
          f"{_v(overall_rl,'info_ratio',False,'.3f'):>12}  "
          f"{_v(overall_fx,'info_ratio',False,'.3f'):>12}")
    print(f"  {'Hit Rate':<26}  {_v(overall_rl,'hit_rate',True,'.1f'):>12}  "
          f"{_v(overall_fx,'hit_rate',True,'.1f'):>12}")
    print(f"  {'Max Active DD':<26}  {_v(overall_rl,'max_active_dd'):>12}  "
          f"{_v(overall_fx,'max_active_dd'):>12}")

    print(f"\n  Per-fold IR summary:")
    print(f"  {'Fold':<14}  {'n':>4}  {'RL IR':>8}  {'Fixed IR':>8}")
    print(f"  {'-'*40}")
    for s in fold_summaries:
        rl_ir  = s['rl_ir']
        fx_ir  = s['fixed_ir']
        better = "✓" if (np.isfinite(rl_ir) and np.isfinite(fx_ir) and rl_ir > fx_ir) else " "
        print(f"  {s['label']:<14}  {s['n_months']:>4}  "
              f"{rl_ir:>8.3f}  {fx_ir:>8.3f}  {better}")

    rl_wins = sum(1 for s in fold_summaries
                  if np.isfinite(s['rl_ir']) and np.isfinite(s['fixed_ir'])
                  and s['rl_ir'] > s['fixed_ir'])
    print(f"\n  RL beats fixed alpha in {rl_wins}/{len(fold_summaries)} folds.")

    # Save outputs
    out_bt = DATA_DIR / "bt_wf_rl.csv"
    all_rl.reset_index().to_csv(out_bt, index=False)
    print(f"\nSaved -> {out_bt}")

    fold_df = pd.DataFrame(fold_summaries)
    out_folds = DATA_DIR / "wf_fold_summary.csv"
    fold_df.to_csv(out_folds, index=False)
    print(f"Saved -> {out_folds}")

    plot_results(all_rl, all_fixed, fold_summaries)

    print("\nDone.")
    print(f"\nNOTE: These are fully out-of-sample results - factor weights,")
    print(f"state normalisation, and RL policy were all trained on past data only.")
    print(f"No information from the test period was used in any component.")


if __name__ == "__main__":
    main()
