"""
5e_dapo_kaggle.py — DAPO/GRPO/DAPOSwitch Walk-Forward (Kaggle CPU version)
===========================================================================
Identical logic to 5e_dapo_agent.py. Only the paths differ.

NOTE: GPU will NOT speed this up — the bottleneck is simulate_month()
which is pure pandas. Kaggle is faster because it has a stronger CPU
and more RAM than a MacBook Air.

╔══════════════════════════════════════════════════════════════════════╗
║  KAGGLE SETUP (4 steps)                                              ║
║                                                                      ║
║  1. Upload these 3 files to a Kaggle Dataset (e.g. "investsoc-rl"):  ║
║       • panel_monthly_enriched.parquet                               ║
║       • factor_selected.csv                                          ║
║       • spx_weights.parquet                                          ║
║                                                                      ║
║  2. Create a new Kaggle Notebook, attach the dataset above.          ║
║     Settings → Accelerator → None (CPU is fine, GPU adds no benefit) ║
║                                                                      ║
║  3. Upload this file and in a cell run:                              ║
║       !pip install hmmlearn -q                                       ║
║       !python 5e_dapo_kaggle.py                                      ║
║     OR paste the whole file into a code cell.                        ║
║                                                                      ║
║  4. When done, download from Output tab:                             ║
║       • dapo_comparison.csv   → put in your local data/              ║
║       • dapo_comparison.png   → put in your local figures/           ║
╚══════════════════════════════════════════════════════════════════════╝

Expected Kaggle CPU time: ~30–45 minutes (vs 60–90 min on MacBook Air)
"""

import subprocess, sys
try:
    import hmmlearn
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "hmmlearn", "-q"])

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

# ── Kaggle paths ───────────────────────────────────────────────────────────────
import os
KAGGLE_INPUT = Path("/kaggle/input")

# Find the attached dataset automatically
_datasets = list(KAGGLE_INPUT.glob("*/panel_monthly_enriched.parquet"))
if _datasets:
    _ds_dir = _datasets[0].parent
else:
    _ds_dir = KAGGLE_INPUT / "investsoc-rl"   # fallback — rename to match yours

DATA_DIR = Path("/kaggle/working")
FIG_DIR  = Path("/kaggle/working")

PANEL_FILE   = _ds_dir / "panel_monthly_enriched.parquet"
FACTORS_FILE = _ds_dir / "factor_selected.csv"
WEIGHTS_FILE = _ds_dir / "spx_weights.parquet"

print(f"Dataset dir : {_ds_dir}")
print(f"Panel file  : {PANEL_FILE}  (exists={PANEL_FILE.exists()})")
print(f"Factors file: {FACTORS_FILE}  (exists={FACTORS_FILE.exists()})")
print(f"Weights file: {WEIGHTS_FILE}  (exists={WEIGHTS_FILE.exists()})")

# ── Walk-forward folds ─────────────────────────────────────────────────────────
FOLDS = [
    ("2013-12-31", "2014-01-01", "2015-12-31", "2014–2015"),
    ("2015-12-31", "2016-01-01", "2017-12-31", "2016–2017"),
    ("2017-12-31", "2018-01-01", "2019-12-31", "2018–2019"),
    ("2019-12-31", "2020-01-01", "2021-12-31", "2020–2021"),
    ("2021-12-31", "2022-01-01", "2025-12-31", "2022–2025"),
]
TRAIN_START_GLOBAL = "2010-01-01"

# ── Hyperparameters (identical to 5e_dapo_agent.py) ───────────────────────────
ALPHA_MIN   = 0.002
ALPHA_MAX   = 0.050
FIXED_ALPHA = 0.010
HIDDEN_DIM  = 64
SEED        = 42

GRPO_EPOCHS  = 400
GRPO_LR      = 3e-4
GRPO_G       = 4
GRPO_KL_BETA = 0.01

DAPO_EPOCHS     = 400
DAPO_LR         = 3e-4
DAPO_EPS_LOW    = 0.20
DAPO_EPS_HIGH   = 0.28
DAPO_G_INIT     = 4
DAPO_G_MIN      = 4
DAPO_G_MAX      = 8
DAPO_VAR_THRESH = 0.05

STATE_COLS = [
    "signal_strength", "signal_dispersion", "bench_vol",
    "recent_active_ret", "regime", "rolling_te",
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
    port_ret  = (port_w           * df["fwd_ret_1m"]).sum()
    bench_ret = (df["spx_weight"] * df["fwd_ret_1m"]).sum()
    return {"port_ret": port_ret, "bench_ret": bench_ret,
            "active_ret": port_ret - bench_ret}


# =============================================================================
# HMM REGIME DETECTION
# =============================================================================

def _fit_hmm(bench_vol_series, bench_ret_series):
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        return None, None
    X = np.column_stack([bench_vol_series.fillna(0).values,
                         bench_ret_series.fillna(0).values])
    if len(X) < 12:
        return None, None
    try:
        model = GaussianHMM(n_components=2, covariance_type="full",
                            n_iter=200, random_state=42)
        model.fit(X)
        states = model.predict(X)
        mean_vol = [X[states == s, 0].mean() for s in range(2)]
        risk_off_state = int(np.argmax(mean_vol))
        return model, risk_off_state
    except Exception:
        return None, None


def _hmm_predict(model, risk_off_state, df):
    X = np.column_stack([df["bench_vol"].fillna(0).values,
                         df["bench_ret"].fillna(0).values])
    try:
        states = model.predict(X)
        return pd.Series((states == risk_off_state).astype(float), index=df.index)
    except Exception:
        return pd.Series(0.0, index=df.index)


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
        norm_params = {}
        for col in [c for c in STATE_COLS if c != "regime"]:
            mu, sg = df[col].mean(), df[col].std() + 1e-8
            norm_params[col] = (mu, sg)
        hmm_model, risk_off_state = _fit_hmm(df["bench_vol"], df["bench_ret"])
        norm_params["_hmm_model"]          = hmm_model
        norm_params["_hmm_risk_off_state"] = risk_off_state if risk_off_state is not None else 1
        if hmm_model is not None:
            df["regime"] = _hmm_predict(hmm_model, norm_params["_hmm_risk_off_state"], df)
            print("    [Regime] HMM fitted — risk-off months:",
                  int(df["regime"].sum()), "/", len(df))
        else:
            vol_med = df["bench_vol"].expanding(min_periods=6).median()
            df["regime"] = (df["bench_vol"] > vol_med).astype(float).fillna(0.0)
            norm_params["_vol_median"] = float(vol_med.iloc[-1])
            print("    [Regime] hmmlearn not available — using vol-threshold fallback")
    else:
        hmm_model      = norm_params.get("_hmm_model")           if norm_params else None
        risk_off_state = norm_params.get("_hmm_risk_off_state", 1) if norm_params else 1
        if hmm_model is not None:
            df["regime"] = _hmm_predict(hmm_model, risk_off_state, df)
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
# NEURAL NETWORK
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
# GRPO AGENT
# =============================================================================

class GRPOAgent:
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
                adv   = torch.FloatTensor((r_arr - r_arr.mean()) / (r_arr.std() + 1e-8))
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
                alphas, log_probs, rewards = self._sample_group(s, row, DAPO_G_INIT)
                if len(rewards) >= 2:
                    r_var = float(np.var(rewards))
                    if r_var > DAPO_VAR_THRESH and len(rewards) < DAPO_G_MAX:
                        a2, lp2, r2 = self._sample_group(s, row, DAPO_G_MAX - len(rewards))
                        alphas += a2; log_probs += lp2; rewards += r2
                if len(rewards) < 2:
                    continue
                ep_rewards.extend(rewards)
                r_arr = np.array(rewards, dtype=np.float32)
                adv   = torch.FloatTensor((r_arr - r_arr.mean()) / (r_arr.std() + 1e-8))
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
                pos_mask  = adv > 0
                clip_high = torch.where(pos_mask,
                                        torch.full_like(ratio, 1.0 + DAPO_EPS_HIGH),
                                        torch.full_like(ratio, 1.0 + DAPO_EPS_LOW))
                clip_low  = torch.full_like(ratio, 1.0 - DAPO_EPS_LOW)
                ratio_clipped = torch.max(torch.min(ratio, clip_high), clip_low)
                loss = -torch.min(ratio * adv, ratio_clipped * adv).mean()
                self.actor_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_opt.step()
            avg_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            if verbose and epoch % 100 == 0:
                print(f"    [DAPO] Epoch {epoch:4d}  avg_r={avg_r:+.4f}")
        return avg_r


# =============================================================================
# DAPO-SWITCH AGENT
# =============================================================================

class DAPOSwitchAgent:
    name = "DAPOSwitch"

    def __init__(self, state_dim):
        self.actor     = GaussianActor(state_dim)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=DAPO_LR)
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

    def _dapo_update(self, s, alphas, log_probs, rewards):
        r_arr = np.array(rewards, dtype=np.float32)
        adv   = torch.FloatTensor((r_arr - r_arr.mean()) / (r_arr.std() + 1e-8))
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
        pos_mask  = adv > 0
        clip_high = torch.where(pos_mask,
                                torch.full_like(ratio, 1.0 + DAPO_EPS_HIGH),
                                torch.full_like(ratio, 1.0 + DAPO_EPS_LOW))
        clip_low  = torch.full_like(ratio, 1.0 - DAPO_EPS_LOW)
        ratio_clipped = torch.max(torch.min(ratio, clip_high), clip_low)
        return -torch.min(ratio * adv, ratio_clipped * adv).mean()

    def _grpo_update(self, s, log_probs, rewards):
        r_arr = np.array(rewards, dtype=np.float32)
        adv   = torch.FloatTensor((r_arr - r_arr.mean()) / (r_arr.std() + 1e-8))
        lp    = torch.cat(log_probs, dim=0).squeeze(-1)
        pg    = -(lp * adv).mean()
        mu1, s1 = self.actor(s)
        with torch.no_grad():
            mu2, s2 = self.ref_actor(s)
        kl = (torch.log(s2 / (s1 + 1e-8))
              + (s1.pow(2) + (mu1 - mu2).pow(2)) / (2 * s2.pow(2) + 1e-8)
              - 0.5).mean()
        return pg + GRPO_KL_BETA * kl

    def train(self, train_rows, verbose=False):
        avg_r = 0.0
        for epoch in range(DAPO_EPOCHS):
            ep_rewards  = []
            prev_regime = None
            for dt, row in train_rows:
                state          = _safe_state(row)
                current_regime = float(row["regime"])
                is_transition  = (prev_regime is not None and
                                  current_regime != prev_regime)
                prev_regime    = current_regime
                s = torch.nan_to_num(torch.FloatTensor(state).unsqueeze(0), nan=0.0)
                if is_transition:
                    _, log_probs, rewards = self._sample_group(s, row, GRPO_G)
                    if len(rewards) < 2:
                        continue
                    loss = self._grpo_update(s, log_probs, rewards)
                else:
                    alphas, log_probs, rewards = self._sample_group(s, row, DAPO_G_INIT)
                    if len(rewards) >= 2:
                        r_var = float(np.var(rewards))
                        if r_var > DAPO_VAR_THRESH and len(rewards) < DAPO_G_MAX:
                            a2, lp2, r2 = self._sample_group(s, row, DAPO_G_MAX - len(rewards))
                            alphas += a2; log_probs += lp2; rewards += r2
                    if len(rewards) < 2:
                        continue
                    loss = self._dapo_update(s, alphas, log_probs, rewards)
                ep_rewards.extend(rewards)
                self.actor_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_opt.step()
            avg_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            if verbose and epoch % 100 == 0:
                print(f"    [DAPOSwitch] Epoch {epoch:4d}  avg_r={avg_r:+.4f}")
        return avg_r


class RuleBasedAgent:
    name = "Rule"
    def __init__(self, state_dim): pass
    def select_action(self, state, deterministic=False):
        regime = state[4] if len(state) > 4 else 0.0
        return (ALPHA_MIN + ALPHA_MAX) / 2.0 * (1.0 - 0.4 * regime)
    def train(self, train_rows, verbose=False): return 0.0


# =============================================================================
# EVALUATE + STATS + PLOT
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
    rl_bt    = pd.DataFrame(rl_rows).set_index("date")    if rl_rows    else pd.DataFrame()
    fixed_bt = pd.DataFrame(fixed_rows).set_index("date") if fixed_rows else pd.DataFrame()
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


COLORS = {"DAPO": "#FF6B9D", "GRPO": "#66BB6A", "DAPOSwitch": "#5B9BD5", "Fixed": "#AAAAAA"}


def plot_comparison(all_bt, fold_summary, algo_names):
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    fig.suptitle("DAPO vs GRPO vs DAPOSwitch — Walk-Forward Out-of-Sample\n"
                 "DAPOSwitch: DAPO on stable regime, GRPO+KL on transitions",
                 fontsize=13, fontweight="bold")
    ax = axes[0]
    for name, bt in all_bt.items():
        if bt.empty:
            continue
        nav = (1 + bt["active_ret"]).cumprod() - 1
        ax.plot(nav.index, nav * 100, label=name, color=COLORS.get(name, "#999"),
                lw=2, linestyle="--" if name == "Fixed" else "-")
    ax.axhline(0, color="black", lw=0.8, linestyle=":")
    ax.set_ylabel("Cumulative Active Return (%)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax    = axes[1]
    lbls  = fold_summary["label"].tolist()
    x     = np.arange(len(lbls))
    order = algo_names + ["Fixed"]
    w     = 0.20
    for i, name in enumerate(order):
        key = f"{name.lower()}_ir"
        if key not in fold_summary.columns:
            continue
        offset = (i - (len(order) - 1) / 2) * (w + 0.02)
        ax.bar(x + offset, fold_summary[key].tolist(), w,
               label=name, color=COLORS.get(name, "#999"), alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(lbls, fontsize=9)
    ax.set_ylabel("Information Ratio"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
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
    print("5e_dapo_kaggle.py — DAPO vs GRPO vs DAPOSwitch (Kaggle CPU)")
    print("=" * 70)

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
    algo_names = ["DAPO", "GRPO", "DAPOSwitch"]
    all_bt     = {n: [] for n in algo_names + ["Fixed"]}
    fold_summary = []

    for fold_idx, (train_end, test_start, test_end, label) in enumerate(FOLDS):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx+1}/5 — {label}  |  train ≤ {train_end}  |  test {test_start}→{test_end}")
        print("="*70)

        panel_train = panel[(panel["date"] >= TRAIN_START_GLOBAL) &
                            (panel["date"] <= train_end)].copy()
        panel_test  = panel[(panel["date"] >= test_start) &
                            (panel["date"] <= test_end)].copy()

        if len(panel_train) < 500 or len(panel_test) < 50:
            print("  Skipping — insufficient data.")
            continue

        print("  Computing factor weights ...")
        signed_w = compute_fold_weights(panel_train, factor_names, signs)
        print("  Building scores ...")
        sc_train = build_scores(panel_train, factor_names, signed_w)
        sc_test  = build_scores(panel_test,  factor_names, signed_w)

        ep_train, norm_p = build_episodes(sc_train, weights, fit_norm=True)
        ep_test,  _      = build_episodes(sc_test,  weights, norm_params=norm_p)

        if ep_train.empty or ep_test.empty:
            print("  Skipping — empty episodes.")
            continue

        train_rows = list(ep_train.iterrows())
        print(f"  Train: {len(train_rows)} months  |  Test: {len(ep_test)} months")

        fold_bt   = {}
        fold_rows = {"label": label, "n_months": len(ep_test)}
        AgentClasses = [DAPOAgent, GRPOAgent, DAPOSwitchAgent] if HAS_TORCH else [RuleBasedAgent] * 3

        for name, AgentCls in zip(algo_names, AgentClasses):
            print(f"\n  [{name}] Training {DAPO_EPOCHS} epochs ...")
            agent   = AgentCls(state_dim)
            agent.train(train_rows, verbose=True)
            bt, fbt = evaluate(agent, ep_test)
            s       = ie_stats(bt)
            fold_bt[name] = bt
            print(f"  [{name}] IR={s.get('info_ratio', float('nan')):.3f}  "
                  f"α={s.get('ann_alpha', 0)*100:.2f}%  TE={s.get('track_err', 0)*100:.2f}%")
            fold_rows[f"{name.lower()}_ir"]    = s.get("info_ratio",   float("nan"))
            fold_rows[f"{name.lower()}_alpha"] = s.get("ann_alpha",    float("nan"))
            fold_rows[f"{name.lower()}_te"]    = s.get("track_err",    float("nan"))
            fold_rows[f"{name.lower()}_maxdd"] = s.get("max_active_dd",float("nan"))

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

    all_bt_concat = {}
    for name in algo_names + ["Fixed"]:
        if all_bt[name]:
            all_bt_concat[name] = pd.concat(all_bt[name]).sort_index()
        else:
            all_bt_concat[name] = pd.DataFrame()

    fold_df = pd.DataFrame(fold_summary)

    print("\n" + "="*70)
    print("WALK-FORWARD RESULTS SUMMARY")
    print("="*70)
    for name in algo_names + ["Fixed"]:
        col = f"{name.lower()}_ir"
        if col in fold_df.columns:
            irs  = fold_df[col].dropna()
            wins = int((fold_df.get(f"{name.lower()}_ir", pd.Series())
                        > fold_df.get("fixed_ir", pd.Series())).sum()) if name != "Fixed" else 0
            print(f"  {name:<12} — avg IR: {irs.mean():.3f}  "
                  f"(folds: {' '.join(f'{v:.3f}' for v in fold_df[col].tolist())})"
                  + (f"  beats Fixed: {wins}/5" if name != "Fixed" else ""))

    fold_df.to_csv(DATA_DIR / "dapo_comparison.csv", index=False)
    print(f"\nSaved -> {DATA_DIR}/dapo_comparison.csv")

    if not fold_df.empty and all_bt_concat:
        plot_comparison(all_bt_concat, fold_df, algo_names)

    print("\nDone. Download dapo_comparison.csv and dapo_comparison.png from the Output tab.")


if __name__ == "__main__":
    main()
