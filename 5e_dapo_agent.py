"""
5e_dapo_agent.py - Hybrid GRPO/DAPO with Multi-Feature Regime Detection
Rebuilds the DAPO portfolio tilt agent with two key improvements over the
prior version:

  1. Proper Market Regime Detection
       Uses a 3-state Gaussian HMM fitted on 5 market features:
         bench_vol, bench_ret, bench_momentum, vol_trend, cs_dispersion
       States: Risk-On (0), Transition (1), Risk-Off (2)
       Falls back to rule-based 3-state if hmmlearn is unavailable.
       Regime confidence (posterior probability) is also passed as a state
       feature so the agent can modulate its own uncertainty.

  2. Hybrid GRPO/DAPO Algorithm
       HybridDAPOAgent switches update rule per timestep based on regime:
         Risk-On  (0) → DAPO: clip-higher (ε↑=0.28), dynamic G (4–8), no KL
         Risk-Off (2) → GRPO: KL-anchored (β=0.01), G=4
         Transition(1) → GRPO: KL-anchored (β=0.02), G=4  [most conservative]
       Single shared actor + reference network (no separate per-regime networks).

State space (8 dims):
  signal_strength   - mean |z-score| of cross-sectional ML scores
  signal_dispersion - std of z-scores (cross-sectional spread)
  bench_vol         - rolling 3m annualised SPX volatility
  recent_active_ret - rolling 3m mean active return at ref alpha
  rolling_te        - rolling 3m tracking error
  bench_momentum    - 3m cumulative SPX return (trend feature)
  regime_id_norm    - regime / 2.0  (0.0 risk-on, 0.5 transition, 1.0 risk-off)
  regime_conf       - HMM posterior probability of predicted state

Walk-forward: 5 expanding folds (same dates as 5c / 5d).
Outputs:
  data/dapo_comparison.csv    - per-fold metrics (HybridDAPO, PureGRPO, PureDAPO, Fixed)
  figures/dapo_comparison.png - cumulative active return + per-fold IR bars
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
import copy
from scipy.stats import pearsonr

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not installed - rule-based fallback only.")

# Paths
DATA_DIR = Path("data")
FIG_DIR  = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

PANEL_FILE   = DATA_DIR / "panel_monthly_enriched.parquet"
FACTORS_FILE = DATA_DIR / "factor_selected.csv"
WEIGHTS_FILE = DATA_DIR / "spx_weights.parquet"

# Walk-forward folds
FOLDS = [
    ("2013-12-31", "2014-01-01", "2015-12-31", "2014–2015"),
    ("2015-12-31", "2016-01-01", "2017-12-31", "2016–2017"),
    ("2017-12-31", "2018-01-01", "2019-12-31", "2018–2019"),
    ("2019-12-31", "2020-01-01", "2021-12-31", "2020–2021"),
    ("2021-12-31", "2022-01-01", "2025-12-31", "2022–2025"),
]
TRAIN_START_GLOBAL = "2010-01-01"

# Hyperparameters
ALPHA_MIN   = 0.002
ALPHA_MAX   = 0.050
FIXED_ALPHA = 0.010
HIDDEN_DIM  = 64
SEED        = 42

GRPO_EPOCHS  = 400
GRPO_LR      = 3e-4
GRPO_G       = 4
GRPO_KL_BETA_STABLE     = 0.01   # Risk-Off: conservative KL
GRPO_KL_BETA_TRANSITION = 0.02   # Transition: tighter KL

DAPO_EPOCHS     = 400            # used for all agents
DAPO_LR         = 3e-4
DAPO_EPS_LOW    = 0.20           # symmetric clip bound for negative advantages
DAPO_EPS_HIGH   = 0.28           # higher clip bound for positive advantages
DAPO_G_INIT     = 4              # initial candidates (matches GRPO_G for fair comparison)
DAPO_G_MAX      = 8              # extend to this if state is hard (high reward variance)
DAPO_VAR_THRESH = 0.05           # reward variance threshold for dynamic sampling

# Regime IDs
REGIME_RISK_ON     = 0
REGIME_TRANSITION  = 1
REGIME_RISK_OFF    = 2

# State column names (excluding regime cols - those are added by build_episodes)
BASE_STATE_COLS = [
    "signal_strength",
    "signal_dispersion",
    "bench_vol",
    "recent_active_ret",
    "rolling_te",
    "bench_momentum",
]
# Full state (after regime features appended)
STATE_COLS = BASE_STATE_COLS + ["regime_id_norm", "regime_conf"]
STATE_DIM  = len(STATE_COLS)

np.random.seed(SEED)
if HAS_TORCH:
    torch.manual_seed(SEED)


# FACTOR COMBO SCORES

def compute_fold_weights(panel_train, factor_names, signs):
    """IC-weighted factor combination. Weights fitted on training data only."""
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
        w = {f: 1.0 / len(factor_names) for f in factor_names}
    else:
        w = {f: v / total for f, v in ics.items()}
    return np.array([w[f] * signs.get(f, 1.0) for f in factor_names], dtype=np.float64)


def build_scores(panel_slice, factor_names, signed_w):
    """Cross-sectional z-score composite scores."""
    available = [f for f in factor_names if f in panel_slice.columns]
    idx       = [factor_names.index(f) for f in available]
    sw        = signed_w[idx]

    dates   = panel_slice["date"].values
    tickers = panel_slice["ticker"].values
    fwds    = panel_slice["fwd_ret_1m"].values.astype(np.float64)
    X_all   = panel_slice[available].values.astype(np.float64)

    codes, uniq = pd.factorize(dates, sort=True)
    scores = np.empty(len(dates), dtype=np.float64)

    for c in range(len(uniq)):
        mask = codes == c
        X    = X_all[mask]
        mu   = np.nanmean(X, axis=0)
        sg   = np.nanstd(X,  axis=0) + 1e-8
        Xz   = np.where(np.isnan(X), 0.0, (X - mu) / sg)
        scores[mask] = Xz @ sw

    return pd.DataFrame({
        "date": dates, "ticker": tickers,
        "score": scores, "fwd_ret_1m": fwds,
    })


# PORTFOLIO SIMULATION

def _prep_month(scores_month, weights_month, top_n=100, bottom_n=100):
    """
    Pre-compute per-month numpy arrays. Called once per month during episode
    construction. The result dict is stored in the episode table and reused
    at every training step (avoids pandas overhead during RL training).
    """
    df = scores_month.merge(
        weights_month[["ticker", "spx_weight"]], on="ticker", how="inner"
    ).dropna(subset=["score", "spx_weight", "fwd_ret_1m"])
    if len(df) < 50:
        return None
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    n  = len(df)
    return {
        "spx_w":   df["spx_weight"].values.astype(np.float64),
        "fwd_ret": df["fwd_ret_1m"].values.astype(np.float64),
        "n":       n,
        "top_n":   min(top_n,    n // 2),
        "bot_n":   min(bottom_n, n // 2),
    }


def simulate_month_fast(prep, alpha):
    """Pure numpy portfolio simulation using precomputed arrays."""
    if prep is None:
        return None
    spx_w   = prep["spx_w"]
    fwd_ret = prep["fwd_ret"]
    n       = prep["n"]
    tilt    = np.zeros(n, dtype=np.float64)
    tilt[:prep["top_n"]]     = +alpha
    tilt[n - prep["bot_n"]:] = -alpha
    raw_w = np.clip(spx_w + tilt, 0.0, None)
    total = raw_w.sum()
    if total < 1e-8:
        return None
    port_w    = raw_w / total
    port_ret  = float(port_w  @ fwd_ret)
    bench_ret = float(spx_w   @ fwd_ret)
    return {"port_ret": port_ret, "bench_ret": bench_ret,
            "active_ret": port_ret - bench_ret}


def simulate_month(scores_month, weights_month, alpha, top_n=100, bottom_n=100):
    prep = _prep_month(scores_month, weights_month, top_n, bottom_n)
    return simulate_month_fast(prep, alpha)


# MARKET REGIME DETECTION

class MarketRegimeDetector:
    """
    3-state regime detector based on a composite market stress score.

    Composite stress score = weighted sum of percentile ranks (fitted on training):
      +0.4 × prank(bench_vol)       - high vol      → stress
      −0.3 × prank(bench_momentum)  - neg momentum  → stress
      +0.2 × prank(vol_trend)       - rising vol    → stress
      +0.1 × prank(cs_dispersion)   - wide cs disp  → stress

    Training: ranks computed within the training set → p33/p67 thresholds stored.
    Prediction: each test month ranked against the TRAINING distribution via
    searchsorted - avoids the sigmoid-clustering bug where everything scored
    near the mean fell into Transition.

    Regimes:
      stress < p33  → Risk-On   (0)
      p33 ≤ stress < p67 → Transition (1)
      stress ≥ p67  → Risk-Off  (2)
    """

    # Feature column indices in X_raw
    _FEAT_COLS = ["bench_vol", "bench_ret", "bench_momentum", "vol_trend", "cs_dispersion"]
    _WEIGHTS   = [0.4, 0.0, -0.3, 0.2, 0.1]   # index-matched to _FEAT_COLS

    def __init__(self):
        self._sorted_train = None   # (n_train, 5) sorted training features per column
        self._thresholds   = (0.33, 0.67)
        self._n_train      = 0

    def _build_features(self, df):
        """5-column raw feature matrix, NaN/inf → 0."""
        X = np.column_stack([df[c].fillna(0.0).values
                             for c in self._FEAT_COLS]).astype(np.float64)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _stress_to_ids(stress, p33, p67):
        ids = np.full(len(stress), REGIME_TRANSITION, dtype=int)
        ids[stress <  p33] = REGIME_RISK_ON
        ids[stress >= p67] = REGIME_RISK_OFF
        return ids

    def _stress_score(self, X_raw, sorted_train=None):
        """
        Compute composite stress score for X_raw rows.
        If sorted_train provided: percentile-rank each column against that
        reference distribution (correct OOS behaviour).
        If not: rank within X_raw itself (used during fit).
        """
        n = len(X_raw)
        stress = np.zeros(n, dtype=np.float64)
        for j, w in enumerate(self._WEIGHTS):
            if w == 0.0:
                continue
            col = X_raw[:, j]
            if sorted_train is not None:
                ref = sorted_train[:, j]
                n_ref = len(ref)
                # count how many training values lie strictly below each test value
                counts = np.searchsorted(ref, col, side="left")
                pranks = (counts + 0.5) / n_ref
            else:
                order = np.argsort(col)
                pranks = np.empty(n, dtype=np.float64)
                pranks[order] = (np.arange(n) + 0.5) / n
            stress += w * pranks
        return stress

    def fit(self, df):
        X_raw = self._build_features(df)
        n = len(X_raw)
        # Sort each column for fast searchsorted in predict()
        self._sorted_train = np.sort(X_raw, axis=0)
        self._n_train = n

        stress = self._stress_score(X_raw, sorted_train=None)
        p33 = float(np.percentile(stress, 33))
        p67 = float(np.percentile(stress, 67))
        self._thresholds = (p33, p67)

        ids = self._stress_to_ids(stress, p33, p67)
        counts = [int((ids == r).sum()) for r in range(3)]
        print(f"    [Regime] Detector fitted ({n} months): "
              f"Risk-On={counts[0]}  Transition={counts[1]}  Risk-Off={counts[2]}")

    def predict(self, df):
        """
        Returns (regime_ids, regime_confs).
        Test months are ranked against the training distribution (searchsorted),
        not against each other - so regimes are consistently calibrated OOS.
        """
        X_raw  = self._build_features(df)
        p33, p67 = self._thresholds
        stress = self._stress_score(X_raw, sorted_train=self._sorted_train)
        ids    = self._stress_to_ids(stress, p33, p67)
        mid    = (p33 + p67) / 2.0
        span   = max(p67 - p33, 1e-6)
        confs  = np.clip(np.abs(stress - mid) / span, 0.0, 1.0)
        return ids, confs


def _attach_regime(df, detector):
    """
    Add regime_id / regime_id_norm / regime_conf columns to an episode df
    in-place using an already-fitted MarketRegimeDetector. Much faster than
    calling build_episodes a second time on the same data.
    """
    if detector is not None and not df.empty:
        ids, confs = detector.predict(df)
    else:
        ids   = np.zeros(len(df), dtype=int)
        confs = np.ones(len(df), dtype=float)
    df["regime_id"]      = ids
    df["regime_id_norm"] = np.clip(ids / 2.0, 0.0, 1.0)
    df["regime_conf"]    = np.clip(confs,      0.0, 1.0)


# EPISODE TABLE

def build_episodes(scores_df, weights_df, scores_panel, ref_alpha=0.01,
                   norm_params=None, fit_norm=False, regime_detector=None):
    """
    Build a per-month episode table with state features and precomputed prep dicts.

    scores_df    - output of build_scores (date, ticker, score, fwd_ret_1m)
    weights_df   - SPX weights parquet (date, ticker, spx_weight)
    scores_panel - raw panel (needed for cs_dispersion from stock returns)
    norm_params  - dict of (mean, std) per BASE_STATE_COLS col; fitted on train
    fit_norm     - if True, fit norm_params from this slice (training fold)
    regime_detector - MarketRegimeDetector instance (fitted externally on train)
    """
    scores_df  = scores_df.copy()
    weights_df = weights_df.copy()
    scores_df["date"]  = pd.to_datetime(scores_df["date"])
    weights_df["date"] = pd.to_datetime(weights_df["date"])

    # year-month key for weight lookup (avoids date boundary issues)
    weights_df["_ym"] = weights_df["date"].dt.year * 100 + weights_df["date"].dt.month
    weights_by_ym     = {ym: grp for ym, grp in weights_df.groupby("_ym")}

    # Cross-sectional dispersion of stock returns from raw panel
    scores_panel = scores_panel.copy()
    scores_panel["date"] = pd.to_datetime(scores_panel["date"])
    cs_disp_map = {}
    if "fwd_ret_1m" in scores_panel.columns:
        for dt, grp in scores_panel.groupby("date"):
            ym = dt.year * 100 + dt.month
            vals = grp["fwd_ret_1m"].dropna().values
            cs_disp_map[ym] = float(vals.std()) if len(vals) > 10 else 0.0

    rows = []
    for dt, sc_m in scores_df.groupby("date"):
        ym  = dt.year * 100 + dt.month
        w_m = weights_by_ym.get(ym, pd.DataFrame())
        if w_m.empty or len(sc_m) < 50:
            continue
        w_m = w_m[["ticker", "spx_weight"]].copy()

        s_vals = sc_m["score"].values
        z      = (s_vals - s_vals.mean()) / (s_vals.std() + 1e-8)
        ref    = simulate_month(sc_m, w_m, alpha=ref_alpha)
        if ref is None:
            continue
        prep = _prep_month(sc_m, w_m)

        rows.append({
            "date":              dt,
            "signal_strength":   float(np.abs(z).mean()),
            "signal_dispersion": float(z.std()),
            "bench_ret":         float(ref["bench_ret"]),
            "active_ret_ref":    float(ref["active_ret"]),
            "cs_dispersion":     cs_disp_map.get(ym, 0.0),
            "_prep":             prep,
        })

    if not rows:
        return pd.DataFrame(), norm_params

    df = pd.DataFrame(rows).set_index("date").sort_index()

    # Rolling market features
    df["bench_vol"]      = (df["bench_ret"]
                            .rolling(3, min_periods=2).std()
                            .fillna(df["bench_ret"].expanding().std())
                            .fillna(0.0)) * np.sqrt(12)
    # 3-month compounded return - use log-sum for speed (no rolling().apply())
    log1r = np.log1p(df["bench_ret"].fillna(0.0))
    df["bench_momentum"] = np.expm1(log1r.rolling(3, min_periods=1).sum())
    df["vol_trend"]      = df["bench_vol"].diff(2).fillna(0.0)
    df["rolling_te"]     = (df["active_ret_ref"]
                            .rolling(3, min_periods=2).std()
                            .fillna(df["active_ret_ref"].expanding().std())
                            .fillna(0.0)) * np.sqrt(12)
    df["recent_active_ret"] = df["active_ret_ref"].rolling(3, min_periods=1).mean().fillna(0.0)

    # Regime detection
    if regime_detector is not None:
        regime_ids, regime_confs = regime_detector.predict(df)
    else:
        # Minimal fallback: everything is Risk-On
        regime_ids  = np.zeros(len(df), dtype=int)
        regime_confs = np.ones(len(df), dtype=float)

    df["regime_id"]      = regime_ids
    df["regime_id_norm"] = regime_ids / 2.0          # 0.0 / 0.5 / 1.0
    df["regime_conf"]    = regime_confs

    # Normalise BASE_STATE_COLS
    if fit_norm:
        norm_params = {}
        for col in BASE_STATE_COLS:
            mu, sg = float(df[col].mean()), float(df[col].std() + 1e-8)
            norm_params[col] = (mu, sg)

    if norm_params:
        for col in BASE_STATE_COLS:
            if col in df.columns:
                mu, sg = norm_params[col]
                df[col] = ((df[col] - mu) / sg).fillna(0.0).clip(-5.0, 5.0)

    # regime_id_norm and regime_conf are NOT z-scored (already on [0,1])
    for col in ["regime_id_norm", "regime_conf"]:
        df[col] = df[col].fillna(0.0).clip(0.0, 1.0)

    return df, norm_params


# NEURAL NETWORK COMPONENTS  (only defined when PyTorch is available)

LOG_STD_MIN = -10
LOG_STD_MAX =   2


def _make_mlp(in_dim, out_dim, hidden=HIDDEN_DIM):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class GaussianActor(nn.Module):
    """Squashed-Gaussian stochastic policy over α ∈ [ALPHA_MIN, ALPHA_MAX]."""

    def __init__(self, state_dim, hidden=HIDDEN_DIM):
        super().__init__()
        self.mean_net    = _make_mlp(state_dim, 1, hidden)
        self.log_std_net = _make_mlp(state_dim, 1, hidden)

    def forward(self, state):
        mean    = self.mean_net(state)
        log_std = self.log_std_net(state).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std.exp()

    def sample(self, state):
        """Returns (alpha, log_prob, mean_alpha)."""
        mean, std = self(state)
        dist  = torch.distributions.Normal(mean, std)
        x_t   = dist.rsample()
        y_t   = torch.tanh(x_t)
        alpha = ALPHA_MIN + (y_t + 1.0) / 2.0 * (ALPHA_MAX - ALPHA_MIN)
        log_p = dist.log_prob(x_t) - torch.log(1.0 - y_t.pow(2) + 1e-6)
        return alpha, log_p.sum(-1, keepdim=True), mean

    def deterministic_action(self, state):
        with torch.no_grad():
            mean, _ = self(state)
            y = torch.tanh(mean)
            return float(ALPHA_MIN + (y + 1.0) / 2.0 * (ALPHA_MAX - ALPHA_MIN))


def _to_state_tensor(row):
    """Convert an episode row to a float32 numpy array, clipped to [-5, 5]."""
    vals = []
    for col in STATE_COLS:
        v = float(row[col]) if col in row.index else 0.0
        vals.append(v)
    arr = np.array(vals, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=5.0, neginf=-5.0).clip(-5.0, 5.0)


def _row_to_torch(row):
    return torch.FloatTensor(_to_state_tensor(row)).unsqueeze(0)


def compute_reward(active_ret):
    """Annualised active return as the RL reward signal."""
    return float(active_ret) * 12.0


# KL divergence helper (Gaussian)

def _gaussian_kl(actor, ref_actor, s):
    """KL( current_policy || ref_policy ) for a Gaussian actor."""
    mu1, s1 = actor(s)
    with torch.no_grad():
        mu2, s2 = ref_actor(s)
    kl = (torch.log(s2 / (s1 + 1e-8))
          + (s1.pow(2) + (mu1 - mu2).pow(2)) / (2 * s2.pow(2) + 1e-8)
          - 0.5)
    return kl.mean()


# SHARED SAMPLING UTILITY

def _sample_group(actor, s, row, n_samples):
    """
    Sample n_samples candidate alphas from the current policy and simulate each.
    Returns (alphas: list[float], log_probs: list[Tensor], rewards: list[float]).
    """
    alphas, log_probs, rewards = [], [], []
    for _ in range(n_samples):
        alpha_t, log_p, _ = actor.sample(s)
        a   = float(alpha_t.squeeze())
        res = simulate_month_fast(row["_prep"], a)
        if res is None:
            continue
        alphas.append(a)
        log_probs.append(log_p)
        rewards.append(compute_reward(res["active_ret"]))
    return alphas, log_probs, rewards


def _group_advantage(rewards):
    """Group-relative advantage: (r - mean) / std."""
    r_arr = np.array(rewards, dtype=np.float32)
    return torch.FloatTensor((r_arr - r_arr.mean()) / (r_arr.std() + 1e-8))


# DAPO OBJECTIVE (clip-higher, no KL)

def _dapo_loss(actor, s, alphas, log_probs_old_list, rewards):
    """
    Clip-higher PPO objective.
    Positive advantages: clipped at 1 + DAPO_EPS_HIGH (more permissive)
    Negative advantages: clipped at 1 - DAPO_EPS_LOW
    """
    adv = _group_advantage(rewards)

    log_probs_old = torch.cat(log_probs_old_list, dim=0).squeeze(-1).detach()
    alphas_t      = torch.FloatTensor(alphas).unsqueeze(-1)

    # Re-derive log probs under current policy parameters
    mean_c, std_c = actor(s.expand(len(alphas), -1))
    y_t = ((alphas_t - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN) * 2.0 - 1.0
           ).clamp(-0.9999, 0.9999)
    x_t = torch.atanh(y_t)
    dist = torch.distributions.Normal(mean_c, std_c)
    log_probs_new = (dist.log_prob(x_t) - torch.log(1.0 - y_t.pow(2) + 1e-6)).squeeze(-1)

    ratio     = (log_probs_new - log_probs_old).exp()
    pos_mask  = adv > 0
    clip_high = torch.where(pos_mask,
                            torch.full_like(ratio, 1.0 + DAPO_EPS_HIGH),
                            torch.full_like(ratio, 1.0 + DAPO_EPS_LOW))
    clip_low  = torch.full_like(ratio, 1.0 - DAPO_EPS_LOW)
    ratio_clp = torch.max(torch.min(ratio, clip_high), clip_low)
    return -torch.min(ratio * adv, ratio_clp * adv).mean()


# GRPO OBJECTIVE (REINFORCE + KL)

def _grpo_loss(actor, ref_actor, s, log_probs_list, rewards, kl_beta):
    """REINFORCE + KL penalty (conservative update rule)."""
    adv = _group_advantage(rewards)
    lp  = torch.cat(log_probs_list, dim=0).squeeze(-1)
    pg  = -(lp * adv).mean()
    kl  = _gaussian_kl(actor, ref_actor, s)
    return pg + kl_beta * kl


# HYBRID DAPO AGENT (main contribution)

class HybridDAPOAgent:
    """
    Regime-adaptive hybrid agent.

    Update rule per timestep:
      REGIME_RISK_ON    → DAPO (clip-higher, dynamic G, no KL)   - explore freely
      REGIME_RISK_OFF   → GRPO (KL β=0.01)                       - stay conservative
      REGIME_TRANSITION → GRPO (KL β=0.02)                       - maximum caution
    """
    name = "HybridDAPO"

    def __init__(self, state_dim):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for HybridDAPOAgent")
        self.actor     = GaussianActor(state_dim)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=DAPO_LR)
        self.ref_actor = copy.deepcopy(self.actor)
        for p in self.ref_actor.parameters():
            p.requires_grad_(False)

    def select_action(self, state_arr, deterministic=True):
        s = torch.FloatTensor(state_arr).unsqueeze(0)
        if deterministic:
            return self.actor.deterministic_action(s)
        with torch.no_grad():
            a, _, _ = self.actor.sample(s)
        return float(a.squeeze())

    def train(self, train_rows, verbose=False):
        """train_rows: list of (dt, row) from episode table."""
        avg_r = 0.0
        for epoch in range(DAPO_EPOCHS):
            ep_rewards = []

            for dt, row in train_rows:
                state   = _to_state_tensor(row)
                s       = torch.FloatTensor(state).unsqueeze(0)
                regime  = int(round(float(row.get("regime_id", 0))))

                if regime == REGIME_RISK_ON:
                    # DAPO: clip-higher, dynamic G
                    alphas, lps, rewards = _sample_group(self.actor, s, row, DAPO_G_INIT)
                    if len(rewards) >= 2 and len(rewards) < DAPO_G_MAX:
                        if float(np.var(rewards)) > DAPO_VAR_THRESH:
                            a2, lp2, r2 = _sample_group(
                                self.actor, s, row, DAPO_G_MAX - len(rewards))
                            alphas += a2; lps += lp2; rewards += r2
                    if len(rewards) < 2:
                        continue
                    loss = _dapo_loss(self.actor, s, alphas, lps, rewards)

                else:
                    # GRPO: KL-anchored (tighter β for transition)
                    kl_beta = (GRPO_KL_BETA_TRANSITION if regime == REGIME_TRANSITION
                               else GRPO_KL_BETA_STABLE)
                    _, lps, rewards = _sample_group(self.actor, s, row, GRPO_G)
                    if len(rewards) < 2:
                        continue
                    loss = _grpo_loss(self.actor, self.ref_actor, s, lps, rewards, kl_beta)

                ep_rewards.extend(rewards)
                self.actor_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_opt.step()

            avg_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            if verbose and epoch % 100 == 0:
                print(f"    [{self.name}] Epoch {epoch:4d}  avg_r={avg_r:+.4f}")

        return avg_r


# PURE GRPO AGENT  (baseline - always KL-anchored)

class PureGRPOAgent:
    name = "PureGRPO"

    def __init__(self, state_dim):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required")
        self.actor     = GaussianActor(state_dim)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=GRPO_LR)
        self.ref_actor = copy.deepcopy(self.actor)
        for p in self.ref_actor.parameters():
            p.requires_grad_(False)

    def select_action(self, state_arr, deterministic=True):
        s = torch.FloatTensor(state_arr).unsqueeze(0)
        if deterministic:
            return self.actor.deterministic_action(s)
        with torch.no_grad():
            a, _, _ = self.actor.sample(s)
        return float(a.squeeze())

    def train(self, train_rows, verbose=False):
        avg_r = 0.0
        for epoch in range(GRPO_EPOCHS):
            ep_rewards = []
            for dt, row in train_rows:
                s = torch.FloatTensor(_to_state_tensor(row)).unsqueeze(0)
                _, lps, rewards = _sample_group(self.actor, s, row, GRPO_G)
                if len(rewards) < 2:
                    continue
                loss = _grpo_loss(self.actor, self.ref_actor, s, lps, rewards,
                                  GRPO_KL_BETA_STABLE)
                ep_rewards.extend(rewards)
                self.actor_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_opt.step()
            avg_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            if verbose and epoch % 100 == 0:
                print(f"    [{self.name}] Epoch {epoch:4d}  avg_r={avg_r:+.4f}")
        return avg_r


# PURE DAPO AGENT  (baseline - always clip-higher, no KL)

class PureDAPOAgent:
    name = "PureDAPO"

    def __init__(self, state_dim):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required")
        self.actor     = GaussianActor(state_dim)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=DAPO_LR)

    def select_action(self, state_arr, deterministic=True):
        s = torch.FloatTensor(state_arr).unsqueeze(0)
        if deterministic:
            return self.actor.deterministic_action(s)
        with torch.no_grad():
            a, _, _ = self.actor.sample(s)
        return float(a.squeeze())

    def train(self, train_rows, verbose=False):
        avg_r = 0.0
        for epoch in range(DAPO_EPOCHS):
            ep_rewards = []
            for dt, row in train_rows:
                s = torch.FloatTensor(_to_state_tensor(row)).unsqueeze(0)
                alphas, lps, rewards = _sample_group(self.actor, s, row, DAPO_G_INIT)
                if len(rewards) >= 2 and len(rewards) < DAPO_G_MAX:
                    if float(np.var(rewards)) > DAPO_VAR_THRESH:
                        a2, lp2, r2 = _sample_group(
                            self.actor, s, row, DAPO_G_MAX - len(rewards))
                        alphas += a2; lps += lp2; rewards += r2
                if len(rewards) < 2:
                    continue
                loss = _dapo_loss(self.actor, s, alphas, lps, rewards)
                ep_rewards.extend(rewards)
                self.actor_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_opt.step()
            avg_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            if verbose and epoch % 100 == 0:
                print(f"    [{self.name}] Epoch {epoch:4d}  avg_r={avg_r:+.4f}")
        return avg_r


# RULE-BASED FALLBACK (when PyTorch is unavailable)

class RuleBasedAgent:
    """Regime-adjusted fixed alpha: lower tilt in risk-off, higher in risk-on."""
    name = "RuleBased"

    def __init__(self, state_dim):
        pass

    def select_action(self, state_arr, deterministic=True):
        regime_norm = float(state_arr[6]) if len(state_arr) > 6 else 0.0
        # risk-on → higher alpha, risk-off → lower alpha
        return ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * (1.0 - regime_norm) * 0.6 + FIXED_ALPHA * 0.4

    def train(self, train_rows, verbose=False):
        return 0.0


# EVALUATION

def evaluate(agent, ep_test):
    """Run agent deterministically on test episodes. Returns (rl_bt, fixed_bt)."""
    rl_rows, fixed_rows = [], []
    for dt in ep_test.index:
        row   = ep_test.loc[dt]
        state = _to_state_tensor(row)

        alpha_rl = agent.select_action(state, deterministic=True)
        r_rl     = simulate_month_fast(row["_prep"], alpha_rl)
        if r_rl:
            rl_rows.append({"date": dt, "alpha_used": alpha_rl,
                            "regime_id": int(row.get("regime_id", 0)), **r_rl})

        r_fx = simulate_month_fast(row["_prep"], FIXED_ALPHA)
        if r_fx:
            fixed_rows.append({"date": dt, "alpha_used": FIXED_ALPHA,
                               "regime_id": int(row.get("regime_id", 0)), **r_fx})

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
    ann_alpha = float((1 + r).prod() ** (12 / n) - 1)
    track_err = float(r.std(ddof=1) * np.sqrt(12))
    ir        = ann_alpha / track_err if track_err > 1e-8 else float("nan")
    nav       = (1 + r).cumprod()
    return dict(
        ann_alpha    = ann_alpha,
        track_err    = track_err,
        info_ratio   = ir,
        hit_rate     = float((r > 0).mean()),
        max_active_dd= float((nav / nav.cummax() - 1).min()),
        n_months     = n,
    )


# PLOTTING

COLORS = {
    "HybridDAPO": "#FF6B9D",
    "PureGRPO":   "#66BB6A",
    "PureDAPO":   "#5B9BD5",
    "Fixed":      "#AAAAAA",
}


def plot_comparison(all_bt, fold_df, algo_names):
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    fig.suptitle(
        "Hybrid GRPO/DAPO vs Pure GRPO vs Pure DAPO - Walk-Forward Out-of-Sample\n"
        "HybridDAPO: DAPO on Risk-On, GRPO+KL on Risk-Off/Transition (3-state HMM regime)",
        fontsize=12, fontweight="bold",
    )

    # Panel 1: cumulative active return
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

    # Panel 2: per-fold IR bars
    ax   = axes[1]
    lbls = fold_df["label"].tolist()
    x    = np.arange(len(lbls))
    order = algo_names + ["Fixed"]
    n_a   = len(order)
    w     = 0.18
    for i, name in enumerate(order):
        key = f"{name.lower()}_ir"
        if key not in fold_df.columns:
            continue
        vals   = fold_df[key].tolist()
        offset = (i - (n_a - 1) / 2) * (w + 0.02)
        ax.bar(x + offset, vals, w, label=name,
               color=COLORS.get(name, "#999"), alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(lbls, fontsize=9)
    ax.set_ylabel("Information Ratio")
    ax.set_title("Per-Fold IR: HybridDAPO vs PureGRPO vs PureDAPO vs Fixed")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = FIG_DIR / "dapo_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


# MAIN

def main():
    print("=" * 72)
    print("5e_dapo_agent.py - Hybrid GRPO/DAPO + 3-State Market Regime Detection")
    print("  HybridDAPO : DAPO on Risk-On, GRPO+KL on Risk-Off, GRPO+2xKL on Transition")
    print("  Regime     : 3-state HMM on vol / ret / momentum / vol-trend / cross-sec-disp")
    print("=" * 72)

    if not HAS_TORCH:
        print("WARNING: PyTorch not available - using rule-based fallback agents.\n")

    # Load data
    print("\nLoading data ...")
    for p in [PANEL_FILE, FACTORS_FILE, WEIGHTS_FILE]:
        if not p.exists():
            raise FileNotFoundError(f"Required data file not found: {p}\n"
                                    "Run the earlier pipeline scripts first.")

    panel   = pd.read_parquet(PANEL_FILE)
    panel["date"] = pd.to_datetime(panel["date"])
    factors = pd.read_csv(FACTORS_FILE)
    weights = pd.read_parquet(WEIGHTS_FILE)
    weights["date"] = pd.to_datetime(weights["date"])

    factor_names = list(factors["factor"])
    signs        = (factors.set_index("factor")["majority_sign"]
                    .map({"+": 1.0, "-": -1.0}).fillna(1.0).to_dict())

    print(f"  Panel  : {panel['date'].min().date()} → {panel['date'].max().date()}")
    print(f"  Factors: {len(factor_names)}")

    algo_names = ["HybridDAPO", "PureGRPO", "PureDAPO"]
    all_bt     = {n: [] for n in algo_names + ["Fixed"]}
    fold_rows  = []

    for fold_idx, (train_end, test_start, test_end, label) in enumerate(FOLDS):
        print(f"\n{'=' * 72}")
        print(f"FOLD {fold_idx + 1}/5 - {label}  |  train ≤ {train_end}  |"
              f"  test {test_start} → {test_end}")
        print("=" * 72)

        panel_train = panel[(panel["date"] >= TRAIN_START_GLOBAL) &
                            (panel["date"] <= train_end)].copy()
        panel_test  = panel[(panel["date"] >= test_start) &
                            (panel["date"] <= test_end)].copy()

        if len(panel_train) < 500 or len(panel_test) < 50:
            print("  Skipping - insufficient data.")
            continue

        # Factor weights (train only)
        print("  Computing factor weights ...")
        signed_w = compute_fold_weights(panel_train, factor_names, signs)

        # Build composite scores
        print("  Building scores ...")
        sc_train = build_scores(panel_train, factor_names, signed_w)
        sc_test  = build_scores(panel_test,  factor_names, signed_w)

        # Slice only the columns needed for cs_dispersion to avoid copying 80-col panel
        panel_train_cs = panel_train[["date", "fwd_ret_1m"]].copy()
        panel_test_cs  = panel_test[["date", "fwd_ret_1m"]].copy()

        # Build episode table for training (regime added in-place after)
        print("  Building training episodes ...")
        ep_train, norm_p = build_episodes(
            sc_train, weights, panel_train_cs,
            fit_norm=True, regime_detector=None)

        # Fit regime detector on training episodes, then attach
        print("  Fitting regime detector ...")
        regime_det = MarketRegimeDetector()
        if not ep_train.empty:
            regime_det.fit(ep_train)
        _attach_regime(ep_train, regime_det)

        # Build test episodes (regime detector applied during build)
        print("  Building test episodes ...")
        ep_test, _ = build_episodes(
            sc_test, weights, panel_test_cs,
            norm_params=norm_p, fit_norm=False, regime_detector=regime_det)

        if ep_train.empty or ep_test.empty:
            print("  Skipping - empty episode table.")
            continue

        train_rows_list = [(dt, ep_train.loc[dt]) for dt in ep_train.index]
        print(f"  Train months: {len(train_rows_list)}  |  Test months: {len(ep_test)}")

        # Regime distribution in test set
        if "regime_id" in ep_test.columns:
            rc = ep_test["regime_id"].value_counts().sort_index()
            names_map = {0: "Risk-On", 1: "Transition", 2: "Risk-Off"}
            dist_str = "  ".join(f"{names_map.get(r, r)}={c}" for r, c in rc.items())
            print(f"  Test regime distribution: {dist_str}")

        # Train agents
        if HAS_TORCH:
            AgentClasses = [HybridDAPOAgent, PureGRPOAgent, PureDAPOAgent]
        else:
            AgentClasses = [RuleBasedAgent] * 3

        fold_bt   = {}
        fold_info = {"label": label, "n_months": len(ep_test)}
        fixed_bt  = None

        for name, AgentCls in zip(algo_names, AgentClasses):
            print(f"\n  [{name}] Training {DAPO_EPOCHS} epochs ...")
            agent    = AgentCls(STATE_DIM)
            avg_r    = agent.train(train_rows_list, verbose=True)
            bt, fbt  = evaluate(agent, ep_test)
            s        = ie_stats(bt)
            fold_bt[name] = bt
            if fixed_bt is None:
                fixed_bt = fbt
            print(f"  [{name}] IR={s.get('info_ratio', float('nan')):.3f}  "
                  f"α={s.get('ann_alpha', 0)*100:.2f}%  "
                  f"TE={s.get('track_err', 0)*100:.2f}%  "
                  f"Hit={s.get('hit_rate', 0)*100:.0f}%")
            fold_info[f"{name.lower()}_ir"]    = s.get("info_ratio",    float("nan"))
            fold_info[f"{name.lower()}_alpha"] = s.get("ann_alpha",     float("nan"))
            fold_info[f"{name.lower()}_te"]    = s.get("track_err",     float("nan"))
            fold_info[f"{name.lower()}_maxdd"] = s.get("max_active_dd", float("nan"))

        # Fixed baseline stats
        if fixed_bt is not None and not fixed_bt.empty:
            fs = ie_stats(fixed_bt)
            fold_bt["Fixed"] = fixed_bt
            fold_info["fixed_ir"]    = fs.get("info_ratio",    float("nan"))
            fold_info["fixed_alpha"] = fs.get("ann_alpha",     float("nan"))
            fold_info["fixed_te"]    = fs.get("track_err",     float("nan"))
            fold_info["fixed_maxdd"] = fs.get("max_active_dd", float("nan"))

        fold_rows.append(fold_info)

        for name in algo_names + ["Fixed"]:
            if name in fold_bt and not fold_bt[name].empty:
                all_bt[name].append(fold_bt[name])

    # Concatenate folds
    all_bt_cat = {}
    for name in algo_names + ["Fixed"]:
        parts = all_bt[name]
        if parts:
            all_bt_cat[name] = pd.concat(parts).sort_index()

    fold_df = pd.DataFrame(fold_rows)

    # Summary table
    print("\n" + "=" * 72)
    print("WALK-FORWARD RESULTS SUMMARY")
    print("=" * 72)
    if not fold_df.empty:
        for name in algo_names + ["Fixed"]:
            col = f"{name.lower()}_ir"
            if col not in fold_df.columns:
                continue
            irs  = fold_df[col].dropna()
            avg  = irs.mean()
            vals = "  ".join(f"{v:+.3f}" for v in fold_df[col].tolist())
            beat = ""
            if name != "Fixed" and "fixed_ir" in fold_df.columns:
                wins = int((fold_df[col] > fold_df["fixed_ir"]).sum())
                beat = f"  beats Fixed: {wins}/5"
            print(f"  {name:<12} avg IR: {avg:+.3f}  [{vals}]{beat}")

    # Save outputs
    if not fold_df.empty:
        out_csv = DATA_DIR / "dapo_comparison.csv"
        fold_df.to_csv(out_csv, index=False)
        print(f"\nSaved -> {out_csv}")

    if fold_df.shape[0] > 0 and all_bt_cat:
        plot_comparison(all_bt_cat, fold_df, algo_names)

    print("\nDone.")


if __name__ == "__main__":
    main()
