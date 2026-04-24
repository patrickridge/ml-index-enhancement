"""
5f_dynamic_portfolio_rl.py - Dynamic Portfolio RL with Asymmetric Tilt
=======================================================================
Conservative extension of 5e: the agent independently controls the
long and short tilt, but n_frac (bucket size) stays fixed at 0.20.

Motivation: the 1D agent (5e) must set the same alpha for both longs and
shorts. In practice, the optimal long and short tilts are not always equal:
  - Bull regime: strong long alpha (exploit the top stocks), smaller short
    alpha (hard to short in a rising market, higher underweight risk)
  - Bear regime: reduce long alpha (protect against wrong calls), larger
    short alpha (more conviction in avoiding the bottom stocks)

  Action dim 1 - alpha_long  ∈ [0.002, 0.05]
    Overweight tilt applied to the top 20% of stocks (by ML score).

  Action dim 2 - alpha_short ∈ [0.002, 0.05]
    Underweight tilt applied to the bottom 20% of stocks.
    Learned independently of alpha_long.

n_frac is intentionally kept fixed at 0.20 (top/bottom 100 stocks).
Once asymmetric tilts are validated, n_frac can be added as action dim 3.

Algorithm: GRPO + KL (best from 5e) - critic-free group advantage.
Regime: 2-state HMM on [bench_vol, bench_ret] per fold (same as 5e).
Walk-forward: same 5-fold expanding window.

Baseline comparisons:
  - Fixed symmetric (alpha_long = alpha_short = 0.01)  ← 5e fixed
  - 1D GRPO from 5e (alpha_long = alpha_short, learned)

Outputs:
  data/dynamic_portfolio_results.csv
  figures/dynamic_portfolio.png

Run:
  python 5f_dynamic_portfolio_rl.py
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
from scipy.stats import pearsonr
import copy

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not installed.")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR     = Path("data")
FIG_DIR      = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

PANEL_FILE   = DATA_DIR / "panel_monthly_enriched.parquet"
FACTORS_FILE = DATA_DIR / "factor_selected.csv"
WEIGHTS_FILE = DATA_DIR / "spx_weights.parquet"

# ── Walk-forward folds (same as 5c/5d/5e) ─────────────────────────────────────
FOLDS = [
    ("2013-12-31", "2014-01-01", "2015-12-31", "2014–2015"),
    ("2015-12-31", "2016-01-01", "2017-12-31", "2016–2017"),
    ("2017-12-31", "2018-01-01", "2019-12-31", "2018–2019"),
    ("2019-12-31", "2020-01-01", "2021-12-31", "2020–2021"),
    ("2021-12-31", "2022-01-01", "2025-12-31", "2022–2025"),
]
TRAIN_START_GLOBAL = "2010-01-01"

# ── Action space bounds ────────────────────────────────────────────────────────
ALPHA_L_MIN, ALPHA_L_MAX = 0.002, 0.050   # long tilt
ALPHA_S_MIN, ALPHA_S_MAX = 0.002, 0.050   # short tilt

ACTION_DIM   = 2      # (alpha_long, alpha_short) - n_frac fixed for now
FIXED_N_FRAC = 0.20   # top/bottom 20% of universe (unchanged from 5e)

# Fixed symmetric baseline (same as 5e fixed)
FIXED_ALPHA_L = 0.010
FIXED_ALPHA_S = 0.010

# ── GRPO hyperparameters ───────────────────────────────────────────────────────
EPOCHS   = 400
LR       = 3e-4
G        = 4        # candidates per state
KL_BETA  = 0.01
HIDDEN   = 64
SEED     = 42

# ── State features (same as 5e) ───────────────────────────────────────────────
STATE_COLS = [
    "signal_strength",
    "signal_dispersion",
    "bench_vol",
    "recent_active_ret",
    "regime",
    "rolling_te",
]

np.random.seed(SEED)
if HAS_TORCH:
    torch.manual_seed(SEED)


# =============================================================================
# FACTOR COMBO SCORES (same as 5e)
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
        tickers = grp["ticker"].tolist()
        fwds    = grp["fwd_ret_1m"].tolist() if "fwd_ret_1m" in grp.columns else [np.nan] * len(grp)
        for i in range(len(grp)):
            rows.append({"date": dt, "ticker": tickers[i],
                         "score": float(sc[i]), "fwd_ret_1m": fwds[i]})
    return pd.DataFrame(rows)


# =============================================================================
# PORTFOLIO SIMULATION - 3D action
# =============================================================================

def _prep_month(scores_month, weights_month):
    """Merge + sort once. Returns numpy arrays for fast simulation."""
    df = scores_month.merge(
        weights_month[["ticker", "spx_weight"]], on="ticker", how="inner"
    ).dropna(subset=["score", "spx_weight", "fwd_ret_1m"])
    if len(df) < 50:
        return None
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return {"spx_w": df["spx_weight"].values.astype(np.float64),
            "fwd_ret": df["fwd_ret_1m"].values.astype(np.float64),
            "n": len(df)}


def simulate_month(scores_month, weights_month, alpha_long, alpha_short, n_frac):
    return simulate_month_fast(_prep_month(scores_month, weights_month),
                               alpha_long, alpha_short, n_frac)


def simulate_month_fast(prep, alpha_long, alpha_short, n_frac):
    if prep is None:
        return None
    spx_w, fwd_ret, n = prep["spx_w"], prep["fwd_ret"], prep["n"]
    n_bkt = max(10, int(np.floor(n * n_frac)))
    tilt  = np.zeros(n, dtype=np.float64)
    tilt[:n_bkt]     = +alpha_long
    tilt[n - n_bkt:] = -alpha_short
    raw_w = np.clip(spx_w + tilt, 0.0, None)
    total = raw_w.sum()
    if total < 1e-8:
        return None
    port_w = raw_w / total
    return {"port_ret": float(port_w @ fwd_ret), "bench_ret": float(spx_w @ fwd_ret),
            "active_ret": float(port_w @ fwd_ret) - float(spx_w @ fwd_ret)}


# =============================================================================
# HMM REGIME DETECTION (same as 5e)
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

def build_episodes(scores, weights, ref_alpha_l=0.01, ref_alpha_s=0.01,
                   ref_n_frac=0.20, norm_params=None, fit_norm=False):
    scores  = scores.copy()
    weights = weights.copy()
    scores["date"]  = pd.to_datetime(scores["date"])
    weights["date"] = pd.to_datetime(weights["date"])

    # Build (year, month) → weight rows lookup so we match even when panel
    # dates differ from weight month-end dates by a day or two.
    weights["_ym"] = weights["date"].dt.year * 100 + weights["date"].dt.month
    weights_by_ym  = {ym: grp for ym, grp in weights.groupby("_ym")}

    rows = []
    for dt, sc_m in scores.groupby("date"):
        ym  = dt.year * 100 + dt.month
        w_m = weights_by_ym.get(ym, pd.DataFrame())[["ticker", "spx_weight"]].copy()
        if len(sc_m) < 50:
            continue
        s_vals = sc_m["score"].values
        z      = (s_vals - s_vals.mean()) / (s_vals.std() + 1e-8)
        ref    = simulate_month(sc_m, w_m,
                                alpha_long=ref_alpha_l,
                                alpha_short=ref_alpha_s,
                                n_frac=ref_n_frac)
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
        else:
            vol_med = df["bench_vol"].expanding(min_periods=6).median()
            df["regime"] = (df["bench_vol"] > vol_med).astype(float).fillna(0.0)
            norm_params["_vol_median"] = float(vol_med.iloc[-1])
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
# NEURAL NETWORK - 3D action output
# =============================================================================

if HAS_TORCH:
    LOG_STD_MIN, LOG_STD_MAX = -10, 2

    class DynamicActor(nn.Module):
        """
        Squashed Gaussian policy over 2D action space:
          dim 0: alpha_long  ∈ [ALPHA_L_MIN, ALPHA_L_MAX]
          dim 1: alpha_short ∈ [ALPHA_S_MIN, ALPHA_S_MAX]

        n_frac is fixed at FIXED_N_FRAC - not learned yet.
        Same architecture as 5e GaussianActor, just with 2D output.
        """
        def __init__(self, state_dim, hidden=HIDDEN):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(state_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),    nn.ReLU(),
            )
            self.mean_head    = nn.Linear(hidden, ACTION_DIM)
            self.log_std_head = nn.Linear(hidden, ACTION_DIM)

            # action bounds as buffers
            lo = torch.tensor([ALPHA_L_MIN, ALPHA_S_MIN], dtype=torch.float32)
            hi = torch.tensor([ALPHA_L_MAX, ALPHA_S_MAX], dtype=torch.float32)
            self.register_buffer("act_lo", lo)
            self.register_buffer("act_hi", hi)

        def forward(self, state):
            h       = self.trunk(state)
            mean    = self.mean_head(h)
            log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
            return mean, log_std.exp()

        def sample(self, state):
            mean, std = self(state)
            dist  = torch.distributions.Normal(mean, std)
            x_t   = dist.rsample()                          # (batch, 3)
            y_t   = torch.tanh(x_t)                        # squash to (−1, 1)
            # scale each dim independently to its [lo, hi] range
            action = self.act_lo + (y_t + 1.0) / 2.0 * (self.act_hi - self.act_lo)
            log_p  = (dist.log_prob(x_t)
                      - torch.log(1.0 - y_t.pow(2) + 1e-6)).sum(-1, keepdim=True)
            return action, log_p, mean

        def select_action(self, state_np, deterministic=False):
            """Returns (alpha_long, alpha_short) as numpy floats. n_frac is fixed."""
            s = torch.FloatTensor(state_np).unsqueeze(0)
            with torch.no_grad():
                mean, std = self(s)
                if deterministic:
                    y_t = torch.tanh(mean)
                else:
                    dist = torch.distributions.Normal(mean, std)
                    y_t  = torch.tanh(dist.rsample())
                action = self.act_lo + (y_t + 1.0) / 2.0 * (self.act_hi - self.act_lo)
            a = action.squeeze(0).cpu().numpy()
            return float(a[0]), float(a[1])


# =============================================================================
# GRPO TRAINING - 3D action
# =============================================================================

def _safe_state(row):
    return np.clip(
        np.array([row[c] for c in STATE_COLS], dtype=np.float32),
        -5.0, 5.0
    )


def train_grpo(actor, train_rows, ref_actor, verbose=True):
    opt = optim.Adam(actor.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        epoch_rewards = []
        for dt, row in train_rows:
            state  = torch.FloatTensor(_safe_state(row)).unsqueeze(0)
            # ── Sample G candidate actions ────────────────────────────────────
            rewards, log_probs = [], []
            for _ in range(G):
                action, log_p, _ = actor.sample(state)
                a = action.squeeze(0).cpu().numpy()
                r = simulate_month(row["_scores"], row["_weights"],
                                   alpha_long=float(a[0]),
                                   alpha_short=float(a[1]),
                                   n_frac=FIXED_N_FRAC)
                rewards.append(float(r["active_ret"]) * 12.0 if r else 0.0)
                log_probs.append(log_p)

            r_arr = np.array(rewards)
            r_std = r_arr.std() + 1e-8
            advs  = (r_arr - r_arr.mean()) / r_std

            # ── Policy gradient + KL penalty ──────────────────────────────────
            loss = torch.tensor(0.0)
            for log_p, adv in zip(log_probs, advs):
                loss = loss - log_p * adv

            # KL vs reference policy
            with torch.no_grad():
                ref_mean, ref_std = ref_actor(state)
            cur_mean, cur_std = actor(state)
            kl = (torch.distributions.Normal(ref_mean, ref_std)
                  .log_prob(cur_mean)
                  - torch.distributions.Normal(cur_mean, cur_std)
                  .log_prob(cur_mean)).sum()
            loss = loss / G + KL_BETA * kl

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
            epoch_rewards.append(float(r_arr.mean()))

        if verbose and (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch+1:4d}  avg_reward={np.mean(epoch_rewards):+.4f}")


# =============================================================================
# EVALUATE
# =============================================================================

def evaluate(actor, ep_test):
    rl_rows, fixed_rows = [], []
    for dt in ep_test.index:
        row = ep_test.loc[dt]
        state = _safe_state(row)

        # Dynamic RL agent
        if HAS_TORCH:
            al, as_ = actor.select_action(state, deterministic=True)
        else:
            al, as_ = FIXED_ALPHA_L, FIXED_ALPHA_S
        r_rl = simulate_month(row["_scores"], row["_weights"],
                              alpha_long=al, alpha_short=as_, n_frac=FIXED_N_FRAC)
        if r_rl:
            rl_rows.append({"date": dt, "alpha_long": al,
                            "alpha_short": as_, **r_rl})

        # Fixed symmetric baseline
        r_fx = simulate_month(row["_scores"], row["_weights"],
                              alpha_long=FIXED_ALPHA_L,
                              alpha_short=FIXED_ALPHA_S,
                              n_frac=FIXED_N_FRAC)
        if r_fx:
            fixed_rows.append({"date": dt, **r_fx})

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


# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(all_bt, fold_summary):
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    fig.suptitle(
        "Dynamic Portfolio RL - 3D Action Space (α_long, α_short, n_frac)\n"
        "GRPO with HMM regime detection · Walk-Forward 5-Fold",
        fontsize=13, fontweight="bold")

    colors = {"Dynamic RL": "#5B9BD5", "Fixed": "#AAAAAA"}

    ax = axes[0]
    for name, bt in all_bt.items():
        if bt.empty or "active_ret" not in bt.columns:
            continue
        nav = (1 + bt["active_ret"]).cumprod() - 1
        ax.plot(nav.index, nav * 100, label=name,
                color=colors.get(name, "#FF6B9D"),
                lw=2, linestyle="--" if name == "Fixed" else "-")
    ax.axhline(0, color="black", lw=0.8, linestyle=":")
    ax.set_ylabel("Cumulative Active Return (%)")
    ax.set_title("Cumulative Active Return - Dynamic RL vs Fixed")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax   = axes[1]
    lbls = fold_summary["label"].tolist()
    x    = np.arange(len(lbls))
    w    = 0.35
    for i, (name, col) in enumerate(
            [("Dynamic RL", "rl_ir"), ("Fixed", "fixed_ir")]):
        if col not in fold_summary.columns:
            continue
        vals = fold_summary[col].tolist()
        ax.bar(x + (i - 0.5) * (w + 0.04), vals, w,
               label=name, color=colors[name], alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(lbls, fontsize=9)
    ax.set_ylabel("Information Ratio")
    ax.set_title("Per-Fold IR - Dynamic RL vs Fixed Symmetric Alpha")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = FIG_DIR / "dynamic_portfolio.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("5f_dynamic_portfolio_rl.py - 3D Action Space Portfolio RL")
    print("Actions: alpha_long ∈ [0.002, 0.05]  |  alpha_short ∈ [0.002, 0.05]"
          "  |  n_frac ∈ [0.05, 0.30]")
    print("Algorithm: GRPO + KL  |  HMM regime detection  |  5-fold walk-forward")
    print("=" * 70)

    if not HAS_TORCH:
        print("ERROR: PyTorch required for this script.")
        return

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

    state_dim    = len(STATE_COLS)
    all_bt       = {"Dynamic RL": [], "Fixed": []}
    fold_summary = []

    for fold_idx, (train_end, test_start, test_end, label) in enumerate(FOLDS):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx+1}/5 - {label}  |  train ≤ {train_end}"
              f"  |  test {test_start} → {test_end}")
        print("="*70)

        panel_train = panel[(panel["date"] >= TRAIN_START_GLOBAL) &
                            (panel["date"] <= train_end)].copy()
        panel_test  = panel[(panel["date"] >= test_start) &
                            (panel["date"] <= test_end)].copy()

        if len(panel_train) < 500 or len(panel_test) < 50:
            print("  Skipping - insufficient data.")
            continue

        print("  Computing factor weights (training data only) ...")
        signed_w = compute_fold_weights(panel_train, factor_names, signs)

        print("  Building scores ...")
        sc_train = build_scores(panel_train, factor_names, signed_w)
        sc_test  = build_scores(panel_test,  factor_names, signed_w)

        ep_train, norm_p = build_episodes(sc_train, weights, fit_norm=True)
        ep_test,  _      = build_episodes(sc_test,  weights, norm_params=norm_p)

        if ep_train.empty or ep_test.empty:
            print("  Skipping - empty episodes.")
            continue

        train_rows = [(dt, ep_train.loc[dt]) for dt in ep_train.index]
        print(f"  Train: {len(train_rows)} months  |  Test: {len(ep_test)} months")

        # ── Train ─────────────────────────────────────────────────────────────
        actor     = DynamicActor(state_dim)
        ref_actor = copy.deepcopy(actor)
        ref_actor.eval()

        print(f"\n  [Dynamic RL] Training {EPOCHS} epochs ...")
        train_grpo(actor, train_rows, ref_actor, verbose=True)

        # ── Evaluate ──────────────────────────────────────────────────────────
        rl_bt, fixed_bt = evaluate(actor, ep_test)

        rl_stats    = ie_stats(rl_bt)
        fixed_stats = ie_stats(fixed_bt)

        if rl_stats:
            all_bt["Dynamic RL"].append(rl_bt)
        if fixed_stats:
            all_bt["Fixed"].append(fixed_bt)

        rl_ir    = rl_stats.get("info_ratio", float("nan"))
        fixed_ir = fixed_stats.get("info_ratio", float("nan"))

        fold_summary.append({
            "label":     label,
            "n_months":  len(ep_test),
            "rl_ir":     rl_ir,
            "fixed_ir":  fixed_ir,
            "rl_alpha":  rl_stats.get("ann_alpha", float("nan")),
            "rl_te":     rl_stats.get("track_err", float("nan")),
        })

        print(f"\n  Fold results:")
        print(f"    Dynamic RL  - IR: {rl_ir:+.3f}  |  "
              f"alpha: {rl_stats.get('ann_alpha', 0)*100:.2f}%  |  "
              f"TE: {rl_stats.get('track_err', 0)*100:.2f}%")
        print(f"    Fixed       - IR: {fixed_ir:+.3f}  |  "
              f"alpha: {fixed_stats.get('ann_alpha', 0)*100:.2f}%  |  "
              f"TE: {fixed_stats.get('track_err', 0)*100:.2f}%")

        # Print mean action choices from RL agent (diagnostic)
        if not rl_bt.empty and "alpha_long" in rl_bt.columns:
            asym = (rl_bt["alpha_long"] - rl_bt["alpha_short"]).mean()
            print(f"    RL avg alpha_long:  {rl_bt['alpha_long'].mean()*100:.2f}%  "
                  f"(range {rl_bt['alpha_long'].min()*100:.2f}–{rl_bt['alpha_long'].max()*100:.2f}%)")
            print(f"    RL avg alpha_short: {rl_bt['alpha_short'].mean()*100:.2f}%  "
                  f"(range {rl_bt['alpha_short'].min()*100:.2f}–{rl_bt['alpha_short'].max()*100:.2f}%)")
            print(f"    Avg asymmetry (long−short): {asym*100:+.2f}%  "
                  f"({'long-biased' if asym > 0 else 'short-biased'})")

    if not fold_summary:
        print("\nNo folds completed.")
        return

    fold_df = pd.DataFrame(fold_summary)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("WALK-FORWARD RESULTS SUMMARY")
    print("="*70)
    rl_irs    = fold_df["rl_ir"].dropna().tolist()
    fixed_irs = fold_df["fixed_ir"].dropna().tolist()
    beats     = sum(a > b for a, b in zip(rl_irs, fixed_irs))

    print(f"  Dynamic RL - avg IR: {np.mean(rl_irs):.3f}  "
          f"(folds: {' '.join(f'{v:.3f}' for v in rl_irs)})  beats Fixed: {beats}/{len(rl_irs)}")
    print(f"  Fixed      - avg IR: {np.mean(fixed_irs):.3f}  "
          f"(folds: {' '.join(f'{v:.3f}' for v in fixed_irs)})")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_csv = DATA_DIR / "dynamic_portfolio_results.csv"
    fold_df.to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv}")

    for name in all_bt:
        if all_bt[name]:
            all_bt[name] = pd.concat(all_bt[name]).sort_index()
        else:
            all_bt[name] = pd.DataFrame()

    plot_results(all_bt, fold_df)
    print("\nDone.")


if __name__ == "__main__":
    main()
