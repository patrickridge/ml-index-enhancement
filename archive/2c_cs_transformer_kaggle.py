"""
2c_cs_transformer_kaggle.py  — GPU-ready standalone version
=============================================================
Identical logic to 2c_cs_transformer.py but with NO dependency on config.py.
All hyperparameters are defined as plain variables at the top — easy to tweak on Kaggle.

╔══════════════════════════════════════════════════════════════════════╗
║  KAGGLE SETUP (5 steps)                                              ║
║                                                                      ║
║  1. Create a Kaggle Dataset called "investsoc-ml-data" and upload:   ║
║       • panel_monthly_enriched.parquet  (~12 MB)                     ║
║       • panel_monthly.parquet           (~10 MB)  [optional]         ║
║                                                                      ║
║  2. In your Kaggle notebook:                                         ║
║       Add Data → Your Datasets → investsoc-ml-data                   ║
║                                                                      ║
║  3. Enable GPU:                                                      ║
║       Settings (right panel) → Accelerator → GPU T4 x2               ║
║                                                                      ║
║  4. Paste this entire file into a code cell and run.                 ║
║     Or upload this file and run: !python 2c_cs_transformer_kaggle.py ║
║                                                                      ║
║  5. When done, download from Output:                                 ║
║       scores_cs_transformer.parquet                                  ║
║       bt_cs_transformer.csv                                          ║
║       bt_cs_transformer_ls.csv                                       ║
║     Copy these into your local data/ folder.                         ║
╚══════════════════════════════════════════════════════════════════════╝

Expected GPU time:  ~5–10 min on Kaggle T4
Expected CPU time:  ~45–90 min (not recommended)
"""

import os
import time as _time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

_t0 = _time.time()

# ── Paths (override via env vars if needed) ───────────────────────────────────
DATA_DIR = Path(os.environ.get("ML_DATA_DIR", "/kaggle/input/investsoc-ml-data"))
OUT_DIR  = Path(os.environ.get("ML_OUT_DIR",  "/kaggle/working"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

PANEL_IN    = DATA_DIR / "panel_monthly_enriched.parquet"
OUT_SCORES  = OUT_DIR  / "scores_cs_transformer.parquet"
OUT_BT_LO   = OUT_DIR  / "bt_cs_transformer.csv"
OUT_BT_LS   = OUT_DIR  / "bt_cs_transformer_ls.csv"

# ── Date splits (must match your local config.py) ─────────────────────────────
START_DATE    = "2010-01-01"
TRAIN_END     = "2020-12-31"
VALID_END     = "2022-12-31"

# ── Portfolio settings ────────────────────────────────────────────────────────
TOP_N         = 50
BOTTOM_N      = 50
LONG_FRAC     = 0.10
RETRAIN_EVERY = 12   # retrain every N test months (expanding window)

# ── Cross-Sectional Transformer hyperparameters ───────────────────────────────
D_MODEL      = 128   # embedding dimension per feature / per stock
N_HEADS_S1   = 4     # attention heads in Stage 1 (feature attention)
N_LAYERS_S1  = 2     # encoder layers in Stage 1
N_HEADS_S2   = 4     # attention heads in Stage 2 (cross-stock attention)
N_LAYERS_S2  = 2     # encoder layers in Stage 2
DROPOUT      = 0.1
LR           = 5e-4
WEIGHT_DECAY = 1e-4
EPOCHS       = 100
PATIENCE     = 15
MAX_STOCKS   = 520   # pad all months to this size

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def perf_stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 6:
        return dict(months=len(r), ann=float("nan"), vol=float("nan"),
                    sharpe=float("nan"), maxdd=float("nan"))
    ann    = (1 + r).prod() ** (12 / len(r)) - 1
    vol    = r.std(ddof=1) * np.sqrt(12)
    sharpe = ann / vol if vol > 0 else float("nan")
    nav    = (1 + r).cumprod()
    maxdd  = (nav / nav.cummax() - 1).min()
    return dict(months=len(r), ann=ann, vol=vol, sharpe=sharpe, maxdd=maxdd)


def print_stats(label: str, r: pd.Series):
    s = perf_stats(r)
    print(f"  {label}")
    print(f"    months={s['months']} | ann={s['ann']*100:.2f}% | "
          f"vol={s['vol']*100:.2f}% | sharpe={s['sharpe']:.2f} | maxdd={s['maxdd']*100:.2f}%")


def long_only_ret(df_month: pd.DataFrame, top_n: int) -> float:
    sub = df_month.dropna(subset=["score", "fwd_ret_1m"])
    if len(sub) < top_n:
        return float("nan")
    return sub.nlargest(top_n, "score")["fwd_ret_1m"].mean()


def long_short_ret(df_month: pd.DataFrame, frac: float) -> float:
    sub = df_month.dropna(subset=["score", "fwd_ret_1m"])
    n = len(sub)
    if n < 20:
        return float("nan")
    k = max(1, int(np.floor(n * frac)))
    sub_s = sub.sort_values("score", ascending=False)
    return sub_s.head(k)["fwd_ret_1m"].mean() - sub_s.tail(k)["fwd_ret_1m"].mean()


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureTokenizer(nn.Module):
    """Per-feature linear projection: scalar → d_model-dim embedding."""
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias   = nn.Parameter(torch.zeros(n_features, d_model))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, n_features) → (N, n_features, d_model)
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class CrossSectionalTransformer(nn.Module):
    """
    Two-stage cross-sectional transformer for monthly stock ranking.
    Stage 1: per-stock feature attention (shared weights across stocks).
    Stage 2: cross-stock attention (all stocks in a month as one sequence).
    """
    def __init__(
        self,
        n_features:  int,
        d_model:     int   = D_MODEL,
        n_heads_s1:  int   = N_HEADS_S1,
        n_layers_s1: int   = N_LAYERS_S1,
        n_heads_s2:  int   = N_HEADS_S2,
        n_layers_s2: int   = N_LAYERS_S2,
        dropout:     float = DROPOUT,
        max_stocks:  int   = MAX_STOCKS,
    ):
        super().__init__()
        self.d_model    = d_model
        self.max_stocks = max_stocks

        # Stage 1
        self.tokenizer    = FeatureTokenizer(n_features, d_model)
        self.cls_token_s1 = nn.Parameter(torch.zeros(1, 1, d_model))
        s1_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads_s1,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer_s1 = nn.TransformerEncoder(
            s1_layer, num_layers=n_layers_s1, enable_nested_tensor=False
        )

        # Stage 2
        self.market_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        s2_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads_s2,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer_s2 = nn.TransformerEncoder(
            s2_layer, num_layers=n_layers_s2, enable_nested_tensor=False
        )

        # Score head
        self.score_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        x:            (MAX_N, F)  — padded feature matrix for one month
        padding_mask: (MAX_N,)    — True = padded/invalid stock
        Returns:      (MAX_N,)    — scores (caller extracts only real stocks)
        """
        MAX_N = x.shape[0]

        # Stage 1 — per-stock feature attention
        tokens = self.tokenizer(x)                           # (MAX_N, F, d)
        cls_s1 = self.cls_token_s1.expand(MAX_N, -1, -1)    # (MAX_N, 1, d)
        tokens = torch.cat([cls_s1, tokens], dim=1)          # (MAX_N, F+1, d)
        out_s1 = self.transformer_s1(tokens)                 # (MAX_N, F+1, d)
        stock_emb = out_s1[:, 0, :]                          # (MAX_N, d)

        # Stage 2 — cross-stock attention
        stock_seq = stock_emb.unsqueeze(0)                   # (1, MAX_N, d)
        mkt_cls   = self.market_cls.expand(1, -1, -1)        # (1, 1, d)
        stock_seq = torch.cat([mkt_cls, stock_seq], dim=1)   # (1, MAX_N+1, d)
        s2_mask   = torch.cat([
            torch.zeros(1, 1, dtype=torch.bool, device=x.device),
            padding_mask.unsqueeze(0),
        ], dim=1)                                             # (1, MAX_N+1)

        out_s2   = self.transformer_s2(stock_seq, src_key_padding_mask=s2_mask)
        enriched = out_s2[0, 1:, :]                          # (MAX_N, d)
        scores   = self.score_head(enriched).squeeze(-1)     # (MAX_N,)
        return scores


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

def build_monthly_cross_sections(panel: pd.DataFrame, feat_cols: list) -> list:
    """Convert panel into list of padded monthly cross-section dicts."""
    months = sorted(panel["date"].unique())
    cross_sections = []

    for m in months:
        sub = panel[panel["date"] == m].dropna(subset=["fwd_ret_1m"]).copy()
        n   = len(sub)
        if n == 0:
            continue
        if n > MAX_STOCKS:
            sub = sub.sort_values("ticker").iloc[:MAX_STOCKS]
            n   = MAX_STOCKS

        X_vals = sub[feat_cols].fillna(0.0).values.astype(np.float32)
        y_vals = sub["fwd_ret_1m"].values.astype(np.float32)

        F     = len(feat_cols)
        X_pad = np.zeros((MAX_STOCKS, F), dtype=np.float32)
        y_pad = np.zeros(MAX_STOCKS,      dtype=np.float32)
        mask  = np.ones(MAX_STOCKS,       dtype=bool)

        X_pad[:n] = X_vals
        y_pad[:n] = y_vals
        mask[:n]  = False

        cross_sections.append({
            "date":    m,
            "X":       torch.from_numpy(X_pad),
            "y":       torch.from_numpy(y_pad),
            "mask":    torch.from_numpy(mask),
            "tickers": sub["ticker"].tolist(),
            "n_valid": n,
        })

    return cross_sections


def masked_mse_loss(pred, target, mask):
    valid_pred   = pred[~mask]
    valid_target = target[~mask]
    if len(valid_pred) == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return F.mse_loss(valid_pred, valid_target)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_cs_model(train_cs: list, valid_cs: list, n_features: int) -> CrossSectionalTransformer:
    model = CrossSectionalTransformer(n_features=n_features).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

    best_val_loss  = float("inf")
    patience_count = 0
    best_state     = None
    rng            = np.random.default_rng(seed=42)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for idx in rng.permutation(len(train_cs)):
            cs   = train_cs[idx]
            X    = cs["X"].to(DEVICE)
            y    = cs["y"].to(DEVICE)
            mask = cs["mask"].to(DEVICE)
            optimiser.zero_grad()
            loss = masked_mse_loss(model(X, mask), y, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            epoch_loss += loss.item()
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for cs in valid_cs:
                X    = cs["X"].to(DEVICE)
                y    = cs["y"].to(DEVICE)
                mask = cs["mask"].to(DEVICE)
                val_losses.append(masked_mse_loss(model(X, mask), y, mask).item())

        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss  = val_loss
            patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d} | train={epoch_loss/len(train_cs):.5f} "
                  f"| val={val_loss:.5f} | patience={patience_count}")

        if patience_count >= PATIENCE:
            print(f"    Early stop at epoch {epoch+1} (best val: {best_val_loss:.6f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_cs(model: CrossSectionalTransformer, cs: dict) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        scores = model(cs["X"].to(DEVICE), cs["mask"].to(DEVICE)).cpu().numpy()
    return scores[:cs["n_valid"]]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"Device: {DEVICE}")
    print(f"Panel:  {PANEL_IN}")

    panel = pd.read_parquet(PANEL_IN)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel["date"] >= START_DATE].reset_index(drop=True)

    exclude    = {"date", "ticker", "fwd_ret_1m"}
    feat_cols  = [c for c in panel.columns if c not in exclude]
    n_features = len(feat_cols)
    print(f"Features: {n_features} | Rows: {len(panel):,} | Tickers: {panel['ticker'].nunique()}")

    months       = sorted(panel["date"].unique())
    train_end    = pd.Timestamp(TRAIN_END)
    valid_end    = pd.Timestamp(VALID_END)
    train_months = [m for m in months if m <= train_end]
    valid_months = [m for m in months if train_end < m <= valid_end]
    test_months  = [m for m in months if m > valid_end]

    print(f"Train: {train_months[0].date()} → {train_months[-1].date()} ({len(train_months)}m)")
    print(f"Valid: {valid_months[0].date()} → {valid_months[-1].date()} ({len(valid_months)}m)")
    print(f"Test:  {test_months[0].date()} → {test_months[-1].date()} ({len(test_months)}m)")

    print("\nBuilding monthly cross-sections...")
    all_cs = build_monthly_cross_sections(panel, feat_cols)
    cs_map = {cs["date"]: cs for cs in all_cs}

    train_cs = [cs_map[m] for m in train_months if m in cs_map]
    valid_cs = [cs_map[m] for m in valid_months if m in cs_map]
    print(f"  Train months: {len(train_cs)} | Valid months: {len(valid_cs)}")
    print(f"  Padded to MAX_STOCKS={MAX_STOCKS}")

    print("\nFitting initial Cross-Sectional Transformer...")
    model = train_cs_model(train_cs, valid_cs, n_features)

    # Walk-forward prediction
    all_scores = []
    for i, m in enumerate(test_months):
        if i > 0 and i % RETRAIN_EVERY == 0:
            seen      = train_months + valid_months + test_months[:i]
            tr_exp    = [cs_map[mo] for mo in seen if mo in cs_map]
            va_new    = [cs_map[mo] for mo in test_months[max(0, i - RETRAIN_EVERY):i]
                         if mo in cs_map]
            if va_new:
                print(f"  Retraining at {m.date()} (train={len(tr_exp)} months)...")
                model = train_cs_model(tr_exp, va_new, n_features)

        if m not in cs_map:
            continue

        cs     = cs_map[m]
        scores = predict_cs(model, cs)
        sub = (panel[panel["date"] == m]
               .dropna(subset=["fwd_ret_1m"])
               .head(cs["n_valid"])
               .copy())
        sub["score"] = scores
        all_scores.append(sub[["date", "ticker", "score", "fwd_ret_1m"]])

    scores_df = pd.concat(all_scores, ignore_index=True)
    scores_df.to_parquet(OUT_SCORES, index=False)
    print(f"\nSaved scores → {OUT_SCORES} ({len(scores_df):,} rows)")

    # Backtests
    lo_rets = scores_df.groupby("date").apply(long_only_ret, top_n=TOP_N).rename("port_ret")
    lo_rets = lo_rets.dropna().reset_index()
    lo_rets.to_csv(OUT_BT_LO, index=False)

    ls_rets = scores_df.groupby("date").apply(long_short_ret, frac=LONG_FRAC).rename("ls_ret")
    ls_rets = ls_rets.dropna().reset_index()
    ls_rets.to_csv(OUT_BT_LS, index=False)

    print(f"Saved backtest → {OUT_BT_LO}")
    print(f"Saved backtest → {OUT_BT_LS}")

    print("\n" + "=" * 65)
    print("CROSS-SECTIONAL TRANSFORMER — TEST PERIOD RESULTS")
    print("=" * 65)
    print_stats(f"Long-Only Top{TOP_N}", lo_rets["port_ret"])
    print_stats(f"Long-Short top/bot {int(LONG_FRAC*100)}%", ls_rets["ls_ret"])
    print("=" * 65)
    print(f"\nDownload these files from Kaggle Output:")
    print(f"  {OUT_SCORES.name}")
    print(f"  {OUT_BT_LO.name}")
    print(f"  {OUT_BT_LS.name}")
    print("Copy them into your local data/ folder and run 4_benchmark_spx.py")
    print(f"\nDone in {(_time.time() - _t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
