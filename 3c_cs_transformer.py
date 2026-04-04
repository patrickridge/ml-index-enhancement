"""
3c_cs_transformer.py
=====================
Cross-Sectional Transformer for S&P 500 stock ranking.

Architecture (two-stage):
  Stage 1 — Per-Stock Feature Attention:
    Each stock's 100+ features are tokenized (one d_model-dim embedding per feature),
    a learnable [CLS] token is prepended, and a TransformerEncoder attends over the
    feature dimension. The CLS output becomes the stock's summary embedding.
    Weights are SHARED across all stocks (like a shared encoder in set transformers).

  Stage 2 — Cross-Stock Attention:
    All stock embeddings for one month are stacked into a sequence of length N_stocks.
    A MARKET_CLS token is prepended. A second TransformerEncoder attends across the
    full cross-section so each stock "sees" its peers before generating its score.
    This is the key advantage over the original FT-Transformer which processed
    each stock independently.

  Score Head:
    LayerNorm → Linear(d_model, 1) → scalar score per stock.

Training:
  Each month is one forward pass (batch=1 for Stage 2).
  Months are padded to MAX_N_STOCKS=520; padding_mask prevents attention to pad positions.
  Loss: masked MSE on non-padded stocks. Shuffle at the month level each epoch.
  Walk-forward expanding window, same as 2b_nn_backtest.py.

Outputs:
  data/scores_cs_transformer.parquet
  data/bt_cs_transformer.csv      (long-only top 50)
  data/bt_cs_transformer_ls.csv   (long-short)

Run AFTER 1_feature_engineering.py (and optionally 1b_orthogonalize.py).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from config import (
    DATA_DIR, START_DATE, TRAIN_END, VALID_END,
    TOP_N, BOTTOM_N, LONG_FRAC, RETRAIN_EVERY,
    TRANSFORMER_CS_PARAMS, USE_ORTHOGONALIZED_FEATURES,
)

# Select feature panel
_panel_file = ("panel_monthly_orthogonalized.parquet" if USE_ORTHOGONALIZED_FEATURES
               else "panel_monthly_enriched.parquet")
PANEL_IN    = DATA_DIR / _panel_file
OUT_SCORES  = DATA_DIR / "scores_cs_transformer.parquet"
OUT_BT_LO   = DATA_DIR / "bt_cs_transformer.csv"
OUT_BT_LS   = DATA_DIR / "bt_cs_transformer_ls.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE HELPERS  (identical to 2b_nn_backtest.py)
# ═══════════════════════════════════════════════════════════════════════════════

def perf_stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 6:
        return dict(months=len(r), ann=np.nan, vol=np.nan, sharpe=np.nan, maxdd=np.nan)
    ann    = (1 + r).prod() ** (12 / len(r)) - 1
    vol    = r.std(ddof=1) * np.sqrt(12)
    sharpe = ann / vol if vol > 0 else np.nan
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
        return np.nan
    return sub.nlargest(top_n, "score")["fwd_ret_1m"].mean()


def long_short_ret(df_month: pd.DataFrame, frac: float) -> float:
    sub = df_month.dropna(subset=["score", "fwd_ret_1m"])
    n = len(sub)
    if n < 20:
        return np.nan
    k = max(1, int(np.floor(n * frac)))
    sub_s = sub.sort_values("score", ascending=False)
    return sub_s.head(k)["fwd_ret_1m"].mean() - sub_s.tail(k)["fwd_ret_1m"].mean()


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureTokenizer(nn.Module):
    """
    Per-feature linear projection: each scalar → d_model-dim embedding.
    Separate weights per feature (same as in FTTransformer).
    """
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

    Stage 1 (feature attention):
      - Input: (MAX_N, n_features) — all stocks in a month (padded)
      - FeatureTokenizer → (MAX_N, n_features, d_model)
      - Prepend CLS_S1 → (MAX_N, n_features+1, d_model)
      - TransformerEncoder over feature dim → (MAX_N, n_features+1, d_model)
      - Extract CLS output → stock_embeddings: (MAX_N, d_model)

    Stage 2 (cross-stock attention):
      - Input: stock_embeddings: (MAX_N, d_model) → treated as seq of length MAX_N
      - Prepend MARKET_CLS → (1, MAX_N+1, d_model)
      - TransformerEncoder with padding_mask → (1, MAX_N+1, d_model)
      - Drop MARKET_CLS → enriched_embeddings: (MAX_N, d_model)

    Score Head:
      - LayerNorm → Linear → scalar → (MAX_N,)
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads_s1: int = 4,
        n_layers_s1: int = 2,
        n_heads_s2: int = 4,
        n_layers_s2: int = 2,
        dropout: float = 0.1,
        max_stocks: int = 520,
    ):
        super().__init__()
        self.d_model    = d_model
        self.max_stocks = max_stocks

        # ── Stage 1: per-stock feature attention ─────────────────────────────
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

        # ── Stage 2: cross-stock attention ───────────────────────────────────
        self.market_cls = nn.Parameter(torch.zeros(1, 1, d_model))

        s2_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads_s2,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer_s2 = nn.TransformerEncoder(
            s2_layer, num_layers=n_layers_s2, enable_nested_tensor=False
        )

        # ── Score head ────────────────────────────────────────────────────────
        self.score_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        x: torch.Tensor,            # (MAX_N, n_features)
        padding_mask: torch.Tensor, # (MAX_N,) bool — True = padded/invalid stock
    ) -> torch.Tensor:              # (MAX_N,) scores
        """
        x:            (MAX_N, F)  — feature matrix for all stocks in one month (padded)
        padding_mask: (MAX_N,)    — True for padded positions (ignored in attention)
        Returns:      (MAX_N,)    — scores; caller masks out padded positions
        """
        MAX_N = x.shape[0]

        # ── Stage 1: embed each stock's features independently ────────────────
        # Treat all stocks as an independent batch (shared encoder weights)
        tokens = self.tokenizer(x)                           # (MAX_N, F, d)
        cls_s1 = self.cls_token_s1.expand(MAX_N, -1, -1)    # (MAX_N, 1, d)
        tokens = torch.cat([cls_s1, tokens], dim=1)          # (MAX_N, F+1, d)
        out_s1 = self.transformer_s1(tokens)                 # (MAX_N, F+1, d)
        stock_emb = out_s1[:, 0, :]                          # (MAX_N, d) — CLS output

        # ── Stage 2: cross-stock attention ────────────────────────────────────
        # Treat the N stocks as a sequence (batch=1, seq_len=MAX_N)
        stock_seq = stock_emb.unsqueeze(0)                   # (1, MAX_N, d)
        mkt_cls   = self.market_cls.expand(1, -1, -1)        # (1, 1, d)
        stock_seq = torch.cat([mkt_cls, stock_seq], dim=1)   # (1, MAX_N+1, d)

        # Mask: MARKET_CLS is never masked (False); padded stocks are masked (True)
        s2_mask = torch.cat([
            torch.zeros(1, 1, dtype=torch.bool, device=x.device),  # MARKET_CLS
            padding_mask.unsqueeze(0),                               # (1, MAX_N)
        ], dim=1)   # (1, MAX_N+1)

        out_s2    = self.transformer_s2(stock_seq, src_key_padding_mask=s2_mask)
        enriched  = out_s2[0, 1:, :]                         # (MAX_N, d) — skip MARKET_CLS
        scores    = self.score_head(enriched).squeeze(-1)    # (MAX_N,)

        return scores


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

def build_monthly_cross_sections(
    panel: pd.DataFrame,
    feat_cols: list,
    max_stocks: int = 520,
) -> list:
    """
    Convert panel (long format) into list of monthly cross-section dicts.
    Each dict contains padded tensors ready for CrossSectionalTransformer.

    Returns: list of dicts:
      { date, X:(MAX_N,F), y:(MAX_N,), mask:(MAX_N,), tickers:[str], n_valid:int }
    """
    months = sorted(panel["date"].unique())
    cross_sections = []

    for m in months:
        sub = panel[panel["date"] == m].dropna(subset=["fwd_ret_1m"]).copy()
        n   = len(sub)

        if n == 0:
            continue

        if n > max_stocks:
            # Very rare: truncate (sorted by ticker for reproducibility)
            sub = sub.sort_values("ticker").iloc[:max_stocks]
            n   = max_stocks

        X_vals = sub[feat_cols].fillna(0.0).values.astype(np.float32)   # (n, F)
        y_vals = sub["fwd_ret_1m"].values.astype(np.float32)             # (n,)

        # Pad to max_stocks
        F      = len(feat_cols)
        X_pad  = np.zeros((max_stocks, F), dtype=np.float32)
        y_pad  = np.zeros(max_stocks, dtype=np.float32)
        mask   = np.ones(max_stocks, dtype=bool)    # True = padded

        X_pad[:n] = X_vals
        y_pad[:n] = y_vals
        mask[:n]  = False   # valid stocks: not padded

        cross_sections.append({
            "date":    m,
            "X":       torch.from_numpy(X_pad),
            "y":       torch.from_numpy(y_pad),
            "mask":    torch.from_numpy(mask),
            "tickers": sub["ticker"].tolist(),
            "n_valid": n,
        })

    return cross_sections


def masked_mse_loss(
    pred:   torch.Tensor,   # (MAX_N,)
    target: torch.Tensor,   # (MAX_N,)
    mask:   torch.Tensor,   # (MAX_N,) bool: True = padded
) -> torch.Tensor:
    """MSE computed only over valid (non-padded) stocks."""
    valid_pred   = pred[~mask]
    valid_target = target[~mask]
    if len(valid_pred) == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return F.mse_loss(valid_pred, valid_target)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_cs_model(
    train_cs:   list,
    valid_cs:   list,
    n_features: int,
) -> CrossSectionalTransformer:
    """
    Train CrossSectionalTransformer on monthly cross-sections.
    Shuffles at the month level each epoch.
    Early stopping on mean validation masked MSE.
    """
    p = TRANSFORMER_CS_PARAMS
    model = CrossSectionalTransformer(
        n_features  = n_features,
        d_model     = p["d_model"],
        n_heads_s1  = p["n_heads_s1"],
        n_layers_s1 = p["n_layers_s1"],
        n_heads_s2  = p["n_heads_s2"],
        n_layers_s2 = p["n_layers_s2"],
        dropout     = p["dropout"],
        max_stocks  = p["max_stocks"],
    ).to(DEVICE)

    optimiser = torch.optim.Adam(
        model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=p["epochs"])

    best_val_loss  = float("inf")
    patience_count = 0
    best_state     = None
    rng            = np.random.default_rng(seed=42)

    for epoch in range(p["epochs"]):
        # ── Train ──
        model.train()
        epoch_loss  = 0.0
        indices     = rng.permutation(len(train_cs))   # shuffle months

        for idx in indices:
            cs   = train_cs[idx]
            X    = cs["X"].to(DEVICE)      # (MAX_N, F)
            y    = cs["y"].to(DEVICE)      # (MAX_N,)
            mask = cs["mask"].to(DEVICE)   # (MAX_N,)

            optimiser.zero_grad()
            scores = model(X, mask)
            loss   = masked_mse_loss(scores, y, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            epoch_loss += loss.item()

        scheduler.step()

        # ── Validate ──
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
            print(f"    Epoch {epoch+1:3d} | train_loss={epoch_loss/len(train_cs):.5f} "
                  f"| val_loss={val_loss:.5f} | patience={patience_count}")

        if patience_count >= p["patience"]:
            print(f"    Early stop at epoch {epoch+1} (best val MSE: {best_val_loss:.6f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model


def predict_cs(
    model: CrossSectionalTransformer,
    cs:    dict,
) -> np.ndarray:
    """
    Predict scores for one cross-section.
    Returns scores only for real (non-padded) stocks: shape (n_valid,).
    """
    model.eval()
    with torch.no_grad():
        X    = cs["X"].to(DEVICE)
        mask = cs["mask"].to(DEVICE)
        scores = model(X, mask).cpu().numpy()   # (MAX_N,)
    return scores[:cs["n_valid"]]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"Device: {DEVICE}")
    print(f"Panel:  {PANEL_IN.name}")

    panel = pd.read_parquet(PANEL_IN)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel["date"] >= START_DATE].reset_index(drop=True)

    exclude   = {"date", "ticker", "fwd_ret_1m"}
    feat_cols = [c for c in panel.columns if c not in exclude]
    n_features = len(feat_cols)

    print(f"Features: {n_features} | Rows: {len(panel):,} | Tickers: {panel['ticker'].nunique()}")

    p = TRANSFORMER_CS_PARAMS

    # ── Time splits ───────────────────────────────────────────────────────────
    months       = sorted(panel["date"].unique())
    train_end    = pd.Timestamp(TRAIN_END)
    valid_end    = pd.Timestamp(VALID_END)
    train_months = [m for m in months if m <= train_end]
    valid_months = [m for m in months if train_end < m <= valid_end]
    test_months  = [m for m in months if m > valid_end]

    print(f"Train: {train_months[0].date()} → {train_months[-1].date()} ({len(train_months)} months)")
    print(f"Valid: {valid_months[0].date()} → {valid_months[-1].date()} ({len(valid_months)} months)")
    print(f"Test:  {test_months[0].date()} → {test_months[-1].date()} ({len(test_months)} months)")

    # ── Build all monthly cross-sections once ─────────────────────────────────
    print("\nBuilding monthly cross-sections...")
    all_cs = build_monthly_cross_sections(panel, feat_cols, max_stocks=p["max_stocks"])
    cs_map = {cs["date"]: cs for cs in all_cs}

    train_cs = [cs_map[m] for m in train_months if m in cs_map]
    valid_cs = [cs_map[m] for m in valid_months if m in cs_map]
    print(f"  Train cross-sections: {len(train_cs)}")
    print(f"  Valid cross-sections: {len(valid_cs)}")
    print(f"  Max stocks per month (padded to): {p['max_stocks']}")

    # ── Initial training ──────────────────────────────────────────────────────
    print("\nFitting initial Cross-Sectional Transformer...")
    model = train_cs_model(train_cs, valid_cs, n_features)

    # ── Walk-forward prediction ───────────────────────────────────────────────
    all_scores = []

    for i, m in enumerate(test_months):
        # Retrain every RETRAIN_EVERY months with expanding window
        if i > 0 and i % RETRAIN_EVERY == 0:
            seen_months  = train_months + valid_months + test_months[:i]
            tr_cs_exp    = [cs_map[mo] for mo in seen_months if mo in cs_map]
            va_cs_new    = [cs_map[mo] for mo in test_months[max(0, i - RETRAIN_EVERY):i]
                            if mo in cs_map]
            if va_cs_new:
                print(f"  Retraining at {m.date()} (train={len(tr_cs_exp)} months)...")
                model = train_cs_model(tr_cs_exp, va_cs_new, n_features)

        if m not in cs_map:
            continue

        cs     = cs_map[m]
        scores = predict_cs(model, cs)   # (n_valid,)

        # Match scores back to valid stock rows in the panel
        sub = (panel[panel["date"] == m]
               .dropna(subset=["fwd_ret_1m"])
               .head(cs["n_valid"])
               .copy())
        sub["score"] = scores
        all_scores.append(sub[["date", "ticker", "score", "fwd_ret_1m"]])

    scores_df = pd.concat(all_scores, ignore_index=True)
    scores_df.to_parquet(OUT_SCORES, index=False)
    print(f"\nSaved scores: {OUT_SCORES} | rows={len(scores_df):,}")

    # ── Backtests ─────────────────────────────────────────────────────────────
    lo_rets = scores_df.groupby("date").apply(long_only_ret, top_n=TOP_N).rename("port_ret")
    lo_rets = lo_rets.dropna().reset_index()
    lo_rets.to_csv(OUT_BT_LO, index=False)

    ls_rets = scores_df.groupby("date").apply(long_short_ret, frac=LONG_FRAC).rename("ls_ret")
    ls_rets = ls_rets.dropna().reset_index()
    ls_rets.to_csv(OUT_BT_LS, index=False)

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("CROSS-SECTIONAL TRANSFORMER — TEST PERIOD RESULTS")
    print("=" * 65)
    print_stats(f"Long-Only Top{TOP_N} (equal weight)", lo_rets["port_ret"])
    print_stats(f"Long-Short top/bot {int(LONG_FRAC*100)}%",  ls_rets["ls_ret"])

    # ── Compare to FT-Transformer if available ────────────────────────────────
    ft_path = DATA_DIR / "scores_transformer.parquet"
    if ft_path.exists():
        ft_sc = pd.read_parquet(ft_path)
        ft_lo = ft_sc.groupby("date").apply(long_only_ret, top_n=TOP_N).dropna()
        ft_ls = ft_sc.groupby("date").apply(long_short_ret, frac=LONG_FRAC).dropna()

        ft_lo_s = perf_stats(ft_lo)
        ft_ls_s = perf_stats(ft_ls)
        cs_lo_s = perf_stats(lo_rets["port_ret"])
        cs_ls_s = perf_stats(ls_rets["ls_ret"])

        print("\n" + "=" * 65)
        print("COMPARISON: FT-Transformer  vs  Cross-Sectional Transformer")
        print("=" * 65)
        print(f"{'Strategy':<32} {'Sharpe':>8}  {'Ann Ret':>8}  {'MaxDD':>8}")
        print("-" * 65)
        print(f"{'FT-Transformer Long-Only':<32} {ft_lo_s['sharpe']:>8.2f}  "
              f"{ft_lo_s['ann']*100:>7.1f}%  {ft_lo_s['maxdd']*100:>7.1f}%")
        print(f"{'CS-Transformer Long-Only':<32} {cs_lo_s['sharpe']:>8.2f}  "
              f"{cs_lo_s['ann']*100:>7.1f}%  {cs_lo_s['maxdd']*100:>7.1f}%")
        print("-" * 65)
        print(f"{'FT-Transformer Long-Short':<32} {ft_ls_s['sharpe']:>8.2f}  "
              f"{ft_ls_s['ann']*100:>7.1f}%  {ft_ls_s['maxdd']*100:>7.1f}%")
        print(f"{'CS-Transformer Long-Short':<32} {cs_ls_s['sharpe']:>8.2f}  "
              f"{cs_ls_s['ann']*100:>7.1f}%  {cs_ls_s['maxdd']*100:>7.1f}%")
        print("=" * 65)


if __name__ == "__main__":
    main()
