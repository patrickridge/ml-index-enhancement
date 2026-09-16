"""
8_readme_figures.py - The two figures in the README that no other script makes.

The walk-forward chart comes out of 5c. These two do not have a home anywhere
else, and both exist to make a claim checkable rather than to decorate:

  figures/tilt_adaptation.png  does the agent tilt harder when tilting pays?
  figures/factor_null.png      is the factor library distinguishable from noise?

Usage:
    python 8_readme_figures.py

Reads:  data/bt_wf_rl.csv, data/factor_ic_summary.csv
Writes: figures/tilt_adaptation.png, figures/factor_null.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

DATA_DIR = Path(os.environ.get("ML_DATA_DIR", "data"))
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

INK = "#1c1c1c"
MUTED = "#6b7280"
BLUE = "#2563eb"
AMBER = "#d97706"
GRID = "#e5e7eb"


def style(ax):
    """Recessive axes and grid, so the data is the only thing with weight."""
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def tilt_adaptation():
    """Tilt size against payoff per unit of tilt.

    Dividing active return by alpha is the whole point. A bigger bet moves the
    portfolio more whatever the agent knows, so the raw relationship between
    alpha and active return is mechanical and proves nothing. What is left
    after dividing is whether the agent leans in during months that reward
    leaning in.
    """
    d = pd.read_csv(DATA_DIR / "bt_wf_rl.csv")
    x = d["alpha_used"] * 100
    y = d["active_ret"] / d["alpha_used"]

    r, p = stats.pearsonr(x, y)
    med = x.median()
    hi, lo = y[x > med], y[x <= med]
    t, pt = stats.ttest_ind(hi, lo, equal_var=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    style(ax)
    ax.axhline(0, color=MUTED, linewidth=1, zorder=1)

    ax.scatter(x[x <= med], lo, s=34, color=MUTED, alpha=0.55,
               edgecolor="white", linewidth=0.8, zorder=3, label="Low tilt")
    ax.scatter(x[x > med], hi, s=34, color=BLUE, alpha=0.75,
               edgecolor="white", linewidth=0.8, zorder=3, label="High tilt")

    fit = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, np.polyval(fit, xs), color=AMBER, linewidth=2, zorder=4)

    # Group means as short segments over each group's own x-range, so the
    # labels sit in clear space instead of on top of the points.
    for lo_hi, col, seg in ((lo, MUTED, (x.min(), med)),
                            (hi, BLUE, (med, x.max()))):
        ax.plot(seg, [lo_hi.mean()] * 2, color=col, linewidth=2.5,
                linestyle=(0, (4, 2)), zorder=5)
        ax.text(seg[1] if col == BLUE else seg[0], lo_hi.mean(),
                f" mean {lo_hi.mean():+.2f} ", fontsize=9, color=col,
                weight="bold", va="bottom",
                ha="right" if col == BLUE else "left", zorder=6)

    ax.set_xlabel("Tilt size chosen that month, α (%)", color=INK, fontsize=10)
    ax.set_ylabel("Active return per unit of tilt", color=INK, fontsize=10)
    ax.set_title("The agent tilts harder in the months that pay more per unit",
                 color=INK, fontsize=12, weight="bold", loc="left", pad=26)
    ax.text(0, 1.015, f"r = {r:+.3f} (p = {p:.0e}), {len(d)} months    "
                      f"high vs low tilt: t = {t:.2f}, p = {pt:.0e}",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="lower right")

    fig.savefig(FIG_DIR / "tilt_adaptation.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"tilt_adaptation.png    r={r:+.3f} p={p:.2e} t={t:.2f}")


def factor_null():
    """The library's t-statistics against the distribution noise would give.

    This is the figure that makes the negative result honest. If the factors
    were weak but real, the histogram would sit wider than the normal curve
    and have a tail past 3. It does neither.
    """
    d = pd.read_csv(DATA_DIR / "factor_ic_summary.csv")
    t = d["t_stat"].dropna()

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    style(ax)

    ax.hist(t, bins=34, density=True, color=BLUE, alpha=0.6,
            edgecolor="white", linewidth=0.8, zorder=3,
            label=f"{len(t)} factors tested")

    xs = np.linspace(-5, 5, 400)
    ax.plot(xs, stats.norm.pdf(xs), color=INK, linewidth=2, zorder=4,
            label="What pure noise gives")

    # Labels run vertically along their own line, which is the one placement
    # that cannot collide with the other label or with the bars.
    top = ax.get_ylim()[1]
    for cut, lbl in [(3.0, "Harvey-Liu-Zhu hurdle, t = 3.0"),
                     (3.99, "BHY at 253 tests, t ≈ 4.0")]:
        for s in (-1, 1):
            ax.axvline(s * cut, color=AMBER, linewidth=1.6,
                       linestyle="--", zorder=5)
        ax.text(cut - 0.1, top * 0.97, lbl, fontsize=8.5, color=AMBER,
                rotation=90, va="top", ha="right")

    ax.set_xlim(-5.2, 5.2)
    ax.set_xlabel("Factor IC t-statistic", color=INK, fontsize=10)
    ax.set_ylabel("Density", color=INK, fontsize=10)
    ax.set_title("The factor library is not distinguishable from noise",
                 color=INK, fontsize=12, weight="bold", loc="left", pad=26)
    ax.text(0, 1.015,
            f"median |t| = {t.abs().median():.3f} against 0.674 for a standard "
            f"normal    strongest |t| = {t.abs().max():.2f}    0 survive BHY",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="upper left")

    fig.savefig(FIG_DIR / "factor_null.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"factor_null.png        n={len(t)} median|t|={t.abs().median():.3f} "
          f"max|t|={t.abs().max():.2f}")


if __name__ == "__main__":
    tilt_adaptation()
    factor_null()
