"""PLOT -- R2.2 / R2.4: the systematic retiming grid, as ONE figure.

Reads ONLY figures/response/data/.

  fig_R22_R24_retiming.png   accuracy vs D_test in a 2 x 2 grid:
      a  D_train = 30, zero-shot        b  D_train = 30, readout-only
      c  D_train = 60, zero-shot        d  D_train = 60, readout-only

The named-cell, asymmetry and 30->200 collapse panels this script used to draw
are dropped; their numbers are still in data/exp3_named_cells.csv,
exp3_asymmetry.csv and exp3_retime_cells.csv.
"""
import numpy as np
import matplotlib.pyplot as plt

import style as S
from common import CHANCE_3STIM, fig_path, read_csv

S.apply()
SRC = "data/exp3_retime_cells.csv (+ _basemodel, _units)"
MODES = [("hippo", "HiPPO", S.SERIES["hippo"]),
         ("rand_complex", "random", S.SERIES["rand_complex"])]
PHASE_LABEL = {"transfer": "zero-shot transfer (no retraining)",
               "recovery": "readout-only recovery (2,000 episodes)"}

def _curves(ax, cells, base, dtr, phase, tag, show_chance=False):
    S.grid(ax, axis="both")
    sel = sorted([c for c in cells if c["D_train"] == dtr and c["phase"] == phase],
                 key=lambda c: c["D_test"])
    xs = [c["D_test"] for c in sel]
    for mode, lab, col in MODES:
        k = "hippo" if mode == "hippo" else "rand"
        mu = np.array([c[f"{k}_mean"] for c in sel])
        sd = np.array([c[f"{k}_sd"] for c in sel])
        ax.fill_between(xs, mu - sd, mu + sd, color=col, alpha=0.15, lw=0, zorder=2)
        ax.plot(xs, mu, color=col, zorder=4, marker="o", ms=3.0,
                markeredgecolor=S.SURFACE, markeredgewidth=0.7)
        pts = [b for b in base if b["D_train"] == dtr and b["phase"] == phase
               and b["mode"] == mode]
        ax.scatter([p["D_test"] for p in pts],
                   [p["acc_pct_mean_over_retime_seeds"] for p in pts],
                   s=2.5, color=col, alpha=0.30, lw=0, zorder=3)
        ax.annotate(lab, (xs[-1] + 3, mu[-1]), color=col, fontsize=8,
                    fontweight="bold", va="center")
    ax.axvline(dtr, color=S.GRID, lw=0.9, zorder=1)
    S.note(ax, dtr + 2, 97, f"trained at\nD = {dtr}", fontsize=7, color=S.INK_2,
            va="top", linespacing=1.4)
    if show_chance:
        S.chance_line(ax, CHANCE_3STIM, "chance 33.3%", x=0.02, ha="left")
    else:
        ax.axhline(CHANCE_3STIM, color=S.INK_MUTED, lw=0.9, zorder=1)
    ax.set_xlim(20, 245)
    ax.set_ylim(28, 100)
    ax.set_xlabel("D_test (steps)", fontsize=8)
    short = {"transfer": "zero-shot", "recovery": "readout-only"}[phase]
    ax.set_title(f"{tag}   D_train = {dtr},  {short}", loc="left", fontsize=9)

def fig_retiming():
    """The four accuracy curves only, in a 2 x 2 grid."""
    cells = read_csv("exp3_retime_cells.csv")
    base = read_csv("exp3_retime_basemodel.csv")

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6), sharey=True,
                             gridspec_kw=dict(hspace=0.28, wspace=0.07))
    for k, (dtr, phase, tag) in enumerate([(30, "transfer", "a"),
                                           (30, "recovery", "b"),
                                           (60, "transfer", "c"),
                                           (60, "recovery", "d")]):
        ax = axes[k // 2][k % 2]
        _curves(ax, cells, base, dtr, phase, tag, show_chance=(k == 0))
        if k % 2 == 0:
            ax.set_ylabel("accuracy (%)")

    S.title(fig, "Retiming accuracy across the grid",
            "1,920 runs.  Line is the mean over 10 base models, band is +/- s.d., "
            "faint dots are the individual base models;\nLambda and B are frozen "
            "in the readout-only arm.")
    S.stamp(fig, "R2.2 / R2.4  |  data/exp3_retime_cells.csv + "
                 "exp3_retime_basemodel.csv", y=-0.04)
    S.save(fig, fig_path("fig_R22_R24_retiming.png"))

def main():
    print("[plot] exp3 / retiming accuracy")
    fig_retiming()

if __name__ == "__main__":
    main()

