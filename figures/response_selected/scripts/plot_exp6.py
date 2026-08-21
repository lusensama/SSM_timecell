"""PLOT -- R2.3: the LSTM under the freeze-and-retime protocol.

Trimmed from figures/response/scripts/plot_exp6.py.  That script's
fig_R23_lstm_retiming is a three-panel figure (curves | gain distributions |
freeze verification); only the first panel is carried here, at a standalone
size, and the drawing code for it is the original's byte-for-byte.  The other
two panels and the four CSVs behind them stay in figures/response/.

Reads ONLY ../data/.

  fig_R23_lstm_retiming.png  the retraining curves: the LSTM stays flat under
                             freeze-and-retime while both SSM arms climb
"""
import numpy as np
import matplotlib.pyplot as plt

import style as S
from common import CHANCE_3STIM, fig_path, read_csv

S.apply()
ARCH = ["LSTM H=50", "SSM HiPPO", "SSM random"]

def fig_retiming():
    curves = read_csv("exp6_retime_curves.csv")

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    S.grid(ax, axis="both")
    for a in ARCH:
        col = S.SERIES[a]
        eps = sorted({r["episode"] for r in curves if r["architecture"] == a})
        M = []
        for bs in sorted({r["base_seed"] for r in curves
                          if r["architecture"] == a}):
            for rs in sorted({r["retime_seed"] for r in curves
                              if r["architecture"] == a and r["base_seed"] == bs}):
                pts = sorted([r for r in curves if r["architecture"] == a
                              and r["base_seed"] == bs and r["retime_seed"] == rs],
                             key=lambda r: r["episode"])
                ax.plot([p["episode"] for p in pts], [p["acc_pct"] for p in pts],
                        color=col, lw=0.7, alpha=0.28, zorder=3)
                M.append([p["acc_pct"] for p in pts])
        M = np.array(M, dtype=float)
        ax.plot(eps, M.mean(axis=0), color=col, lw=2.4, zorder=5)
        ax.annotate(a, (eps[-1] + 40, M.mean(axis=0)[-1]), color=col,
                    fontsize=8.5, fontweight="bold", va="center")
    S.chance_line(ax, CHANCE_3STIM, "chance 33.3%", x=0.02, ha="left")
    ax.set_xlim(-80, 2520)
    ax.set_ylim(28, 88)
    ax.set_xlabel("readout-retraining episode at delay 100")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Retiming 30 -> 100 under frozen recurrence", loc="left")

    S.title(fig, "The LSTM recovers nothing when the recurrence is frozen and retimed",
            "Freeze the recurrent and input weights (the LSTM analogue of freezing "
            "Lambda and B), retrain only\nthe readout for 2,000 episodes at delay "
            "100.  3 base models x 3 retiming seeds = 9 units per architecture.")
    S.stamp(fig, "R2.3  |  data/exp6_retime_curves.csv", y=-0.16)
    S.save(fig, fig_path("fig_R23_lstm_retiming.png"))

def main():
    print("[plot] exp6 / LSTM retiming")
    fig_retiming()

if __name__ == "__main__":
    main()

