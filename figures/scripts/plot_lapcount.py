import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import style as S
from common import data_path, fig_path, read_csv

SRC = "exp5b_lapcount_sweep.csv"
OUT = None
TABLE = "exp5b_lapcount_table.csv"

ACC_C, VP_C = "#2a78d6", "#eb6834"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
SURFACE = "#ffffff"

def main():
    global OUT
    OUT = fig_path("lapcount_performance")

    rows = read_csv(SRC)
    for r in rows:
        r["K"], r["seed"] = int(r["K"]), int(r["seed"])
    Ks = sorted({r["K"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    train_lo, train_hi = 2, 6

    def stats(key, scale=1.0):
        m = np.array([np.mean([r[key] for r in rows if r["K"] == K]) for K in Ks]) * scale
        s = np.array([np.std([r[key] for r in rows if r["K"] == K]) for K in Ks]) * scale
        return m, s

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.axvspan(train_lo - 0.4, train_hi + 0.4, color="#000000", alpha=0.045,
               linewidth=0, zorder=0)
    ax.grid(axis="y", color=INK3, alpha=0.28, linewidth=0.7, zorder=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK3); ax.spines[s].set_linewidth(0.9)

    for key, scale, color, label in [("count_acc", 1.0, ACC_C, "Count accuracy"),
                                     ("vp", 100.0, VP_C, "VP alignment (×100)")]:
        m, sd = stats(key, scale)
        ax.errorbar(Ks, m, yerr=sd, color=color, lw=2.0, marker="o", ms=6.5,
                    mec=SURFACE, mew=1.4, elinewidth=1.3, capsize=3.5, capthick=1.3,
                    ecolor=color, zorder=3, clip_on=False, label=label)
        ax.annotate(label.split(" (")[0], xy=(Ks[-1], m[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=9.5, color=color, zorder=5, annotation_clip=False)

    ax.set_ylabel("Percent of maximum (%)", fontsize=11, color=INK)
    ax.set_xlabel("Number of laps in the episode (K)", fontsize=11, color=INK)
    ax.set_ylim(0, 108)
    ax.set_xticks(Ks)
    ax.set_xlim(Ks[0] - 0.5, Ks[-1] + 0.6)
    ax.tick_params(colors=INK2, labelsize=10, length=3, width=0.9)

    ax.annotate("trained range\nK ~ U{2..6}", xy=(4, 40), ha="center", va="center",
                fontsize=9.5, color=INK2, zorder=5, linespacing=1.4)
    ax.annotate("extrapolation", xy=(11, 62), ha="center", va="center",
                fontsize=9.5, color=INK2, zorder=5)

    ax.set_title("Lap-counting performance versus number of laps\n"
                 "exp5b agents: mean ± s.d. over 3 seeds, 150 episodes per condition",
                 fontsize=12.5, color=INK, loc="left", pad=10)

    ax.legend(loc="lower left", frameon=False, fontsize=10, ncol=2,
              handlelength=2.2, columnspacing=2.0, labelcolor=INK2,
              bbox_to_anchor=(0.0, 0.02))

    fig.text(0.5, 0.018,
             "Both measures are normalised (1.0 = perfect), so they share one axis.\n"
             "At K=10 count accuracy is still 99.6% while VP has fallen to 0.25: "
             "false alarms never increment the counter.",
             fontsize=8.5, color=INK2, ha="center", va="bottom", linespacing=1.5)

    fig.subplots_adjust(left=0.085, right=0.815, top=0.865, bottom=0.205)
    bbox = None
    if S.NOTEXT:
        S.strip_text(fig)
        bbox = "tight"
    plt.savefig(f"{OUT}.png", dpi=S.DPI, facecolor=SURFACE, bbox_inches=bbox)
    plt.savefig(f"{OUT}.pdf", facecolor=SURFACE, bbox_inches=bbox)
    plt.close(fig)
    print(f"  wrote lapcount_performance.png / .pdf")

    with open(data_path(TABLE), "w") as f:
        f.write("K,in_training_range,"
                + ",".join(f"count_acc_s{s},vp_s{s}" for s in seeds)
                + ",count_acc_mean,count_acc_sd,vp_mean,vp_sd\n")
        for K in Ks:
            cells = []
            for s in seeds:
                g = lambda k: next(r[k] for r in rows if r["K"] == K and r["seed"] == s)
                cells += [f"{g('count_acc'):.2f}", f"{g('vp'):.4f}"]
            ca = [r["count_acc"] for r in rows if r["K"] == K]
            vp = [r["vp"] for r in rows if r["K"] == K]
            f.write(f"{K},{'yes' if train_lo <= K <= train_hi else 'no'},"
                    + ",".join(cells)
                    + f",{np.mean(ca):.2f},{np.std(ca):.2f},{np.mean(vp):.4f},{np.std(vp):.4f}\n")
    print(f"  wrote data/{TABLE} (table view, per-seed values retained)")

if __name__ == "__main__":
    main()

