import os
import numpy as np
import matplotlib.pyplot as plt

import style as S
from common import MODE_ORDER, MODE_LABEL, DATA, fig_path

S.apply()

def fig_heatmaps():
    z = np.load(os.path.join(DATA, "exp1_heatmap_matrices.npz"))
    phases = [("initial_d30", "at initialization (untrained)"),
              ("trained_d30", "after 250,000 episodes")]
    present = [m for m in MODE_ORDER if f"{m}__initial_d30" in z.files]

    fig, axes = plt.subplots(len(phases), len(present),
                             figsize=(1.55 * len(present) + 1.0, 4.8),
                             squeeze=False,
                             gridspec_kw=dict(hspace=0.32, wspace=0.10))
    im = None
    for i, (ph, plab) in enumerate(phases):
        for j, m in enumerate(present):
            ax = axes[i][j]
            M = z[f"{m}__{ph}"]
            pk = z[f"{m}__{ph}__peakcounts"]
            im = ax.imshow(M, aspect="auto", cmap=S.HEATMAP_CMAP, vmin=-1, vmax=1,
                           interpolation=S.HEATMAP_INTERP)
            n_cascade = int((pk <= 2).sum())
            if 0 < n_cascade < len(pk):
                ax.axhline(n_cascade - 0.5, color="k", lw=1.8, ls="-", alpha=0.55)
                ax.axhline(n_cascade - 0.5, color="w", lw=1.0, ls="--")
            ax.text(0.97, 0.03, f"{n_cascade}/{len(pk)}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=6.6, color="w",
                    bbox=dict(facecolor="k", alpha=0.5, edgecolor="none", pad=1.2))
            ax.set_xticks([0, M.shape[1] - 1])
            ax.set_xticklabels(["1", str(M.shape[1])], fontsize=7)
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(MODE_LABEL[m], fontsize=8, pad=4)
            if j == 0:
                ax.set_ylabel(plab, fontsize=8)
                ax.set_yticks([0, M.shape[0] - 1])
                ax.set_yticklabels(["1", str(M.shape[0])], fontsize=7)

    fig.text(0.5, 0.045, "delay step", ha="center", fontsize=8.5, color=S.INK_2)
    cb = fig.colorbar(im, ax=axes, fraction=0.014, pad=0.012, ticks=[-1, 0, 1])
    cb.set_label("normalized activation", fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7)

    S.title(fig, "Delay-period activity, units grouped by frequency then by peak time",
            "Manuscript ordering (sort_freq_resp) and the repo's jet colormap.  "
            "A diagonal in the upper block is a "
            "time-cell cascade;\nthe dashed rule is where units with three or more "
            "peaks begin, and the corner count is how many of 50 are single- or "
            "double-peaked.\nreal_diagonal and rand_complex are 50/50 single-peaked "
            "but all peak in the same bin; alt_basis is mostly high-frequency.")
    S.stamp(fig, "R2.1 / R3.1  |  data/exp1_heatmap_matrices.npz (+ .csv), "
                 "sort = freq", y=-0.005)
    S.save(fig, fig_path("fig_R31_cascade_heatmaps.png"))

def main():
    print("[plot] exp1 / cascade heatmaps")
    fig_heatmaps()

if __name__ == "__main__":
    main()

