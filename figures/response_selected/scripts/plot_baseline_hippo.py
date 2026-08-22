import os

import numpy as np
import matplotlib.pyplot as plt

import style as S
from common import DATA, fig_path

S.apply()

PANELS = [("hippo__trained_d30", "delay 30"),
          ("hippo__retimed_d100", "delay 100")]

def fig_baseline():
    z = np.load(os.path.join(DATA, "exp1_heatmap_matrices.npz"))

    fig, axes = plt.subplots(1, len(PANELS), figsize=(9.6, 3.9),
                             gridspec_kw=dict(width_ratios=[1.0, 1.55],
                                              wspace=0.16))
    im = None
    for ax, (key, lab) in zip(axes, PANELS):
        M = z[key]
        n_unit, n_step = M.shape
        im = ax.imshow(M, aspect="auto", cmap=S.HEATMAP_CMAP, vmin=-1, vmax=1,
                       interpolation=S.HEATMAP_INTERP)
        ax.set_xticks([0, n_step // 2, n_step - 1])
        ax.set_xticklabels(["1", str(n_step // 2 + 1), str(n_step)], fontsize=8)
        ax.set_yticks([0, n_unit - 1])
        ax.set_yticklabels(["1", str(n_unit)], fontsize=8)
        ax.set_xlabel("delay step")
        ax.set_title(lab, loc="left")
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel("unit (sorted)")

    cb = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.016, ticks=[-1, 0, 1])
    cb.set_label("normalized activation", fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7)

    S.title(fig, "The baseline: a time-cell cascade across the delay period",
            "Delay-period activity of the 50 HiPPO-LegS SSM units after 250,000 "
            "episodes of the 3-stimulus task (seed 1), at the\ntrained delay of "
            "30 and retimed to 100.  Units are grouped by peak count then ordered "
            "by peak time (sort_freq_resp,\nthe manuscript's convention).  Each "
            "unit is min-max normalized to [-1, 1], so colour is relative to that "
            "unit's own range.")
    S.stamp(fig, "baseline  |  data/exp1_heatmap_matrices.npz (+ .csv), "
                 "keys = hippo__trained_d30 + hippo__retimed_d100, sort = freq",
            y=-0.14)
    S.save(fig, fig_path("fig_baseline_hippo.png"))

def main():
    print("[plot] baseline / HiPPO SSM at delay 30 and 100")
    fig_baseline()

if __name__ == "__main__":
    main()

