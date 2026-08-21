"""PLOT -- the baseline model: HiPPO-LegS SSM, at delay 30 and delay 100.

The reference point every other figure in this folder is measured against.
fig_R31_cascade_heatmaps puts the delay-30 condition in a 9-wide grid of
thumbnails at 1.5 in per column; here the baseline gets the whole canvas, at
both delays, so the cascade is legible as a figure in its own right rather than
as one cell of a comparison.

Same arrays, same conventions, no recomputation: `hippo__trained_d30` and
`hippo__retimed_d100` out of data/exp1_heatmap_matrices.npz -- 50 units x 30 and
50 units x 100 delay steps, per-unit min-max normalized to [-1, 1],
trial-averaged over the delay period of the 3-stimulus task, seed 1.  Ordering
is the manuscript's sort_freq_resp (utils/utils_analysis.py:13): units grouped
by frequency content -- the number of local maxima in the normalized profile,
counted with a prominence floor at 60% of the unit's amplitude -- then ordered
by peak time within the pooled one-or-two-peak group.  Colormap is jet with
interpolation="none", which is what seven of the repo's eight activity imshow
calls use.

Each panel is sorted on its own, which is what extract_exp1.py stored and what
fig_R31_cascade_heatmaps does cell by cell: row 7 of the delay-30 panel is not
necessarily row 7 of the delay-100 panel.  The panels show that a cascade spans
each delay, not how one unit moved between them.

Plain heatmaps: no rules, no callouts, no in-panel labels.

Reads ONLY ../data/.

  fig_baseline_hippo.png  the baseline's delay-period activity at both delays
"""
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

