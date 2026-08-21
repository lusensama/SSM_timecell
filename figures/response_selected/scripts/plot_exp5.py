"""PLOT -- R3.3: the lap-phase sequential basis under randomised timing.

Trimmed from figures/response/scripts/plot_exp5.py: only the one figure this
folder carries is kept, byte-for-byte.  Reads ONLY ../data/.

  fig_R33_lap_heatmap.png  the sequential basis renormalises to each lap
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import style as S
from common import DATA, fig_path

S.apply()

def fig_lap_heatmap():
    p = os.path.join(DATA, "exp5_heatmap_matrices.npz")
    if not os.path.isfile(p):
        print("  ! data/exp5_heatmap_matrices.npz missing "
              "(run export_exp5_heatmap.py once, with checkpoints present)")
        return
    z = np.load(p)
    n_lap, n_phase = int(z["n_lap"]), int(z["n_phase"])

    panels = [("trained_s2__globalnorm",
               "trained — one sort_freq_resp order and one normalization across "
               "all laps"),
              ("trained_s2__perlapnorm",
               "trained — sorted AND renormalized WITHIN each lap "
               "(basic_lap_state.sort_sub's convention)"),
              ("untrained_s2__globalnorm",
               "untrained, identical architecture — same global order")]
    panels = [(k, t) for k, t in panels if k in z.files]

    fig, axes = plt.subplots(len(panels), 1, figsize=(11.0, 2.35 * len(panels)),
                             squeeze=False, gridspec_kw=dict(hspace=0.42))
    im = None
    for i, (key, lab) in enumerate(panels):
        ax = axes[i][0]
        M = z[key]
        im = ax.imshow(M, aspect="auto", cmap="jet", vmin=-1, vmax=1,
                       interpolation="none")
        for k in range(1, n_lap):
            ax.axvline(k * n_phase - 0.5, color="white", lw=1.2, ls="--")
        ax.set_xticks([(k + 0.5) * n_phase for k in range(n_lap)])
        ax.set_xticklabels([f"lap {k + 1}" for k in range(n_lap)], fontsize=8)
        ax.set_yticks([0, M.shape[0] - 1])
        ax.set_yticklabels(["1", str(M.shape[0])], fontsize=7.5)
        ax.set_ylabel("unit", fontsize=8)
        ax.set_title(f"{'abc'[i]}   {lab}", loc="left", fontsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[-1][0].set_xlabel("lap phase (position within lap)", fontsize=8.5)
    cb = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.012, ticks=[-1, 0, 1])
    cb.set_label("normalized activation", fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7)

    S.title(fig, "The sequential basis renormalises to each lap though durations vary six-fold",
            "Lap duration is U{10..60} steps drawn independently per lap.  Sort "
            "orders come from a HELD-OUT HALF of the episodes,\nso no panel can show "
            "a diagonal that is an artifact of sorting the data being displayed.  "
            "Lap identity is evidenced by the\ndecoding, not by the heatmap: the "
            "manuscript convention in panel b divides the cross-lap gain differences "
            "out.\nOrdering is the manuscript's sort_freq_resp (grouped by peak "
            "count, then peak time) and the colormap is jet.  Panel a takes one "
            "order for the\nwhole span; panel b is basic_lap_state.sort_sub's "
            "convention, sorting and renormalizing each lap independently.")
    S.stamp(fig, "R3.3  |  data/exp5_heatmap_matrices.npz (+ .csv) "
                 "+ exp5_lap_selectivity.csv", y=-0.03)
    S.save(fig, fig_path("fig_R33_lap_heatmap.png"))

def main():
    print("[plot] exp5 / lap-phase heatmap")
    fig_lap_heatmap()

if __name__ == "__main__":
    main()

