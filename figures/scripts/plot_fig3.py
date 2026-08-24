import os
import numpy as np
import matplotlib.pyplot as plt

import style as S
from common import DATA, fig_path, read_csv

S.apply()

PANELS = [
    ("Lambda", r"$|\Delta\lambda| \, / \, |\lambda_{init}|$",
     r"a   State matrix  $\Lambda$"),
    ("B", r"$|\Delta B| \, / \, |B_{init}|$", "b   Input matrix  B"),
    ("C_tilde", r"$|\Delta C| \, / \, |C_{init}|$", "c   Readout matrix  C"),
]
COND = [("hippo", "HiPPO", S.MODE_COLOR["hippo"]),
        ("rand_complex", "random", S.MODE_COLOR["rand_complex"])]

def _hist(ax, z, param, symbol, title, stats):
    data = {}
    for mode, _lab, _col in COND:
        v = np.asarray(z[f"{param}__{mode}"], dtype=float)
        data[mode] = v[np.isfinite(v)]

    pooled = np.concatenate(list(data.values()))
    q1, q99 = np.percentile(pooled, [1, 99])
    bins = np.linspace(q1, q99, 50)

    for mode, lab, col in COND:
        ax.hist(data[mode], bins=bins, alpha=0.6, color=col,
                density=True, edgecolor="black", linewidth=0.5, zorder=3)
    for mode, lab, col in COND:
        m = data[mode].mean()
        ax.axvline(m, color=col, linestyle="--", linewidth=2, alpha=0.9, zorder=5)

    ax.grid(True, ls="--", alpha=0.4, axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel(symbol)
    ax.set_ylabel("Density (Frequency)")
    ax.set_title(title, loc="left")

    for mode, lab, col in COND:
        sub = stats[(param, mode)]["submitted_single_run"]
        if sub in ("", None):
            continue
        sv = float(sub)
        if bins[0] <= sv <= bins[-1]:
            ax.axvline(sv, color=col, ls=":", lw=1.6, alpha=0.9, zorder=5)

    hi, lo = data["hippo"].mean(), data["rand_complex"].mean()
    return hi, lo

def main():
    print("[plot] manuscript Fig. 3 relative change")
    z = np.load(os.path.join(DATA, "fig3_relative_change.npz"))
    rows = read_csv("fig3_relative_change_stats.csv")
    stats = {(r["parameter"], r["mode"]): r for r in rows if r["scope"] == "pooled"}

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(12.0, 12.6),
                             gridspec_kw=dict(hspace=0.30))
    ratios = {}
    for ax, (param, symbol, title) in zip(np.atleast_1d(axes), PANELS):
        ratios[param] = _hist(ax, z, param, symbol, title, stats)

    lh, lr = ratios["Lambda"]
    bh, br = ratios["B"]
    S.title(fig, "Manuscript Fig. 3b/3c, re-measured over the 10-seed cohort",
            "Elementwise relative change, the quantity the submitted panel "
            "histograms, pooled over all 10 seeds per condition instead of one "
            "run.\nDashed rules are the means; dotted rules and the boxed "
            f"\"submitted\" line are the single-run values quoted in the paper.  "
            f"The ordering holds:\nLambda {lh:.2f} vs {lr:.2f} ({lr / lh:.1f}x) and "
            f"B {bh:.2f} vs {br:.2f} ({br / bh:.1f}x), against 4.3x and 2.6x in the "
            "submitted run.")
    S.stamp(fig, "R3.2 / manuscript Fig. 3b-3c  |  data/fig3_relative_change.npz "
                 "+ fig3_relative_change_stats.csv (+ .csv, every element)",
            y=-0.055)
    S.save(fig, fig_path("fig_R32_fig3_relative_change.png"))

if __name__ == "__main__":
    main()

