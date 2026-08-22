"""PLOT -- Fig. 5d / 5e: lap identity under randomised lap timing.

Redraws the two panels from ../data/ alone -- no checkpoints, no 6.7 GB rollout
cache.  The drawing code is analysis_plot/plot_fig5de_lap_svm.py's panel_d() and
panel_e() unchanged; only the inputs are swapped from live arrays to the CSVs
that scripts/provenance/extract_fig5de.py boiled that run down to.

  fig5d_lap_confusion.png     pooled 4x4 confusion, cross-validated, 108,030
                              states over 3 seeds x 31 EXACT timesteps
  fig5e_lap_discriminant.png  the state at one exact timestep on the decoder's
                              own discriminant axes; basis and boundary fit on a
                              train half, points are the held-out half

Both are decoded at an exact absolute timestep, so every lap class shares the
same elapsed time and the published panels' time confound is gone by
construction.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import style as S
from common import DATA, fig_path, read_csv

LAP_COLORS = ["#3b6ea5", "#d1793d", "#4f9d6a", "#a4508b", "#8a8a8a", "#c4413f"]

def clf():
    """One-vs-one linear decoder; see plot_fig5de_lap_svm.clf for why not OvR."""
    return make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))

def _meta():
    with open(os.path.join(DATA, "fig5de_meta.json")) as f:
        return json.load(f)

def panel_d(meta):
    rows = read_csv("fig5de_confusion.csv")
    kinds = sorted({r["kind"] for r in rows})
    n = int(max(r["true_lap"] for r in rows))
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.2 * len(kinds), 4.6))
    axes = np.atleast_1d(axes)
    for ax, kind in zip(axes, kinds):
        cm = np.zeros((n, n), dtype=int)
        for r in rows:
            if r["kind"] == kind:
                cm[int(r["true_lap"]) - 1, int(r["predicted_lap"]) - 1] = \
                    int(r["count"])
        row = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
        im = ax.imshow(row, cmap="Blues", vmin=0, vmax=100)
        for i in range(row.shape[0]):
            for j in range(row.shape[1]):
                fg = "white" if row[i, j] > 55 else "#222222"
                ax.text(j, i - 0.10, f"{row[i, j]:.2f}", ha="center",
                        va="center", fontsize=11, color=fg)
                ax.text(j, i + 0.16, f"n={cm[i, j]:,}", ha="center",
                        va="center", fontsize=7.5, color=fg, alpha=0.75)
        lab = [f"Lap {i + 1}" for i in range(row.shape[0])]
        ax.set_xticks(range(row.shape[1])); ax.set_xticklabels(lab)
        ax.set_yticks(range(row.shape[0])); ax.set_yticklabels(lab)
        ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
        ax.set_title(f"{kind}\nbalanced acc "
                     f"{meta['confusion_balanced_acc_pct']:.1f}%  (chance "
                     f"{meta['confusion_chance_pct']:.1f}%)\n"
                     f"n = {cm.sum():,} states, "
                     f"{meta['confusion_timesteps_per_seed']} "
                     f"exact timesteps",
                     fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, label="% of true-lap states")
    fig.suptitle("Fig. 5d | Lap identity decoded at an exact timestep "
                 "(randomised lap durations)", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig5d_lap_confusion.png", keep_cells=True)

def panel_e(meta, mode="lda"):
    pts = [r for r in read_csv("fig5de_projection.csv")
           if r["projection"] == mode]
    kinds = sorted({r["kind"] for r in pts})
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.8 * len(kinds), 5.1))
    axes = np.atleast_1d(axes)
    for ax, kind in zip(axes, kinds):
        sub = [r for r in pts if r["kind"] == kind]
        tr = [r for r in sub if r["split"] == "train"]
        te = [r for r in sub if r["split"] == "test"]
        Ptr = np.array([[r["dim1"], r["dim2"]] for r in tr])
        ytr = np.array([int(r["lap"]) - 1 for r in tr])
        Pte = np.array([[r["dim1"], r["dim2"]] for r in te])
        yte = np.array([int(r["lap"]) - 1 for r in te])
        keep = sorted(set(ytr.tolist()))
        m = clf().fit(Ptr, ytr)
        pad = 0.08 * (Pte.max(0) - Pte.min(0) + 1e-9)
        xx, yy = np.meshgrid(
            np.linspace(Pte[:, 0].min() - pad[0], Pte[:, 0].max() + pad[0], 400),
            np.linspace(Pte[:, 1].min() - pad[1], Pte[:, 1].max() + pad[1], 400))
        G = m.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        cmap = ListedColormap([LAP_COLORS[c] for c in keep])
        ax.contourf(xx, yy, np.searchsorted(keep, G), levels=len(keep),
                    cmap=cmap, alpha=0.18)
        for c in keep:
            s = yte == c
            ax.scatter(Pte[s, 0], Pte[s, 1], s=15, alpha=0.85,
                       edgecolors="none", color=LAP_COLORS[c],
                       label=f"Lap {c + 1}  (n={s.sum()})")
        names = meta[mode]["axes"]
        ax.set_xlabel(names[0]); ax.set_ylabel(names[1])
        ax.set_title(f"{kind}\nheld-out balanced acc in 2-D: "
                     f"{meta[mode]['heldout_balanced_acc_pct']:.1f}%  "
                     f"(chance {meta[mode]['chance_pct']:.1f}%)", fontsize=10)
        ax.legend(frameon=False, fontsize=9)
    what = ("decoder discriminant axes" if mode == "lda"
            else "top-2 principal components")
    fig.suptitle(f"Fig. 5e | Lap geometry at exact timestep "
                 f"t = {meta['panel_e_t']} (seed {meta['panel_e_seed']}), {what}\n"
                 f"projection fit on a train half, points are the held-out half",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, "fig5e_lap_discriminant.png" if mode == "lda"
                else "fig5e_lap_pca_boundary.png")

def _save(fig, name, keep_cells=False):
    path = fig_path(name)
    if S.NOTEXT:
        S.strip_text(fig, keep_in_axes=keep_cells)
    fig.savefig(path, dpi=S.DPI, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.basename(path)}")

def main():
    print("[plot] fig 5d / 5e / lap identity at an exact timestep")
    meta = _meta()
    panel_d(meta)
    panel_e(meta, "lda")

if __name__ == "__main__":
    main()

