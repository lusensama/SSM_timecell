"""
New Fig. 5d / 5e: lap-identity decoding under RANDOMISED lap timing.

Why a new version is needed
---------------------------
The published 5d/5e come from analysis_plot/svm_time_classification.py and
analysis_plot/visualize_svm_real_data.py, which were run on the FIXED ~30-step
lap data (figures/eval_lap_best_model/fixed_baseline). There the four "lap"
classes are four fixed absolute-time windows ([12-16], [45-49], [78-82],
[112-116]), so corr(lap index, absolute time) ~ 1 and the 100% confusion matrix
is decodable from elapsed time alone -- it does not establish a lap code.

This script rebuilds both panels under the exp5 randomised-timing regime
(K~U{2..6}, lap durations U(10,60), 0-2 mid-lap pauses U(0,25)) and decodes at
an EXACT absolute timestep, so every class shares the same elapsed time and the
time confound is removed by construction. The untrained (random-init, never
trained) network is run as the control that a passive leaky integrator would
pass.

  5d  pooled confusion matrix, trained vs untrained, at exact timesteps
  5e  the state at one exact timestep projected onto a 2-D subspace, with
      linear-SVM decision regions, trained vs untrained

Two projections are produced for 5e:

  --proj lda   the DECODER's discriminant axes (regularised LDA). This is the
               subspace the lap code actually lives in. The basis is fit on a
               TRAIN half and the plotted points are the HELD-OUT half, so the
               separation cannot be an artefact of fitting 80 dimensions to a
               few hundred samples -- and the untrained network run through the
               identical pipeline is therefore a real control.
  --proj pca   top-2 principal components, i.e. what the published 5e did.
               Under randomised timing this shows NO lap separation: the
               leading variance directions carry within-lap temporal phase,
               not lap identity. Kept because that contrast is the result.

Rollouts are cached to --cache_dir, so re-plotting is instant after the first
run.

Usage
-----
  python analysis_plot/plot_fig5de_lap_svm.py --seeds 2 3 4 --n_episodes 500 \
      --min_per_class 40 --save_dir figures/exp5_random/fig5de
"""

import argparse, json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_lap_identity import load, collect

LAP_COLORS = ["#3b6ea5", "#d1793d", "#4f9d6a", "#a4508b", "#8a8a8a", "#c4413f"]

def clf():
    """Linear multiclass decoder, ONE-VS-ONE.

    Must not be LinearSVC. LinearSVC is one-vs-rest, and the lap code turns out
    to be a single monotonically ordered axis (lap 1 .. lap 4 sit at decreasing
    LD1), so a middle class cannot be isolated by any one half-plane: each OvR
    binary fit yields a half-plane and the argmax swallows laps 2 and 3. Measured
    at t=90, seed 2, in the full 80-D space, on identical data:
        LinearSVC  one-vs-rest  54.17%
        SVC linear one-vs-one   97.24%   (nearest-centroid agrees, 98.17%)
    The 54% was an artefact of the classifier, not a property of the state.
    SVC(kernel="linear") pairs classes off, so ordered classes are handled.
    """
    return make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))

def cv_confusion(X, y, labels, folds=4):
    """Cross-validated predictions -> confusion matrix. Returns (cm, bal_acc)."""
    counts = np.bincount(y)
    counts = counts[counts > 0]
    folds = int(min(folds, counts.min()))
    if folds < 2:
        return None, None
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    yp = np.empty_like(y)
    for tr, te in skf.split(X, y):
        yp[te] = clf().fit(X[tr], y[tr]).predict(X[te])
    cm = confusion_matrix(y, yp, labels=labels)
    per_class = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(len(labels)),
                          where=cm.sum(1) > 0)
    return cm, per_class[cm.sum(1) > 0].mean() * 100

def timestep_samples(d, t, min_per_class, n_lap):
    """States at exact absolute timestep t, labelled by lap index.

    One sample per episode, so a stratified split over samples is also an
    episode-level split -- no leakage between the fit and held-out halves.
    """
    lap = d["lap_t"][:, t]
    m = (lap >= 0) & (lap < n_lap)
    X, y = d["Xt"][m, t], lap[m]
    keep = [c for c in range(n_lap) if (y == c).sum() >= min_per_class]
    if len(keep) < 2:
        return None
    m2 = np.isin(y, keep)
    return X[m2], y[m2], keep

def rollout(kind, seed, args):
    """collect(), memoised to --cache_dir."""
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        tag = (f"{kind}_s{seed}_n{args.n_episodes}_k{args.k_range[0]}-"
               f"{args.k_range[1]}_l{args.lap_len_range[0]}-{args.lap_len_range[1]}"
               f"_p{args.pause_count_range[1]}-{args.pause_len_range[1]}"
               f"_L{args.n_lap_plot}_P{args.n_phase}")
        path = os.path.join(args.cache_dir, tag + ".npz")
        if os.path.exists(path):
            z = np.load(path)
            print(f"    cache hit {path}")
            return dict(Xt=z["Xt"], lap_t=z["lap_t"], K=z["K"], T=int(z["T"]))
    net = load(kind, seed, args.n_neurons, args.ckpt_dir)
    d = collect(net, args, seed=4000 + seed, oracle=(kind == "untrained"))
    if args.cache_dir:
        np.savez_compressed(path, Xt=d["Xt"], lap_t=d["lap_t"], K=d["K"],
                            T=d["T"])
        print(f"    cached -> {path}")
    return d

def panel_d(per_kind, n_lap, path, ts_used):
    kinds = list(per_kind)
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.2 * len(kinds), 4.6))
    axes = np.atleast_1d(axes)
    for ax, kind in zip(axes, kinds):
        cm, bal, chance, n = per_kind[kind]
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
        ax.set_title(f"{kind}\nbalanced acc {bal:.1f}%  (chance {chance:.1f}%)\n"
                     f"n = {n:,} states, {ts_used} exact timesteps", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, label="% of true-lap states")
    fig.suptitle("Fig. 5d | Lap identity decoded at an exact timestep "
                 "(randomised lap durations)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")

def fit_projection(Xtr, ytr, mode, shrinkage):
    """Return a callable mapping 80-D states -> 2-D, fit on the TRAIN half."""
    sc = StandardScaler().fit(Xtr)
    if mode == "pca":
        red = PCA(n_components=2, random_state=0).fit(sc.transform(Xtr))
        var = red.explained_variance_ratio_[:2]
        names = (f"PC1 ({var[0]:.1%} var)", f"PC2 ({var[1]:.1%} var)")
    else:
        n_c = len(np.unique(ytr))
        red = LDA(solver="eigen", shrinkage=shrinkage,
                  n_components=min(2, n_c - 1)).fit(sc.transform(Xtr), ytr)
        if min(2, n_c - 1) == 2:
            names = ("LD1 (discriminant axis 1)", "LD2 (discriminant axis 2)")
            aux = None
        else:
            names = ("LD1 (discriminant axis)", "PC1 of residual")
            aux = PCA(n_components=1, random_state=0).fit(sc.transform(Xtr))

        def proj_lda(X):
            Z = sc.transform(X)
            L = red.transform(Z)
            if aux is None:
                return L[:, :2]
            return np.c_[L[:, 0], aux.transform(Z)[:, 0]]
        return proj_lda, names

    def proj_pca(X):
        return red.transform(sc.transform(X))
    return proj_pca, names

def panel_e(per_kind, path, t, mode, shrinkage, test_size, seed_label):
    kinds = list(per_kind)
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.8 * len(kinds), 5.1))
    axes = np.atleast_1d(axes)
    out = {}
    for ax, kind in zip(axes, kinds):
        X, y, keep = per_kind[kind]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                     random_state=0)
        tr, te = next(sss.split(X, y))
        proj, names = fit_projection(X[tr], y[tr], mode, shrinkage)
        Ptr, Pte = proj(X[tr]), proj(X[te])
        m = clf().fit(Ptr, y[tr])
        acc = balanced_accuracy_score(y[te], m.predict(Pte)) * 100
        pad = 0.08 * (Pte.max(0) - Pte.min(0) + 1e-9)
        xx, yy = np.meshgrid(
            np.linspace(Pte[:, 0].min() - pad[0], Pte[:, 0].max() + pad[0], 400),
            np.linspace(Pte[:, 1].min() - pad[1], Pte[:, 1].max() + pad[1], 400))
        G = m.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        cmap = ListedColormap([LAP_COLORS[c] for c in keep])
        ax.contourf(xx, yy, np.searchsorted(keep, G), levels=len(keep),
                    cmap=cmap, alpha=0.18)
        for c in keep:
            s = y[te] == c
            ax.scatter(Pte[s, 0], Pte[s, 1], s=15, alpha=0.85,
                       edgecolors="none", color=LAP_COLORS[c],
                       label=f"Lap {c + 1}  (n={s.sum()})")
        ax.set_xlabel(names[0]); ax.set_ylabel(names[1])
        ax.set_title(f"{kind}\nheld-out balanced acc in 2-D: {acc:.1f}%  "
                     f"(chance {100 / len(keep):.1f}%)", fontsize=10)
        ax.legend(frameon=False, fontsize=9)
        out[kind] = dict(heldout_balanced_acc_pct=float(acc),
                         chance_pct=100 / len(keep),
                         n_train=int(len(tr)), n_test=int(len(te)),
                         axes=list(names))
    what = ("decoder discriminant axes" if mode == "lda"
            else "top-2 principal components")
    fig.suptitle(f"Fig. 5e | Lap geometry at exact timestep t = {t} "
                 f"({seed_label}), {what}\n"
                 f"projection fit on a train half, points are the held-out half",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[2])
    p.add_argument("--kinds", nargs="+", default=["trained", "untrained"])
    p.add_argument("--ckpt_dir", default="training/exp5_random")
    p.add_argument("--save_dir", default="figures/exp5_random/fig5de")
    p.add_argument("--cache_dir", default="figures/exp5_random/fig5de/_cache")
    p.add_argument("--n_neurons", type=int, default=80)
    p.add_argument("--n_episodes", type=int, default=400)
    p.add_argument("--n_lap_plot", type=int, default=4)
    p.add_argument("--n_phase", type=int, default=40)
    p.add_argument("--k_range", type=int, nargs=2, default=[2, 6])
    p.add_argument("--lap_len_range", type=int, nargs=2, default=[10, 60])
    p.add_argument("--pause_count_range", type=int, nargs=2, default=[0, 2])
    p.add_argument("--pause_len_range", type=int, nargs=2, default=[0, 25])
    p.add_argument("--hit_window", type=int, default=3)
    p.add_argument("--t_start", type=int, default=10)
    p.add_argument("--t_step", type=int, default=20)
    p.add_argument("--min_per_class", type=int, default=20)
    p.add_argument("--proj", nargs="+", default=["lda", "pca"],
                   choices=["lda", "pca"], help="projections to draw for 5e")
    p.add_argument("--shrinkage", default="auto",
                   help="LDA shrinkage: 'auto' or a float in [0,1]")
    p.add_argument("--test_size", type=float, default=0.5)
    p.add_argument("--panel_e_t", type=int, default=-1,
                   help="timestep for 5e; -1 picks most classes, then most samples")
    p.add_argument("--panel_e_seed", type=int, default=-1,
                   help="seed for 5e; -1 uses the first of --seeds")
    p.add_argument("--cache_only", action="store_true",
                   help="roll out and populate --cache_dir, then exit without "
                        "decoding or plotting. Use for the sbatch array; the "
                        "plotting pass is then a cache-hit and runs in seconds.")
    args = p.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    shrink = args.shrinkage if args.shrinkage == "auto" else float(args.shrinkage)
    e_seed = args.panel_e_seed if args.panel_e_seed > 0 else args.seeds[0]

    if args.cache_only:
        for kind in args.kinds:
            for seed in args.seeds:
                print(f"[{kind}_s{seed}] {args.n_episodes} episodes (cache only) ...")
                rollout(kind, seed, args)
        print("  cache_only: done")
        return

    n_lap = args.n_lap_plot
    cm_by_kind, e_cand, report = {}, {}, {}
    ts_used = 0
    for kind in args.kinds:
        cms, bals, chances, ns = [], [], [], 0
        for seed in args.seeds:
            print(f"[{kind}_s{seed}] {args.n_episodes} episodes ...")
            d = rollout(kind, seed, args)
            per_t = []
            for t in range(args.t_start, d["T"], args.t_step):
                s = timestep_samples(d, t, args.min_per_class, n_lap)
                if s is None:
                    continue
                X, y, keep = s
                cm, bal = cv_confusion(X, y, labels=list(range(n_lap)))
                if cm is None:
                    continue
                per_t.append(dict(t=t, cm=cm, bal=bal,
                                  chance=100 / len(keep), n=len(y)))
                if seed == e_seed:
                    e_cand.setdefault(kind, {})[t] = (X, y, keep)
            if not per_t:
                print("    no qualifying timestep"); continue
            cms.append(sum(r["cm"] for r in per_t))
            bals += [r["bal"] for r in per_t]
            chances += [r["chance"] for r in per_t]
            ns += sum(r["n"] for r in per_t)
            ts_used = max(ts_used, len(per_t))
            print(f"    {len(per_t)} timesteps, balanced acc "
                  f"{np.mean([r['bal'] for r in per_t]):.2f}% "
                  f"(chance {np.mean([r['chance'] for r in per_t]):.2f}%)")
        if not cms:
            continue
        CM = sum(cms)
        cm_by_kind[kind] = (CM, float(np.mean(bals)), float(np.mean(chances)), ns)
        report[kind] = dict(balanced_acc_pct=float(np.mean(bals)),
                            chance_pct=float(np.mean(chances)),
                            n_states=int(ns), n_timesteps=len(bals),
                            confusion=CM.tolist())

    if cm_by_kind:
        panel_d(cm_by_kind, n_lap, f"{args.save_dir}/fig5d_lap_confusion.png",
                ts_used)

    if e_cand:
        shared = set.intersection(*(set(v) for v in e_cand.values()))
        if args.panel_e_t in shared:
            t_e = args.panel_e_t
        else:
            ref = e_cand[list(e_cand)[0]]
            t_e = max(shared, key=lambda t: (len(ref[t][2]), len(ref[t][1])))
        per_kind = {k: e_cand[k][t_e] for k in e_cand}
        for mode in args.proj:
            name = ("fig5e_lap_discriminant" if mode == "lda"
                    else "fig5e_lap_pca_boundary")
            report.setdefault("panel_e", {})[mode] = panel_e(
                per_kind, f"{args.save_dir}/{name}.png", t_e, mode, shrink,
                args.test_size, f"seed {e_seed}")
        report.setdefault("panel_e", {})["t"] = int(t_e)

    with open(f"{args.save_dir}/fig5de_summary.json", "w") as f:
        json.dump(dict(config=vars(args), results=report), f, indent=2)
    print(f"  wrote {args.save_dir}/fig5de_summary.json")

if __name__ == "__main__":
    main()

