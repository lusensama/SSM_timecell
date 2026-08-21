"""EXTRACT -- Fig. 5d / 5e (lap identity under randomised lap timing).

This is the one figure pair in this folder whose upstream artifact is not a CSV:
analysis_plot/plot_fig5de_lap_svm.py rolls the exp5_random checkpoints out over
5,000 episodes and caches ~1 GB of 80-D state per (kind, seed) under
figures/exp5_random/fig5de_n5000/_cache/.  That cache is far too large to carry
here, so this script boils it down to the two small tables the panels actually
need:

  fig5de_confusion.csv    the pooled 4x4 lap confusion matrix (counts), lifted
                          straight from the run's fig5de_summary.json -- these
                          are cross-validated predictions pooled over 3 seeds x
                          31 exact timesteps, 108,030 states.
  fig5de_projection.csv   panel e: every plotted point.  The 80-D state at the
                          single exact timestep t = 90, seed 2, projected to 2-D
                          by a basis fit on the TRAIN half; both halves are
                          written out (split = train | test) so the decision
                          regions can be refit downstream exactly as drawn.
  fig5de_meta.json        accuracies, chance levels, axis names, sample counts.

The panel-d subtitle counts DISTINCT timesteps per seed (11), not the 31
seed-timestep decodes that summary.json's n_timesteps records; that number is
recovered here by replaying the qualifying-timestep test on lap_t alone, which
np.load reads lazily and so costs nothing.

Reproduction is checked, not assumed: the balanced accuracies recomputed here
are asserted against the ones the original run wrote to fig5de_summary.json.

Needs the run tree (the 6.7 GB rollout cache).  Nothing downstream of data/
does.

  python extract_fig5de.py
"""
import json
import os
import sys

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

HERE = os.path.dirname(os.path.abspath(__file__))
SEL = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(SEL, "data")
ROOT = os.path.dirname(os.path.dirname(SEL))

RUN = os.path.join(ROOT, "figures/exp5_random/fig5de_n5000_trained")
CACHE = os.path.join(ROOT, "figures/exp5_random/fig5de_n5000/_cache")

def clf():
    """Linear multiclass decoder, ONE-VS-ONE.  Must not be LinearSVC: the lap
    code is a single monotonically ordered axis, so a one-vs-rest argmax
    swallows the middle classes (54% vs 97% on identical data)."""
    return make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))

def timestep_samples(d, t, min_per_class, n_lap):
    """States at exact absolute timestep t, labelled by lap index."""
    lap = d["lap_t"][:, t]
    m = (lap >= 0) & (lap < n_lap)
    X, y = d["Xt"][m, t], lap[m]
    keep = [c for c in range(n_lap) if (y == c).sum() >= min_per_class]
    if len(keep) < 2:
        return None
    m2 = np.isin(y, keep)
    return X[m2], y[m2], keep

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

def _cache_path(cfg, kind, seed):
    """The rollout cache file plot_fig5de_lap_svm.rollout() would have written."""
    tag = (f"{kind}_s{seed}_n{cfg['n_episodes']}_k{cfg['k_range'][0]}-"
           f"{cfg['k_range'][1]}_l{cfg['lap_len_range'][0]}-"
           f"{cfg['lap_len_range'][1]}_p{cfg['pause_count_range'][1]}-"
           f"{cfg['pause_len_range'][1]}_L{cfg['n_lap_plot']}_P{cfg['n_phase']}")
    path = os.path.join(CACHE, tag + ".npz")
    if not os.path.isfile(path):
        sys.exit(f"missing rollout cache {path}")
    return path

def main():
    summ_path = os.path.join(RUN, "fig5de_summary.json")
    if not os.path.isfile(summ_path):
        sys.exit(f"missing {summ_path} -- run analysis_plot/plot_fig5de_lap_svm.py")
    S = json.load(open(summ_path))
    cfg, res = S["config"], S["results"]
    kind = cfg["kinds"][0]
    n_lap = cfg["n_lap_plot"]
    t_e = res["panel_e"]["t"]
    e_seed = cfg["panel_e_seed"] if cfg["panel_e_seed"] > 0 else cfg["seeds"][0]

    cm = np.array(res[kind]["confusion"], dtype=int)
    with open(os.path.join(DATA, "fig5de_confusion.csv"), "w") as f:
        f.write("kind,true_lap,predicted_lap,count,row_pct\n")
        for i in range(cm.shape[0]):
            tot = max(cm[i].sum(), 1)
            for j in range(cm.shape[1]):
                f.write(f"{kind},{i + 1},{j + 1},{cm[i, j]},"
                        f"{cm[i, j] / tot * 100:.6f}\n")
    print(f"  fig5de_confusion.csv   {cm.sum():,} states, "
          f"balanced acc {res[kind]['balanced_acc_pct']:.2f}%")

    per_seed = []
    for seed in cfg["seeds"]:
        z = np.load(_cache_path(cfg, kind, seed))
        lap, T = z["lap_t"], int(z["T"])
        n = 0
        for t in range(cfg["t_start"], T, cfg["t_step"]):
            l = lap[:, t]
            y = l[(l >= 0) & (l < n_lap)]
            if sum((y == c).sum() >= cfg["min_per_class"]
                   for c in range(n_lap)) >= 2:
                n += 1
        per_seed.append(n)
    ts_used = max(per_seed)
    assert sum(per_seed) == res[kind]["n_timesteps"], (per_seed,
                                                       res[kind]["n_timesteps"])
    print(f"  timesteps per seed {per_seed} -> {ts_used} distinct, "
          f"{sum(per_seed)} seed-timestep decodes")

    path = _cache_path(cfg, kind, e_seed)
    print(f"  loading {path} ...")
    z = np.load(path)
    d = dict(Xt=z["Xt"], lap_t=z["lap_t"], T=int(z["T"]))

    X, y, keep = timestep_samples(d, t_e, cfg["min_per_class"], n_lap)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=cfg["test_size"],
                                 random_state=0)
    tr, te = next(sss.split(X, y))

    rows, meta = [], {}
    shrink = (cfg["shrinkage"] if cfg["shrinkage"] == "auto"
              else float(cfg["shrinkage"]))
    for mode in cfg["proj"]:
        proj, names = fit_projection(X[tr], y[tr], mode, shrink)
        Ptr, Pte = proj(X[tr]), proj(X[te])
        m = clf().fit(Ptr, y[tr])
        acc = balanced_accuracy_score(y[te], m.predict(Pte)) * 100
        ref = res["panel_e"][mode][kind]["heldout_balanced_acc_pct"]
        assert abs(acc - ref) < 1e-6, f"{mode}: {acc} != {ref} (run's value)"
        for split, P, idx in (("train", Ptr, tr), ("test", Pte, te)):
            for k in range(len(idx)):
                rows.append((kind, mode, split, int(y[idx[k]]) + 1,
                             P[k, 0], P[k, 1]))
        meta[mode] = dict(axes=list(names),
                          heldout_balanced_acc_pct=float(acc),
                          chance_pct=100.0 / len(keep),
                          n_train=int(len(tr)), n_test=int(len(te)))
        print(f"  panel e / {mode:3s}  held-out balanced acc {acc:.2f}% "
              f"(run: {ref:.2f}%)  axes = {names}")

    with open(os.path.join(DATA, "fig5de_projection.csv"), "w") as f:
        f.write("kind,projection,split,lap,dim1,dim2\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.17g},{r[5]:.17g}\n")
    print(f"  fig5de_projection.csv  {len(rows):,} points")

    meta.update(kind=kind, n_lap=n_lap, laps_present=[c + 1 for c in keep],
                panel_e_t=int(t_e), panel_e_seed=int(e_seed),
                confusion_balanced_acc_pct=res[kind]["balanced_acc_pct"],
                confusion_chance_pct=res[kind]["chance_pct"],
                confusion_n_states=res[kind]["n_states"],
                confusion_timesteps_per_seed=int(ts_used),
                confusion_n_seed_timestep_decodes=res[kind]["n_timesteps"],
                seeds=cfg["seeds"], n_episodes=cfg["n_episodes"],
                source=os.path.relpath(RUN, ROOT))
    with open(os.path.join(DATA, "fig5de_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("  fig5de_meta.json")

if __name__ == "__main__":
    main()

