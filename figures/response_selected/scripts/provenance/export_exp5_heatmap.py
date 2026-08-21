"""EXPORT (needs checkpoints) -- lap-phase heatmap matrices for R3.3's Fig. 5.

This is the one piece of the response-figure pipeline whose numbers are not
already sitting in a saved artifact: plot_lap_identity.py rendered the heatmaps
straight to PNG and only persisted its scalar summary.  Run this ONCE while the
exp5_random checkpoints are present; it writes the matrices into
figures/response/data/, after which plot_exp5.py needs no model.

The roll-out, phase binning, split-half sort and normalization are imported from
plot_lap_identity.py rather than reimplemented, so the exported matrices are the
same arrays that produced figures/exp5_random/lapid_*.png.

plot_lap_identity imports scikit-learn at module scope for its pairwise SVM.  In
this environment sklearn's compiled extensions are ABI-incompatible with the
installed numpy, and the heatmap path never touches them, so the sklearn names are
stubbed before the import.  If sklearn imports cleanly the stubs are not used.

Ordering follows the manuscript, i.e. utils.utils_analysis.sort_freq_resp, the
same routine ssm_observer_1d.py and basic_lap_state.sort_sub use.  Two variants,
both mirroring plot_landmark_heatmaps.py:

  globalnorm  one sort_freq_resp order for the whole multi-lap span, taken from a
              held-out half of the episodes and applied to the other half, with a
              single normalization across all laps (cross-lap gain survives)
  perlapnorm  the manuscript convention of basic_lap_state.sort_sub: split into
              laps, sort AND renormalize each lap independently, concatenate --
              again with the order from half A and the matrix from half B

Writes:
  data/exp5_heatmap_matrices.npz   global-norm and per-lap-norm matrices
  data/exp5_heatmap_matrices.csv   the same, long format
  data/exp5_lap_selectivity.csv    per-unit between-lap variance fraction

Usage:
  python export_exp5_heatmap.py --seeds 2 --kinds trained untrained --n_episodes 400
"""
import os
import sys
import types
import argparse
import numpy as np

from common import ROOT, write_csv, data_path, sort_freq_resp

def _stub_sklearn():
    """Install no-op sklearn modules if the real ones cannot be imported."""
    try:
        import sklearn.svm
        import sklearn.model_selection
        return False
    except Exception:
        pass

    def _unavailable(*_a, **_k):
        raise RuntimeError("sklearn is stubbed here; this exporter only builds "
                           "heatmap matrices, not the SVM decodes")

    for name, attrs in {
            "sklearn": {},
            "sklearn.svm": {"LinearSVC": _unavailable},
            "sklearn.preprocessing": {"StandardScaler": _unavailable},
            "sklearn.pipeline": {"make_pipeline": _unavailable},
            "sklearn.model_selection": {"cross_val_score": _unavailable,
                                        "StratifiedGroupKFold": _unavailable},
    }.items():
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod
    return True

def _halves(n, seed=0):
    """Split episode indices in two; order comes from A, the matrix from B."""
    idx = np.random.RandomState(seed).permutation(n)
    return idx[:n // 2], idx[n // 2:]

def _norm_rows(mat):
    """Per-unit min-max to [-1, 1], as sort_freq_resp does internally."""
    lo = mat.min(axis=1, keepdims=True)
    rng = np.ptp(mat, axis=1, keepdims=True)
    return (mat - lo) / np.where(rng == 0, 1.0, rng) * 2 - 1

def _global(resp, seed=0):
    """One sort_freq_resp order for the whole span, from a held-out half.

    Mirror of plot_landmark_heatmaps.sorted_matrix_global.  A single
    normalization across all laps, so between-lap gain differences survive.
    """
    a, b = _halves(resp.shape[0], seed)
    order, *_ = sort_freq_resp(resp[a], norm=True)
    return _norm_rows(resp[b].mean(axis=0).T)[order], order

def _perlap(resp, n_seg, seed=0):
    """The manuscript convention: sort AND renormalize each lap independently.

    Mirror of plot_landmark_heatmaps.sorted_matrix_splithalf, which is itself a
    mirror of basic_lap_state.sort_sub -- sort_freq_resp is called on each lap
    segment separately, so each lap gets its own order and its own scaling.
    """
    a, b = _halves(resp.shape[0], seed)
    w = resp.shape[1] // n_seg
    mats, orders = [], []
    for i in range(n_seg):
        order, *_ = sort_freq_resp(resp[a][:, i * w:(i + 1) * w, :], norm=True)
        held = _norm_rows(resp[b][:, i * w:(i + 1) * w, :].mean(axis=0).T)
        mats.append(held[order])
        orders.append(order)
    return np.concatenate(mats, axis=1), np.asarray(orders)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2])
    ap.add_argument("--kinds", nargs="+", default=["trained", "untrained"])
    ap.add_argument("--ckpt_dir", default=os.path.join(ROOT, "training", "exp5_random"))
    ap.add_argument("--n_neurons", type=int, default=80)
    ap.add_argument("--n_episodes", type=int, default=400)
    ap.add_argument("--n_lap_plot", type=int, default=4)
    ap.add_argument("--n_phase", type=int, default=40)
    ap.add_argument("--k_range", type=int, nargs=2, default=[2, 6])
    ap.add_argument("--lap_len_range", type=int, nargs=2, default=[10, 60])
    ap.add_argument("--pause_count_range", type=int, nargs=2, default=[0, 2])
    ap.add_argument("--pause_len_range", type=int, nargs=2, default=[0, 25])
    ap.add_argument("--hit_window", type=int, default=3)
    args = ap.parse_args()

    stubbed = _stub_sklearn()
    if stubbed:
        print("  (sklearn stubbed; heatmap path only)")
    sys.path.insert(0, ROOT)
    import plot_lap_identity as P

    L, Ph = args.n_lap_plot, args.n_phase
    mats, long_rows, sel_rows = {}, [], []
    for kind in args.kinds:
        for seed in args.seeds:
            tag = f"{kind}_s{seed}"
            print(f"[{tag}] rolling out {args.n_episodes} episodes")
            net = P.load(kind, seed, args.n_neurons, args.ckpt_dir)
            d = P.collect(net, args, seed=4000 + seed, oracle=(kind == "untrained"))
            pl = d["per_lap"]
            print(f"    {pl.shape[0]} episodes with K >= {L}")

            resp = pl.reshape(pl.shape[0], L * Ph, pl.shape[3])
            M_glob, order = _global(resp)
            M_lap, per_lap_orders = _perlap(resp, L)
            mean_B = pl[_halves(pl.shape[0])[1]].mean(axis=0)

            panels = {"globalnorm": M_glob, "perlapnorm": M_lap}
            for pname, M in panels.items():
                key = f"{tag}__{pname}"
                mats[key] = M.astype(np.float32)
                for u in range(M.shape[0]):
                    for t in range(M.shape[1]):
                        long_rows.append([kind, seed, pname, u, t,
                                          t // Ph + 1, t % Ph,
                                          round(float(M[u, t]), 5)])
            mats[f"{tag}__order_global"] = np.asarray(order)
            mats[f"{tag}__order_perlap"] = np.asarray(per_lap_orders)
            mats[f"{tag}__n_episodes_used"] = np.asarray(pl.shape[0])

            sel = P.lap_selectivity(mean_B)
            pref = np.argmax(mean_B.mean(axis=1), axis=0)
            for u, v in enumerate(sel):
                sel_rows.append([kind, seed, u, int(pref[u]) + 1, round(float(v), 5)])

    mats["n_lap"] = np.asarray(L)
    mats["n_phase"] = np.asarray(Ph)
    np.savez_compressed(data_path("exp5_heatmap_matrices.npz"), **mats)
    print(f"  wrote data/exp5_heatmap_matrices.npz  ({len(mats)} arrays)")
    write_csv("exp5_heatmap_matrices.csv",
              ["kind", "seed", "panel", "unit_row", "column", "lap", "phase_bin",
               "value"], long_rows)
    write_csv("exp5_lap_selectivity.csv",
              ["kind", "seed", "unit", "preferred_lap", "selectivity"], sel_rows)

if __name__ == "__main__":
    main()

