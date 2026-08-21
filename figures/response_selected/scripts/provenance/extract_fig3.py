"""EXTRACT -- manuscript Figure 3b/3c: per-element relative parameter change.

Reproduces the computation in analysis_plot/visualization_of_relative_change.py
exactly -- `load_complex_series` then `compute_relative_change`, i.e. the
ELEMENTWISE ratio

    |theta_final - theta_init| / |theta_init|

over every entry of Lambda (50), Lambda_bar (50), B (50 x 3) and C_tilde
(50 x 50) -- and runs it over the whole 10-seed exp1 cohort instead of the
single pair of runs (seeds 200 / 201) the submitted figure used.

This is the quantity the submitted Fig. 3b histograms.  It is NOT the same as
the per-run scalar `plasticity` field in exp1_summary.jsonl, which is a ratio of
whole-matrix norms; that one is in exp1_ladder_perseed.csv.

Reads:
  training/exp1/{hippo,rand_complex}_s<seed>/base/ssm_*_delay30/
      {initial,final}_{lambda,lambda_bar,B,C_tilde}_<seed>.pt

Writes into figures/response/data/:
  fig3_relative_change.npz       pooled per-element ratios, per (parameter, mode)
  fig3_relative_change.csv       the same, long format (every element, every seed)
  fig3_relative_change_stats.csv min / mean / median / max / n, pooled and per seed
"""
import os
import glob
import numpy as np

from common import ROOT, write_csv, data_path, r

T = os.path.join(ROOT, "training", "exp1")
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 72]
MODES = ["hippo", "rand_complex"]

PARAMS = [
    ("lambda",     "Lambda",     "|dlambda| / |lambda_init|"),
    ("lambda_bar", "Lambda_bar", "|dlambda_bar| / |lambda_bar_init|"),
    ("B",          "B",          "|dB| / |B_init|"),
    ("C_tilde",    "C_tilde",    "|dC| / |C_init|"),
]

SUBMITTED = {("hippo", "Lambda"): 0.37, ("rand_complex", "Lambda"): 1.60,
             ("hippo", "B"): 0.83, ("rand_complex", "B"): 2.16}

def load_complex_series(pt_path):
    """Verbatim behaviour of visualization_of_relative_change.py's loader.

    The exp1 checkpoints store one complex tensor per file inside a dict, which
    is the `isinstance(obj, dict)` branch of the original: take the first tensor
    and ravel it.
    """
    import torch
    obj = torch.load(pt_path, map_location="cpu")
    arr = None
    if torch.is_tensor(obj):
        arr = obj.detach().cpu().numpy()
    elif isinstance(obj, dict):
        for v in obj.values():
            if torch.is_tensor(v):
                arr = v.detach().cpu().numpy()
                break
    if arr is None:
        raise ValueError(f"cannot parse {pt_path}")
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        return arr.astype(np.complex128).ravel()
    return arr.astype(np.float64).ravel() + 0j

def compute_relative_change(initial, final):
    """Verbatim from visualization_of_relative_change.py:62."""
    size = min(initial.size, final.size)
    deltas = np.abs(final[:size] - initial[:size])
    init_mag = np.abs(initial[:size])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(init_mag > 0, deltas / init_mag, np.inf)

def run_dir(mode, seed):
    hits = glob.glob(os.path.join(T, f"{mode}_s{seed}", "base", "ssm_*_delay30"))
    return hits[0] if hits else None

def main():
    print("[fig3] per-element relative parameter change")
    arrays, long_rows, stat_rows = {}, [], []

    for stem, plab, symbol in PARAMS:
        for mode in MODES:
            pooled, per_seed = [], {}
            for seed in SEEDS:
                d = run_dir(mode, seed)
                if d is None:
                    continue
                pi = os.path.join(d, f"initial_{stem}_{seed}.pt")
                pf = os.path.join(d, f"final_{stem}_{seed}.pt")
                if not (os.path.isfile(pi) and os.path.isfile(pf)):
                    print(f"  ! missing {mode} s{seed} {stem}")
                    continue
                rel = compute_relative_change(load_complex_series(pi),
                                              load_complex_series(pf))
                per_seed[seed] = rel
                pooled.append(rel)
                for i, v in enumerate(rel):
                    long_rows.append([plab, mode, seed, i,
                                      "" if not np.isfinite(v) else round(float(v), 6)])
            if not pooled:
                continue
            allv = np.concatenate(pooled)
            arrays[f"{plab}__{mode}"] = allv.astype(np.float32)
            fin = allv[np.isfinite(allv)]
            stat_rows.append([plab, symbol, mode, "pooled", len(per_seed), fin.size,
                              r(fin.min(), 6), r(fin.mean(), 6),
                              r(np.median(fin), 6), r(fin.max(), 6),
                              SUBMITTED.get((mode, plab), "")])
            for seed, rel in sorted(per_seed.items()):
                f = rel[np.isfinite(rel)]
                stat_rows.append([plab, symbol, mode, f"seed {seed}", 1, f.size,
                                  r(f.min(), 6), r(f.mean(), 6),
                                  r(np.median(f), 6), r(f.max(), 6), ""])

    np.savez_compressed(data_path("fig3_relative_change.npz"), **arrays)
    print(f"  wrote data/fig3_relative_change.npz  ({len(arrays)} pooled arrays)")
    write_csv("fig3_relative_change.csv",
              ["parameter", "mode", "seed", "element_index", "relative_change"],
              long_rows)
    write_csv("fig3_relative_change_stats.csv",
              ["parameter", "symbol", "mode", "scope", "n_seeds", "n_elements",
               "min", "mean", "median", "max", "submitted_single_run"], stat_rows)

    print("  pooled over 10 seeds, elementwise (the submitted Fig. 3b quantity):")
    for row in stat_rows:
        if row[3] != "pooled":
            continue
        sub = f"   submitted {row[10]}" if row[10] != "" else ""
        print(f"    {row[0]:11s} {row[2]:13s} mean {row[7]:8.4f}  "
              f"median {row[8]:8.4f}  n = {row[5]:6d}{sub}")

if __name__ == "__main__":
    main()

