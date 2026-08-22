"""EXTRACT -- exp5b lap-count sweep: count accuracy and VP versus K.

Reads the merged sweep produced by eval_lapcount_sweep.py (45 cells: 3 seeds x
K = 2..16, 150 episodes each) and flattens it to one CSV, so plot_lapcount.py
needs neither the per-cell JSONs under figures/exp5_random/sweep_cells/ nor the
checkpoints they were rolled out from.

  figures/exp5_random/lapcount_sweep.json  ->  data/exp5b_lapcount_sweep.csv

Checks, not assumptions: the grid is asserted complete (no missing cells, 3
seeds x 15 K), and the per-K means recomputed here are asserted against the ones
the original run wrote to figures/exp5_random/lapcount_performance.csv.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import ROOT, data_path

SRC = os.path.join(ROOT, "figures/exp5_random/lapcount_sweep.json")
REF = os.path.join(ROOT, "figures/exp5_random/lapcount_performance.csv")

COLS = ["seed", "K", "kind", "n_episodes", "count_acc", "vp", "abs_count_err",
        "hit_rate", "miss_rate", "fa_per_episode", "fa_per_lap",
        "extra_per_episode", "mean_pred_count", "std_pred_count",
        "mean_episode_len"]

def main():
    with open(SRC) as f:
        blob = json.load(f)
    rows = blob["rows"]

    assert not blob["missing_cells"], f"incomplete sweep: {blob['missing_cells']}"
    seeds = sorted({r["seed"] for r in rows})
    Ks = sorted({r["K"] for r in rows})
    assert len(rows) == len(seeds) * len(Ks) == 45, \
        f"expected 3 seeds x 15 K = 45 cells, got {len(rows)}"

    out = data_path("exp5b_lapcount_sweep.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for r in sorted(rows, key=lambda r: (r["seed"], r["K"])):
            w.writerow([r[c] for c in COLS])
    print(f"  wrote {os.path.basename(out)} ({len(rows)} rows, "
          f"seeds {seeds}, K {Ks[0]}..{Ks[-1]})")

    ref = {int(d["K"]): d for d in csv.DictReader(open(REF))}
    for K in Ks:
        ca = [r["count_acc"] for r in rows if r["K"] == K]
        vp = [r["vp"] for r in rows if r["K"] == K]
        for key, got in (("count_acc_mean", np.mean(ca)), ("count_acc_sd", np.std(ca)),
                         ("vp_mean", np.mean(vp)), ("vp_sd", np.std(vp))):
            want = float(ref[K][key])
            assert abs(got - want) < 5e-3, f"K={K} {key}: {got} vs run's {want}"
    print(f"  checked {len(Ks)} K x 4 summary stats against "
          f"figures/exp5_random/lapcount_performance.csv")

if __name__ == "__main__":
    main()

