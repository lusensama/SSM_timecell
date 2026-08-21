"""EXTRACT -- Experiment 3: the systematic retiming grid.

Answers R2.2 and R2.4.  32 (D_train, D_test) cells x 2 initializations x 10 base
seeds x 3 retiming seeds = 1,920 retiming runs.

Reads:
  figures/exp3/exp3_retime_grid_units.csv   per-unit (base_seed x retime_seed)
  figures/exp3/exp3_retime_grid.csv         the aggregate the collector already wrote
  figures/exp1/fig4b_matched.csv            the 30->{30..100} curve of Fig. 4b

Writes into figures/response/data/:
  exp3_retime_units.csv       every run (the datapoints), verbatim copy
  exp3_retime_basemodel.csv   retiming seeds averaged within each base model (n=10)
  exp3_retime_cells.csv       per-cell mean/s.d. over base models, gap, Welch +
                              Mann-Whitney p, and a one-sample test against chance
  exp3_named_cells.csv        the seven cells quoted in the R2.2 table
  exp3_asymmetry.csv          matched-on-D_test vs matched-on-ratio comparison (R2.2)
  fig4b_matched.csv           the superseded-figure replacement curve, copied

All p-values are computed on the 10 independent base-model means, so the three
retiming seeds of one base model are never treated as independent.
"""
import os
import csv
import itertools
import numpy as np
from scipy import stats

from common import ROOT, CHANCE_3STIM, write_csv, r

UNITS = os.path.join(ROOT, "figures", "exp3", "exp3_retime_grid_units.csv")
FIG4B = os.path.join(ROOT, "figures", "exp1", "fig4b_matched.csv")

MODES = ["hippo", "rand_complex"]
PHASES = ["transfer", "recovery"]

NAMED = [(30, 60), (30, 100), (30, 150), (30, 200),
         (60, 100), (60, 150), (60, 200)]

def load_units():
    rows = []
    with open(UNITS) as f:
        for row in csv.DictReader(f):
            rows.append(dict(
                D_train=int(row["D_train"]), D_test=int(row["D_test"]),
                mode=row["mode"], base_seed=int(row["base_seed"]),
                retime_seed=int(row["retime_seed"]),
                transfer=float(row["transfer"]), recovery=float(row["recovery"])))
    return rows

def base_model_means(units):
    """{(D_train, D_test, mode, phase): {base_seed: mean over retiming seeds}}"""
    acc = {}
    for u in units:
        for ph in PHASES:
            acc.setdefault((u["D_train"], u["D_test"], u["mode"], ph), {}) \
               .setdefault(u["base_seed"], []).append(u[ph])
    return {k: {s: float(np.mean(v)) for s, v in d.items()} for k, d in acc.items()}

def main():
    print("[exp3] retiming grid")
    units = load_units()
    write_csv("exp3_retime_units.csv",
              ["D_train", "D_test", "ratio", "mode", "base_seed", "retime_seed",
               "zeroshot_transfer_pct", "readout_only_recovery_pct"],
              [[u["D_train"], u["D_test"], r(u["D_test"] / u["D_train"], 4),
                u["mode"], u["base_seed"], u["retime_seed"],
                u["transfer"], u["recovery"]] for u in sorted(
                   units, key=lambda x: (x["D_train"], x["D_test"], x["mode"],
                                         x["base_seed"], x["retime_seed"]))])

    bm = base_model_means(units)

    rows = []
    for (dtr, dte, mode, ph), d in sorted(bm.items()):
        for s, v in sorted(d.items()):
            rows.append([dtr, dte, r(dte / dtr, 4), mode, ph, s, r(v, 3),
                         len([u for u in units if u["D_train"] == dtr
                              and u["D_test"] == dte and u["mode"] == mode
                              and u["base_seed"] == s])])
    write_csv("exp3_retime_basemodel.csv",
              ["D_train", "D_test", "ratio", "mode", "phase", "base_seed",
               "acc_pct_mean_over_retime_seeds", "n_retime_seeds"], rows)

    cells = sorted({(k[0], k[1]) for k in bm})
    crows = []
    for dtr, dte in cells:
        for ph in PHASES:
            h = np.array([bm[(dtr, dte, "hippo", ph)][s]
                          for s in sorted(bm.get((dtr, dte, "hippo", ph), {}))])
            g = np.array([bm[(dtr, dte, "rand_complex", ph)][s]
                          for s in sorted(bm.get((dtr, dte, "rand_complex", ph), {}))])
            if len(h) == 0 or len(g) == 0:
                continue
            tw, pw = stats.ttest_ind(h, g, equal_var=False)
            try:
                _, pmw = stats.mannwhitneyu(h, g, alternative="two-sided")
            except ValueError:
                pmw = np.nan
            _, ph_chance = stats.ttest_1samp(h, CHANCE_3STIM)
            _, pg_chance = stats.ttest_1samp(g, CHANCE_3STIM)
            crows.append([
                dtr, dte, r(dte / dtr, 4), ph, len(h),
                r(h.mean(), 2), r(h.std(ddof=1), 2),
                r(g.mean(), 2), r(g.std(ddof=1), 2),
                r(h.mean() - g.mean(), 2),
                r(tw, 3), r(pw, 5), r(pmw, 5),
                r(ph_chance, 5), r(pg_chance, 5),
                r(CHANCE_3STIM, 2)])
    write_csv("exp3_retime_cells.csv",
              ["D_train", "D_test", "ratio", "phase", "n_base_models",
               "hippo_mean", "hippo_sd", "rand_mean", "rand_sd", "gap_hippo_minus_rand",
               "welch_t", "welch_p", "mannwhitney_p",
               "hippo_vs_chance_p", "rand_vs_chance_p", "chance_pct"], crows)

    idx = {(int(c[0]), int(c[1]), c[3]): c for c in crows}
    nrows = []
    for dtr, dte in NAMED:
        for ph in PHASES:
            c = idx.get((dtr, dte, ph))
            if c:
                nrows.append([f"{dtr} -> {dte}"] + c)
    write_csv("exp3_named_cells.csv",
              ["cell", "D_train", "D_test", "ratio", "phase", "n_base_models",
               "hippo_mean", "hippo_sd", "rand_mean", "rand_sd", "gap_hippo_minus_rand",
               "welch_t", "welch_p", "mannwhitney_p",
               "hippo_vs_chance_p", "rand_vs_chance_p", "chance_pct"], nrows)

    GAP_COL = 9

    def gap(dtr, dte, ph="recovery"):
        c = idx.get((dtr, dte, ph))
        return float(c[GAP_COL]) if c else None

    arows = []
    for dte in (100, 150, 200):
        arows.append(["matched_on_D_test", f"D_test={dte}", 30, dte,
                      r(dte / 30, 3), r(gap(30, dte), 2), 60, dte, r(dte / 60, 3),
                      r(gap(60, dte), 2)])
    for ratio, (a, b) in {2.0: ((30, 60), (60, 120)),
                          3.0: ((30, 90), (60, 180))}.items():
        arows.append(["matched_on_ratio", f"ratio={ratio}", a[0], a[1], ratio,
                      r(gap(*a), 2), b[0], b[1], ratio, r(gap(*b), 2)])
    write_csv("exp3_asymmetry.csv",
              ["comparison", "matched_at", "A_D_train", "A_D_test", "A_ratio",
               "A_gap", "B_D_train", "B_D_test", "B_ratio", "B_gap"], arows)

    with open(FIG4B) as f:
        f4 = list(csv.DictReader(f))
    write_csv("fig4b_matched.csv", list(f4[0].keys()),
              [list(row.values()) for row in f4])

    print("  R2.2 table check (readout-only recovery):")
    for dtr, dte in NAMED:
        c = idx[(dtr, dte, "recovery")]
        print(f"    {dtr:3d} -> {dte:3d}   hippo {c[5]:5.1f} +/- {c[6]:4.1f}   "
              f"rand {c[7]:5.1f} +/- {c[8]:4.1f}   gap {c[9]:+5.1f}   p={c[11]}")

if __name__ == "__main__":
    main()

