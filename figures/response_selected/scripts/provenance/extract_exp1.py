"""EXTRACT -- Experiment 1: the nine-initialization ladder.

Answers R2.1 / R2.5 / R3.1.

Reads (via analysis_plot/exp1_sources.py, which owns the source routing:
s4d_inv comes from training/exp1_s4d_inv_fixedN/, hippo+rand_complex delay-100
columns come from the --fixed_delay rerun):

  training/exp1/exp1_summary.jsonl                      scalar metrics per (mode,seed)
  training/exp1/exp1_delay100_fixed_summary.jsonl       honest delay-100 columns
  training/exp1_s4d_inv_fixedN/exp1_summary.jsonl       corrected s4d_inv cohort
  training/<root>/<mode>_s<seed>/pretrain_seq.log       cascade metrics at init
  training/<root>/<mode>_s<seed>/states_trained_d30.log cascade metrics after training
  training/<root>/<mode>_s<seed>/states/*.npz           cached delay-period activity
  training/exp1/<mode>_s<seed>/base.log                 training curves

Writes into figures/response/data/:
  exp1_ladder_summary.csv        per-mode means +/- s.d. over 10 seeds
  exp1_ladder_perseed.csv        every seed's raw numbers (the datapoints)
  exp1_property_verdicts.csv     the property-by-property table of R3.1 (a)-(f)
  exp1_heatmap_matrices.csv      sorted heatmaps, long format, both orderings
  exp1_heatmap_matrices.npz      same, as arrays (what the plotter reads)
  exp1_training_curves.csv       eval accuracy vs episode, per (mode, seed)
  fig3b_param_change.csv         the manuscript Fig. 3b comparison, re-measured
                                 over 10 seeds: |dtheta|/|theta_init| for Lambda
                                 (A), B and C, HiPPO vs random, with Welch tests
                                 and the submitted single-run values alongside
"""
import os
import re
import sys
import glob
import numpy as np

from scipy import stats

from common import (ROOT, MODE_ORDER, write_csv, data_path, r,
                    sort_freq_resp)

sys.path.insert(0, os.path.join(ROOT, "analysis_plot"))
import exp1_sources as S

def sort_resp(total_resp, norm=True):
    """Verbatim reimplementation of utils.utils_analysis.sort_resp.

    Inlined rather than imported: utils_analysis pulls in sklearn, whose build in
    this environment is ABI-incompatible with the installed numpy, and the
    ordering used by the paper's heatmaps is pure numpy.  Trial-average, per-cell
    min-max normalize to [-1, 1], sort cells by argmax time.
    """
    np.seterr(divide="ignore", invalid="ignore")
    n_neurons = np.shape(total_resp)[2]
    segments = np.moveaxis(total_resp, 0, 1)
    unsorted_matrix = np.zeros((n_neurons, len(segments)))
    sorted_matrix = np.zeros((n_neurons, len(segments)))
    normalized_matrix = unsorted_matrix
    for i in range(len(segments)):
        unsorted_matrix[:, i] = np.transpose(np.mean(segments[i], axis=0))
        if norm is True:
            scaled = (unsorted_matrix - np.min(unsorted_matrix, axis=1, keepdims=True)) \
                / np.ptp(unsorted_matrix, axis=1, keepdims=True)
            normalized_matrix = scaled * 2 - 1
            cell_nums = np.argsort(np.argmax(normalized_matrix, axis=1))
            for j, i_cell in enumerate(list(cell_nums)):
                sorted_matrix[j] = normalized_matrix[i_cell]
        else:
            cell_nums = np.argsort(np.argmax(unsorted_matrix, axis=1))
            for j, i_cell in enumerate(list(cell_nums)):
                sorted_matrix[j] = unsorted_matrix[i_cell]
    return cell_nums, sorted_matrix, normalized_matrix

OUTROOT = os.path.join(ROOT, "training", "exp1")
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 72]

SCALARS = [("pretrain_seq", "pretrain_seq"),
           ("base_acc", "base_acc30"),
           ("freeze_ab_acc", "freeze_ab"),
           ("transfer_acc", "transfer100"),
           ("retimed_acc", "retimed100")]

PHASES = [("initial_d30", "initial_d30"),
          ("trained_d30", "trained_d30"),
          ("retimed_d100", "retimed_d100")]

PROPERTIES = [
    ("a", "Optimal history compression", "NOT responsible",
     "spectrum_matched,freq_matched",
     "Destroying the Legendre construction (spectrum_matched) or randomizing the "
     "decay rates (freq_matched) reproduces HiPPO on both accuracy and coverage."),
    ("b", "Eigenvalue spectrum (imaginary part)", "SUFFICIENT",
     "freq_matched,s4d_inv",
     "HiPPO's real parts are all equal, so freq_matched isolates the frequencies; "
     "S4D-Inv approximates only the spectrum asymptotics and lands on HiPPO."),
    ("c", "Multiple timescales", "NECESSARY BUT NOT SUFFICIENT",
     "real_diagonal",
     "Multi-timescale by construction but non-rotational: coverage 0.033, all "
     "active units peak in one of the 30 bins, peak-time entropy exactly 0."),
    ("d", "Complex rotations", "NECESSARY",
     "real_diagonal,rand_complex",
     "Zeroing the imaginary parts destroys the cascade (0.033 vs 0.457) and "
     "250,000 episodes of training cannot rebuild it (0.082)."),
    ("e", "Orthogonality / frequency-to-input pairing", "NOT responsible",
     "spectrum_matched",
     "Breaking the pairing between rotation frequency and input channel while "
     "holding B at its HiPPO value changes neither cascade nor accuracy."),
    ("f", "Presence of a structured temporal basis", "NOT sufficient",
     "alt_basis",
     "Strongest cascade at initialization (0.702) yet cannot learn the task "
     "(65.6 +/- 12.5%, 2 of 10 seeds at chance)."),
]

_EVAL = re.compile(r"Eval @ episode (\d+): ([\d.]+)%")

SUBMITTED_FIG3B = {("hippo", "plast_Lambda"): 0.37, ("hippo", "plast_B"): 0.83,
                   ("rand_complex", "plast_Lambda"): 1.60,
                   ("rand_complex", "plast_B"): 2.16}
PARAMS = [("plast_Lambda", "Lambda (A)"), ("plast_B", "B"), ("plast_C", "C")]

def _state_file(mode, seed, phase):
    """Cached activity npz for (mode, seed, phase); honest protocol preferred."""
    root = S.root_for(mode, OUTROOT)
    d = os.path.join(root, f"{mode}_s{seed}", "states")
    if phase == "retimed_d100":
        p = os.path.join(d, "retimed_d100_fixedDelay.npz")
        if os.path.isfile(p):
            return p
    p = os.path.join(d, f"{phase}.npz")
    return p if os.path.isfile(p) else None

def per_seed_rows(recs, cov_init, cov_trained):
    rows = []
    for m in MODE_ORDER:
        for s in SEEDS:
            rec = recs.get((m, s))
            if rec is None:
                continue
            row = [m, s]
            for key, _ in SCALARS:
                row.append(r(rec.get(key), 4))
            ci = cov_init.get(m, {}).get(s)
            ct = cov_trained.get(m, {}).get(s)
            row += [r(ci[0] if ci is not None else None),
                    r(ct[0] if ct is not None else None),
                    r(ci[2] if ci is not None else None),
                    r(ct[2] if ct is not None else None),
                    r(ct[1] if ct is not None else None),
                    r(ci[4] if ci is not None else None, 1),
                    r(ct[4] if ct is not None else None, 1)]
            pl = rec.get("plasticity") or {}
            row += [r(pl.get("Lambda")), r(pl.get("B")), r(pl.get("C"))]
            row.append(S.root_for(m, "training/exp1"))
            rows.append(row)
    return rows

def summarise(perseed_rows, header):
    """Collapse the per-seed table to mean/std/n per mode."""
    idx = {h: i for i, h in enumerate(header)}
    num_cols = [h for h in header if h not in ("mode", "seed", "source_root")]
    by = {}
    for row in perseed_rows:
        by.setdefault(row[0], []).append(row)
    out = []
    for m in MODE_ORDER:
        if m not in by:
            continue
        rows = by[m]
        rec = [m, len(rows)]
        for h in num_cols:
            vals = [row[idx[h]] for row in rows if row[idx[h]] != ""]
            vals = np.array(vals, dtype=float)
            if len(vals) == 0:
                rec += ["", "", 0]
            else:
                rec += [r(vals.mean()), r(vals.std(ddof=0)), len(vals)]
        out.append(rec)
    header_out = ["mode", "n_seeds"]
    for h in num_cols:
        header_out += [f"{h}_mean", f"{h}_std", f"{h}_n"]
    return header_out, out

def main():
    print("[exp1] initialization ladder")
    recs = S.load_records(OUTROOT)
    if not recs:
        raise SystemExit(f"no exp1 records under {OUTROOT}")
    if not S.delay100_ok(recs):
        raise SystemExit("an artifact-protocol delay-100 value survived; refusing")

    cov_init = {m: S.scrape_seq_by_seed(m, "pretrain_seq.log", OUTROOT)
                for m in MODE_ORDER}
    cov_trained = {m: S.scrape_seq_by_seed(m, "states_trained_d30.log", OUTROOT)
                   for m in MODE_ORDER}

    header = (["mode", "seed"] + [n for _, n in SCALARS]
              + ["cascade_coverage_init", "cascade_coverage_trained",
                 "peak_entropy_init", "peak_entropy_trained", "sparsity_trained",
                 "n_active_init", "n_active_trained",
                 "plast_Lambda", "plast_B", "plast_C", "source_root"])
    rows = per_seed_rows(recs, cov_init, cov_trained)
    write_csv("exp1_ladder_perseed.csv", header, rows)

    sh, srows = summarise(rows, header)
    ci, ct = sh.index("cascade_coverage_init_mean"), sh.index("cascade_coverage_trained_mean")
    sh = sh + ["peak_bins_init_of30", "peak_bins_trained_of30", "outcome_class"]
    from common import MODE_CLASS
    for row in srows:
        row += [r(float(row[ci]) * 30, 1), r(float(row[ct]) * 30, 1),
                MODE_CLASS[row[0]]]
    write_csv("exp1_ladder_summary.csv", sh, srows)

    write_csv("exp1_property_verdicts.csv",
              ["property_id", "property", "verdict", "conditions", "evidence"],
              [list(p) for p in PROPERTIES])

    mats, long_rows = {}, []
    for m in MODE_ORDER:
        for phase, label in PHASES:
            picked = None
            for s in SEEDS:
                p = _state_file(m, s, phase)
                if p:
                    picked, picked_seed = p, s
                    break
            if picked is None:
                print(f"  ! no cached states for {m}/{label}")
                continue
            d = np.load(picked)
            arr = d["delay1"] if "delay1" in d.files else d[d.files[0]]
            _, freq_mat, _, peaks = sort_freq_resp(arr, norm=True)
            _, time_mat, _ = sort_resp(arr, norm=True)
            key = f"{m}__{label}"
            mats[key] = freq_mat.astype(np.float32)
            mats[key + "__timesorted"] = time_mat.astype(np.float32)
            mats[key + "__peakcounts"] = peaks.astype(np.int16)
            mats[key + "__seed"] = np.array(picked_seed)
            for which, M in (("freq", freq_mat), ("time", time_mat)):
                for u in range(M.shape[0]):
                    for t in range(M.shape[1]):
                        long_rows.append([m, label, picked_seed, which, u, t,
                                          int(peaks[u]) if which == "freq" else "",
                                          round(float(M[u, t]), 5)])
    np.savez_compressed(data_path("exp1_heatmap_matrices.npz"), **mats)
    print(f"  wrote data/exp1_heatmap_matrices.npz  "
          f"({sum(1 for k in mats if k.count('__') == 1)} panels x 2 orderings)")
    write_csv("exp1_heatmap_matrices.csv",
              ["mode", "phase", "seed", "sort", "unit_rank", "time_bin",
               "peak_count", "value"], long_rows)

    curves = []
    for m in MODE_ORDER:
        root = S.root_for(m, OUTROOT)
        for s in SEEDS:
            p = os.path.join(root, f"{m}_s{s}", "base.log")
            if not os.path.isfile(p):
                continue
            for ep, acc in _EVAL.findall(open(p, errors="ignore").read()):
                curves.append([m, s, int(ep), float(acc)])
    write_csv("exp1_training_curves.csv",
              ["mode", "seed", "episode", "eval_acc_pct"], curves)

    idx = {h: i for i, h in enumerate(header)}
    f3 = []
    for field, plab in PARAMS:
        vals = {}
        for m in ("hippo", "rand_complex"):
            vals[m] = np.array([float(row[idx[field]]) for row in rows
                                if row[0] == m and row[idx[field]] != ""])
        t, pv = stats.ttest_ind(vals["hippo"], vals["rand_complex"], equal_var=False)
        for m in ("hippo", "rand_complex"):
            v = vals[m]
            f3.append([plab, field, m, len(v), r(v.mean(), 4), r(v.std(ddof=1), 4),
                       r(v.min(), 4), r(v.max(), 4),
                       SUBMITTED_FIG3B.get((m, field), ""),
                       r(t, 3), float(f"{pv:.3g}"),
                       " ".join(f"{x:.4f}" for x in v)])
    write_csv("fig3b_param_change.csv",
              ["parameter", "field", "mode", "n_seeds", "mean", "sd", "min", "max",
               "submitted_single_run", "welch_t_hippo_vs_random",
               "welch_p_hippo_vs_random", "per_seed_values"], f3)

    print("  manuscript Fig. 3b, re-measured (n = 10 per condition):")
    for row in f3:
        sub = f"   submitted {row[8]}" if row[8] != "" else ""
        print(f"    {row[0]:11s} {row[2]:13s} {row[4]:6.4f} +/- {row[5]:.4f}"
              f"   Welch p = {row[10]:.3g}{sub}")

if __name__ == "__main__":
    main()

