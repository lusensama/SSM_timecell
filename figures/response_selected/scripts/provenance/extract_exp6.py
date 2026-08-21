"""EXTRACT -- Experiment 6: the LSTM baseline under the 3-stimulus protocol.

Answers R2.3.

Reads:
  training/exp6_lstm_baseline/lstm_s*/base/.../best_eval_acc_*.txt     H = 50
  training/exp6_lstm_matched/lstm_s*/base/.../best_eval_acc_*.txt      H = 32
  training/exp6_lstm_baseline/lstm_s*/retime_curve_matched_d100/rseed*/curve.json
  training/exp1/retime_curve_matched_d100_summary_r*.jsonl   (SSM comparator,
      restricted to base seeds 72/1/2 so the cohorts are matched)
  training/exp6_lstm_*/slurm_*.log                           training curves
  figures/exp6_lstm/*.npz                                    sorted heatmaps
  checkpoints (state_dicts only, for parameter counts and the freeze check)

Writes into figures/response/data/:
  exp6_base_accuracy.csv         per-seed best eval accuracy at delay 30
  exp6_retime_units.csv          9 units per architecture: zero-shot, 2,000 ep, gain
  exp6_retime_curves.csv         accuracy vs retraining episode, every unit
  exp6_retime_stats.csv          Welch tests on per-base-model means (n = 3 vs 3)
  exp6_param_counts.csv          trainable parameters per architecture
  exp6_freeze_check.csv          how far frozen vs trainable blocks actually moved
  exp6_training_curves.csv       eval accuracy vs episode during base training
  exp6_heatmap_matrices.npz      LSTM sorted delay-period heatmaps
  exp6_heatmap_matrices.csv      the same, long format
"""
import os
import re
import glob
import json
import numpy as np
from scipy import stats

from common import ROOT, CHANCE_3STIM, write_csv, data_path, load_jsonl, r

T = os.path.join(ROOT, "training")
LSTM_SEEDS = [72, 1, 2]
_EVAL = re.compile(r"Eval @ episode (\d+): ([\d.]+)%")

COHORTS = [
    ("LSTM H=50", os.path.join(T, "exp6_lstm_baseline"), "lstm_50_*_delay30"),
    ("LSTM H=32", os.path.join(T, "exp6_lstm_matched"), "lstm_32_*_delay30"),
]

def _torch():
    import torch
    return torch

def base_accuracy():
    rows = []
    for label, root, pat in COHORTS:
        for s in LSTM_SEEDS:
            hits = glob.glob(os.path.join(root, f"lstm_s{s}", "base", pat,
                                          f"best_eval_acc_{s}.txt"))
            if hits:
                rows.append([label, "LSTM", s, r(float(open(hits[0]).read()), 2)])
    per = {}
    for m in ("hippo", "rand_complex"):
        for rec in load_jsonl(os.path.join(T, "exp1", "exp1_summary.jsonl")):
            if rec.get("complete") and rec.get("mode") == m:
                per.setdefault(m, {})[int(rec["seed"])] = float(rec["base_acc"])
    for m, label in (("hippo", "SSM HiPPO"), ("rand_complex", "SSM random")):
        for s in LSTM_SEEDS:
            if s in per.get(m, {}):
                rows.append([label, "SSM", s, r(per[m][s], 2)])
    return rows, per

def retiming_units():
    """9 LSTM units + 9 units for each SSM initialization, matched on base seed."""
    rows = []
    for p in sorted(glob.glob(os.path.join(
            T, "exp6_lstm_baseline", "lstm_s*", "retime_curve_matched_d100",
            "rseed*", "curve.json"))):
        d = json.load(open(p))
        pts = {q["episode"]: q["acc"] for q in d["points"]}
        rows.append(["LSTM H=50", int(d["base_seed"]), int(d["retime_seed"]),
                     pts.get(0), pts.get(2000)])
    for p in sorted(glob.glob(os.path.join(
            T, "exp1", "retime_curve_matched_d100_summary_r*.jsonl"))):
        for d in load_jsonl(p):
            if not d.get("complete") or int(d.get("base_seed", -1)) not in LSTM_SEEDS:
                continue
            label = {"hippo": "SSM HiPPO", "rand_complex": "SSM random"}.get(d["mode"])
            if label is None:
                continue
            pts = {q["episode"]: q["acc"] for q in d["points"]}
            rows.append([label, int(d["base_seed"]), int(d["retime_seed"]),
                         pts.get(0), pts.get(2000)])
    return [[a, b, c, r(z, 2), r(f, 2), r(f - z, 2)] for a, b, c, z, f in rows]

def retiming_curves():
    rows = []
    for p in sorted(glob.glob(os.path.join(
            T, "exp6_lstm_baseline", "lstm_s*", "retime_curve_matched_d100",
            "rseed*", "curve.json"))):
        d = json.load(open(p))
        for q in d["points"]:
            rows.append(["LSTM H=50", int(d["base_seed"]), int(d["retime_seed"]),
                         int(q["episode"]), r(q["acc"], 2)])
    for p in sorted(glob.glob(os.path.join(
            T, "exp1", "retime_curve_matched_d100_summary_r*.jsonl"))):
        for d in load_jsonl(p):
            if not d.get("complete") or int(d.get("base_seed", -1)) not in LSTM_SEEDS:
                continue
            label = {"hippo": "SSM HiPPO", "rand_complex": "SSM random"}.get(d["mode"])
            if label is None:
                continue
            for q in d["points"]:
                rows.append([label, int(d["base_seed"]), int(d["retime_seed"]),
                             int(q["episode"]), r(q["acc"], 2)])
    return rows

def retiming_stats(units):
    """Welch t-tests on per-base-model means, so retiming seeds are not pseudo-n."""
    by = {}
    for arch, bs, _rs, zs, fin, gain in units:
        by.setdefault(arch, {}).setdefault(bs, []).append((zs, fin, gain))
    agg = {a: {s: np.array(v, dtype=float).mean(axis=0) for s, v in d.items()}
           for a, d in by.items()}

    rows = []
    for arch, d in agg.items():
        u = np.array([x for _, x in sorted(d.items())])
        per_unit = np.array([[z, f, g] for a, _, _, z, f, g in units if a == arch],
                            dtype=float)
        rows.append([arch, len(d), len(per_unit),
                     r(per_unit[:, 0].mean(), 2), r(per_unit[:, 0].std(ddof=1), 2),
                     r(per_unit[:, 1].mean(), 2), r(per_unit[:, 1].std(ddof=1), 2),
                     r(per_unit[:, 2].mean(), 2), r(per_unit[:, 2].std(ddof=1), 2),
                     r(per_unit[:, 2].min(), 2), r(per_unit[:, 2].max(), 2),
                     r(u[:, 2].mean(), 2), r(u[:, 2].std(ddof=1), 2)])

    comp = []
    for other in ("SSM HiPPO", "SSM random"):
        if "LSTM H=50" not in agg or other not in agg:
            continue
        a = np.array([v[2] for _, v in sorted(agg["LSTM H=50"].items())])
        b = np.array([v[2] for _, v in sorted(agg[other].items())])
        t, p = stats.ttest_ind(b, a, equal_var=False)
        comp.append([f"{other} vs LSTM H=50", "gain (2,000 ep - zero-shot)",
                     len(b), len(a), r(b.mean(), 2), r(a.mean(), 2),
                     r(b.mean() - a.mean(), 2), r(t, 3), r(p, 5)])
    return rows, comp, agg

def param_counts():
    torch = _torch()
    targets = [
        ("LSTM H=50", glob.glob(os.path.join(
            T, "exp6_lstm_baseline", "lstm_s1", "base", "lstm_50_*_delay30",
            "best_eval_1.pt"))),
        ("LSTM H=32", glob.glob(os.path.join(
            T, "exp6_lstm_matched", "lstm_s1", "base", "lstm_32_*_delay30",
            "best_eval_1.pt"))),
        ("SSM (S5 P=100)", glob.glob(os.path.join(
            T, "exp1", "hippo_s1", "base", "ssm_*_delay30", "best_eval_1.pt"))),
    ]
    rows = []
    for label, hits in targets:
        if not hits:
            print(f"  ! no checkpoint for {label}")
            continue
        sd = torch.load(hits[0], map_location="cpu")
        n = int(sum(v.numel() * (2 if v.is_complex() else 1)
                    for v in sd.values() if hasattr(v, "numel")))
        rows.append([label, n, os.path.relpath(hits[0], ROOT)])
    if rows:
        base = [x for x in rows if x[0].startswith("SSM")]
        base = base[0][1] if base else None
        for x in rows:
            x.append(r(x[1] / base, 3) if base else "")
    return rows

FREEZE_BLOCKS = {
    "LSTM H=50": {
        "frozen": ["cell1.weight_ih", "cell1.weight_hh",
                   "cell1.bias_ih", "cell1.bias_hh"],
        "trainable": ["readout.weight", "readout.bias", "actor.weight", "actor.bias"],
    },
    "SSM HiPPO": {
        "frozen": ["ssm_cell1.Lambda_param", "ssm_cell1.B"],
        "trainable": ["ssm_cell1.C_tilde", "actor.weight", "actor.bias"],
    },
    "SSM random": {
        "frozen": ["ssm_cell1.Lambda_param", "ssm_cell1.B"],
        "trainable": ["ssm_cell1.C_tilde", "actor.weight", "actor.bias"],
    },
}

FREEZE_RUNS = [
    ("LSTM H=50",
     "exp6_lstm_baseline/lstm_s{s}/base/lstm_50_*_delay30/best_eval_{s}.pt",
     "exp6_lstm_baseline/lstm_s{s}/retime_curve_matched_d100/rseed{rs}/"
     "lstm_50_*_delay100_fixedDelay/final_model_{rs}_freezeLambda_freezeB.pt"),
    ("SSM HiPPO",
     "exp1/hippo_s{s}/base/ssm_*_delay30/best_eval_{s}.pt",
     "exp1/hippo_s{s}/retime_curve_matched_d100/rseed{rs}/"
     "ssm_*_delay100_fixedDelay/final_model_{rs}_freezeLambda_freezeB.pt"),
    ("SSM random",
     "exp1/rand_complex_s{s}/base/ssm_*_delay30/best_eval_{s}.pt",
     "exp1/rand_complex_s{s}/retime_curve_matched_d100/rseed{rs}/"
     "ssm_*_delay100_fixedDelay/final_model_{rs}_freezeLambda_freezeB.pt"),
]

def freeze_check():
    """Did the frozen blocks really stay put, and did the readout really move?

    R2.3's control that the LSTM's flat retiming curve is not a failed
    optimization.  Metric is the relative Frobenius change per parameter tensor,
    ||theta_retimed - theta_base||_F / ||theta_base||_F, between the base
    checkpoint and the end of the 2,000-episode retiming run.  Complex tensors are
    compared as complex.  Computed identically for both architectures so the
    LSTM's readout movement can be read against the SSM's.
    """
    torch = _torch()
    rows = []
    for arch, base_pat, fin_pat in FREEZE_RUNS:
        blocks = FREEZE_BLOCKS[arch]
        for s in LSTM_SEEDS:
            b = glob.glob(os.path.join(T, base_pat.format(s=s, rs="*")))
            if not b:
                continue
            sb = torch.load(b[0], map_location="cpu")
            for rs in (90, 91, 92):
                f = glob.glob(os.path.join(T, fin_pat.format(s=s, rs=rs)))
                if not f:
                    continue
                sf = torch.load(f[0], map_location="cpu")
                for role, keys in (("frozen", blocks["frozen"]),
                                   ("trainable", blocks["trainable"])):
                    for k in keys:
                        if k not in sb or k not in sf or sb[k].shape != sf[k].shape:
                            continue
                        den = float(torch.linalg.vector_norm(sb[k]))
                        num = float(torch.linalg.vector_norm(sf[k] - sb[k]))
                        rows.append([arch, s, rs, role, k,
                                     r(num / den if den else np.nan, 5)])
    return rows

def training_curves():
    rows = []
    for label, root, _ in COHORTS:
        for p in sorted(glob.glob(os.path.join(root, "slurm_*.log"))):
            txt = open(p, errors="ignore").read()
            m = re.search(r"lstm_s(\d+)", txt)
            seed = int(m.group(1)) if m else -1
            for ep, acc in _EVAL.findall(txt):
                rows.append([label, seed, int(ep), float(acc)])
    return rows

def heatmaps():
    mats, long_rows = {}, []
    for p in sorted(glob.glob(os.path.join(ROOT, "figures", "exp6_lstm", "*.npz"))):
        stem = os.path.basename(p)
        m = re.search(r"lstm(\d+)_(best_eval_\d+|untrained).*_(delay\d)_", stem)
        if not m:
            continue
        cohort = f"LSTM H={m.group(1)}"
        kind = "trained" if m.group(2).startswith("best_eval") else "untrained"
        period = m.group(3)
        d = np.load(p)
        for which in ("sorted", "unsorted"):
            if which not in d.files:
                continue
            key = f"{cohort}__{kind}__{period}__{which}".replace(" ", "_")
            arr = np.asarray(d[which], dtype=np.float32)
            mats[key] = arr
            if which == "sorted":
                for u in range(arr.shape[0]):
                    for t in range(arr.shape[1]):
                        long_rows.append([cohort, kind, period, u, t,
                                          round(float(arr[u, t]), 5)])
    return mats, long_rows

def main():
    print("[exp6] LSTM baseline")
    rows, per = base_accuracy()
    write_csv("exp6_base_accuracy.csv",
              ["cohort", "architecture", "seed", "best_eval_acc_pct"], rows)

    units = retiming_units()
    write_csv("exp6_retime_units.csv",
              ["architecture", "base_seed", "retime_seed",
               "zeroshot_pct", "after_2000ep_pct", "gain_pct"], units)
    write_csv("exp6_retime_curves.csv",
              ["architecture", "base_seed", "retime_seed", "episode", "acc_pct"],
              retiming_curves())

    srows, comp, _ = retiming_stats(units)
    write_csv("exp6_retime_stats.csv",
              ["architecture", "n_base_models", "n_units",
               "zeroshot_mean", "zeroshot_sd", "after_2000ep_mean", "after_2000ep_sd",
               "gain_mean", "gain_sd", "gain_min", "gain_max",
               "gain_mean_of_basemodel_means", "gain_sd_of_basemodel_means"], srows)
    write_csv("exp6_retime_welch.csv",
              ["comparison", "statistic", "n_a", "n_b", "mean_a", "mean_b",
               "difference", "welch_t", "welch_p"], comp)

    write_csv("exp6_param_counts.csv",
              ["architecture", "n_parameters", "checkpoint", "ratio_to_SSM"],
              param_counts())
    write_csv("exp6_freeze_check.csv",
              ["architecture", "base_seed", "retime_seed", "role", "parameter",
               "relative_frobenius_change"], freeze_check())
    write_csv("exp6_training_curves.csv",
              ["cohort", "seed", "episode", "eval_acc_pct"], training_curves())

    mats, long_rows = heatmaps()
    np.savez_compressed(data_path("exp6_heatmap_matrices.npz"), **mats)
    print(f"  wrote data/exp6_heatmap_matrices.npz  ({len(mats)} arrays)")
    write_csv("exp6_heatmap_matrices.csv",
              ["cohort", "kind", "delay_period", "unit_rank", "time_bin", "value"],
              long_rows)

    print(f"  chance = {CHANCE_3STIM:.1f}%")
    for row in srows:
        print(f"    {row[0]:12s} zero-shot {row[3]:6.2f} +/- {row[4]:5.2f}   "
              f"2,000 ep {row[5]:6.2f} +/- {row[6]:5.2f}   "
              f"gain {row[7]:+6.2f} +/- {row[8]:4.2f}")
    for c in comp:
        print(f"    {c[0]:28s} t = {c[7]}, p = {c[8]}")

if __name__ == "__main__":
    main()

