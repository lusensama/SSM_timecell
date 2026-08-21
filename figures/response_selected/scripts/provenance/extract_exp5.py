"""EXTRACT -- Experiment 5: lap counting under randomised timing.

Answers R3.3.

Reads:
  figures/eval_lap_best_model/{fixed_baseline,zeroshot_varied}/
      laps_evaluation_results.json        the published model's confound
  figures/exp5_random/lap_count.json      decoding, task chance, behaviour (exp5b)
  figures/exp5_random/lap_identity_visual.json   pairwise lap decode + selectivity
  training/exp5_random/seed_*/best_eval_metrics.json     behaviour at the best ckpt
  training/exp5_random/seed_*/eval_curve.jsonl           behaviour vs episode
  training/exp5_landmark/seed_*/duration_invariance_grid.json   unseen-duration grid

Writes into figures/response/data/:
  exp5_confound.csv               95.4% -> 12.5% under variable durations
  exp5_behaviour.csv              per-seed behaviour on the rebuilt task
  exp5_behaviour_curve.csv        hit rate / count accuracy vs training episode
  exp5_duration_grid.csv          zero-shot transfer to never-trained durations
  exp5_decoding.csv               every decoding comparison, trained vs untrained
  exp5_decoding_pairwise.csv      the full lap x lap pairwise decode matrices
  exp5_decoding_crossK.csv        fit on one lap count, test on another
  exp5_task_chance.csv            naive policies run through the environment
  exp5_timestep_profile.csv       per-timestep multi-class decode (the datapoints)
"""
import os
import glob
import json
import numpy as np

from common import ROOT, write_csv, load_jsonl, r

F = os.path.join(ROOT, "figures")
T = os.path.join(ROOT, "training")
LAPCOUNT = os.path.join(F, "exp5_random", "lap_count.json")
LAPID = os.path.join(F, "exp5_random", "lap_identity_visual.json")

def confound():
    rows = []
    for tag, label in (("fixed_baseline", "fixed 30-step laps (as published)"),
                       ("zeroshot_varied", "variable lap duration, zero-shot")):
        p = os.path.join(F, "eval_lap_best_model", tag, "laps_evaluation_results.json")
        if not os.path.isfile(p):
            continue
        d = json.load(open(p))
        rows.append([label, tag, d["model_path"], d["n_eval_episodes"],
                     r(d["count_matching_accuracy_pct"], 2),
                     r(d.get("mean_vp_score"), 4),
                     bool(d.get("vary_lap_len")),
                     str(d.get("lap_len_range")), str(d.get("pause_range"))])
    return rows

def behaviour():
    rows = []
    for p in sorted(glob.glob(os.path.join(T, "exp5_random", "seed_*",
                                           "best_eval_metrics.json"))):
        seed = int(os.path.basename(os.path.dirname(p)).split("_")[1])
        m = json.load(open(p))
        rows.append([seed, m.get("episode"), m["n_episodes"],
                     r(m["count_accuracy_pct"], 2), r(m["hit_rate"], 4),
                     r(m["miss_rate"], 4), r(m["fa_per_episode"], 4),
                     r(m["fa_per_100_steps"], 5), r(m["vp_score"], 4),
                     r(m["mean_pred_count"], 3), r(m["std_pred_count"], 3),
                     r(m["mean_hit_lag"], 3), r(m["frac_presses_early"], 5),
                     r(m["mean_episode_len"], 1)])
    return rows

def behaviour_curve():
    rows = []
    for p in sorted(glob.glob(os.path.join(T, "exp5_random", "seed_*",
                                           "eval_curve.jsonl"))):
        seed = int(os.path.basename(os.path.dirname(p)).split("_")[1])
        for d in load_jsonl(p):
            rows.append([seed, d.get("episode"),
                         r(d.get("count_accuracy_pct"), 2), r(d.get("hit_rate"), 4),
                         r(d.get("fa_per_episode"), 4), r(d.get("vp_score"), 4)])
    return rows

def duration_grid():
    rows = []
    for p in sorted(glob.glob(os.path.join(T, "exp5_landmark", "seed_*",
                                           "duration_invariance_grid.json"))):
        seed = int(os.path.basename(os.path.dirname(p)).split("_")[1])
        d = json.load(open(p))
        for cond, m in d["conditions"].items():
            rows.append([seed, cond, str(m.get("lap_len_range")),
                         m["n_episodes"], r(m["count_accuracy_pct"], 2),
                         r(m["hit_rate"], 5), r(m["fa_per_episode"], 5),
                         r(m["fa_per_100_steps"], 6), r(m["vp_score"], 5),
                         r(m["mean_episode_len"], 1)])
    return rows

def _pairwise_split(summary, kind):
    """Adjacent-lap and >=2-apart pairwise decode, pooled over seeds."""
    adj, far, cells = [], [], []
    for tag, v in summary.items():
        if not tag.startswith(kind):
            continue
        M = np.array([[np.nan if x is None else x for x in row]
                      for row in v["pairwise"]], dtype=float)
        N = np.array(v["pairwise_n_timesteps"], dtype=int)
        for i in range(M.shape[0]):
            for j in range(i + 1, M.shape[1]):
                if not np.isfinite(M[i, j]):
                    continue
                cells.append([tag, kind, i + 1, j + 1, j - i,
                              r(100 * M[i, j], 2), int(N[i, j])])
                (adj if j - i == 1 else far).append(M[i, j])
    return adj, far, cells

def decoding():
    d = json.load(open(LAPCOUNT))
    li = json.load(open(LAPID))["summary"]
    res = d["results"]

    def m(kind, fn):
        v = [fn(e) for e in res if e["kind"] == kind]
        v = [x for x in v if x is not None and np.isfinite(x)]
        return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                len(v))

    rows, pairwise_rows = [], []

    def add(label, chance, tr, un, note=""):
        rows.append([label, r(100 * tr[0], 2), r(100 * tr[1], 2), tr[2],
                     r(100 * un[0], 2), r(100 * un[1], 2), un[2],
                     r(100 * chance, 2), note])

    add("multi-class lap decode (exact timestep)",
        m("trained", lambda e: e["exact_timestep_mean_chance"])[0],
        m("trained", lambda e: e["exact_timestep_mean_acc"]),
        m("untrained", lambda e: e["exact_timestep_mean_acc"]),
        "permutation null 95th pct "
        f"{100 * m('trained', lambda e: e['perm_null']['p95'])[0]:.1f}%")

    tr_adj, tr_far, cells_t = _pairwise_split(li, "trained")
    un_adj, un_far, cells_u = _pairwise_split(li, "untrained")
    pairwise_rows = cells_t + cells_u

    def stat(v):
        return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                len(v))

    add("adjacent laps, pairwise", 0.5, stat(tr_adj), stat(un_adj),
        "load-bearing comparison; passive decay cannot produce it")
    add("laps >= 2 apart, pairwise", 0.5, stat(tr_far), stat(un_far),
        "untrained keeps a weak residual here only")

    for cc in res[0]["cross_condition"]:
        add(cc.replace("_", " ").replace("->", " -> "),
            m("trained", lambda e, c=cc: e["cross_condition"][c]["chance"])[0],
            m("trained", lambda e, c=cc: e["cross_condition"][c]["acc"]),
            m("untrained", lambda e, c=cc: e["cross_condition"][c]["acc"]),
            "fit in one temporal condition, test in the other")

    def ck(e, same):
        v = [x["acc"] for k, x in e["cross_K"].items()
             if (k.split("->")[0] == k.split("->")[1]) == same]
        return float(np.mean(v)) if v else None

    add("within the same lap count K", 0.5,
        m("trained", lambda e: ck(e, True)), m("untrained", lambda e: ck(e, True)),
        "decoder fitted and tested on episodes of the same K")
    add("fit on one K, test on another", 0.5,
        m("trained", lambda e: ck(e, False)), m("untrained", lambda e: ck(e, False)),
        "transfer is not worse than within: lap 2 is lap 2 regardless of K")

    add("TOTAL lap count K (matched t and ordinal)",
        m("trained", lambda e: e["cardinality_matched_t_and_ordinal"]["chance"])[0],
        m("trained", lambda e: e["cardinality_matched_t_and_ordinal"]["acc"]),
        m("untrained", lambda e: e["cardinality_matched_t_and_ordinal"]["acc"]),
        "NOT decodable -- a null by construction; the claim is CURRENT LAP")

    crossK = []
    for e in res:
        for k, v in e["cross_K"].items():
            a, b = k.split("->")
            crossK.append([e["label"], e["kind"], e["seed"], int(a), int(b),
                           a == b, r(100 * v["acc"], 2), r(100 * v["chance"], 2),
                           v["n_timesteps"]])

    profile = []
    for e in res:
        for q in e["exact_timestep_profile"]:
            profile.append([e["label"], e["kind"], e["seed"], q["t"],
                            r(100 * q["acc"], 2), r(100 * q["chance"], 2),
                            q["n_classes"], q["n"]])

    chance_rows = []
    for name, v in d["task_chance"].items():
        chance_rows.append([name, r(v["count_acc_pct"], 2),
                            r(v["fa_per_episode"], 3),
                            r(v["presses_per_episode"], 3)])

    return rows, pairwise_rows, crossK, profile, chance_rows, d["config"]

def main():
    print("[exp5] lap counting under randomised timing")
    write_csv("exp5_confound.csv",
              ["condition", "tag", "model", "n_episodes", "count_accuracy_pct",
               "mean_vp_score", "vary_lap_len", "lap_len_range", "pause_range"],
              confound())
    write_csv("exp5_behaviour.csv",
              ["seed", "best_episode", "n_eval_episodes", "count_accuracy_pct",
               "hit_rate", "miss_rate", "fa_per_episode", "fa_per_100_steps",
               "vp_score", "mean_pred_count", "std_pred_count", "mean_hit_lag",
               "frac_presses_early", "mean_episode_len"], behaviour())
    write_csv("exp5_behaviour_curve.csv",
              ["seed", "episode", "count_accuracy_pct", "hit_rate",
               "fa_per_episode", "vp_score"], behaviour_curve())
    write_csv("exp5_duration_grid.csv",
              ["seed", "condition", "lap_len_range", "n_episodes",
               "count_accuracy_pct", "hit_rate", "fa_per_episode",
               "fa_per_100_steps", "vp_score", "mean_episode_len"], duration_grid())

    dec, pw, crossK, profile, chance, cfg = decoding()
    write_csv("exp5_decoding.csv",
              ["comparison", "trained_pct", "trained_sd", "trained_n",
               "untrained_pct", "untrained_sd", "untrained_n", "chance_pct", "note"],
              dec)
    write_csv("exp5_decoding_pairwise.csv",
              ["run", "kind", "lap_i", "lap_j", "separation",
               "balanced_acc_pct", "n_timesteps"], pw)
    write_csv("exp5_decoding_crossK.csv",
              ["run", "kind", "seed", "fit_K", "test_K", "same_K",
               "balanced_acc_pct", "chance_pct", "n_timesteps"], crossK)
    write_csv("exp5_timestep_profile.csv",
              ["run", "kind", "seed", "timestep", "balanced_acc_pct",
               "chance_pct", "n_classes", "n_episodes"], profile)
    write_csv("exp5_task_chance.csv",
              ["policy", "count_accuracy_pct", "fa_per_episode",
               "presses_per_episode"], chance)

    print("  R3.3 check:")
    for row in dec:
        print(f"    {row[0]:44s} trained {row[1]:6.2f}   untrained {row[4]:6.2f}   "
              f"chance {row[7]:5.2f}")

if __name__ == "__main__":
    main()

