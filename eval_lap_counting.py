import argparse
import json

import numpy as np
import torch
from torch.distributions import Categorical

from envs.lap_random import Laps_Random
from agents.model_ssm_RL_laps import AC_SSM_stack
from utils.vp import vp_score

DEV = torch.device("cpu")

def build_net(n_neurons, seed):
    torch.manual_seed(1234 + seed)
    np.random.seed(1234 + seed)
    ssm_params = dict(P=n_neurons * 2, C_init="trunc_standard_normal",
                      discretization="zoh", dt_min=0.001, dt_max=0.1,
                      conj_sym=True, step_rescale=1.0,
                      spike=False, layer2=False, lap_count=4)
    return AC_SSM_stack(input_dimensions=2, action_dimensions=2, batch_size=1,
                        hidden_dim=n_neurons, ssm_params=ssm_params,
                        p_dropout=0.1).to(DEV)

@torch.no_grad()
def evaluate_at_K(net, K, a, seed):
    env = Laps_Random(seed=seed, k_range=(K, K),
                      lap_len_range=tuple(a.lap_len_range),
                      pause_count_range=tuple(a.pause_count_range),
                      pause_len_range=tuple(a.pause_len_range),
                      hit_window=a.hit_window)
    correct = err = hits = misses = fas = extras = cues = 0
    vps, preds, lens = [], [], []
    for _ in range(a.n_episodes):
        obs = env.reset()
        net.reinit_hid()
        stage = None
        while stage != 'done':
            pol, _v, _l = net.forward(torch.tensor([obs], dtype=torch.float32))
            action = int(Categorical(pol).sample().item())
            obs, _r, stage = env.step2(action)
        s = env.episode_summary()
        correct += s["correct"]
        err += abs(s["pred_count"] - s["true_count"])
        hits += s["n_hit"]
        misses += s["n_miss"]
        fas += s["n_fa"]
        extras += s["n_extra"]
        cues += s["true_count"]
        preds.append(s["pred_count"])
        lens.append(s["episode_len"])
        vps.append(vp_score(s["press_times"], s["lap_ends"], tol=a.vp_tol))
    n = a.n_episodes
    return {
        "K": int(K),
        "count_acc": 100.0 * correct / n,
        "abs_count_err": err / n,
        "hit_rate": hits / cues if cues else float('nan'),
        "miss_rate": misses / cues if cues else float('nan'),
        "fa_per_episode": fas / n,
        "extra_per_episode": extras / n,
        "vp": float(np.mean(vps)),
        "mean_pred_count": float(np.mean(preds)),
        "std_pred_count": float(np.std(preds)),
        "mean_episode_len": float(np.mean(lens)),
    }

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="models/lap_counting_best.pt")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--n_neurons", type=int, default=80)
    p.add_argument("--k_min", type=int, default=2)
    p.add_argument("--k_max", type=int, default=6)
    p.add_argument("--n_episodes", type=int, default=150)
    p.add_argument("--lap_len_range", type=int, nargs=2, default=[10, 60])
    p.add_argument("--pause_count_range", type=int, nargs=2, default=[0, 2])
    p.add_argument("--pause_len_range", type=int, nargs=2, default=[0, 25])
    p.add_argument("--hit_window", type=int, default=3)
    p.add_argument("--vp_tol", type=float, default=3.0)
    p.add_argument("--trained_k_range", type=int, nargs=2, default=[2, 6],
                   help="marks which K are in-distribution in the printed table")
    p.add_argument("--out", default=None, help="also write the rows as JSON here")
    a = p.parse_args()

    net = build_net(a.n_neurons, a.seed)
    net.load_state_dict(torch.load(a.ckpt, map_location=DEV))
    net.eval()

    print(f"ckpt   {a.ckpt}")
    print(f"env    lap_len {a.lap_len_range}  pauses {a.pause_count_range} "
          f"of {a.pause_len_range}  hit_window {a.hit_window}")
    print(f"eval   K {a.k_min}..{a.k_max}, {a.n_episodes} episodes each, "
          f"seed {a.seed}, vp_tol {a.vp_tol}\n")
    print(f"{'K':>4} {'count_acc':>10} {'vp':>8} {'hit':>7} {'miss':>7} "
          f"{'fa/ep':>8} {'pred':>13} {'ep_len':>8}")

    rows = []
    for K in range(a.k_min, a.k_max + 1):
        r = evaluate_at_K(net, K, a, a.seed)
        rows.append(r)
        flag = "" if a.trained_k_range[0] <= K <= a.trained_k_range[1] else "  <- unseen K"
        print(f"{K:>4} {r['count_acc']:>9.2f}% {r['vp']:>8.4f} "
              f"{r['hit_rate']:>7.4f} {r['miss_rate']:>7.4f} "
              f"{r['fa_per_episode']:>8.2f} "
              f"{r['mean_pred_count']:>6.2f}+-{r['std_pred_count']:<5.2f} "
              f"{r['mean_episode_len']:>8.1f}{flag}")

    inside = [r for r in rows if a.trained_k_range[0] <= r['K'] <= a.trained_k_range[1]]
    outside = [r for r in rows if not (a.trained_k_range[0] <= r['K'] <= a.trained_k_range[1])]
    print()
    if inside:
        print(f"mean count_acc, trained K: {np.mean([r['count_acc'] for r in inside]):6.2f}%"
              f"   (vp {np.mean([r['vp'] for r in inside]):.4f})")
    if outside:
        print(f"mean count_acc, unseen  K: {np.mean([r['count_acc'] for r in outside]):6.2f}%"
              f"   (vp {np.mean([r['vp'] for r in outside]):.4f})")
    print(f"mean count_acc, all     K: {np.mean([r['count_acc'] for r in rows]):6.2f}%")

    if a.out:
        with open(a.out, "w") as f:
            json.dump({"ckpt": a.ckpt, "seed": a.seed, "rows": rows}, f, indent=2)
        print(f"\nwrote {a.out}")

if __name__ == "__main__":
    main()
