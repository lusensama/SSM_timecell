"""
Experiment 5 -- train the SSM agent under the landmark-driven reward and measure
duration invariance.

New script. Does not import or modify envs/lap_counting.py, train_and_plot_laps.py
or ssm_run_laps.py.

Two things differ from the existing pipeline besides the reward:

  * CHECKPOINT SELECTION IS ON VP, NOT ACCURACY. With fixed_laps=4 and no count
    randomization, predicted_lap_count is just the tally of emissions, so any
    policy that emits exactly 4 times anywhere scores 100% count accuracy. Every
    best_eval.pt produced by train_and_plot_laps.py (which selects on `acc`) was
    therefore chosen by a criterion blind to whether emissions were
    landmark-locked. `--select_on` defaults to vp here.

  * The reported metrics are the hit/miss/false-alarm decomposition plus VP at an
    explicit tolerance, plus the anticipation rate. Count accuracy is still
    reported but flagged as degenerate.

Usage
-----
  train:
    python train_landmark_laps.py --mode train --save_dir training/exp5_landmark/seed_2 \
        --seed 2 --n_total_episodes 60000 --eval_every 2000

  duration-invariance grid on a checkpoint (the headline test):
    python train_landmark_laps.py --mode grid --ckpt <path>/best_eval.pt \
        --save_dir training/exp5_landmark/seed_2 --n_eval_episodes 2000
"""

import argparse
import json
import os
import numpy as np
import torch
from torch.distributions import Categorical
from tqdm import tqdm

from envs.lap_landmark import Laps_Landmark
from envs.lap_random import Laps_Random
from agents.model_ssm_RL_laps import AC_SSM_stack, SavedAction, finish_run
from utils.vp import vp_score, q_from_tol

def build_env(args, vary_lap_len=None, lap_len_range=None, lap_length=None,
              pause_range=None, seed=None):
    if args.env == "random":
        return Laps_Random(
            seed=args.seed if seed is None else seed,
            k_range=tuple(args.k_range),
            lap_len_range=tuple(args.lap_len_range) if lap_len_range is None
                          else tuple(lap_len_range),
            pause_count_range=tuple(args.pause_count_range),
            pause_len_range=tuple(args.pause_len_range),
            inter_lap_pause_range=tuple(args.inter_lap_pause_range),
            total_steps=args.total_steps,
            hit_window=args.hit_window,
            hit_rwd=args.hit_rwd,
            miss_cost=args.miss_cost,
            fa_cost=args.fa_cost,
            term_rwd=args.term_rwd,
        )
    return Laps_Landmark(
        seed=args.seed if seed is None else seed,
        fixed_laps=args.lap_count,
        vary_lap_len=args.vary_lap_len if vary_lap_len is None else vary_lap_len,
        lap_length=args.lap_length if lap_length is None else lap_length,
        lap_len_range=tuple(args.lap_len_range) if lap_len_range is None else tuple(lap_len_range),
        pause_range=tuple(args.pause_range) if pause_range is None else tuple(pause_range),
        hit_window=args.hit_window,
        hit_rwd=args.hit_rwd,
        miss_cost=args.miss_cost,
        fa_cost=args.fa_cost,
        term_rwd=args.term_rwd,
    )

def build_net(args, device):
    ssm_params = {
        "P": args.n_neurons * 2,
        "C_init": "trunc_standard_normal",
        "discretization": "zoh",
        "dt_min": 0.001,
        "dt_max": 0.1,
        "conj_sym": True,
        "step_rescale": 1.0,
        "spike": False,
        "layer2": args.layer2,
        "lap_count": args.lap_count,
    }
    net = AC_SSM_stack(
        input_dimensions=2,
        action_dimensions=2,
        batch_size=1,
        hidden_dim=args.n_neurons,
        ssm_params=ssm_params,
        p_dropout=args.p_dropout,
    ).to(device)
    net.reinit_hid()
    return net

def entropy_at(episode, total, start, end):
    """Linear anneal, matching get_param_linear's contract in the existing code."""
    if episode <= 1:
        return start
    if episode >= total:
        return end
    return start + (end - start) * (episode - 1) / (total - 1)

def signed_lead_lag(press_times, lap_ends):
    """(press time - nearest landmark time) for each press. Negative = early."""
    if len(press_times) == 0 or len(lap_ends) == 0:
        return []
    ends = np.asarray(lap_ends, dtype=float)
    return [float(p - ends[np.argmin(np.abs(ends - p))]) for p in press_times]

def anticipation_count(fa_times, lap_ends, lead):
    """False alarms landing within `lead` steps BEFORE an upcoming landmark."""
    if len(fa_times) == 0:
        return 0
    ends = np.asarray(lap_ends, dtype=float)
    n = 0
    for t in fa_times:
        future = ends[ends > t]
        if future.size and (future.min() - t) <= lead:
            n += 1
    return n

@torch.no_grad()
def evaluate(net, env, device, n_episodes, vp_tol, ant_lead, vp_hard_window=False,
             progress=False):
    """Greedy-sampled rollouts; returns an aggregate metric dict."""
    net.eval()
    n_correct = 0
    hits = misses = fas = extras = cues = 0
    vps, pred_counts, lags, all_lead_lag = [], [], [], []
    n_ant = 0
    n_press = 0
    total_steps = 0

    it = range(n_episodes)
    for _ in (tqdm(it, desc="eval", leave=False) if progress else it):
        obs = env.reset()
        done = False
        if hasattr(net, 'reinit_hid'):
            net.reinit_hid()
        while not done:
            obs_t = torch.tensor([obs], dtype=torch.float32).to(device)
            pol, _val, _lin = net.forward(obs_t)
            action = int(Categorical(pol).sample().item())
            obs, _r, stage = env.step2(action)
            done = (stage == 'done')

        s = env.episode_summary()
        n_correct += s["correct"]
        hits += s["n_hit"]
        misses += s["n_miss"]
        fas += s["n_fa"]
        extras += s["n_extra"]
        cues += s["true_count"]
        n_press += len(s["press_times"])
        total_steps += s["episode_len"]
        pred_counts.append(s["pred_count"])
        lags.extend(s["hit_lags"])
        all_lead_lag.extend(signed_lead_lag(s["press_times"], s["lap_ends"]))
        n_ant += anticipation_count(s["fa_times"], s["lap_ends"], ant_lead)
        vps.append(vp_score(s["press_times"], s["lap_ends"], tol=vp_tol,
                            hard_window=vp_hard_window))

    net.train()
    ll = np.asarray(all_lead_lag, dtype=float) if all_lead_lag else np.zeros(0)
    return {
        "n_episodes": int(n_episodes),
        "vp_score": float(np.mean(vps)),
        "vp_tol": float(vp_tol),
        "vp_q": float(q_from_tol(vp_tol)),
        "vp_hard_window": bool(vp_hard_window),
        "count_accuracy_pct": 100.0 * n_correct / float(n_episodes),
        "count_accuracy_is_degenerate": True,
        "hit_rate": float(hits / cues) if cues else 0.0,
        "miss_rate": float(misses / cues) if cues else 0.0,
        "fa_per_episode": float(fas / n_episodes),
        "fa_per_100_steps": float(100.0 * fas / total_steps) if total_steps else 0.0,
        "extra_per_episode": float(extras / n_episodes),
        "presses_per_episode": float(n_press / n_episodes),
        "mean_episode_len": float(total_steps / n_episodes),
        "mean_pred_count": float(np.mean(pred_counts)),
        "std_pred_count": float(np.std(pred_counts)),
        "mean_hit_lag": float(np.mean(lags)) if lags else float('nan'),
        "anticipation_rate": float(n_ant / n_press) if n_press else 0.0,
        "frac_presses_early": float(np.mean(ll < 0)) if ll.size else 0.0,
        "mean_signed_lead_lag": float(np.mean(ll)) if ll.size else float('nan'),
        "lead_lag_hist": np.histogram(
            ll, bins=np.arange(-20.5, 21.5, 1.0))[0].tolist() if ll.size else [],
        "lead_lag_hist_bin_centers": np.arange(-20, 21, 1).tolist(),
    }

def selection_value(metrics, select_on):
    if select_on == 'vp':
        return metrics["vp_score"]
    if select_on == 'acc':
        return metrics["count_accuracy_pct"]
    if select_on == 'hit_minus_fa':
        return metrics["hit_rate"] - metrics["fa_per_episode"]
    raise ValueError(select_on)

def train(args, device):
    os.makedirs(args.save_dir, exist_ok=True)
    env = build_env(args)
    net = build_net(args, device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    best_path = os.path.join(args.save_dir, 'best_eval.pt')
    best_json = os.path.join(args.save_dir, 'best_eval_metrics.json')
    curve_path = os.path.join(args.save_dir, 'eval_curve.jsonl')
    best_value = -np.inf

    print(f"[exp5] reward: hit=+{args.hit_rwd} miss=-{args.miss_cost} "
          f"fa=-{args.fa_cost} term=+/-{args.term_rwd} window={args.hit_window}")
    if args.env == "random":
        nk = args.k_range[1] - args.k_range[0] + 1
        print(f"[exp5] env=random: K~U{tuple(args.k_range)} "
              f"lap_len~U{tuple(args.lap_len_range)} "
              f"mid_lap_pauses~U{tuple(args.pause_count_range)} of "
              f"len~U{tuple(args.pause_len_range)} "
              f"inter_lap~U{tuple(args.inter_lap_pause_range)}")
        print(f"[exp5] episodes padded to {env.total_steps} steps; "
              f"fixed-count task chance = {100.0/nk:.1f}%")
    else:
        print(f"[exp5] env=landmark: vary_lap_len={args.vary_lap_len} "
              f"range={tuple(args.lap_len_range)} pause={tuple(args.pause_range)} "
              f"laps={args.lap_count}")
    print(f"[exp5] selecting checkpoints on '{args.select_on}' "
          f"(vp_tol={args.vp_tol}, q={q_from_tol(args.vp_tol):.4f})")

    for i_ep in tqdm(range(args.n_total_episodes), desc="train"):
        obs = env.reset()
        if hasattr(net, 'reinit_hid'):
            net.reinit_hid()
        rewards, saved = [], []
        done = False
        while not done:
            obs_t = torch.tensor([obs], dtype=torch.float32).to(device)
            pol, val, _lin = net.forward(obs_t)
            dist = Categorical(pol)
            a = dist.sample()
            saved.append(SavedAction(dist.log_prob(a), val, pol))
            obs, r, stage = env.step2(int(a.item()))
            rewards.append(r)
            done = (stage == 'done')

        net.rewards = rewards
        net.saved_actions = saved
        finish_run(net, args.gamma, optimizer,
                   entropy_weight=entropy_at(i_ep + 1, args.n_total_episodes,
                                             args.entropy, args.entropy_end))

        if args.eval_every > 0 and (i_ep + 1) % args.eval_every == 0:
            m = evaluate(net, env, device, args.n_eval_episodes,
                         args.vp_tol, args.ant_lead, args.vp_hard_window)
            m["episode"] = i_ep + 1
            with open(curve_path, 'a') as f:
                f.write(json.dumps(m) + "\n")
            print(f"  @{i_ep+1}: vp={m['vp_score']:.4f} "
                  f"hit={m['hit_rate']:.3f} miss={m['miss_rate']:.3f} "
                  f"fa/ep={m['fa_per_episode']:.2f} "
                  f"acc={m['count_accuracy_pct']:.1f}% "
                  f"pred={m['mean_pred_count']:.2f} "
                  f"early={m['frac_presses_early']:.3f}")
            v = selection_value(m, args.select_on)
            if v > best_value:
                best_value = v
                torch.save(net.state_dict(), best_path)
                m["selected_on"] = args.select_on
                m["selection_value"] = float(v)
                with open(best_json, 'w') as f:
                    json.dump(m, f, indent=2)
                print(f"  new best {args.select_on}={v:.4f} -> {best_path}")

    torch.save(net.state_dict(), os.path.join(args.save_dir, 'final.pt'))
    return net

GRID = [
    ("fixed_30",   False, 30,         (20, 45),      (0, 0)),
    ("trained_20_45", True, 30,       (20, 45),      (0, 15)),
    ("unseen_short_10_19", True, 30,  (10, 19),      (0, 15)),
    ("unseen_long_46_70",  True, 30,  (46, 70),      (0, 15)),
    ("unseen_wide_15_70",  True, 30,  (15, 70),      (0, 15)),
]

def run_grid(args, device):
    """
    Duration-invariance test. Train once, evaluate with no retraining across lap
    length ranges -- including ranges never seen. Flat hit_rate / vp across all
    five is the claim: the agent responds to landmarks, whenever they arrive.
    """
    if args.env == "random":
        raise SystemExit(
            "--mode grid is defined for --env landmark only: its conditions are "
            "fixed vs varied lap-length RANGES, which is not the manipulation "
            "Laps_Random makes (there lap count and length are both random every "
            "episode). Use --mode train, then decode at exact timesteps -- padding "
            "makes every episode the same length, so no time-banding is needed.")
    os.makedirs(args.save_dir, exist_ok=True)
    net = build_net(args, device)
    state = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(state)
    print(f"[exp5] loaded {args.ckpt}")

    results = {}
    for name, vary, lap_length, lr_range, p_range in GRID:
        env = build_env(args, vary_lap_len=vary, lap_length=lap_length,
                        lap_len_range=lr_range, pause_range=p_range,
                        seed=args.seed + 1000)
        m = evaluate(net, env, device, args.n_eval_episodes, args.vp_tol,
                     args.ant_lead, args.vp_hard_window, progress=args.progress)
        m["condition"] = name
        m["vary_lap_len"] = vary
        m["lap_len_range"] = list(lr_range) if vary else [lap_length, lap_length]
        m["pause_range"] = list(p_range)
        results[name] = m
        print(f"  {name:22s} hit={m['hit_rate']:.3f} miss={m['miss_rate']:.3f} "
              f"fa/100st={m['fa_per_100_steps']:.2f} vp={m['vp_score']:.4f} "
              f"acc={m['count_accuracy_pct']:.1f}% pred={m['mean_pred_count']:.2f} "
              f"len={m['mean_episode_len']:.0f}")

    out = os.path.join(args.save_dir, 'duration_invariance_grid.json')
    with open(out, 'w') as f:
        json.dump({"ckpt": args.ckpt, "conditions": results}, f, indent=2)
    print(f"[exp5] wrote {out}")

    hits = [results[n]["hit_rate"] for n, *_ in GRID]
    fas = [results[n]["fa_per_100_steps"] for n, *_ in GRID]
    print(f"\n[exp5] hit-rate spread:        min={min(hits):.3f} max={max(hits):.3f} "
          f"range={max(hits)-min(hits):.3f}")
    print(f"[exp5] fa/100-steps spread:    min={min(fas):.2f} max={max(fas):.2f}")
    print("[exp5] flat across conditions => landmark-driven; "
          "collapse on unseen ranges => still a timer.")
    print("[exp5] compare hit_rate and fa_per_100_steps across conditions; "
          "vp_score is not length-normalized.")
    return results

def main():
    p = argparse.ArgumentParser(description="Exp 5: landmark-driven lap counting")
    p.add_argument("--mode", choices=["train", "grid"], default="train")
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--ckpt", type=str, default=None, help="checkpoint for --mode grid")
    p.add_argument("--seed", type=int, default=2)

    p.add_argument("--env", choices=["landmark", "random"], default="landmark",
                   help="'landmark' = Laps_Landmark (fixed lap count, post-landmark "
                        "pauses indistinguishable from running -- the completed exp5 "
                        "run). 'random' = Laps_Random: random lap count, random lap "
                        "length, MID-LAP pauses as the empty signal [0,0], all "
                        "episodes padded to one length.")
    p.add_argument("--k_range", type=int, nargs=2, default=[2, 6],
                   help="--env random: lap count drawn uniformly from this range")
    p.add_argument("--pause_count_range", type=int, nargs=2, default=[0, 2],
                   help="--env random: number of MID-LAP pauses per lap")
    p.add_argument("--pause_len_range", type=int, nargs=2, default=[0, 20],
                   help="--env random: length of each mid-lap pause")
    p.add_argument("--inter_lap_pause_range", type=int, nargs=2, default=[0, 0],
                   help="--env random: post-landmark dwell; off by default")
    p.add_argument("--total_steps", type=int, default=None,
                   help="--env random: padded episode length; default is the "
                        "worst-case schedule so K stays exactly uniform")
    p.add_argument("--lap_count", type=int, default=4)
    p.add_argument("--vary_lap_len", action="store_true", default=True)
    p.add_argument("--no_vary_lap_len", dest="vary_lap_len", action="store_false")
    p.add_argument("--lap_length", type=int, default=30, help="used when not varying")
    p.add_argument("--lap_len_range", type=int, nargs=2, default=[20, 45])
    p.add_argument("--pause_range", type=int, nargs=2, default=[0, 15])

    p.add_argument("--hit_window", type=int, default=3,
                   help="one-sided response window, in steps AFTER the cue")
    p.add_argument("--hit_rwd", type=float, default=1.0)
    p.add_argument("--miss_cost", type=float, default=1.0)
    p.add_argument("--fa_cost", type=float, default=0.1,
                   help="class-imbalance knob; ~n_cues/n_noncues")
    p.add_argument("--term_rwd", type=float, default=2.0)

    p.add_argument("--n_neurons", type=int, default=80)
    p.add_argument("--layer2", action="store_true", default=False)
    p.add_argument("--p_dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--entropy", type=float, default=0.1)
    p.add_argument("--entropy_end", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.9,
                   help="the reward is dense and local, so a short horizon is "
                        "appropriate; the old timing reward used gamma=1")
    p.add_argument("--n_total_episodes", type=int, default=60000)

    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--n_eval_episodes", type=int, default=300)
    p.add_argument("--vp_tol", type=float, default=3.0,
                   help="VP misalignment tolerance in steps; q = 2/tol")
    p.add_argument("--vp_hard_window", action="store_true", default=False,
                   help="forbid VP matches beyond tol (anti-aliasing)")
    p.add_argument("--ant_lead", type=int, default=5,
                   help="a false alarm this close before a landmark counts as "
                        "anticipatory")
    p.add_argument("--progress", action="store_true", default=False,
                   help="per-episode tqdm bar during grid eval (floods slurm logs)")
    p.add_argument("--select_on", choices=["vp", "acc", "hit_minus_fa"],
                   default="vp",
                   help="count accuracy is degenerate at fixed lap_count; "
                        "default selects on VP")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[exp5] device={device}")

    if args.mode == "train":
        train(args, device)
    else:
        if not args.ckpt:
            raise SystemExit("--mode grid requires --ckpt")
        run_grid(args, device)

if __name__ == "__main__":
    main()

