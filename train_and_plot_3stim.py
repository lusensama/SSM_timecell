"""
Train and evaluate a spiking SSM actor-critic on a 3-stimulus interval discrimination task.
Use intermediate choice task to train.
"""

import os
import sys

# Go two levels up to get to the project's root directory ('deeprl-timecells')
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the project root to the system path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import re
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from tqdm import tqdm

from envs.int_discrim import IntDiscrim3_Intermediate
from agents.model_ssm_stack_RL import *
from utils.utils_analysis import sort_resp
from ssm_observer_1d import analyze_model_example


def get_param_linear(epoch: int, start: float = 1.0, end: float = 0.1, total_epochs: int = 5000) -> float:
    if epoch <= 1:
        return start
    if epoch >= total_epochs:
        return end
    frac = (epoch - 1) / (total_epochs - 1)
    return start + (end - start) * frac


def train_3stim(
    n_total_episodes: int,
    n_neurons: int,
    lr: float,
    weight_decay: float,
    entropy_weight: float,
    seed: int,
    delay: int,
    spike: bool,
    layer2: bool,
    device: torch.device,
    save_dir: str,
    eval_every: int,
    n_eval_episodes: int,
):
    env = IntDiscrim3_Intermediate(seed=seed, delay=delay, fixed_delay=False)

    ssm_params = {
        "P": n_neurons * 2,
        "C_init": "trunc_standard_normal",
        "discretization": "zoh",
        "dt_min": 0.001,
        "dt_max": 0.1,
        "conj_sym": True,
        "step_rescale": 1.0,
        "spike": spike,
        "layer2": layer2,
    }
    net = AC_SSM_stack(
        input_dimensions=3,
        action_dimensions=3,
        batch_size=1,
        hidden_dim=n_neurons,
        ssm_params=ssm_params,
        p_dropout=0.1,
    ).to(device)
    net.reinit_hid()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    policy_hist, loss_hist, p_loss_hist, v_loss_hist = [], [], [], []
    correct_trial = np.zeros(n_total_episodes, dtype=np.int8)

    # best eval tracking
    best_ckpt = os.path.join(save_dir, 'best_eval.pt')
    best_acc_path = os.path.join(save_dir, 'best_eval_acc.txt')
    best_eval_acc = -1.0
    if os.path.exists(best_acc_path):
        try:
            with open(best_acc_path, 'r') as f:
                best_eval_acc = float(f.read().strip())
        except Exception:
            best_eval_acc = -1.0

    for i_episode in tqdm(range(n_total_episodes)):
        done = False
        env.reset()
        net.reinit_hid()

        episode_rewards = []
        episode_saved_actions = []
        while not done:
            if spike:
                if layer2:
                    pol, val, lin_act, _, _ = net.forward(
                        torch.unsqueeze(torch.Tensor(env.observation).float(), dim=0).to(device)
                    )
                else:
                    pol, val, lin_act, _ = net.forward(
                        torch.unsqueeze(torch.Tensor(env.observation).float(), dim=0).to(device)
                    )
            else:
                pol, val, lin_act = net.forward(
                    torch.unsqueeze(torch.Tensor(env.observation).float(), dim=0).to(device)
                )

            action_to_take = None
            if env.task_stage == 'init':
                action_to_take = 1
                new_obs, reward, done = env.step(action_to_take)

            elif env.task_stage == 'intermediate_choice_init':
                if net.action_d < 2:
                    raise ValueError("Network action dimension must be at least 2 for intermediate choice")
                intermediate_pol_logits = net.actor(lin_act)[:, :2]
                intermediate_pol = torch.nn.functional.softmax(intermediate_pol_logits, dim=1)
                intermediate_dist = Categorical(intermediate_pol)
                intermediate_act_tensor = intermediate_dist.sample()
                action_to_take = intermediate_act_tensor.item()
                log_prob_intermediate = intermediate_dist.log_prob(intermediate_act_tensor)
                value_intermediate = net.critic(lin_act)
                episode_saved_actions.append(SavedAction(log_prob_intermediate, value_intermediate, intermediate_pol))
                new_obs, reward, done = env.step(action_to_take)
                episode_rewards.append(reward)

            elif env.task_stage == 'choice_init':
                final_dist = Categorical(pol)
                final_act_tensor = final_dist.sample()
                action_to_take = final_act_tensor.item()
                log_prob_final = final_dist.log_prob(final_act_tensor)
                value_final = val
                episode_saved_actions.append(SavedAction(log_prob_final, value_final, pol))
                new_obs, reward, done = env.step(action_to_take)
                episode_rewards.append(reward)
                correct_trial[i_episode] = env.correct_trial
                policy_hist.append(pol.detach().cpu().numpy().squeeze())
            else:
                new_obs, reward, done = env.step()

            env.observation = new_obs

        net.rewards = episode_rewards
        net.saved_actions = episode_saved_actions
        p_loss, v_loss, total_loss = finish_trial(
            net,
            0.99,
            optimizer,
            entropy_weight=get_param_linear(i_episode + 1, start=entropy_weight, total_epochs=n_total_episodes),
        )
        loss_hist.append(total_loss.detach().cpu().numpy().squeeze())
        p_loss_hist.append(p_loss.detach().cpu().numpy().squeeze())
        v_loss_hist.append(v_loss.detach().cpu().numpy().squeeze())

        # periodic evaluation based on evaluation-only performance
        if eval_every > 0 and (i_episode + 1) % eval_every == 0:
            acc = eval_accuracy_3stim(
                net=net,
                env=env,
                device=device,
                n_episodes=n_eval_episodes,
                layer2=layer2,
            )
            print(f"Eval @ episode {i_episode + 1}: {acc:.3f}% over {n_eval_episodes} episodes")
            if acc > best_eval_acc:
                best_eval_acc = acc
                torch.save(net.state_dict(), best_ckpt)
                with open(best_acc_path, 'w') as f:
                    f.write(f"{best_eval_acc:.6f}")
                print(f"New best eval acc {best_eval_acc:.3f}% → saved {best_ckpt}")
    return net, env


@torch.no_grad()
def eval_accuracy_3stim(
    net: torch.nn.Module,
    env: IntDiscrim3_Intermediate,
    device: torch.device,
    n_episodes: int,
    layer2: bool,
):
    net.eval()
    correct = 0
    for i_episode in tqdm(range(n_episodes)):
        done = False
        env.reset()
        net.reinit_hid()
        while not done:
            pol, val, lin_act,_ = net.forward(
                torch.unsqueeze(torch.Tensor(env.observation).float(), dim=0).to(device)
            )
            if env.task_stage == 'init':
                new_obs, reward, done = env.step(1)
            elif env.task_stage == 'intermediate_choice_init':
                logits = net.actor(lin_act)[:, :2]
                probs = torch.nn.functional.softmax(logits, dim=1)
                act = Categorical(probs).sample().item()
                new_obs, reward, done = env.step(act)
            elif env.task_stage == 'choice_init':
                act = Categorical(pol).sample().item()
                new_obs, reward, done = env.step(act)
            else:
                new_obs, reward, done = env.step()
            env.observation = new_obs
        correct += int(env.correct_trial)
    acc = 100.0 * correct / float(n_episodes)
    return acc





def main():
    parser = argparse.ArgumentParser(description="Train SSM (3stim) and plot hidden states")
    parser.add_argument("--n_total_episodes", type=int, default=200000)
    parser.add_argument("--n_eval_episodes", type=int, default=100)
    parser.add_argument("--n_neurons", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--entropy", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--delay", type=int, default=30)
    parser.add_argument("--spike", action='store_true', default=False)
    parser.add_argument("--layer2", action='store_true', default=False)
    parser.add_argument("--save_dir", type=str, default="./training/3stim")
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--load_model", action='store_true', default=False,
                        help="Load a pretrained checkpoint and plot without training.")
    parser.add_argument("--model_path", type=str, default="3stim_best_model_spiking.pt",
                        help="Path to checkpoint to load when --load_best is set.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    save_dir_str = f'ssm_{args.n_neurons}_{args.lr}'
    if args.weight_decay != 0:
        save_dir_str += f'_wd{args.weight_decay}'
    if args.spike:
        save_dir_str += f'_spiking'
    if args.layer2:
        save_dir_str += f'_2layer'
    save_dir_str += f'_delay{args.delay}'
    save_dir = os.path.join(args.save_dir, save_dir_str)
    os.makedirs(save_dir, exist_ok=True)

    if args.load_model:
        # Initialize env and net, then load checkpoint
        env = IntDiscrim3_Intermediate(seed=args.seed, delay=args.delay, fixed_delay=False)
        ssm_params = {
            "P": args.n_neurons * 2,
            "C_init": "trunc_standard_normal",
            "discretization": "zoh",
            "dt_min": 0.001,
            "dt_max": 0.1,
            "conj_sym": True,
            "step_rescale": 1.0,
            "spike": args.spike,
            "layer2": args.layer2,
        }
        net = AC_SSM_stack(
            input_dimensions=3,
            action_dimensions=3,
            batch_size=1,
            hidden_dim=args.n_neurons,
            ssm_params=ssm_params,
            p_dropout=0.1,
        ).to(device)
        net.reinit_hid()
        if os.path.isfile(args.model_path):
            state = torch.load(args.model_path, map_location=device)
            net.load_state_dict(state)
            net.eval()
            print(f"Loaded checkpoint from {args.model_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {args.model_path}")
    else:
        net, env = train_3stim(
            n_total_episodes=args.n_total_episodes,
            n_neurons=args.n_neurons,
            lr=args.lr,
            weight_decay=args.weight_decay,
            entropy_weight=args.entropy,
            seed=args.seed,
            delay=args.delay,
            spike=args.spike,
            layer2=args.layer2,
            device=device,
            save_dir=save_dir,
            eval_every=args.eval_every,
            n_eval_episodes=args.n_eval_episodes,
        )

    # Evaluate accuracy with parameters frozen (no updates)
    eval_acc = eval_accuracy_3stim(
        net=net,
        env=env,
        device=device,
        n_episodes=args.n_eval_episodes,
        layer2=args.layer2,
    )
    print(f"Evaluation accuracy over {args.n_eval_episodes} episodes: {eval_acc:.3f}%")
    # Save only if improved vs existing best (skip when loading checkpoint)
    if not args.load_model:
        best_ckpt = os.path.join(save_dir, 'best_eval.pt')
        best_acc_path = os.path.join(save_dir, 'best_eval_acc.txt')
        prev_best = -1.0
        if os.path.exists(best_acc_path):
            try:
                with open(best_acc_path, 'r') as f:
                    prev_best = float(f.read().strip())
            except Exception:
                prev_best = -1.0
        if eval_acc > prev_best:
            torch.save(net.state_dict(), best_ckpt)
            with open(best_acc_path, 'w') as f:
                f.write(f"{eval_acc:.6f}")
            print(f"Saved evaluated model to {best_ckpt}")
        else:
            print(f"No improvement over best eval acc ({prev_best:.3f}%). Keeping {best_ckpt}.")
    analyze_model_example(model_path=args.model_path, n_neurons=args.n_neurons, delay=args.delay, fixed_delay=True, seed=args.seed)


if __name__ == "__main__":
    main()


