import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

def save_lambda_bar_from_model(model: torch.nn.Module, save_path: str, default_dt: float = 0.05) -> str:

    def _ensure_lambda_bar(cell: torch.nn.Module) -> torch.Tensor:
        lam = getattr(cell, 'lambda_bar', None)
        if lam is not None:
            return lam
        dzoh = ddirac = dasync = None
        try:
            from agents.model_ssm_stack_RL import discretize_zoh as _dzoh, discretize_dirac as _ddirac, discretize_async as _dasync
            dzoh, ddirac, dasync = _dzoh, _ddirac, _dasync
        except Exception:
            pass
        if dzoh is None:
            try:
                from agents.model_ssm_RL_laps import discretize_zoh as _dzoh2, discretize_dirac as _ddirac2, discretize_async as _dasync2
                dzoh, ddirac, dasync = _dzoh2, _ddirac2, _dasync2
            except Exception:
                pass
        fn_name = getattr(getattr(cell, 'discretize_fn', None), '__name__', None)
        if fn_name == 'discretize_dirac' and ddirac is not None:
            disc_fn = ddirac
        elif fn_name == 'discretize_async' and dasync is not None:
            disc_fn = dasync
        else:
            disc_fn = dzoh if dzoh is not None else None
        if disc_fn is None:
            raise RuntimeError("Discretization functions not available for computing lambda_bar.")
        step = cell.step_rescale * torch.exp(cell.log_step)
        lam_bar, _ = disc_fn(cell.Lambda_param, step, default_dt)
        cell.lambda_bar = lam_bar
        return lam_bar

    out = {}
    if hasattr(model, 'ssm_cell1'):
        lam1 = _ensure_lambda_bar(model.ssm_cell1)
        out['ssm_cell1_lambda_bar'] = lam1.detach().cpu()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save(out, save_path)
    return save_path

def save_lambda_from_model(model: torch.nn.Module, save_path: str) -> str:
    out = {}
    if hasattr(model, 'ssm_cell1') and hasattr(model.ssm_cell1, 'Lambda_param'):
        out['ssm_cell1_lambda'] = model.ssm_cell1.Lambda_param.detach().cpu()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save(out, save_path)
    return save_path

def save_B_from_model(model: torch.nn.Module, save_path: str) -> str:
    out = {}
    if hasattr(model, 'ssm_cell1') and hasattr(model.ssm_cell1, 'B'):
        out['ssm_cell1_B'] = model.ssm_cell1.B.detach().cpu()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save(out, save_path)
    return save_path

def save_C_tilde_from_model(model: torch.nn.Module, save_path: str) -> str:
    out = {}
    if hasattr(model, 'ssm_cell1') and hasattr(model.ssm_cell1, 'C_tilde'):
        out['ssm_cell1_C_tilde'] = model.ssm_cell1.C_tilde.detach().cpu()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save(out, save_path)
    return save_path

def freeze_ssm_params(net, freeze_lambda=False, freeze_B=False):
    if freeze_lambda:
        if hasattr(net, 'ssm_cell1') and hasattr(net.ssm_cell1, 'Lambda_param'):
            net.ssm_cell1.Lambda_param.requires_grad_(False)
            print("Froze Lambda matrix (A) for the SSM layer.")
    if freeze_B:
        if hasattr(net, 'ssm_cell1') and hasattr(net.ssm_cell1, 'B'):
            net.ssm_cell1.B.requires_grad_(False)
            print("Froze input projection B for the SSM layer.")

def build_net(hidden_type, n_neurons, spike, device,
              init_mode="hippo", perturb_eps=0.1, rand_init=False,
              rnn_spike=False, p_dropout=0.1):
    if hidden_type == "ssm":
        ssm_params = {
            "P": n_neurons * 2,
            "C_init": "trunc_standard_normal",
            "discretization": "zoh",
            "dt_min": 0.001,
            "dt_max": 0.1,
            "conj_sym": True,
            "rand_init": rand_init,
            "init_mode": init_mode,
            "init_perturb_eps": perturb_eps,
            "step_rescale": 1.0,
            "spike": spike,
        }
        return AC_SSM_stack(
            input_dimensions=3,
            action_dimensions=3,
            batch_size=1,
            hidden_dim=n_neurons,
            ssm_params=ssm_params,
            p_dropout=p_dropout,
        ).to(device)

    from agents.model_lstm_RL import AC_RNN
    return AC_RNN(
        input_dimensions=3,
        action_dimensions=3,
        hidden_dim=n_neurons,
        hidden_type=hidden_type,
        batch_size=1,
        p_dropout=p_dropout,
        spike=spike,
        rnn_spike=rnn_spike,
    ).to(device)

def freeze_backbone_params(net, hidden_type, freeze_lambda=False, freeze_B=False):
    if hidden_type == "ssm":
        freeze_ssm_params(net, freeze_lambda=freeze_lambda, freeze_B=freeze_B)
    else:
        from agents.model_lstm_RL import freeze_rnn_params
        freeze_rnn_params(net, freeze_lambda=freeze_lambda, freeze_B=freeze_B)

def build_freeze_suffix(freeze_lambda: bool, freeze_B: bool) -> str:
    tags = []
    if freeze_lambda:
        tags.append("freezeLambda")
    if freeze_B:
        tags.append("freezeB")
    return f"_{'_'.join(tags)}" if tags else ""

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
    device: torch.device,
    save_dir: str,
    eval_every: int,
    n_eval_episodes: int,
    rand_init: bool,
    resume_checkpoint: str | None,
    freeze_lambda: bool,
    freeze_B: bool,
    init_mode: str = "hippo",
    perturb_eps: float = 0.1,
    fixed_delay: bool = False,
    hidden_type: str = "ssm",
    rnn_spike: bool = False,
):
    env = IntDiscrim3_Intermediate(seed=seed, delay=delay, fixed_delay=fixed_delay)

    net = build_net(
        hidden_type=hidden_type,
        n_neurons=n_neurons,
        spike=spike,
        device=device,
        init_mode=init_mode,
        perturb_eps=perturb_eps,
        rand_init=rand_init,
        rnn_spike=rnn_spike,
    )
    if resume_checkpoint:
        if os.path.isfile(resume_checkpoint):
            state = torch.load(resume_checkpoint, map_location=device)
            net.load_state_dict(state)
            print(f"Loaded checkpoint for continued training from {resume_checkpoint}")
        else:
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
    if freeze_lambda or freeze_B:
        freeze_backbone_params(net, hidden_type,
                               freeze_lambda=freeze_lambda, freeze_B=freeze_B)
    file_suffix = build_freeze_suffix(freeze_lambda, freeze_B)
    try:
        torch.save(net.state_dict(), os.path.join(save_dir, f'initial_rand_model_{seed}{file_suffix}.pt'))
        save_lambda_from_model(net, os.path.join(save_dir, f'initial_lambda_{seed}{file_suffix}.pt'))
        save_lambda_bar_from_model(net, os.path.join(save_dir, f'initial_lambda_bar_{seed}{file_suffix}.pt'))
        save_B_from_model(net, os.path.join(save_dir, f'initial_B_{seed}{file_suffix}.pt'))
        save_C_tilde_from_model(net, os.path.join(save_dir, f'initial_C_tilde_{seed}{file_suffix}.pt'))
    except Exception as e:
        print(f"Warning: could not save initial model/lambda info from loaded model: {e}")
    net.reinit_hid()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    policy_hist, loss_hist, p_loss_hist, v_loss_hist = [], [], [], []
    correct_trial = np.zeros(n_total_episodes, dtype=np.int8)

    best_ckpt = os.path.join(save_dir, f'best_eval_{seed}{file_suffix}.pt')
    best_acc_path = os.path.join(save_dir, f'best_eval_acc_{seed}{file_suffix}.txt')
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
            pol, val, lin_act = net.forward(
                torch.unsqueeze(torch.Tensor(env.observation).float(), dim=0).to(device)
            )[:3]

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

        if eval_every > 0 and (i_episode + 1) % eval_every == 0:
            acc = eval_accuracy_3stim(
                net=net,
                env=env,
                device=device,
                n_episodes=n_eval_episodes,
            )
            print(f"Eval @ episode {i_episode + 1}: {acc:.3f}% over {n_eval_episodes} episodes")
            if acc > best_eval_acc:
                best_eval_acc = acc
                torch.save(net.state_dict(), best_ckpt)
                with open(best_acc_path, 'w') as f:
                    f.write(f"{best_eval_acc:.6f}")
                print(f"New best eval acc {best_eval_acc:.3f}% → saved {best_ckpt}")
    try:
        torch.save(net.state_dict(), os.path.join(save_dir, f'final_model_{seed}{file_suffix}.pt'))
        save_lambda_from_model(net, os.path.join(save_dir, f'final_lambda_{seed}{file_suffix}.pt'))
        save_lambda_bar_from_model(net, os.path.join(save_dir, f'final_lambda_bar_{seed}{file_suffix}.pt'))
        save_B_from_model(net, os.path.join(save_dir, f'final_B_{seed}{file_suffix}.pt'))
        save_C_tilde_from_model(net, os.path.join(save_dir, f'final_C_tilde_{seed}{file_suffix}.pt'))
    except Exception as e:
        print(f"Warning: could not save final model/lambda info: {e}")
    return net, env

@torch.no_grad()
def eval_accuracy_3stim(
    net: torch.nn.Module,
    env: IntDiscrim3_Intermediate,
    device: torch.device,
    n_episodes: int,
):
    net.eval()
    correct = 0
    for i_episode in tqdm(range(n_episodes)):
        done = False
        env.reset()
        net.reinit_hid()
        while not done:
            pol, val, lin_act = net.forward(
                torch.unsqueeze(torch.Tensor(env.observation).float(), dim=0).to(device)
            )[:3]
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
    parser.add_argument("--n_total_episodes", type=int, default=250000)
    parser.add_argument("--n_eval_episodes", type=int, default=100)
    parser.add_argument("--n_neurons", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--entropy", type=float, default=0.3)
    parser.add_argument("--thr", type=float, default=0.4)
    parser.add_argument("--fig_index", type=str, default="2b",
                        help="Index string used in saved figure filenames (e.g., '2b').")
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--delay", type=int, default=30)
    parser.add_argument("--spike", action='store_true', default=True,
                        help="Spiking SSM (surrogate-gradient STE). On by default.")
    parser.add_argument("--no_spike", dest='spike', action='store_false',
                        help="Disable the spiking nonlinearity.")
    parser.add_argument("--init_method", type=str,
                        choices=["hippo", "rand_complex", "spectrum_matched",
                                 "freq_matched", "perturbed_hippo", "alt_basis",
                                 "s4d_lin", "s4d_inv", "real_diagonal"],
                        default="hippo",
                        help="SSM initialization mode.")
    parser.add_argument("--perturb_eps", type=float, default=0.1,
                        help="Noise scale for --init_method perturbed_hippo.")
    parser.add_argument("--hidden_type", type=str,
                        choices=["ssm", "lstm"],
                        default="ssm",
                        help="Recurrent backbone. 'ssm' is the manuscript's "
                             "AC_SSM_stack; the rest use the AC_RNN backbone "
                             "extracted from linclab/deeprl-timecells "
                             "(agents/model_lstm_RL.py). --init_method and "
                             "--perturb_eps are SSM-only and ignored otherwise.")
    parser.add_argument("--rnn_spike", action='store_true', default=False,
                        help="Apply the surrogate-gradient spike to the LSTM "
                             "backbone's cell output too. Off by default: "
                             "--spike is an SSM setting and only fixes the "
                             "forward() return arity for the LSTM backbone. See "
                             "agents/model_lstm_RL.py.")
    parser.add_argument("--partial", action='store_true', default=False)
    parser.add_argument("--save_dir", type=str, default="./training/3stim")
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--freeze_lambda", action='store_true', default=False,
                        help="Freeze Lambda (A) parameters during (re)training.")
    parser.add_argument("--freeze_B", action='store_true', default=False,
                        help="Freeze B projection matrices during (re)training.")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to a checkpoint to load before continuing training.")
    parser.add_argument("--load_model", action='store_true', default=False,
                        help="Load a pretrained checkpoint and plot without training.")
    parser.add_argument("--model_path", type=str, default="../data/3stim_best_model_spiking.pt",
                        help="Path to checkpoint to load when --load_best is set.")
    parser.add_argument("--heatmap", action='store_true', default=False,
                        help="Freeze B projection matrices during (re)training.")
    parser.add_argument("--fixed_delay", action='store_true', default=False,
                        help="Use a single fixed delay of --delay steps every episode. "
                             "Default (False) matches the env's historical behavior: "
                             "delay_set = range(10, delay, 10), which SAMPLES SHORTER "
                             "delays and never actually tests the requested delay itself.")
    args = parser.parse_args()

    from agents.model_ssm_stack_RL import AC_SSM_stack as _AC_SSM_stack, finish_trial as _finish_trial, SavedAction as _SavedAction
    globals().update(AC_SSM_stack=_AC_SSM_stack, finish_trial=_finish_trial, SavedAction=_SavedAction)
    rand_init_flag = (args.init_method == "rand_complex")
    run_suffix = build_freeze_suffix(args.freeze_lambda, args.freeze_B)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    save_dir_str = f'{args.hidden_type}_{args.n_neurons}_{args.lr}'
    if args.weight_decay != 0:
        save_dir_str += f'_wd{args.weight_decay}'
    if args.spike:
        save_dir_str += f'_spiking'
    if args.rnn_spike and args.hidden_type != "ssm":
        save_dir_str += f'_rnnspike'
    save_dir_str += f'_delay{args.delay}'
    if args.fixed_delay:
        save_dir_str += '_fixedDelay'
    save_dir = os.path.join(args.save_dir, save_dir_str)
    os.makedirs(save_dir, exist_ok=True)

    if args.load_model:
        env = IntDiscrim3_Intermediate(seed=args.seed, delay=args.delay, fixed_delay=args.fixed_delay)
        net = build_net(
            hidden_type=args.hidden_type,
            n_neurons=args.n_neurons,
            spike=args.spike,
            device=device,
            init_mode=args.init_method,
            perturb_eps=args.perturb_eps,
            rand_init=rand_init_flag,
            rnn_spike=args.rnn_spike,
        )
        if args.freeze_lambda or args.freeze_B:
            freeze_backbone_params(net, args.hidden_type,
                                   freeze_lambda=args.freeze_lambda,
                                   freeze_B=args.freeze_B)
        net.reinit_hid()

        if os.path.isfile(args.model_path):
            state = torch.load(args.model_path, map_location=device)
            net.load_state_dict(state)
            net.eval()
            print(f"Loaded checkpoint from {args.model_path}")
            if args.partial:
                net.partial_reset3()
                net.to(device)

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
            device=device,
            save_dir=save_dir,
            eval_every=args.eval_every,
            n_eval_episodes=args.n_eval_episodes,
            rand_init=rand_init_flag,
            resume_checkpoint=args.resume_checkpoint,
            freeze_lambda=args.freeze_lambda,
            freeze_B=args.freeze_B,
            init_mode=args.init_method,
            perturb_eps=args.perturb_eps,
            fixed_delay=args.fixed_delay,
            hidden_type=args.hidden_type,
            rnn_spike=args.rnn_spike,
        )

    eval_acc = eval_accuracy_3stim(
        net=net,
        env=env,
        device=device,
        n_episodes=args.n_eval_episodes,
    )
    print(f"Evaluation accuracy over {args.n_eval_episodes} episodes: {eval_acc:.3f}%")
    if not args.load_model:
        best_ckpt = os.path.join(save_dir, f'best_eval_{args.seed}{run_suffix}.pt')
        best_acc_path = os.path.join(save_dir, f'best_eval_acc_{args.seed}{run_suffix}.txt')
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
    if args.heatmap:
        analyze_model_example(
            model_path=args.model_path,
            n_neurons=args.n_neurons,
            delay=args.delay,
            n_inferences=1000,
            fixed_delay=True,
            seed=args.seed,
            thr=args.thr,
            fig_index=args.fig_index,
        )

if __name__ == "__main__":
    main()

