"""
Train/evaluate SSM actor-critic for lap counting and generate analysis plots.

"""
import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.distributions import Categorical
import pandas as pd
from envs.lap_counting import Laps_Counting
from agents.model_ssm_RL_laps import *
from basic_lap_state import sort_sub
import seaborn as sns


def vp_distance(pred, gt, q):
    """
    Victor–Purpura distance between two 1‑D event lists.
    pred, gt: 1‑D numpy arrays of event times (in seconds, sorted).
    q: cost per second for shifting an event.
    """
    m, n = len(pred), len(gt)
    # DP matrix: (m+1)×(n+1)
    D = np.zeros((m + 1, n + 1))
    D[0, :] = np.arange(n + 1)            # cost of n inserts
    D[:, 0] = np.arange(m + 1)            # cost of m deletes

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost_shift = D[i - 1, j - 1] + q * abs(pred[i - 1] - gt[j - 1])
            cost_del   = D[i - 1, j] + 1
            cost_ins   = D[i, j - 1] + 1
            D[i, j] = min(cost_shift, cost_del, cost_ins)

    return D[m, n]

def vp_score(pred, gt, q):
    dist = vp_distance(np.asarray(pred), np.asarray(gt), q)
    return 1.0 - dist / (len(pred) + len(gt))

def get_param_linear(epoch: int, start: float = 1.0, end: float = 0.1, total_epochs: int = 5000) -> float:
    if epoch <= 1:
        return start
    if epoch >= total_epochs:
        return end
    frac = (epoch - 1) / (total_epochs - 1)
    return start + (end - start) * frac

def plot_lap_choices(net, env, n_total_episodes=1000, spike=False, layer2=False, save_path='./figures'):
    net.eval()
    device = torch.device("cuda:0")
    all_laps = np.zeros([n_total_episodes, env.base_lap_count*env.lap_len+50])

    for i_episode in tqdm(range(n_total_episodes)):
        done = False
        observation = env.reset()
        if hasattr(net, 'reinit_hid'):
            net.reinit_hid()

        # Lists to store data for the current episode
        episode_rewards = []
        episode_saved_actions = []
         # Tracks correct (1) vs incorrect (0) lap reports within the episode
        pred = []
        while not done:
            # The observation needs to be a tensor for the network
            obs_tensor = torch.tensor([observation], dtype=torch.float32).to(device)

            # Forward pass through the network
            if spike:
                # The original forward pass calls are kept, assuming they match your network's interface
                if layer2:
                    pol, val, lin_act, _,_ = net.forward(obs_tensor)
                else:
                    pol, val, lin_act, _ = net.forward(obs_tensor)

            else:
                pol, val, lin_act = net.forward(obs_tensor)

            dist = Categorical(pol)
            action_tensor = dist.sample()
            action_to_take = action_tensor.item()
            log_prob = dist.log_prob(action_tensor)


            episode_saved_actions.append(SavedAction(log_prob, val, pol))

            if action_to_take==1:
                pred.append(env.elapsed_t)
            # Take action in the environment
            # MODIFICATION: The `step` function from `lap_counting.py` returns obs, reward, and task_stage.
            new_obs, reward, task_stage = env.step2(action_to_take)

            all_laps[i_episode][env.elapsed_t]= env.predicted_lap_count
            # print(reward)
            done = (task_stage == 'done') # The episode is 'done' only when the task_stage says so.
            episode_rewards.append(reward)

            observation = new_obs
    avg_laps = np.mean(all_laps, axis=0)  # Shape: [timestep]
    plt.clf()
    df = pd.DataFrame(all_laps)
    df_long = df.melt(var_name='Timestep', value_name='Value')
    df_long['Timestep'] = df_long['Timestep'].astype(int)

    # Plot using seaborn with confidence interval (standard deviation)
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_long, x='Timestep', y='Value', errorbar='sd', color='blue', linewidth=2)
    # Plot the summed result
    # plt.legend()
    plt.plot(avg_laps,)
    plt.xlabel("Elapsed Timestep")
    plt.ylabel("Average Predicted Count")
    plt.title(f"Average Predicted Count Over {env.base_lap_count} Laps")
    plt.grid(True)
    plt.tight_layout()
    save_path=os.path.join(save_path, f"figure_5c_{env.base_lap_count}_lap_choices.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Line successfully saved to {save_path}")


def train_laps(
    n_total_episodes: int,
    n_neurons: int,
    lr: float,
    weight_decay: float,
    entropy_weight: float,
    seed: int,
    lap_length: int,
    lap_count: int,
    spike: bool,
    layer2: bool,
    approx: bool,
    device: torch.device,
    save_dir: str,
    eval_every: int,
    n_eval_episodes: int,
):
    env = Laps_Counting(
        final_rwd=10,
        seed=seed,
        lap_length=lap_length,
        fixed_laps=lap_count,
        approx=approx,
        jitter=1.5,
        randomize_laps=False,
    )

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
        "lap_count": lap_count,
    }
    net = AC_SSM_stack(
        input_dimensions=2,
        action_dimensions=2,
        batch_size=1,
        hidden_dim=n_neurons,
        ssm_params=ssm_params,
        p_dropout=0.1,
    ).to(device)
    net.reinit_hid()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

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
        observation = env.reset()
        if hasattr(net, 'reinit_hid'):
            net.reinit_hid()

        episode_rewards = []
        episode_saved_actions = []

        while not done:
            obs_tensor = torch.tensor([observation], dtype=torch.float32).to(device)
            pol, val, lin_act = net.forward(obs_tensor)
            dist = Categorical(pol)
            action_tensor = dist.sample()
            action_to_take = action_tensor.item()
            log_prob = dist.log_prob(action_tensor)
            episode_saved_actions.append(SavedAction(log_prob, val, pol))

            new_obs, reward, task_stage = env.step2(action_to_take)
            done = (task_stage == 'done')
            episode_rewards.append(reward)
            observation = new_obs

        correct_trial[i_episode] = 1 if env.predicted_lap_count == env.true_lap_count else 0
        net.rewards = episode_rewards
        net.saved_actions = episode_saved_actions
        p_loss, v_loss, total_loss = finish_run(
            net, 1, optimizer, entropy_weight=get_param_linear(i_episode + 1, start=entropy_weight, total_epochs=n_total_episodes)
        )
        # periodic evaluation based on evaluation-only performance (now logs VP score too)
        if eval_every > 0 and (i_episode + 1) % eval_every == 0:
            acc, mean_vp, _, _ = eval_accuracy_and_vp_laps(
                net=net,
                env=env,
                device=device,
                n_episodes=n_eval_episodes,
            )
            print(f"Eval @ episode {i_episode + 1}: acc={acc:.3f}% vp={mean_vp:.3f} over {n_eval_episodes} episodes")
            if acc > best_eval_acc:
                best_eval_acc = acc
                torch.save(net.state_dict(), best_ckpt)
                with open(best_acc_path, 'w') as f:
                    f.write(f"{best_eval_acc:.6f}")
                print(f"New best eval acc {best_eval_acc:.3f}% → saved {best_ckpt}")

    return net, env


@torch.no_grad()
def collect_hidden_laps(net, env, device, n_episodes: int, lap_length: int, lap_count: int, spike: bool = False, layer2: bool = False):
    """
    Collect hidden states and spiking data from the network during evaluation episodes.

    Parameters:
    -----------
    net : neural network model
        The SSM network to collect hidden states from
    env : environment
        The laps counting environment
    device : torch.device
        Device to run the network on
    n_episodes : int
        Number of evaluation episodes
    lap_length : int
        Length of each lap
    lap_count : int
        Number of laps
    spike : bool, optional
        Whether the network is spiking (default: False)
    layer2 : bool, optional
        Whether the network has a second layer (default: False)

    Returns:
    --------
    full_resp1 : np.ndarray
        Hidden states from the first layer
    full_resp2 : np.ndarray or None
        Hidden states from the second layer (if layer2=True)
    spiking_entries1 : list
        Spiking data from the first layer (if spike=True)
    spiking_entries2 : list or None
        Spiking data from the second layer (if spike=True and layer2=True)
    """
    # Initialize arrays for storing hidden state data
    total_T = lap_length * lap_count + lap_count - 1 + 10
    n_neurons = net.hidden_dim
    full_resp1 = np.zeros((n_episodes, total_T, n_neurons), dtype=np.complex64)
    if layer2:
        full_resp2 = np.zeros((n_episodes, total_T, n_neurons), dtype=np.complex64)

    # Initialize containers for storing spiking data across episodes
    spiking_entries1, spiking_entries2 = [], []

    for i_episode in tqdm(range(n_episodes)):
        done = False
        observation = env.reset()

        if hasattr(net, 'reinit_hid'):
            net.reinit_hid()

        # Reset per-episode spiking logs
        if spike:
            spiking_row1 = [[]]
            if layer2:
                spiking_row2 = [[]]

        while not done:
            obs_tensor = torch.tensor([observation], dtype=torch.float32).to(device)

            if spike:
                if layer2:
                    pol, val, lin_act, new_spike_log1, new_spike_log2 = net.forward(obs_tensor)
                    spiking_row1[0].append(new_spike_log1.detach().cpu().numpy())
                    spiking_row2[0].append(new_spike_log2.detach().cpu().numpy())
                else:
                    pol, val, lin_act, new_spike_log1 = net.forward(obs_tensor)
                    spiking_row1[0].append(new_spike_log1.detach().cpu().numpy())
            else:
                pol, val, lin_act = net.forward(obs_tensor)

            # Collect hidden state data
            hidden_data1 = net.hidden_state1.clone().detach().cpu().numpy().squeeze()
            t_idx = env.elapsed_t
            if t_idx < full_resp1.shape[1]:
                full_resp1[i_episode, t_idx, :] = hidden_data1

            if layer2:
                hidden_data2 = net.hidden_state2.clone().detach().cpu().numpy().squeeze()
                if t_idx < full_resp2.shape[1]:
                    full_resp2[i_episode, t_idx, :] = hidden_data2

            # Sample action and step environment
            dist = Categorical(pol)
            action_tensor = dist.sample()
            action_to_take = action_tensor.item()

            new_obs, reward, task_stage = env.step2(action_to_take)
            done = (task_stage == 'done')
            observation = new_obs

        # Store spiking data for this episode
        if spike:
            spiking_entries1.append(spiking_row1)
            if layer2:
                spiking_entries2.append(spiking_row2)

    return full_resp1, full_resp2 if layer2 else None, spiking_entries1, spiking_entries2 if layer2 else None


def evaluate_and_plot_model(
    model_path: str = "best_laps.pt",
    n_eval_episodes: int = 1000,
    n_neurons: int = 80,
    seed: int = 2,
    lap_length: int = 30,
    lap_count: int = 4,
    spike: bool = False,
    layer2: bool = False,
    approx: bool = False,
    save_dir: str = "./figures",
):
    """
    Evaluate an existing model and generate plots.

    Parameters:
    -----------
    model_path : str
        Path to the saved model checkpoint (default: "best_laps.pt")
    n_eval_episodes : int
        Number of episodes to evaluate (default: 1000)
    n_neurons : int
        Number of neurons in the hidden layer (default: 80)
    seed : int
        Random seed for reproducibility (default: 2)
    lap_length : int
        Length of each lap in timesteps (default: 30)
    lap_count : int
        Number of laps to count (default: 4)
    spike : bool
        Whether the model is spiking (default: False)
    layer2 : bool
        Whether the model has a second layer (default: False)
    approx : bool
        Whether to use approximate timing (default: False)
    save_dir : str
        Directory to save results and plots (default: "./evaluation_results")
    """
    print(f"Evaluating model: {model_path}")

    # Set up device and seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create evaluation environment
    eval_env = Laps_Counting(
        final_rwd=10,
        seed=seed,
        lap_length=lap_length,
        fixed_laps=lap_count,
        approx=False,
        jitter=1.5,  # Use minimal jitter for evaluation
        randomize_laps=False,
        eval_hold=True
    )

    # Create the network architecture
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
        "lap_count": lap_count,
    }

    net = AC_SSM_stack(
        input_dimensions=2,
        action_dimensions=2,
        batch_size=1,
        hidden_dim=n_neurons,
        ssm_params=ssm_params,
        p_dropout=0.1,
    ).to(device)

    # Load the model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from {model_path}")
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    # Evaluate accuracy and VP score
    print(f"Evaluating accuracy over {n_eval_episodes} episodes...")
    eval_acc, mean_vp, last_pred, last_gt = eval_accuracy_and_vp_laps(net, eval_env, device, n_eval_episodes)
    print(f"Acc: {eval_acc:.3f}% | vp_score: {mean_vp:.3f} | pred(last): {last_pred} | gt(last): {last_gt}")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Save evaluation results
    results_file = os.path.join(save_dir, "laps_evaluation_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Evaluation Episodes: {n_eval_episodes}\n")
        f.write(".6f")
        f.write(f"Parameters:\n")
        f.write(f"  Neurons: {n_neurons}\n")
        f.write(f"  Lap Length: {lap_length}\n")
        f.write(f"  Lap Count: {lap_count}\n")
        f.write(f"  Spike: {spike}\n")
        f.write(f"  Layer2: {layer2}\n")
        f.write(f"  Approx: {approx}\n")
        f.write(f"  Seed: {seed}\n")

    print(f"Results saved to: {results_file}")

    # Collect hidden states for plotting
    print("Collecting hidden states for analysis...")
    full_resp1, _, _, _ = collect_hidden_laps(
        net=net,
        env=eval_env,
        device=device,
        n_episodes=max(100, n_eval_episodes),
        lap_length=lap_length,
        lap_count=lap_count,
        spike=spike,
        layer2=layer2,
    )
    np.save(os.path.join(save_dir, f'./lap_counting_{seed}_activity.npy'),full_resp1)
    # Generate hidden state plots
    print("Generating hidden state plots...")
    plot_sorted_laps(full_resp1, save_dir=save_dir, lap_length=lap_length, lap_count=lap_count)

    print(f"All plots saved to: {save_dir}")
    print("Evaluation complete!")

    return eval_acc, save_dir


@torch.no_grad()
def eval_accuracy_laps(net, env, device, n_episodes: int):
    net.eval()
    correct = 0
    for i_episode in range(n_episodes):
        done = False
        observation = env.reset()
        if hasattr(net, 'reinit_hid'):
            net.reinit_hid()
        while not done:
            obs_tensor = torch.tensor([observation], dtype=torch.float32).to(device)
            pol, val, lin_act = net.forward(obs_tensor)
            dist = Categorical(pol)
            action_tensor = dist.sample()
            action_to_take = action_tensor.item()
            new_obs, reward, task_stage = env.step2(action_to_take)
            done = (task_stage == 'done')
            observation = new_obs
        correct += int(env.predicted_lap_count == env.true_lap_count)
    acc = 100.0 * correct / float(n_episodes)
    return acc


@torch.no_grad()
def eval_accuracy_and_vp_laps(net, env, device, n_episodes: int):
    net.eval()
    correct = 0
    vps = []
    last_pred = []
    last_gt = []
    for i_episode in tqdm(range(n_episodes)):
        done = False
        observation = env.reset()
        if hasattr(net, 'reinit_hid'):
            net.reinit_hid()
        pred = []
        while not done:
            obs_tensor = torch.tensor([observation], dtype=torch.float32).to(device)
            pol, val, lin_act = net.forward(obs_tensor)
            dist = Categorical(pol)
            action_tensor = dist.sample()
            action_to_take = action_tensor.item()

            new_obs, reward, task_stage = env.step2(action_to_take)
            if action_to_take == 1 and env.hold<2:
                pred.append(env.elapsed_t)
            done = (task_stage == 'done')
            observation = new_obs
        correct += int(env.predicted_lap_count == env.true_lap_count)
        vps.append(vp_score(pred, env.lap_ends, 0.1))
        if i_episode == n_episodes - 1:
            last_pred = pred
            last_gt = list(env.lap_ends)
    acc = 100.0 * correct / float(n_episodes)
    mean_vp = float(np.mean(vps)) if len(vps) > 0 else 0.0
    return acc, mean_vp, last_pred, last_gt


def _split_units_by_change(data: np.ndarray, start_idx: int, end_idx: int, threshold: float):
    """
    Separate unit indices into high- and low-change groups within a time window.
    """
    end_idx = min(end_idx, data.shape[1])
    if start_idx >= end_idx or end_idx - start_idx < 2:
        all_indices = np.arange(data.shape[0]).tolist()
        return [], all_indices

    data_window = data[:, start_idx:end_idx]
    abs_change = np.abs(np.diff(data_window, axis=1))
    is_high = np.any(abs_change > threshold, axis=1)
    indices = np.arange(data.shape[0])
    high = indices[is_high].tolist()
    low = indices[~is_high].tolist()
    return high, low


def _save_lap_heatmap(
    matrix: np.ndarray,
    lap_boundaries: list,
    lap_labels: list,
    title: str,
    file_path: str,
):
    """
    Save a heatmap with lap-specific x-axis labels.
    """
    fig, ax = plt.subplots(figsize=(20, 6))
    cax = ax.imshow(matrix, aspect='auto', cmap='jet', interpolation='none')
    divider = make_axes_locatable(ax)
    cb_ax = divider.append_axes("right", size="2.5%", pad=0.5)
    fig.colorbar(cax, cax=cb_ax, label='Normalized Activation')
    for boundary in lap_boundaries[1:-1]:
        ax.axvline(boundary, color='white', lw=1, linestyle='--')
    midpoints = [
        (lap_boundaries[i] + lap_boundaries[i + 1]) / 2
        for i in range(len(lap_labels))
    ]
    ax.set_xticks(midpoints)
    ax.set_xticklabels(lap_labels, rotation=0, fontsize=10, fontweight='bold')
    ax.set_xlabel('Lap')
    ax.set_ylabel('Unit # (Sorted by Peak Activity)')
    ax.set_title(title)
    plt.tight_layout(rect=[0, 0, 0.93, 1])
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path, dpi=300)
    plt.close(fig)


def _save_lap_panel(matrix: np.ndarray, lap_boundaries: list, lap_labels: list, title: str, file_path: str):
    """
    Plot each lap slice in a single figure (subplots), removing all-NaN rows.
    """
    n_laps = len(lap_labels)
    fig, axes = plt.subplots(1, n_laps, figsize=(4 * n_laps, 4), sharey=True)
    if n_laps == 1:
        axes = [axes]
    cax = None
    for i, ax in enumerate(axes):
        start = lap_boundaries[i]
        end = lap_boundaries[i + 1]
        slice_mat = matrix[:, start:end]
        # Drop units that have no data in this lap slice (all NaN across time).
        row_mask = ~np.all(np.isnan(slice_mat), axis=1)
        slice_mat = slice_mat[row_mask]
        if slice_mat.size == 0:
            ax.set_visible(False)
            continue
        slice_mat = np.nan_to_num(slice_mat, nan=0.0)
        cax = ax.imshow(slice_mat, aspect='auto', cmap='jet', interpolation='none')
        ax.set_title(lap_labels[i])
        ax.set_xlabel('Time')
        if i == 0:
            ax.set_ylabel('Unit #')
    fig.suptitle(title)
    # No colorbar for panel plots to avoid overlap.
    plt.tight_layout(rect=[0, 0, 0.98, 0.95])
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path, dpi=300)
    plt.close(fig)


def plot_sorted_laps(full_resp1: np.ndarray, save_dir: str, lap_length: int, lap_count: int):
    n_neurons = full_resp1.shape[2]
    for i_neuron in range(n_neurons):
        # swap 0's in stim1_resp and stim2_resp with nan
        full_resp1[:, :, i_neuron][full_resp1[:, :, i_neuron] == 0] = np.nan
        # normalize across stim1_resp, stim2_resp, and delay_resp
        min_act = np.nanmin(np.concatenate(
            (full_resp1[:, :, i_neuron],), axis=1))
        max_act = np.nanmax(np.concatenate(
            (full_resp1[:, :, i_neuron],), axis=1))
        full_resp1[:, :, i_neuron] = (full_resp1[:, :, i_neuron] - min_act) / (max_act - min_act)

        full_resp1[:, :, i_neuron][np.isnan(full_resp1[:, :, i_neuron])] = 0
    # restrict to a reasonable window (e.g., drop trailing zeros if any)
    sorted_real_matrix, _, _, _ = sort_sub(full_resp1, laps=lap_count)

    total_timesteps = sorted_real_matrix.shape[1]
    if lap_count == 0:
        raise ValueError("lap_count must be greater than zero.")
    lap_segment = total_timesteps // lap_count
    lap_boundaries = [i * lap_segment for i in range(lap_count + 1)]
    lap_labels = [f"L{i + 1}" for i in range(lap_count)]

    os.makedirs(save_dir, exist_ok=True)
    _save_lap_heatmap(
        sorted_real_matrix,
        lap_boundaries,
        lap_labels,
        title="Sorted Lap Hidden State",
        file_path=os.path.join(save_dir, "figure_5c_sorted.png"),
    )
    return


def main():
    parser = argparse.ArgumentParser(description="Train SSM (laps), evaluate model, and plot hidden states")
    parser.add_argument("--mode", type=str, choices=['train', 'eval'], default='eval',
                       help="Mode: 'train' for training or 'eval' for evaluation")
    parser.add_argument("--model_path", type=str, default="../data/lap_best_model.pt",
                       help="Path to model for evaluation (default: best_laps.pt)")
    # Training parameters
    parser.add_argument("--n_total_episodes", type=int, default=10000)
    parser.add_argument("--n_eval_episodes", type=int, default=5000)
    parser.add_argument("--n_neurons", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--entropy", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--lap_length", type=int, default=30)
    parser.add_argument("--lap_count", type=int, default=4)
    parser.add_argument("--spike", action='store_true', default=False)
    parser.add_argument("--layer2", action='store_true', default=False)
    parser.add_argument("--approx", action='store_true', default=False)
    parser.add_argument("--save_dir", type=str, default="./training/lap_counting")
    parser.add_argument("--eval_every", type=int, default=100)

    # Evaluation parameters
    parser.add_argument("--eval_save_dir", type=str, default="./figures/",
                       help="Directory to save evaluation results")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.mode == 'eval':
        # Evaluation mode
        eval_acc, save_dir = evaluate_and_plot_model(
            model_path=args.model_path,
            n_eval_episodes=args.n_eval_episodes,
            n_neurons=args.n_neurons,
            seed=args.seed,
            lap_length=args.lap_length,
            lap_count=args.lap_count,
            spike=args.spike,
            layer2=args.layer2,
            approx=args.approx,
            save_dir=args.eval_save_dir,
        )
        return

    # Training mode
    save_dir_str = f'ssm_{args.n_neurons}_{args.lr}'
    if args.weight_decay != 0:
        save_dir_str += f'_wd{args.weight_decay}'
    if args.spike:
        save_dir_str += f'_spiking'
    if args.layer2:
        save_dir_str += f'_2layer'
    save_dir_str += f'_lap{args.lap_length}'
    save_dir = os.path.join(args.save_dir, save_dir_str)
    os.makedirs(save_dir, exist_ok=True)

    net, env = train_laps(
        n_total_episodes=args.n_total_episodes,
        n_neurons=args.n_neurons,
        lr=args.lr,
        weight_decay=args.weight_decay,
        entropy_weight=args.entropy,
        seed=args.seed,
        lap_length=args.lap_length,
        lap_count=args.lap_count,
        spike=args.spike,
        layer2=args.layer2,
        approx=args.approx,
        device=device,
        save_dir=save_dir,
        eval_every=args.eval_every,
        n_eval_episodes=args.n_eval_episodes,
    )

    # Evaluate accuracy with parameters frozen (no updates)
    eval_acc, mean_vp, last_pred, last_gt = eval_accuracy_and_vp_laps(
        net=net,
        env=env,
        device=device,
        n_episodes=args.n_eval_episodes,
    )
    print(f"Evaluation over {args.n_eval_episodes} eps — Acc: {eval_acc:.3f}% | vp_score: {mean_vp:.3f} | pred(last): {last_pred} | gt(last): {last_gt}")
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

    full_resp1, _, _, _ = collect_hidden_laps(
        net=net,
        env=env,
        device=device,
        n_episodes=args.n_eval_episodes,
        lap_length=args.lap_length,
        lap_count=args.lap_count,
        spike=args.spike,
        layer2=args.layer2,
    )
    plot_sorted_laps(full_resp1, save_dir=save_dir, lap_length=args.lap_length, lap_count=args.lap_count)
    # Skip additional plots to keep only figure_5c_sorted.png


if __name__ == "__main__":
    main()


