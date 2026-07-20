import os
import random
import argparse
import re
import gc
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.distributions import Categorical
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
# Use the same model and environment as train_and_plot_3stim.py
from agents.model_ssm_stack_RL import *
from envs.int_discrim import IntDiscrim3_Intermediate

# def avg_resp(total_resp):
#     """
#     Average the responses across episodes
#     """
#     np.seterr(divide='ignore', invalid='ignore')
#     n_neurons = np.shape(total_resp)[2]
#     segments = np.moveaxis(total_resp, 0, 1)
#     unsorted_matrix = np.zeros((n_neurons, len(segments)))  # len(segments) is also len_delay
#     for i in range(len(segments)):  # at timestep i
#         averages = np.mean(segments[i],
#                            axis=0)  # 1 x n_neurons, each entry is the average response of this neuron at this time step across episodes
#         unsorted_matrix[:, i] = np.transpose(
#             averages)
#     return unsorted_matrix
#
# def select_idx(resp, threshold=0.1):
#     indices = []
#     resp = avg_resp(resp)
#     # Performs Fourier Transform on each feature A 2D NumPy array with shape [n_features, n_timesteps]
#     # , return a list of indices of channels with frequency above threshold
#     # frequencies = np.fft.rfftfreq(n_timesteps, d=1)
#     #
#     # for i in range(n_features):
#     #     # Perform Fast Fourier Transform (FFT) for real input
#     #     fft_coeffs = np.fft.rfft(data[i, :])
#     #
#     #     # record indices corresponding to frequencies above the threshold
#     #
#     #     # Perform Inverse Fast Fourier Transform (IFFT)
#     #     # n=n_timesteps ensures the output has the original length, handling potential
#     #     # odd/even length discrepancies with rfft.
#     #     filtered_data[i, :] = np.fft.irfft(fft_coeffs, n=n_timesteps)
#     return indices
#
# def run_sim(net, env, indices):
#     acc = 0.0
#     # new_obs, reward, done = env.step()
#     # new_obs, reward, done = env.step(action_to_take)
#     correct_trial = []
#     for i_episode in range(100):
#         done = False
#         env.reset()
#         net.reinit_hid()
#         while not done:
#             pol, val, lin_act = net.forward(torch.unsqueeze(torch.Tensor(env.observation).float(), dim=0).to(device), lesion_idx=indices)
#             if env.task_stage == 'init':
#                 action_to_take = 1
#                 new_obs, reward, done = env.step(action_to_take)
#             elif env.task_stage == 'choice_init':
#                 final_dist = Categorical(pol)
#                 final_act_tensor = final_dist.sample()
#                 action_to_take = final_act_tensor.item()
#                 new_obs, reward, done = env.step(action_to_take)
#                 correct_trial[i_episode] = env.correct_trial
#                 # record accuracy in correct trial
#             else:
#                 new_obs, reward, done = env.step()
#     return acc
#
# def sweep_lesion(net, resp, env):
#     indices = select_idx(resp)
#     acc_log = []
#     for i in range(len(indices)):
#         acc_log.append(net, env, indices[:i])
#     return acc_log
def plot_feature_spectra(data: np.ndarray):

    all_blocks = [data]
    concat     = np.concatenate(all_blocks, axis=1)
    labels     = ['delay1']
    lengths    = [m.shape[1] for m in all_blocks]
    print("final shape:", concat.shape)  # → (50, 50 + 30 + T2 + 30 + T3)

    # 1) Compute cumulative boundaries and total time
    boundaries = np.cumsum([0] + lengths)   # e.g. [0, 50, 80, …, total_time]
    total_time = boundaries[-1]

    fig, ax = plt.subplots(figsize=(20, 6))
    cax = ax.imshow(concat, aspect='auto', cmap='jet')
    fig.colorbar(cax, ax=ax, label='Normalized Activation')

    # 1) Draw separators
    for x in boundaries:
        ax.axvline(x, color='white', lw=1)

    # 2) Numeric ticks at the boundaries
    ax.set_xlim(0, total_time)
    ax.set_xticks(boundaries)
    ax.set_xticklabels(boundaries, rotation=0)
    ax.tick_params(axis='x', which='major', pad=8)  # move numbers up a bit

    # 3) Phase labels below the axis
    for start, end, lbl in zip(boundaries[:-1], boundaries[1:], labels):
        mid = (start + end) / 2
        ax.text(
            mid,                            # x position
            -0.06,                          # y position in axes coords
            lbl,
            transform=ax.get_xaxis_transform(),
            ha='center', va='top',
            fontsize=10
        )

    # 4) Tidy up
    ax.set_ylabel('Unit # (sorted by stim1)')
    ax.set_xlabel('Time (timesteps)')
    ax.set_title('SSM 50 units hidden state 1 activity')
    plt.subplots_adjust(bottom=0.1)

    # 2) nudge the x‐axis label further down
    ax.set_xlabel('Time (timesteps)', labelpad=20)

    # 3) if you find the phase‐text still too close, move it further down:
    for txt in ax.texts:
        # only adjust the phase‐labels (they all have va='top' at y<0)
        x, y = txt.get_position()
        if y < 0:
            txt.set_y(-0.1)   # try -0.12, -0.15, … until it’s clear of the xlabel

    plt.draw()
    plt.tight_layout()
    # plt.savefig(f'./sorted_hi_freq.png')
    plt.show()
    plt.close()

def avg_resp(total_resp):
    """
    Average the responses across episodes.
      total_resp: np.ndarray of shape (n_episodes, n_timesteps, n_neurons)
    returns:
      unsorted_matrix: np.ndarray of shape (n_neurons, n_timesteps)
    """
    # ignore divide-by-zero warnings just in case some neuron is silent
    np.seterr(divide='ignore', invalid='ignore')

    # move episodes axis to position 1: (timesteps, episodes, neurons)
    segments = np.moveaxis(total_resp, 0, 1)
    n_neurons = total_resp.shape[2]
    n_timesteps = segments.shape[0]

    out = np.zeros((n_neurons, n_timesteps))
    for t in range(n_timesteps):
        # segments[t] is shape (n_episodes, n_neurons)
        out[:, t] = segments[t].mean(axis=0)
    return out

def select_high_frequency_neurons_sorted(data, sampling_rate=1.0, min_high_frequency=0.1, power_threshold=0.1):
    """
    Selects neuron IDs that exhibit significant power in high-frequency bands,
    and returns them sorted by their highest significant frequency (from high to low).

    Args:
        data (np.ndarray): A 2D NumPy array of shape [n_neurons, n_timesteps]
                           containing the activity of each neuron over time.
        sampling_rate (float): The sampling rate of the data (e.g., in Hz or samples per second).
        min_high_frequency (float): The minimum frequency (in the same units as sampling_rate)
                                    to be considered "high". Only frequencies above this value
                                    will be analyzed.
        power_threshold (float): The power threshold. If any frequency component in the
                                 high-frequency range for a neuron has power exceeding
                                 this threshold, the neuron is selected.

    Returns:
        list: A list of integer neuron indices, sorted by their highest significant
              frequency (descending).
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Input 'data' must be a 2D NumPy array.")
    if data.shape[0] == 0 or data.shape[1] == 0:
        print("Warning: Input data is empty. Returning empty list.")
        return []
    if data.shape[1] < 2:  # Need at least 2 timesteps for meaningful FFT
        print("Warning: Insufficient timesteps for FFT. Returning empty list.")
        return []
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive.")
    if min_high_frequency < 0:
        raise ValueError("Minimum high frequency cannot be negative.")
    if power_threshold < 0:
        raise ValueError("Power threshold cannot be negative.")

    n_neurons, n_timesteps = data.shape
    freqs = np.fft.rfftfreq(n_timesteps, d=1.0 / sampling_rate)
    high_freq_indices = np.where(freqs > min_high_frequency)[0]

    if len(high_freq_indices) == 0:
        print(f"Warning: No frequency bins found above {min_high_frequency} Hz. "
              f"Max resolvable frequency (Nyquist): {freqs[-1]:.2f} Hz. "
              "Returning empty list.")
        return []

    # Collect (neuron_id, highest_significant_freq) for neurons that exceed power_threshold
    neuron_freq_pairs = []
    for i in range(n_neurons):
        neuron_activity = data[i, :]
        fft_values = np.fft.rfft(neuron_activity)
        power_spectrum = np.abs(fft_values) ** 2

        # Extract power in the high-frequency band
        high_power = power_spectrum[high_freq_indices]
        if np.any(high_power > power_threshold):
            # Find the index (within high_freq_indices) of the highest frequency bin where power > threshold
            significant_bins = high_freq_indices[np.where(high_power > power_threshold)[0]]
            max_high_freq_idx = significant_bins.max()
            max_high_freq = freqs[max_high_freq_idx]
            neuron_freq_pairs.append((i, max_high_freq))

    if not neuron_freq_pairs:
        return []

    # Sort by highest significant frequency in descending order
    neuron_freq_pairs.sort(key=lambda x: x[1], reverse=True)

    # Extract and return only the neuron indices
    sorted_neuron_ids = [pair[0] for pair in neuron_freq_pairs]
    return sorted_neuron_ids

def select_high_frequency_neurons(data, sampling_rate=1.0, min_high_frequency=0.1, power_threshold=0.1):
    """
    Selects neuron IDs that exhibit significant power in high-frequency bands.

    Args:
        data (np.ndarray): A 2D NumPy array of shape [n_neurons, n_timesteps]
                           containing the activity of each neuron over time.
        sampling_rate (float): The sampling rate of the data (e.g., in Hz or samples per second).
                               This is crucial for interpreting frequency values.
        min_high_frequency (float): The minimum frequency (in the same units as implied by
                                    sampling_rate, e.g., Hz) to be considered "high".
                                    Only frequencies *above* this value will be analyzed.
        power_threshold (float): The power threshold. If any frequency component in the
                                 high-frequency range for a neuron has power exceeding
                                 this threshold, the neuron is selected.

    Returns:
        list: A list of integer indices (neuron IDs) that meet the criteria.
              Returns an empty list if no neurons meet the criteria or if
              input conditions are not met (e.g., insufficient timesteps).
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Input 'data' must be a 2D NumPy array.")
    if data.shape[0] == 0 or data.shape[1] == 0:
        print("Warning: Input data is empty. Returning empty list.")
        return []
    if data.shape[1] < 2: # Need at least 2 timesteps for meaningful FFT
        print("Warning: Insufficient timesteps for FFT. Returning empty list.")
        return []
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive.")
    if min_high_frequency < 0: # Can be 0 if we want to include DC, but usually positive
        raise ValueError("Minimum high frequency cannot be negative.")
    if power_threshold < 0:
        raise ValueError("Power threshold cannot be negative.")
    n_neurons, n_timesteps = data.shape
    selected_neuron_ids = []
    freqs = np.fft.rfftfreq(n_timesteps, d=1.0/sampling_rate)
    high_freq_indices = np.where(freqs > min_high_frequency)[0]
    if len(high_freq_indices) == 0:
        print(f"Warning: No frequency bins found above {min_high_frequency} Hz. "
              f"Max resolvable frequency (Nyquist): {freqs[-1]:.2f} Hz. "
              "Returning empty list.")
        return []
    for i in range(n_neurons):
        neuron_activity = data[i, :]
        fft_values = np.fft.rfft(neuron_activity)
        power_spectrum = np.abs(fft_values)**2
        neuron_high_freq_power = power_spectrum[high_freq_indices]
        if np.any(neuron_high_freq_power > power_threshold):
            selected_neuron_ids.append(i)
    return selected_neuron_ids

def run_sim(net, env, lesion_idx, samples=1000):
    """
    net: your model, with .forward(obs, lesion_idx=…) returning (pol, val, lin_act)
    env: your environment, with .reset(), .step(action), .task_stage and .correct_trial
    lesion_idx: list of neuron‐indices to disable
    returns: float accuracy over 100 episodes
    """
    correct_trials = []
    for _ in tqdm(range(samples)):
        env.reset()
        net.reinit_hid()
        done = False
        while not done:
            # forward supports lesion_idx and returns 3–5 values depending on spike/layer2
            out = net.forward(torch.unsqueeze(torch.Tensor(env.observation).float(), dim=0).to(device), lesion_idx=lesion_idx)
            pol = out[0]
            if env.task_stage == 'init':
                action_to_take = 1
                new_obs, reward, done = env.step(action_to_take)
            elif env.task_stage == 'choice_init':
                final_dist = Categorical(pol)
                final_act_tensor = final_dist.sample()
                action_to_take = final_act_tensor.item()
                new_obs, reward, done = env.step(action_to_take)
                correct_trials.append(env.correct_trial)
            elif env.task_stage == 'intermediate_choice_init':
                # choose one of the two intermediate actions deterministically to advance
                new_obs, reward, done = env.step(0)
            else:
                new_obs, reward, done = env.step()
    return float(np.mean(correct_trials))



def load_net_env(model_path: str,
                 n_neurons: int,
                 device: torch.device,
                 spike: bool = False,
                 seed: int = 0,
                 delay: int = 30,
                 layer2: bool = False):
    """
    Instantiate your network and environment, load pretrained weights if given.

    Args:
        model_path:   path to a .pt or .pth file with state_dict (or None to skip loading)
        n_neurons:    number of hidden units / neurons in the SSM
        device:       torch.device('cuda') or torch.device('cpu')
        spike:        whether to use spiking mode in the SSM
        seed:         random seed for the environment

    Returns:
        net, env
    """
    # 1) env
    env = IntDiscrim3_Intermediate(seed=seed, delay=delay)
    # 2) ssm params & network
    ssm_params = {
        "P":                n_neurons * 2,
        "C_init":           "trunc_standard_normal",
        "discretization":   "zoh",
        "dt_min":           0.001,
        "dt_max":           0.1,
        "conj_sym":         True,
        "step_rescale":     1.0,
        "spike":            spike,
        "layer2":           layer2
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
    # 3) load pretrained weights if available
    if model_path is not None and os.path.isfile(model_path):
        state = torch.load(model_path, map_location=device)
        net.load_state_dict(state)
    return net, env


def load_and_process_data(data_dir: str,
                          n_neurons: int):
    """
    Load numpy data and normalize responses per neuron.

    Args:
        data_dir:            folder containing the .npz file
        pretrained_filename: name of the .npz (e.g. 'my_model.npz')
        n_neurons:           number of neurons in the 3rd axis of each resp array

    Returns:
        dict with keys:
            action_hist (…),
            correct_trials (…),
            stim (…),
            stim1_resp1, stim1_resp2,
            stim2_resp1, stim2_resp2,
            stim3_resp1, stim3_resp2,
            delay1_resp1, delay1_resp2,
            delay2_resp1, delay2_resp2
        — all with the same shapes as in the file, but with resp1/x arrays normalized.
    """
    data = np.load(data_dir, allow_pickle=True)

    # unpack
    out = {
        "action_hist":   data["action_hist"],
        "correct_trials":data["correct_trial"],
        "stim":          data["stim"],
    }
    # response arrays
    keys = [
        "stim1_resp1_hx",
        "stim2_resp1_hx",
        "stim3_resp1_hx",
        "delay1_resp1_hx",
        "delay2_resp1_hx",
        # "stim1_resp2_hx",
        # "stim2_resp2_hx",
        # "stim3_resp2_hx",
        # "delay1_resp2_hx",
        # "delay2_resp2_hx",
    ]
    for k in keys:
        out[k] = data[k]

    # normalize resp1 arrays per neuron
    for i in range(n_neurons):
        # swap zeros → nan on each resp1
        for phase in ("stim1", "stim2", "stim3", "delay1", "delay2"):
            arr = out[f"{phase}_resp1_hx"]
            arr[:,:,i][arr[:,:,i] == 0] = np.nan

        # compute global min/max over all five phases for this neuron
        pooled = np.concatenate([
            out[f"{phase}_resp1_hx"][:,:,i] for phase in ("stim1","stim2","stim3","delay1","delay2")
        ], axis=1)
        mn, mx = np.nanmin(pooled), np.nanmax(pooled)
        span = mx - mn if mx != mn else 1.0

        # normalize and swap nan → 0
        for phase in ("stim1","stim2","stim3","delay1","delay2"):
            arr = out[f"{phase}_resp1_hx"]
            arr[:,:,i] = (arr[:,:,i] - mn) / span
            arr[:,:,i][np.isnan(arr[:,:,i])] = 0

    return out
def plot_lesion_metrics(metrics: np.ndarray,
                        title: str = 'Lesion Study',
                        xlabel: str = 'Number of Lesioned Units',
                        ylabel: str = 'Accuracy'):
    """
    Plot a line of cumulative lesioned units against accuracy.

    Args:
        metrics:  1D array where metrics[k] is the accuracy after k+1
                  units have been lesioned.
        title:    Plot title.
        xlabel:   X-axis label.
        ylabel:   Y-axis label.
    """
    x = np.arange(1, len(metrics) + 1)  # 1, 2, …, len(metrics)

    plt.figure(figsize=(6,4))
    plt.plot(x, metrics, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.xticks(x)               # show every integer lesion count
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    # plt.show()
    plt.savefig('lesion_study_re.png')

def sweep_lesion_progressive(net,
                             env,
                             max_lesion_units: int,
                             samples: int = 100,
                             reverse_index_order: bool = False,
                             lesion_indices=None):
    """
    Lesion units progressively and measure accuracy.

    Args:
        max_lesion_units: number of leading units to consider when lesion_indices is None.
        samples:          number of rollouts per lesion setting.
        reverse_index_order: reverse the lesion order.
        lesion_indices:   optional explicit list of unit indices to lesion in order.

    Returns:
        np.ndarray of length N+1 with accuracies, where N is the number of lesion steps.
    """
    acc_log = []
    if lesion_indices is not None:
        indices = list(lesion_indices)
        total_lesions = len(indices)
    else:
        indices = list(range(net.hidden_dim))
        total_available = len(indices) if max_lesion_units is None else min(max_lesion_units, len(indices))
        total_lesions = total_available
        indices = indices[:total_available]

    if reverse_index_order:
        indices = indices[::-1]

    for k in tqdm(range(0, total_lesions + 1)):
        subset = None if k == 0 else indices[:k]
        acc = run_sim(net, env, subset, samples=samples)
        acc_log.append(acc)
    return np.array(acc_log, dtype=float)


def lesion_single_units(net,
                        env,
                        indices,
                        samples: int = 100):
    """
    Lesion one unit at a time (no cumulative lesioning) and record accuracy.

    Args:
        indices: iterable of unit indices to lesion individually.
        samples: episodes per lesion setting.

    Returns:
        np.ndarray of accuracies aligned with order of ``indices``.
    """
    acc_log = []
    for idx in tqdm(indices, desc="Single-unit lesions"):
        acc_log.append(run_sim(net, env, [idx], samples=samples))
    return np.array(acc_log, dtype=float)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Lesion study in 3-stim Interval Discrimination task using saved SSM model"
    )
    parser.add_argument("--seed", type=int, default=444)
    parser.add_argument("--load_model_path", type=str, required=True, help="Path to saved model .pt file")
    parser.add_argument("--n_neurons", type=int, required=True, help="Number of hidden units in the SSM")
    parser.add_argument("--delay", type=int, default=30, help="Task delay to match training")
    parser.add_argument("--spike", action='store_true', default=False, help="Use spiking mode (match training)")
    parser.add_argument("--layer2", action='store_true', default=False, help="Use second SSM layer (match training)")
    parser.add_argument("--samples", type=int, default=25000, help="Episodes per lesion setting")
    parser.add_argument("--max_lesion_units", type=int, default=None, help="Max number of leading units to lesion")
    parser.add_argument("--lesion_indices", type=str, default=None,
                        help="Comma-separated list of unit indices to lesion instead of sweeping all units")
    parser.add_argument("--single_unit_lesion", action='store_true', default=False,
                        help="If set, lesion one unit at a time and plot accuracy histogram.")
    args = parser.parse_args()

    lesion_indices = None
    if args.lesion_indices:
        tokens = [tok.strip("[] ") for tok in re.split(r'[,\s]+', args.lesion_indices) if tok.strip("[] ")]
        try:
            lesion_indices = [int(tok) for tok in tokens]
        except ValueError as exc:
            raise ValueError("Failed to parse --lesion_indices. Provide comma-separated integers.") from exc
        if len(lesion_indices) == 0:
            lesion_indices = None

    # device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # EXTRACT FILENAME for unique output folder
    filename = args.load_model_path.replace("\\", "/").split("/")[-1]      # e.g. "model123.pt"
    base_name = filename.rsplit(".", 1)[0]                                  # "model123"

    # create new subfolder to avoid overwrite
    out_dir_root = os.path.dirname(os.path.abspath(args.load_model_path)) or "."
    out_dir = os.path.join(out_dir_root, f"lesion_{base_name}")
    os.makedirs(out_dir, exist_ok=True)

    # 1) load network + env matching training setup
    net, env = load_net_env(
        model_path=args.load_model_path,
        n_neurons=args.n_neurons,
        device=device,
        seed=args.seed,
        spike=args.spike,
        delay=args.delay,
        layer2=args.layer2,
    )
    net.eval()

    # determine which units are eligible for lesioning based on CLI args
    if lesion_indices is None:
        base_indices = list(range(args.n_neurons))
    else:
        base_indices = list(lesion_indices)

    if args.max_lesion_units is not None:
        max_slice = min(args.max_lesion_units, len(base_indices))
        base_indices = base_indices[:max_slice]
    max_units = len(base_indices)

    if max_units == 0:
        raise ValueError("No units available for lesioning after applying filters.")

    if args.single_unit_lesion:
        single_results = lesion_single_units(
            net,
            env,
            base_indices,
            samples=args.samples
        )
        acc_out_path_single = os.path.join(out_dir, f"lesion_single_acc_{base_name}.npy")
        np.save(acc_out_path_single, single_results)
        print(f"Saved single-unit accuracy list to {acc_out_path_single}")

        plt.figure(figsize=(8, 4))
        plt.bar(base_indices, single_results, width=0.8)
        plt.title("Single-unit lesion accuracy")
        plt.xlabel("Unit index")
        plt.ylabel("Accuracy")
        plt.grid(True, axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        hist_path = os.path.join(out_dir, f"lesion_single_hist_{base_name}_delay{args.delay}.png")
        plt.savefig(hist_path, dpi=300)
        plt.close()
        print(f"Saved single-unit lesion histogram to {hist_path}")
        sys.exit(0)

    # 2) progressive lesion sweep over first k units (both orders)
    lesion_indices_for_sweep = None if lesion_indices is None else base_indices
    results1 = sweep_lesion_progressive(
        net,
        env,
        max_units,
        samples=args.samples,
        reverse_index_order=False,
        lesion_indices=lesion_indices_for_sweep
    )
    results2 = sweep_lesion_progressive(
        net,
        env,
        max_units,
        samples=args.samples,
        reverse_index_order=True,
        lesion_indices=lesion_indices_for_sweep
    )

    # 3) save averaged accuracy lists and plots to model directory
    acc_out_path1 = os.path.join(out_dir, f"lesion_study_acc1_{base_name}.npy")
    np.save(acc_out_path1, results1)
    print(f"Saved averaged accuracy list to {acc_out_path1}")

    acc_out_path2 = os.path.join(out_dir, f"lesion_study_acc2_{base_name}.npy")
    np.save(acc_out_path2, results2)
    print(f"Saved averaged accuracy list to {acc_out_path2}")

    # plot for forward order
    plt.figure(figsize=(6, 4))
    x = np.arange(0, len(results1))
    plt.plot(x, results1, linewidth=0.5)
    plt.title("Lesion Study (forward order)")
    plt.xlabel("Number of lesioned units")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    out_path1 = os.path.join(out_dir, f"lesion_study1_{base_name}_delay{args.delay}.png")
    plt.savefig(out_path1, dpi=300)
    plt.close()
    print(f"Saved lesion study plot to {out_path1}")

    # plot for reverse order
    plt.figure(figsize=(6, 4))
    x = np.arange(0, len(results2))
    plt.plot(x, results2, linewidth=0.5)
    plt.title("Lesion Study (reverse order)")
    plt.xlabel("Number of lesioned units")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    out_path2 = os.path.join(out_dir, f"lesion_study2_{base_name}_delay{args.delay}.png")
    plt.savefig(out_path2, dpi=300)
    plt.close()
    print(f"Saved lesion study plot to {out_path2}")

