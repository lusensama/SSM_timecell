"""
Helper functions for 3 stimulation task analysis
"""

import os
import random
import argparse
import gc
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.distributions.categorical import Categorical
from agents.model_ssm_stack_RL import AC_SSM_stack
from envs.int_discrim import IntDiscrim3_Intermediate
from utils.utils_analysis import sort_resp

def setup_directories(base_dir="plots"):
    """Creates necessary directories for saving plots and returns the path."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    return base_dir


def load_net_env(model_path: str, n_neurons: int, device: torch.device, delay: int, fixed_delay: bool):
    """Initializes the network and environment, and loads a pre-trained model."""
    has_layer2 = False
    is_spiking = False
    if model_path:
        has_layer2 = '_2layer' in model_path
        is_spiking = '_spiking' in model_path

    env = IntDiscrim3_Intermediate(seed=0, delay=delay, fixed_delay=fixed_delay)

    ssm_params = {
        "P": n_neurons * 2,
        "C_init": "trunc_standard_normal",
        "discretization": "zoh",
        "dt_min": 0.001,
        "dt_max": 0.1,
        "conj_sym": True,
        "step_rescale": 1.0,
        "spike": is_spiking,
        "layer2": has_layer2
    }
    net = AC_SSM_stack(
        input_dimensions=3,
        action_dimensions=3,
        hidden_dim=n_neurons,
        ssm_params=ssm_params
    ).to(device)

    if model_path and os.path.isfile(model_path):
        net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    print(f"Model loaded. Spiking: {net.spiking}, Layer 2: {net.layer2}")
    print(f"Environment initialized with Delay: {delay}, Fixed Delay: {fixed_delay}")
    return net, env


def collect_hidden_states(net, env, n_inferences, n_neurons, stim_dur, delay_dur, device):
    """
	Runs inference and collects raw data: complex hidden states, spike logs, and trial outcomes.
	"""
    phase_info = {
        'stim1': stim_dur, 'delay1': delay_dur, 'stim2': stim_dur,
        'delay2': delay_dur, 'stim3': stim_dur
    }
    env_stage_map = {
        'first_stim': 'stim1', 'delay': 'delay1', 'second_stim': 'stim2',
        'second_delay': 'delay2', 'third_stim': 'stim3'
    }

    # Initialize data structures to store complex numbers
    raw_data = {}
    for phase_name, duration in phase_info.items():
        raw_data[f"{phase_name}_resp1"] = np.zeros((n_inferences, duration, n_neurons), dtype=np.complex128)
        if net.layer2:
            raw_data[f"{phase_name}_resp2"] = np.zeros((n_inferences, duration, n_neurons), dtype=np.complex128)

    spike_logs = {'layer1': [], 'layer2': []} if net.layer2 else {'layer1': []}
    correct_trials_log = []
    spiking_entries1, spiking_entries2 = [], []

    for i in tqdm(range(n_inferences), desc="Running Inference"):
        env.reset()
        net.reinit_hid()
        done = False
        spiking_row1, spiking_row2 = [[] for _ in range(5)], [[] for _ in range(5)]
        seg_idx = 0
        ZERO = (0, 0, 0)
        prev_zero = tuple(env.observation) == ZERO
        while not done:
            # if env.task_stage == 'delay':
            #     print('debug')
            with torch.no_grad():
                obs = torch.tensor(env.observation, dtype=torch.float32).unsqueeze(0).to(device)
                outputs = net.forward(obs)
                pol = outputs[0]

            action = 1 if env.task_stage == 'init' else (
                Categorical(pol).sample().item() if env.task_stage == 'choice_init' else 0)

            if env.task_stage in env_stage_map and env.elapsed_t > 0:
                phase_name, t_idx = env_stage_map[env.task_stage], env.elapsed_t - 1
                if t_idx < raw_data[f"{phase_name}_resp1"].shape[1]:
                    # Collect the full complex hidden state
                    raw_data[f"{phase_name}_resp1"][i, t_idx,
                    :] = net.hidden_state1.clone().detach().cpu().numpy().squeeze()
                    if net.layer2:
                        raw_data[f"{phase_name}_resp2"][i, t_idx,
                        :] = net.hidden_state2.clone().detach().cpu().numpy().squeeze()

                if net.spiking and len(outputs) > 3:
                    spiking_row1[seg_idx].append(outputs[3].detach().cpu().numpy())
                    if net.layer2 and len(outputs) > 4:
                        spiking_row2[seg_idx].append(outputs[4].detach().cpu().numpy())
                    current_zero = tuple(env.observation) == ZERO
                    if current_zero != prev_zero:
                        seg_idx += 1
                        assert seg_idx < 5, f"too many segments: {seg_idx}"
                    prev_zero = current_zero

            _, _, done = env.step(action)

        correct_trials_log.append(env.correct_trial)
        if net.spiking:
            spiking_entries1.append(spiking_row1)
            if net.layer2: spiking_entries2.append(spiking_row2)

    return raw_data, spiking_entries1, correct_trials_log


def reshape_resp(total_resp):
    mean_over_episodes = np.mean(total_resp, axis=0)
    return mean_over_episodes.T

def plot_sorted_activity(data, phases, layer, delay, normalized, save_path, cmap='jet',
                         n_total_episodes=100, pre_sort=None):
    """
	Builds and saves sorted/unsorted heatmaps. 
	"""
    mats_complex = {}
    mats_real = {}
    for phase in phases:
        key = f"{phase}_resp{layer}"
        if key not in data:
            continue

        # 1) grab & store the full complex array
        arr_complex = data[key].copy()
        mats_complex[phase] = arr_complex

        # 2) extract & mask the real parts for plotting
        arr_real = arr_complex.real.copy()
        arr_real[arr_real == 0] = np.nan
        mats_real[phase] = arr_real

    if not mats_real:
        print(f"No data for layer {layer}. Skipping sorted plot.")
        return

    if normalized:
        all_concat = np.concatenate([mats_real[p] for p in phases if p in mats_real],
                                    axis=1)
        mins, maxs = np.nanmin(all_concat, axis=(0, 1)), np.nanmax(all_concat, axis=(0, 1))
        rng = (maxs - mins).clip(min=1e-6)
        for p in phases:
            if p in mats_real:
                mats_real[p] = (mats_real[p] - mins) / rng
                mats_real[p][np.isnan(mats_real[p])] = 0
    else:
        for p in phases:
            if p in mats_real:
                mats_real[p][np.isnan(mats_real[p])] = 0

    unsorted_mats, sorted_mats, lengths, raw_mats = [], [], [], []
    for p in phases:
        if p in mats_real:

            # RESTORED: Use sort_resp for self-contained sorting, as in the original script.
            # This ensures this plot's behavior is unchanged.
            _, sorted_phase, unsorted_phase = sort_resp(mats_real[p], norm=normalized)
            raw_mats.append(reshape_resp(np.abs(mats_complex[p])))
            unsorted_mats.append(unsorted_phase)
            if p == "delay1":
                sorted_indices, sorted_phase, unsorted_phase = sort_resp(mats_real[p], norm=normalized)
                resp1 = sorted_phase
            elif p == "delay2":
                sorted_indices, sorted_phase, unsorted_phase = sort_resp(mats_real[p], norm=normalized)
                resp2 = sorted_phase

            sorted_mats.append(sorted_phase)
            lengths.append(sorted_phase.shape[1])

    concat_unsorted = np.concatenate(unsorted_mats, axis=1)
    concat_sorted = np.concatenate(sorted_mats, axis=1)
    concat_raw_unsorted_resp = np.concatenate(raw_mats, axis=1)
    boundaries = np.cumsum([0] + lengths)

    def draw_heatmap(mat, title, file_path,value_range=None, ):
        vmin, vmax = (value_range if value_range is not None else (None, None))
        fig, ax = plt.subplots(figsize=(20, 6))
        cbar_label = 'Normalized Activation' if normalized else 'Activation'
        cax = ax.imshow(mat, aspect='auto', cmap=cmap, interpolation='none',vmin=vmin,vmax=vmax)
        fig.colorbar(cax, ax=ax, label=cbar_label)
        for x in boundaries[1:-1]: ax.axvline(x, color='white', lw=1.5, linestyle='--')
        ax.set_xticks([(st + en) / 2 for st, en in zip(boundaries[:-1], boundaries[1:])])
        ax.set_xticklabels(phases, fontsize=12, fontweight='bold')
        ax.set_xlabel('Task Phase', labelpad=15, fontsize=12)
        ax.set_ylabel('Unit # (Sorted by Peak Activity)', fontsize=12)
        ax.set_title(title, fontsize=16, pad=20)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path)
        plt.close(fig)

    norm_str = "Normalized" if normalized else "Raw"
    # unsorted_title = f'Unsorted {norm_str} Activity (L{layer}, Delay {delay}, {n_total_episodes} eps)'
    sorted_title = f'Sorted {norm_str} Activity (L{layer}, Delay {delay}, {n_total_episodes} eps)'

    # draw_heatmap(concat_unsorted, unsorted_title,
    #              os.path.join(save_path, f'L{layer}_delay{delay}_unsorted_{norm_str.lower()}.png'))
    draw_heatmap(concat_sorted, sorted_title,
                 os.path.join(save_path, f'figure_2b_L{layer}_delay{delay}_sorted_{norm_str.lower()}.png'))
    # draw_heatmap(concat_raw_unsorted_resp, unsorted_title,
    #              os.path.join(save_path, f'L{layer}_delay{delay}_unsorted_amp_{norm_str.lower()}.png'),
    #              value_range=(0, 0.5),)
    print(f"Saved sorted heatmaps for Layer {layer} to {os.path.join(save_path, f'figure_2b_L{layer}_delay{delay}_sorted_{norm_str.lower()}.png')}")
    return sorted_indices


def extract_lambda_bars(model_path: str, plot_dir: str, args, untrained: bool=False, n_models: int=5):
    import glob
    os.makedirs(plot_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_paths = glob.glob(os.path.join(model_path, "*.pt"))

    if untrained:
        for i in range(n_models):
            ssm_params = {
                "P": args.n_neurons * 2,
                "C_init": "trunc_standard_normal",
                "discretization": "zoh",
                "dt_min": 0.001,
                "dt_max": 0.1,
                "conj_sym": True,
                "step_rescale": 1.0,
                "spike": True,
                "layer2": False
            }
            torch.manual_seed(i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(i)
                torch.cuda.manual_seed_all(i) # For multi-GPU setups
            np.random.seed(i)
            random.seed(i)

            net = AC_SSM_stack(
                input_dimensions=3,
                action_dimensions=3,
                hidden_dim=args.n_neurons,
                ssm_params=ssm_params
            ).to(device)

            lambda_bar = net.ssm_cell1.get_lambda_bar().detach().cpu().numpy()
            out_path = os.path.join(plot_dir, f"{i}th_untrained_lambda_bar58888.npy")
            np.save(out_path, lambda_bar)
        exit(0)

    if not model_paths:
        raise FileNotFoundError(f"No .pt files found in {model_path}")

    for model_path in model_paths:
        net, _ = load_net_env(
            model_path,
            args.n_neurons,
            device,
            args.delay,
            args.fixed_delay,
        )

        lambda_bar = net.ssm_cell1.get_lambda_bar().detach().cpu().numpy()

        model_base = os.path.splitext(os.path.basename(model_path))[0]
        out_path = os.path.join(plot_dir, f"{model_base}_lambda_bar.npy")

        np.save(out_path, lambda_bar)
    exit(0)

def analyze_model_example(model_path, n_neurons, delay, fixed_delay, seed, n_inferences=1000, normalized=True, mode='plot'):
    """
    Run analysis and plotting for a single model file.
    """
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    # Create root plots directory and then a subfolder per model
    # root_dir = setup_directories("./figures")
    # plot_dir = os.path.join(root_dir, base_name)
    plot_dir = setup_directories("./figures")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net, env = load_net_env(
        model_path,
        n_neurons,
        device,
        delay,
        fixed_delay
    )
    stim_duration = 50

    raw_data, spiking_log, correct_trials = collect_hidden_states(net, env, n_inferences, n_neurons,
                                                        stim_duration, delay, device)
    # np.save(plot_dir+'100_delay_lambda_bar_retrained.npy', net.ssm_cell1.lambda_bar.cpu().numpy())
    if mode == 'plot':
        phases = ['stim1', 'delay1', 'stim2', 'delay2', 'stim3']
        num_layers = 2 if getattr(net, 'layer2', False) else 1

        for layer in range(1, num_layers + 1):
            print(f"\n--- Generating plots for Layer {layer} ({base_name}) ---")

            sorting_indices = plot_sorted_activity(
                data=raw_data,
                phases=phases,
                layer=layer,
                delay=delay,
                normalized=normalized,
                save_path=plot_dir,
                n_total_episodes=n_inferences
            )
            exit(0)
    elif mode == 'lesion':
        print("\n--- Lesion study functionality is not yet implemented. ---")

    if correct_trials:
        print(f"\n> Average accuracy over {n_inferences} inferences: {np.mean(correct_trials):.2%}")

    gc.collect()
    print("\nAnalysis complete for", base_name)



