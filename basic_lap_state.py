"""
Helper functions for lap counting analysis
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from envs.lap_counting import Laps_Counting
from tqdm import tqdm
import torch
from agents.model_ssm_RL_laps import *

# ==============================================================================
# UNMODIFIED USER-PROVIDED FUNCTIONS
# These functions are used exactly as you provided them.
# ==============================================================================

def draw_heatmap(mat: np.ndarray, title: str, save_path: str, boundaries=None):
    """
    Original heatmap drawing function. Relies on global variables:
    `cmap`, `boundaries`, `total_time`, `phases`.
    """
    fig, ax = plt.subplots(figsize=(20, 6))
    # This function will use the globally defined 'cmap' variable
    cax = ax.imshow(mat, aspect='auto', cmap=cmap)
    fig.colorbar(cax, ax=ax, label='Normalized Activation')
    # These loops and settings will use globally defined variables

    for x in boundaries:
        ax.axvline(x, color='white', lw=1)
    ax.set_xticks(boundaries)
    ax.set_xticklabels(boundaries, rotation=0)
    ax.tick_params(axis='x', which='major', pad=8)
    for start, end, lbl in zip(boundaries[:-1], boundaries[1:], phases):
        mid = (start + end) / 2
        ax.text(
            mid, -0.06, lbl,
            transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=10
        )
    ax.set_ylabel(f'Unit #')
    ax.set_xlabel('Time (timesteps)', labelpad=20)
    ax.set_title(title)
    plt.subplots_adjust(bottom=0.1)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Heatmap successfully saved to {save_path}")
def normalize_activity(matrix: np.ndarray) -> np.ndarray:
    """
    Normalizes the rows of a matrix based on their min and max values.

    The normalization performs two steps:
    1. Scales each row to the range [0, 1] (min-max scaling).
    2. Stretches and shifts the [0, 1] range to [-1, 1].

    Args:
        matrix (np.ndarray): The input matrix to normalize, where each row
                             represents a series of observations (e.g., a neuron's activity).

    Returns:
        np.ndarray: The normalized matrix with values in the range [-1, 1].
    """
    # Calculate the minimum and peak-to-peak (max - min) values for each row.
    # `keepdims=True` ensures the output shape is (n_rows, 1) for broadcasting.
    min_vals = np.min(matrix, axis=1, keepdims=True)
    range_vals = np.ptp(matrix, axis=1, keepdims=True)

    # IMPORTANT: Handle rows with no activity variation (range is 0) to avoid
    # division by zero. We set the range to 1 in these cases. The scaled
    # result for that row will correctly be 0, as (value - min) will be 0.
    range_vals[range_vals == 0] = 1.0

    # 1. Scale the matrix rows to the range [0, 1]
    scaled_matrix = (matrix - min_vals) / range_vals

    # 2. Stretch and shift the range from [0, 1] to [-1, 1]
    normalized_matrix = scaled_matrix * 2 - 1

    return normalized_matrix

def sort_resp(total_resp, norm=True):
    """
    Original function to average, normalize, and sort responses.
    This function is used as is, including its original logic.
    """
    np.seterr(divide='ignore', invalid='ignore')
    n_neurons = np.shape(total_resp)[2]
    segments = np.moveaxis(total_resp, 0, 1)
    unsorted_matrix = np.zeros((n_neurons, len(segments)))
    sorted_matrix = np.zeros((n_neurons, len(segments)))
    normalized_matrix = np.zeros_like(unsorted_matrix) # Initialize to avoid reference-before-assignment
    for i in range(len(segments)):
        averages = np.mean(segments[i],
                           axis=0)
        unsorted_matrix[:, i] = np.transpose(
            averages)
        if norm is True:
            # Original logic for normalization and sorting is preserved
            scaled = (unsorted_matrix
                      - np.min(unsorted_matrix, axis=1, keepdims=True)) \
                     / np.ptp(unsorted_matrix, axis=1, keepdims=True)
            normalized_matrix = scaled * 2 - 1
            max_indeces = np.argmax(normalized_matrix, axis=1)
            cell_nums = np.argsort(max_indeces)
            for j, i_cell in enumerate(list(cell_nums)):
                sorted_matrix[j] = normalized_matrix[i_cell]
        else:
            max_indeces = np.argmax(unsorted_matrix, axis=1)
            cell_nums = np.argsort(max_indeces)
            for j, i_cell in enumerate(list(cell_nums)):
                sorted_matrix[j] = unsorted_matrix[i_cell]
    assert len(sorted_matrix) == n_neurons
    return cell_nums, sorted_matrix, normalized_matrix
# def normalize_activity(matrix):
#     """
#     Normalize each row of the matrix to the range [-1, 1].
#     """
#     # Avoid division by zero in case of constant rows
#     ptp = np.ptp(matrix, axis=1, keepdims=True)
#     ptp[ptp == 0] = 1.0
#
#     scaled = (matrix - np.min(matrix, axis=1, keepdims=True)) / ptp
#     normalized = scaled * 2 - 1
#     return normalized
# ==============================================================================
# GLOBAL VARIABLES REQUIRED BY `draw_heatmap`
# These must be defined before `draw_heatmap` is called.
# ==============================================================================
cmap = 'jet'
boundaries = []
total_time = 0
phases = []

# ==============================================================================
# NEW ENCAPSULATING FUNCTION
# This is the single new function you requested.
# ==============================================================================
def reshape_resp(total_resp):
    # Compute mean over the first axis (episodes): shape → (n_timesteps, n_neurons)
    mean_over_episodes = np.mean(total_resp, axis=0)
    # Transpose so rows are neurons, columns are time-steps: shape → (n_neurons, n_timesteps)
    return mean_over_episodes.T

def sort_sub(resp, laps=4, t=30):
    b, t_total, c = resp.shape
    # t = t_total // laps
    resp = resp.real
    t=t+1
    t_total = laps * t
    sorted_segments = []
    sorted_mat_raw_segments = []
    unsorted_mat_segments = []
    unsorted_mat_raw_segments = []
    from utils.utils_analysis import sort_freq_resp
    for i in range(laps):
        seg = resp[:, i*t:(i+1)*t, :]  # [b, t, c]
        sorted_indices, sorted_real_matrix, _ = sort_freq_resp(seg, norm=True)  # assumed to be numpy-based

        unsorted_mat_raw = reshape_resp(seg)  # [b*c, t] or similar depending on reshape_resp

        sorted_mat_raw = unsorted_mat_raw[sorted_indices]
        unsorted_mat = normalize_activity(unsorted_mat_raw)

        sorted_segments.append(sorted_real_matrix)         # assume shape: [b, t, c]
        sorted_mat_raw_segments.append(sorted_mat_raw)     # assume shape: [b*c, t] or similar
        unsorted_mat_segments.append(unsorted_mat)         # same shape as above
        unsorted_mat_raw_segments.append(unsorted_mat_raw) # same shape as above

    sorted_real_matrix = np.concatenate(sorted_segments, axis=1)       # along time
    sorted_mat_raw = np.concatenate(sorted_mat_raw_segments, axis=1)   # along time
    unsorted_mat = np.concatenate(unsorted_mat_segments, axis=1)       # along time
    unsorted_mat_raw = np.concatenate(unsorted_mat_raw_segments, axis=1)

    return sorted_real_matrix, sorted_mat_raw, unsorted_mat, unsorted_mat_raw

def load_and_plot_real(
    npz_path: str,
    array_key: str,
    title: str,
    save_path: str
):
    """
    Loads a complex array from an .npz file, processes its real part using the
    original `sort_resp` function, and plots it using the original
    `draw_heatmap` function.

    Args:
        npz_path (str): Path to the input .npz file.
        array_key (str): The key for the data array within the .npz file.
        title (str): The title for the output plot.
        save_path (str): The file path where the plot image will be saved.
    """
    global boundaries, total_time, phases, cmap

    print(f"--- Starting process for {npz_path} ---")

    # 1. Load the data
    print(f"Step 1: Loading data from '{npz_path}' with key '{array_key}'...")
    try:
        data = np.load(npz_path)
        complex_array = data[array_key]
    except FileNotFoundError:
        print(f"Error: The file was not found at {npz_path}")
        return
    except KeyError:
        print(f"Error: The key '{array_key}' was not found in the file.")
        print(f"Available keys: {list(data.keys())}")
        return

    # 2. Extract the real part
    print("Step 2: Extracting real part of the complex array.")
    real_part = np.real(complex_array)
    real_part = real_part[:,:170,:]
    boundaries = [30, 60, 90, 120]
    # 3. Call the original `sort_resp` function
    print("Step 3: Processing data with the original `sort_resp` function...")

    sorted_real_matrix, sorted_mat_raw, unsorted_mat, unsorted_mat_raw = sort_sub(real_part, laps=4)
    # 4. Set global variables required by `draw_heatmap`
    print("Step 4: Preparing global variables for `draw_heatmap`...")
    total_time = sorted_real_matrix.shape[1]
    # Define some generic phases for plotting purposes
    # boundaries = [0, total_time // 2, total_time]
    # phases = ['First Half', 'Second Half']
    cmap = 'jet' # Let's set a specific colormap for this plot

    # 5. Call the original `draw_heatmap` function
    print("Step 5: Calling `draw_heatmap` to generate the plot...")
    draw_heatmap(
        mat=sorted_real_matrix,
        title="sorted Lap Hidden State",
        save_path=save_path+"/sorted.png"
    )
    draw_heatmap(
        mat=sorted_mat_raw,
        title="sorted Lap raw Hidden State",
        save_path=save_path+"/sorted_raw.png"
    )
    draw_heatmap(
        mat=unsorted_mat,
        title="unsorted Lap Hidden State",
        save_path=save_path+"/unsorted.png"
    )
    draw_heatmap(
        mat=unsorted_mat_raw,
        title="unsorted Lap raw Hidden State",
        save_path=save_path+"/unsorted_raw.png"
    )
    print("--- Process finished ---")
def plot_lap_choices(net, env, n_total_episodes=1000, spike=False, layer2=False, save_path=''):
    net.eval()
    device = torch.device("cuda:0")
    all_laps = np.zeros([n_total_episodes, env.lap_count*env.lap_len+100])
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

            all_laps[i_episode][env.elapsed_t]+= env.predicted_lap_count
            # print(reward)
            done = (task_stage == 'done') # The episode is 'done' only when the task_stage says so.
            episode_rewards.append(reward)

            observation = new_obs
    all_laps=all_laps[:,:-10]
    n_episodes, n_timesteps = all_laps.shape
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
    plt.title(f"Average Predicted Count Over {env.lap_count} Laps")
    plt.grid(True)
    plt.tight_layout()
    save_path=save_path+f"/figure_3c_{env.lap_count}_lap_choices.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Line successfully saved to {save_path}")

# --- Example Usage ---
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Head-fixed Interval Discrimination task simulation")
    parser.add_argument("--path", type=str, default='None'       , help="path RELATIVE TO $SCRATCH/timecell/training/timing")
    args = parser.parse_args()
    # Define file paths and parameters
    NPZ_PATH = args.path
    ARRAY_KEY = 'hidden1'
    OUTPUT_TITLE = "Sorted Lap Running Hidden State"
    OUTPUT_SAVE_PATH = "./laps_plots/"

    # # A. Create a dummy .npz file to simulate the scenario
    # print("--- Setting up a dummy data file for demonstration ---")
    # n_episodes, n_timesteps, n_neurons = 10, 244, 128
    # # Generate some data
    # real_data = np.random.randn(n_episodes, n_timesteps, n_neurons)
    # imag_data = np.random.randn(n_episodes, n_timesteps, n_neurons)
    # dummy_complex_array = real_data + 1j * imag_data
    # # Save it to the path
    # os.makedirs(os.path.dirname(DUMMY_NPZ_PATH), exist_ok=True)
    # np.savez_compressed(DUMMY_NPZ_PATH, hidden_state1=dummy_complex_array)
    # print(f"Dummy file created at '{DUMMY_NPZ_PATH}'\n")
    n_neurons= 80
    lap_length = 30
    lap_count=4
    approx=True
    spike=False
    layer2=False
    env = Laps_Counting(final_rwd=10,
                 seed=100,
                 lap_length=lap_length,
                 fixed_laps=lap_count,
                 approx=approx)
    ssm_params = {
            "P": n_neurons*2,
            "C_init": "trunc_standard_normal",
            "discretization": "zoh",
            "dt_min": 0.001,
            "dt_max": 0.1,
            "conj_sym": True,
            "step_rescale": 1.0,
            "spike":spike,
            "layer2":layer2,
            "lap_count":lap_count,
        }
    # Instantiate the AC network using SSM as core.
    net = AC_SSM_stack(input_dimensions=2, action_dimensions=2, batch_size=1,
                 hidden_dim=n_neurons, ssm_params=ssm_params, p_dropout=0.1).to('cuda:0')
    
    plot_lap_choices(net, env, save_path=OUTPUT_SAVE_PATH)
    # exit(0)
    # B. Run the main function
    load_and_plot_real(
        npz_path=NPZ_PATH,
        array_key=ARRAY_KEY,
        title=OUTPUT_TITLE,
        save_path=OUTPUT_SAVE_PATH
    )


