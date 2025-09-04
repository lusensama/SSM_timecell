# -*- coding: utf-8 -*-
"""
This script analyzes and visualizes the "retiming" of time cells from a State-Space Model (SSM),
as described in the paper 'State Space Models Naturally Produce Traveling Waves, Time Cells...'.

Specifically, it corresponds to Figure 3D of the paper. The script performs the following steps:
1.  Loads hidden state activation data (tuning curves) from two separate experiments:
    - A model trained on a short delay (e.g., 30 timesteps).
    - The same model retrained on a longer delay (e.g., 100 timesteps).
2.  For each neuron in the short-delay model, it finds the most similar neuron in the
    long-delay model by comparing their tuning curves using Mean Squared Error (MSE).
3.  It plots the top N most similar pairs, aligning their tuning curves by their peak
    and trough activities to visually demonstrate the "stretching" of the temporal receptive field.

To use this script, simply update the file paths in the 'Configuration & Parameters'
section and run it. It will save the final comparison figure as a PNG file.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

FILE_PATH_SHORT_DELAY = './datapoints/L1_delay30_sorted.npz'
FILE_PATH_LONG_DELAY = './datapoints/L1_delay100_sorted.npz'
# ------------------------------------

# --- Plotting and Analysis Parameters ---
N_TOP_PAIRS_TO_PLOT = 50  # Number of best-matching neuron pairs to visualize.
COMMON_RESAMPLE_LENGTH = 200 # Resample curves to this length for fair MSE comparison.
OUTPUT_FIGURE_FILENAME = "../figures/figure_3d_retiming_comparison.png" # Output filename.

# --- Color Scheme for Plots ---
# Consistent colors help distinguish between the two conditions.
COLOR_DELAY_SHORT = 'royalblue'
COLOR_DELAY_LONG = 'red'


def load_tuning_curves(file_path: str, data_key: str = 'delay2') -> np.ndarray:
    """
    Loads neuron tuning curves from a specified .npz file.

    Args:
        file_path (str): The full path to the .npz file.
        data_key (str): The key within the .npz file that holds the data.

    Returns:
        np.ndarray: An array where each row is a neuron's activation curve.
    """
    try:
        with np.load(file_path) as data:
            print(f"Successfully loaded data from '{file_path}'.")
            return data[data_key]
    except FileNotFoundError:
        print(f"Error: The file was not found at '{file_path}'.")
        print("Please update FILE_PATH_SHORT_DELAY and FILE_PATH_LONG_DELAY.")
        exit()
    except KeyError:
        print(f"Error: The key '{data_key}' was not found in '{file_path}'.")
        exit()


def find_best_matching_pairs(curves_short_delay: np.ndarray, curves_long_delay: np.ndarray) -> list:
    """
    Finds the best matching neuron pairs between the two delay conditions based on MSE.
    This procedure is described in the paper's Methods section under 'Tuning Curve Similarity Analysis'[cite: 361].

    Args:
        curves_short_delay (np.ndarray): Activation curves from the short delay condition.
        curves_long_delay (np.ndarray): Activation curves from the long delay condition.

    Returns:
        list: A list of dictionaries, sorted by MSE, containing the matched unit indices and MSE value.
    """
    print("Starting similarity search to find best matching units...")
    
    # Resample all long-delay curves to a common length for fair comparison.
    # This normalization is a key step before calculating MSE [cite: 363-364].
    x_resample = np.linspace(0, 1, COMMON_RESAMPLE_LENGTH)
    resampled_long_curves = [np.interp(x_resample, np.linspace(0, 1, len(c)), c) for c in curves_long_delay]

    similarity_results = []
    
    for i, curve_short in enumerate(tqdm(curves_short_delay, desc="Finding matches")):
        # Resample the short-delay curve to the same common length.
        resampled_curve_short = np.interp(x_resample, np.linspace(0, 1, len(curve_short)), curve_short)
        
        min_mse, best_match_idx = float('inf'), -1
        
        # Compare with all resampled long-delay curves to find the best match.
        for j, resampled_curve_long in enumerate(resampled_long_curves):
            mse = np.mean((resampled_curve_short - resampled_curve_long)**2)
            if mse < min_mse:
                min_mse, best_match_idx = mse, j

        similarity_results.append({'unit_short': i, 'unit_long': best_match_idx, 'mse': min_mse})

    # Sort pairs by lowest MSE to find the best matches[cite: 366].
    best_pairs = sorted(similarity_results, key=lambda x: x['mse'])
    print("Similarity search complete.")
    return best_pairs


def plot_aligned_comparison(ax: plt.Axes, curve_short: np.ndarray, curve_long: np.ndarray, align_on: str):
    """
    Plots two tuning curves on the same axes, aligned by their peak or trough.
    This visualization technique clearly shows the stretching of the receptive field.

    Args:
        ax (plt.Axes): The matplotlib axes object to plot on.
        curve_short (np.ndarray): The activation curve from the short delay condition.
        curve_long (np.ndarray): The activation curve from the long delay condition.
        align_on (str): Either 'peak' or 'trough' to specify the alignment point.
    """
    # Determine the alignment point (index of max or min activation).
    if align_on == 'peak':
        align_idx_short, align_idx_long = np.argmax(curve_short), np.argmax(curve_long)
    else: # 'trough'
        align_idx_short, align_idx_long = np.argmin(curve_short), np.argmin(curve_long)

    # Create shifted x-axes so that alignment points are at x=0.
    x_short_aligned = np.arange(len(curve_short)) - align_idx_short
    x_long_aligned = np.arange(len(curve_long)) - align_idx_long

    # Plot the two curves.
    ax.plot(x_short_aligned, curve_short, color=COLOR_DELAY_SHORT, lw=3.0)
    ax.plot(x_long_aligned, curve_long, color=COLOR_DELAY_LONG, lw=3.0)
    
    # --- Style the plot for publication quality ---
    ax.set_ylabel("Normalized Activation", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Create the bottom x-axis for the long delay condition.
    ax.set_xlabel(f'Time Step (Delay = {len(curve_long)})', fontsize=12, color=COLOR_DELAY_LONG)
    ax.spines['bottom'].set_color(COLOR_DELAY_LONG)
    ax.tick_params(axis='x', colors=COLOR_DELAY_LONG)
    bottom_ticks = np.linspace(0, len(curve_long), 5, dtype=int)
    ax.set_xticks(bottom_ticks - align_idx_long)
    ax.set_xticklabels(bottom_ticks)

    # Create a twin y-axis to host the second x-axis at the top.
    ax2 = ax.twiny()
    ax2.set_xlabel(f'Time Step (Delay = {len(curve_short)})', fontsize=12, color=COLOR_DELAY_SHORT)
    ax2.spines['top'].set_color(COLOR_DELAY_SHORT)
    # Hide other spines for clarity.
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(axis='x', colors=COLOR_DELAY_SHORT)
    # Ensure the limits match the bottom axis.
    ax2.set_xlim(ax.get_xlim()) 
    top_ticks = np.linspace(0, len(curve_short), 4, dtype=int)
    ax2.set_xticks(top_ticks - align_idx_short)
    ax2.set_xticklabels(top_ticks)


def main():
    """Main execution function."""
    
    # 1. Load data from the specified files.
    curves_short = load_tuning_curves(FILE_PATH_SHORT_DELAY)
    curves_long = load_tuning_curves(FILE_PATH_LONG_DELAY)
    
    # 2. Find the best matching pairs.
    best_pairs = find_best_matching_pairs(curves_short, curves_long)
    
    # 3. Create the final comparison figure.
    # The figure will have N rows (for N pairs) and 2 columns (peak-aligned, trough-aligned).
    fig, axes = plt.subplots(
        N_TOP_PAIRS_TO_PLOT, 2, 
        figsize=(10, 4 * N_TOP_PAIRS_TO_PLOT), 
        squeeze=False
    )
    
    for i in range(N_TOP_PAIRS_TO_PLOT):
        pair_info = best_pairs[i]
        short_idx = pair_info['unit_short']
        long_idx = pair_info['unit_long']
        
        curve1 = curves_short[short_idx]
        curve2 = curves_long[long_idx]
        
        # Plot peak-aligned comparison on the left column.
        ax_peak = axes[i, 0]
        plot_aligned_comparison(ax_peak, curve1, curve2, align_on='peak')
        
        # Add a title indicating the match rank and unit IDs.
        title_text = (f"Match Rank {i+1}\n"
                      f"Unit {short_idx} (Short) vs. Unit {long_idx} (Long)\n"
                      f"MSE: {pair_info['mse']:.4f}")
        ax_peak.set_title(title_text, fontsize=12, loc='left', y=1.2)
        
        # Plot trough-aligned comparison on the right column.
        ax_trough = axes[i, 1]
        plot_aligned_comparison(ax_trough, curve1, curve2, align_on='trough')

    plt.tight_layout(h_pad=6.0, w_pad=3.0)
    plt.savefig(OUTPUT_FIGURE_FILENAME, dpi=300, bbox_inches='tight')
    
    print(f"\nSuccessfully created and saved the comparison figure as '{OUTPUT_FIGURE_FILENAME}'.")

if __name__ == "__main__":
    main()
