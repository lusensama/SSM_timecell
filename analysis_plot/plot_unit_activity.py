# -*- coding: utf-8 -*-
"""
This script generates a grid plot visualizing the tuning curves of all individual
units (neurons) from a State-Space Model's (SSM) hidden state activations.

The visualization is intended for exploratory analysis to identify representative
examples of emergent cell types (ramping, oscillatory, time cells), as showcased
in Figure 2C of the paper 'State Space Models Naturally Produce Traveling Waves, Time Cells...'.

Each unit's activity over a time interval (e.g., a delay period) is plotted
on its own subplot. The line is color-coded by its activation value, providing an
intuitive view of the dynamics (e.g., red for high activation, blue for low).

To use this script, update the file path and data key in the 'Configuration'
section and run it. It will save the grid plot as a PNG file.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# --- 1. Configuration ---
# Keep using the "unsorted" data for both delay phases
DATA_KEY = 'unsorted'
# Provide file paths for delay1 and delay2 separately
DELAY_CONFIG = [
    {
        "label": "delay1",
        "file_path": "./figures/figure_3a2_L1_delay30_delay1_normalized_sorted_unsorted.npz",
        "output_title": "Unit Activity Tuning Curves (Delay 1)",
        "output_filename": "./figures/figure_2s_all_unit_tuning_curves_delay1.png",
    },
    {
        "label": "delay2",
        "file_path": "./figures/figure_3a2_L1_delay30_delay2_normalized_sorted_unsorted.npz",
        "output_title": "Unit Activity Tuning Curves (Delay 2)",
        "output_filename": "./figures/figure_2s_all_unit_tuning_curves_delay2.png",
    },
]
# ------------------------------------


def load_activation_data(file_path: str, data_key: str) -> np.ndarray:
    """
    Loads hidden state activation data from a specified .npz file.

    Args:
        file_path (str): The full path to the .npz file.
        data_key (str): The key within the .npz file that holds the data.

    Returns:
        np.ndarray: A NumPy array of the activation data.
    """
    try:
        with np.load(file_path) as data:
            print(f"Successfully loaded data from '{file_path}' using key '{data_key}'.")
            return data[data_key]
    except FileNotFoundError:
        print(f"Error: The file was not found at '{file_path}'.")
        print("Please update the FILE_PATH variable.")
        exit()
    except KeyError:
        print(f"Error: The key '{data_key}' was not found in '{file_path}'.")
        print("Please check the DATA_KEY variable.")
        exit()


def plot_tuning_curves_grid(data: np.ndarray, title: str, output_filename: str):
    """
    Creates and saves a grid plot for each unit's tuning curve.
    The line of each plot is colored based on its activation value (y-value).

    Args:
        data (np.ndarray): A 2D or 3D NumPy array of shape (n_units, n_timesteps)
                           or (n_units, n_timesteps, 1).
        title (str): The main title for the entire figure.
        output_filename (str): The filename for the saved figure.
    """
    # Make the code robust to input data shape (handle both 2D and 3D).
    if data.ndim == 3 and data.shape[2] == 1:
        data = np.squeeze(data, axis=2)

    n_units, n_timesteps = data.shape

    # Dynamically determine grid size.
    n_cols = 5
    n_rows = int(np.ceil(n_units / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 2), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=18, y=0.99)
    axes = axes.flatten()

    # Normalize activation values from -1 to 1 for consistent coloring.
    norm = Normalize(vmin=-1.0, vmax=1.0)
    cmap = plt.get_cmap('coolwarm') # Red for high, blue for low.

    for i in range(n_units):
        ax = axes[i]
        unit_activity = data[i, :]
        x = np.arange(n_timesteps)
        y = unit_activity

        # To color the line segment by segment, we use LineCollection.
        # This is a powerful technique for detailed visualization.
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Color each segment by the average activation of its start and end points.
        color_values = (y[:-1] + y[1:]) / 2

        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(color_values)
        lc.set_linewidth(2)

        ax.add_collection(lc)
        ax.autoscale_view()

        ax.set_title(f'Unit {i}', fontsize=10)
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, linestyle=':', alpha=0.6)

    # Hide any unused subplots in the grid.
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    # Add shared labels to the figure to avoid clutter.
    fig.text(0.5, 0.0, 'Time Step', ha='center', va='center', fontsize=14)
    fig.text(0.0, 0.5, 'Normalized Activation', ha='center', va='center', rotation='vertical', fontsize=14)

    plt.tight_layout(rect=[0.02, 0.02, 1, 0.96])
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print(f"\nSuccessfully created and saved the grid plot as '{output_filename}'.")
    # plt.show()


def main():
    """Main execution function."""
    for cfg in DELAY_CONFIG:
        print(f"\nProcessing {cfg['label']} using '{DATA_KEY}' key")
        activation_data = load_activation_data(cfg["file_path"], DATA_KEY)
        plot_tuning_curves_grid(activation_data, cfg["output_title"], cfg["output_filename"])


if __name__ == "__main__":
    main()
