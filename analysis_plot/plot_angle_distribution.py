# -*- coding: utf-8 -*-
"""
This script generates the eigenvalue phase angle distribution plot, corresponding
to Figure 3C in the paper 'State Space Models Naturally Produce Traveling Waves, Time Cells...'.

The plot visualizes a key finding: when the SSM is retrained on a longer temporal
task (e.g., from a 30-step delay to a 100-step delay), the distribution of the
absolute phase angles of its learned state matrix eigenvalues (lambda) shifts
systematically toward zero. This indicates that the model's internal oscillators
learn to rotate more slowly to measure longer durations[cite: 236, 240].

The script performs the following steps:
1.  Loads the complex eigenvalues from two separate model training runs.
2.  Calculates the absolute value of the phase angle for each eigenvalue.
3.  Creates a dual Y-axis plot:
    - The left axis shows the raw frequency counts as a histogram.
    - The right axis shows the smoothed probability density (using a Gaussian KDE).
4.  Saves the final figure as a high-quality PNG file.

To use, update the file paths in the 'Configuration' section and run the script.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

FILE_PATH_DELAY_30 = './datapoints/lambda_30/seed_700_epi49999_lambda_bar.npy'
FILE_PATH_DELAY_100 = './datapoints/lambda_100/seed_700_epi99998_lambda_bar.npy'

DATA_KEY_EIGENVALUES = 'lambdas'
OUTPUT_FIGURE_FILENAME = "../figures/figure_3c_angle_distribution.png"
BINS_COUNT = 40  # Number of bins for the histograms.


def load_eigenvalue_data(file_path: str, data_key: str = None) -> np.ndarray:
    """
    Loads complex eigenvalue data from a specified .npy or .npz file.

    Args:
        file_path (str): The full path to the .npy or .npz file.
        data_key (str): The key within the .npz file that holds the data (ignored for .npy files).

    Returns:
        np.ndarray: A NumPy array of complex numbers.
    """
    try:
        if file_path.endswith('.npy'):
            data = np.load(file_path)
            print(f"Successfully loaded .npy data from '{file_path}'.")
            return data
        else:
            # Handle .npz files
            with np.load(file_path) as data:
                print(f"Successfully loaded .npz data from '{file_path}'.")
                return data[data_key] if data_key else data[list(data.keys())[0]]
    except FileNotFoundError:
        print(f"Error: The file was not found at '{file_path}'.")
        print("Please update the file path variables.")
        exit()
    except KeyError:
        print(f"Error: The key '{data_key}' was not found in '{file_path}'.")
        print("Please check the DATA_KEY_EIGENVALUES variable.")
        exit()


def plot_angle_distribution(data_30: np.ndarray, data_100: np.ndarray, output_filename: str):
    """
    Creates and saves the dual-axis histogram and density plot for phase angles.

    Args:
        data_30 (np.ndarray): Complex eigenvalues from the 30-step delay model.
        data_100 (np.ndarray): Complex eigenvalues from the 100-step delay model.
        output_filename (str): The filename for the saved figure.
    """
    # Step 1: Extract the absolute phase angles from the complex eigenvalues (centered at π/2).
    abs_angles_30 = np.abs(np.angle(data_30) + np.pi/2)
    abs_angles_100 = np.abs(np.angle(data_100) + np.pi/2)

    # Step 2: Set up the plot with dual Y-axes.
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()  # Create a second y-axis that shares the x-axis.

    # Step 3: Plot data for the 100-step delay condition.
    # Plot histogram on the primary axis (ax1).
    ax1.hist(abs_angles_100, bins=BINS_COUNT, alpha=0.6, color='lightcoral', label='100 delay (Histogram)')
    # Calculate and plot the Kernel Density Estimate (KDE) on the secondary axis (ax2).
    kde_100 = gaussian_kde(abs_angles_100)
    x_grid = np.linspace(0, np.pi, 500)
    ax2.plot(x_grid, kde_100(x_grid), color='red', lw=3, label='100 delay (Fitted Curve)')
    
    # Step 4: Plot data for the 30-step delay condition.
    ax1.hist(abs_angles_30, bins=BINS_COUNT, alpha=0.6, color='lightblue', label='30 delay (Histogram)')
    kde_30 = gaussian_kde(abs_angles_30)
    ax2.plot(x_grid, kde_30(x_grid), color='royalblue', lw=3, label='30 delay (Fitted Curve)')
    
    # Step 5: Customize labels, ticks, and legend for publication quality.
    ax1.set_xlabel('Absolute Value of Phase', fontsize=14)
    ax1.set_ylabel('Frequency (Counts)', fontsize=14)
    ax2.set_ylabel('Density', fontsize=14, rotation=270, labelpad=15)
    
    ax1.tick_params(axis='y', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.set_ylim(bottom=0)

    # Set custom x-axis ticks to represent radians.
    ax1.set_xticks(
        [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
        [r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
    )
    ax1.set_xlim(0, np.pi)
    ax1.tick_params(axis='x', labelsize=16)

    # Combine legends from both axes into a single box.
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    # Reversing order to match the paper's legend style
    ax1.legend(handles1[::-1] + handles2[::-1], labels1[::-1] + labels2[::-1], fontsize=12, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print(f"\n Successfully created and saved the distribution plot as '{output_filename}'.")
    # plt.show()


def main():
    """Main execution function."""
    
    # 1. Load data for both conditions.
    eigenvalues_30 = load_eigenvalue_data(FILE_PATH_DELAY_30, DATA_KEY_EIGENVALUES)
    eigenvalues_100 = load_eigenvalue_data(FILE_PATH_DELAY_100, DATA_KEY_EIGENVALUES)
    
    # 2. Generate and save the plot.
    plot_angle_distribution(eigenvalues_30, eigenvalues_100, OUTPUT_FIGURE_FILENAME)


if __name__ == "__main__":
    main()
