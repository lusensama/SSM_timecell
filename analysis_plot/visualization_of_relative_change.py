import os
import warnings

try:
    import torch
except Exception as e:
    raise SystemExit("PyTorch is required to read .pt files, please install it first: pip install torch") from e

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True, linewidth=200)
data_dir = "/home/lugroup/Documents/Sen_Code/time_cell_submit/capsule-0490840-code/data/figure_3_relative_change"
plot_dir = "/home/lugroup/Documents/Sen_Code/time_cell_submit/capsule-0490840-code/figures"
def load_complex_series(pt_path: str) -> np.ndarray:
    obj = torch.load(pt_path, map_location="cpu")
    arr = None

    if torch.is_tensor(obj):
        arr = obj.detach().cpu().numpy()
    elif isinstance(obj, dict):
        for v in obj.values():
            if torch.is_tensor(v):
                arr = v.detach().cpu().numpy()
                break
            if isinstance(v, (list, tuple, np.ndarray)):
                arr = np.asarray(v)
                break
        if arr is None:
            raise ValueError(f"The content of {pt_path} is not a tensor or array and cannot be parsed.")
    elif isinstance(obj, (list, tuple, np.ndarray)):
        arr = np.asarray(obj)
    else:
        raise ValueError(f"The content type {type(obj)} of {pt_path} is not supported.")

    arr = np.asarray(arr)

    if np.iscomplexobj(arr):
        series = arr.astype(np.complex128).ravel()
    elif arr.ndim >= 2 and arr.shape[-1] == 2:
        series = (arr[..., 0] + 1j * arr[..., 1]).reshape(-1)
    elif arr.ndim >= 2 and arr.shape[0] == 2:
        series = (arr[0] + 1j * arr[1]).reshape(-1)
    elif arr.ndim == 1:
        warnings.warn(f"{pt_path} appears to be a real-valued sequence, treating imaginary part as 0.")
        series = arr.astype(np.float64) + 0j
    else:
        raise ValueError(f"The shape {arr.shape} of {pt_path} cannot be parsed as a complex sequence.")
    return series


def compute_change_magnitude(initial: np.ndarray, final: np.ndarray) -> np.ndarray:
    if initial.size != final.size:
        warnings.warn(
            f"{initial.size=} and {final.size=} are not equal, calculating change magnitude based on the shorter length."
        )
    size = min(initial.size, final.size)
    delta = final[:size] - initial[:size]
    return np.abs(delta)


def compute_relative_change(initial: np.ndarray, final: np.ndarray) -> np.ndarray:
    deltas = compute_change_magnitude(initial, final)
    if deltas.size == 0:
        return deltas
    init_mag = np.abs(initial[:deltas.size])
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(init_mag > 0, deltas / init_mag, np.inf)
    return rel


def plot_relative_change_histogram(ax, hippo_rel: np.ndarray, rand_rel: np.ndarray, title: str, ylabel: str = "Relative change") -> None:
    """Plot frequency histogram of relative change rate"""
    # Filter out non-finite values
    hippo_finite = hippo_rel[np.isfinite(hippo_rel)]
    rand_finite = rand_rel[np.isfinite(rand_rel)]
    
    if hippo_finite.size == 0 and rand_finite.size == 0:
        ax.text(0.5, 0.5, "No finite data to plot", 
                transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title)
        return
    
    # Determine appropriate bins range
    all_data = np.concatenate([hippo_finite, rand_finite]) if hippo_finite.size > 0 and rand_finite.size > 0 else (hippo_finite if hippo_finite.size > 0 else rand_finite)
    
    # Use quantiles to exclude extreme outliers for clearer histogram
    q1, q99 = np.percentile(all_data, [1, 99])
    bins = np.linspace(q1, q99, 50)
    
    # Plot histogram
    ax.hist(hippo_finite, bins=bins, alpha=0.6, color="#d62728", label="hippo", density=True, edgecolor="black", linewidth=0.5)
    ax.hist(rand_finite, bins=bins, alpha=0.6, color="#1f77b4", label="rand", density=True, edgecolor="black", linewidth=0.5)
    
    ax.set_xlabel(ylabel)
    ax.set_ylabel("Density (Frequency)")
    ax.set_title(title)
    ax.grid(True, ls="--", alpha=0.4, axis='y')
    ax.legend(loc="upper right")
    
    # Add statistical information
    def format_stats(values):
        if values.size == 0:
            return None
        return values.min(), values.mean(), values.max()
    
    hippo_stats = format_stats(hippo_finite)
    rand_stats = format_stats(rand_finite)
    
    if hippo_stats:
        h_min, h_mean, h_max = hippo_stats
        ax.axvline(h_mean, color="#d62728", linestyle="--", linewidth=2, alpha=0.8, label=f"hippo mean={h_mean:.4f}")
        ax.text(
            0.02,
            0.95,
            f"hippo:\nmin={h_min:.4f}\nmean={h_mean:.4f}\nmax={h_max:.4f}\nn={hippo_finite.size}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="#d62728",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75),
        )
    
    if rand_stats:
        r_min, r_mean, r_max = rand_stats
        ax.axvline(r_mean, color="#1f77b4", linestyle="--", linewidth=2, alpha=0.8, label=f"rand mean={r_mean:.4f}")
        ax.text(
            0.98,
            0.95,
            f"rand:\nmin={r_min:.4f}\nmean={r_mean:.4f}\nmax={r_max:.4f}\nn={rand_finite.size}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color="#1f77b4",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75),
        )


def print_relative_summary(label: str, rel_values: np.ndarray) -> None:
    print(f"=== {label} relative change summary (len={rel_values.size}) ===")
    finite_vals = rel_values[np.isfinite(rel_values)]
    if finite_vals.size == 0:
        print("All ratios undefined (initial magnitude zero).")
        return
    print(f"min={finite_vals.min():.6f}, max={finite_vals.max():.6f}, mean={finite_vals.mean():.6f}")


def main():

    # Define file paths for Lambda
    hippo_lambda_init_path  = os.path.join(data_dir, "initial_lambda_200.pt")
    hippo_lambda_final_path = os.path.join(data_dir, "final_lambda_200.pt")
    rand_lambda_init_path   = os.path.join(data_dir, "initial_lambda_201.pt")
    rand_lambda_final_path  = os.path.join(data_dir, "final_lambda_201.pt")

    # Define file paths for Lambda_bar
    hippo_lambda_bar_init_path  = os.path.join(data_dir, "initial_lambda_bar_200.pt")
    hippo_lambda_bar_final_path = os.path.join(data_dir, "final_lambda_bar_200.pt")
    rand_lambda_bar_init_path   = os.path.join(data_dir, "initial_lambda_bar_201.pt")
    rand_lambda_bar_final_path  = os.path.join(data_dir, "final_lambda_bar_201.pt")

    # Define file paths for B
    hippo_B_init_path  = os.path.join(data_dir, "initial_B_200.pt")
    hippo_B_final_path = os.path.join(data_dir, "final_B_200.pt")
    rand_B_init_path   = os.path.join(data_dir, "initial_B_201.pt")
    rand_B_final_path  = os.path.join(data_dir, "final_B_201.pt")

    # Define file paths for C_tilde
    hippo_C_tilde_init_path  = os.path.join(data_dir, "initial_C_tilde_200.pt")
    hippo_C_tilde_final_path = os.path.join(data_dir, "final_C_tilde_200.pt")
    rand_C_tilde_init_path   = os.path.join(data_dir, "initial_C_tilde_201.pt")
    rand_C_tilde_final_path  = os.path.join(data_dir, "final_C_tilde_201.pt")

    # Load Lambda data
    hippo_lambda_init  = load_complex_series(hippo_lambda_init_path)
    hippo_lambda_final = load_complex_series(hippo_lambda_final_path)
    rand_lambda_init   = load_complex_series(rand_lambda_init_path)
    rand_lambda_final  = load_complex_series(rand_lambda_final_path)

    # Load Lambda_bar data
    hippo_lambda_bar_init  = load_complex_series(hippo_lambda_bar_init_path)
    hippo_lambda_bar_final = load_complex_series(hippo_lambda_bar_final_path)
    rand_lambda_bar_init   = load_complex_series(rand_lambda_bar_init_path)
    rand_lambda_bar_final  = load_complex_series(rand_lambda_bar_final_path)

    # Load B data
    hippo_B_init  = load_complex_series(hippo_B_init_path)
    hippo_B_final = load_complex_series(hippo_B_final_path)
    rand_B_init   = load_complex_series(rand_B_init_path)
    rand_B_final  = load_complex_series(rand_B_final_path)

    # Load C_tilde data
    hippo_C_tilde_init  = load_complex_series(hippo_C_tilde_init_path)
    hippo_C_tilde_final = load_complex_series(hippo_C_tilde_final_path)
    rand_C_tilde_init   = load_complex_series(rand_C_tilde_init_path)
    rand_C_tilde_final  = load_complex_series(rand_C_tilde_final_path)

    # Compute relative changes for Lambda
    hippo_lambda_rel = compute_relative_change(hippo_lambda_init, hippo_lambda_final)
    rand_lambda_rel  = compute_relative_change(rand_lambda_init, rand_lambda_final)

    # Compute relative changes for Lambda_bar
    hippo_lambda_bar_rel = compute_relative_change(hippo_lambda_bar_init, hippo_lambda_bar_final)
    rand_lambda_bar_rel  = compute_relative_change(rand_lambda_bar_init, rand_lambda_bar_final)

    # Compute relative changes for B
    hippo_B_rel = compute_relative_change(hippo_B_init, hippo_B_final)
    rand_B_rel  = compute_relative_change(rand_B_init, rand_B_final)

    # Compute relative changes for C_tilde
    hippo_C_tilde_rel = compute_relative_change(hippo_C_tilde_init, hippo_C_tilde_final)
    rand_C_tilde_rel  = compute_relative_change(rand_C_tilde_init, rand_C_tilde_final)

    # Print relative change summaries
    print("\n" + "="*80)
    print("RELATIVE CHANGE SUMMARY")
    print("="*80 + "\n")
    
    print_relative_summary("Lambda (hippo)",     hippo_lambda_rel)
    print_relative_summary("Lambda_bar (hippo)", hippo_lambda_bar_rel)
    print_relative_summary("Lambda (rand)",      rand_lambda_rel)
    print_relative_summary("Lambda_bar (rand)",  rand_lambda_bar_rel)
    print()
    print_relative_summary("B (hippo)", hippo_B_rel)
    print_relative_summary("B (rand)",  rand_B_rel)
    print()
    print_relative_summary("C_tilde (hippo)", hippo_C_tilde_rel)
    print_relative_summary("C_tilde (rand)",  rand_C_tilde_rel)

    # Figure 1: Lambda relative change histogram
    fig1, axes1 = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
    plot_relative_change_histogram(
        axes1[0],
        hippo_lambda_rel,
        rand_lambda_rel,
        "Relative change |Δλ|/|λ| (Lambda hippo vs rand)",
        ylabel="|Δλ|/|λ|"
    )
    plot_relative_change_histogram(
        axes1[1],
        hippo_lambda_bar_rel,
        rand_lambda_bar_rel,
        "Relative change |Δλ̄|/|λ̄| (Lambda_bar hippo vs rand)",
        ylabel="|Δλ̄|/|λ̄|"
    )
    fig1.suptitle("Lambda Relative Change Histogram (hippo vs rand)", fontsize=16, y=0.97)
    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    fig1.savefig(os.path.join(plot_dir, "Lambda_relative_change.png"), dpi=300, bbox_inches="tight")
    print("\n[Figure saved: Lambda_relative_change.png]")
    print("Summary: Histogram showing the distribution of relative changes in Lambda and Lambda_bar parameters.")
    print("         HiPPO initialization shows the change in eigenvalues compared to random initialization.")
    plt.clf()

    # Figure 2: B relative change histogram
    fig2, axes2 = plt.subplots(1, 1, figsize=(12, 6))
    plot_relative_change_histogram(
        axes2,
        hippo_B_rel,
        rand_B_rel,
        "Relative change |ΔB|/|B| (hippo vs rand)",
        ylabel="|ΔB|/|B|"
    )
    fig2.suptitle("B Matrix Relative Change Histogram (hippo vs rand)", fontsize=16, y=0.95)
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    fig2.savefig(os.path.join(plot_dir, "B_relative_change.png"), dpi=300, bbox_inches="tight")
    print("\n[Figure saved: B_relative_change.png]")
    print("Summary: Histogram showing the distribution of relative changes in B matrix parameters.")
    print("         Compares how much B matrix values changed during training for HiPPO vs random initialization.")
    plt.clf()

    # Figure 3: C_tilde relative change histogram
    fig3, axes3 = plt.subplots(1, 1, figsize=(12, 6))
    plot_relative_change_histogram(
        axes3,
        hippo_C_tilde_rel,
        rand_C_tilde_rel,
        "Relative change |ΔC̃|/|C̃| (hippo vs rand)",
        ylabel="|ΔC̃|/|C̃|"
    )
    fig3.suptitle("C_tilde Matrix Relative Change Histogram (hippo vs rand)", fontsize=16, y=0.95)
    fig3.tight_layout(rect=[0, 0, 1, 0.93])
    fig3.savefig(os.path.join(plot_dir, "C_tilde_relative_change.png"), dpi=300, bbox_inches="tight")
    print("\n[Figure saved: C_tilde_relative_change.png]")
    print("Summary: Histogram showing the distribution of relative changes in C_tilde matrix parameters.")
    print("         Compares how much C_tilde matrix values changed during training for HiPPO vs random initialization.")
    plt.clf()

    print("\n" + "="*80)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("="*80)


if __name__ == "__main__":
    main()
