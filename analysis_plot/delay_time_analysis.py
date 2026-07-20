import argparse
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy import stats
from tqdm import tqdm


DELAY_KEYS = ("delay1_resp1", "delay2_resp1")


def calculate_tuning_curves(resp: np.ndarray) -> np.ndarray:
    """Trial-averaged tuning curve for each neuron."""
    return np.mean(resp, axis=0).T


def calculate_tuning_curves_single_neuron(resp: np.ndarray) -> np.ndarray:
    """Trial-averaged tuning curve for one neuron."""
    return np.mean(resp, axis=0)


def shuffle_activity_single_neuron(delay_resp: np.ndarray) -> np.ndarray:
    """Shuffle activity within each episode for a single neuron."""
    if delay_resp.ndim != 2:
        raise ValueError("shuffle_activity_single_neuron expects 2D input (episodes, time).")
    shuffled_delay_resp = np.empty_like(delay_resp)
    n_episodes, len_delay = delay_resp.shape
    shift = np.random.randint(np.floor(len_delay * 0.3), np.ceil(len_delay * 0.7), size=n_episodes)
    for i_eps in range(n_episodes):
        shuffled_delay_resp[i_eps, :] = np.roll(delay_resp[i_eps, :], shift=shift[i_eps])
    return shuffled_delay_resp


def lin_reg_ramping(resp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit linear regression to trial-averaged tuning curve of each neuron to identify ramping cells.
    Returns p-values, slopes, intercepts, and Pearson r for each neuron.
    """
    len_delay = resp.shape[1]
    n_neurons = resp.shape[2]
    p_result = np.zeros(n_neurons)
    slope_result = np.zeros(n_neurons)
    intercept_result = np.zeros(n_neurons)
    pearson_r_result = np.zeros(n_neurons)
    tuning_curves = calculate_tuning_curves(resp)
    t = np.arange(len_delay)
    for i_neuron in tqdm(range(n_neurons)):
        slope, intercept, r_val, p_val, _std_err = stats.linregress(t, tuning_curves[i_neuron])
        slope_result[i_neuron] = slope
        intercept_result[i_neuron] = intercept
        pearson_r_result[i_neuron] = r_val
        p_result[i_neuron] = p_val
    return p_result, slope_result, intercept_result, pearson_r_result


def trial_reliability_vs_shuffle_score(
    resp: np.ndarray, *, split: str = "odd-even", percentile: float = 95.0, n_shuff: int = 1000
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute trial-to-trial reliability and shuffle-based thresholds.
    Returns observed reliability and shuffle percentile threshold for each neuron.
    """
    if split not in {"random", "odd-even"}:
        raise ValueError("split must be 'random' or 'odd-even'")
    n_episodes = resp.shape[0]
    n_neurons = resp.shape[2]
    reliability = np.zeros(n_neurons)
    reliability_thresh = np.zeros(n_neurons)
    for i_neuron in tqdm(range(n_neurons)):
        if split == "random":
            split_1_idx = np.random.choice(n_episodes, n_episodes // 2, replace=False)
            mask = np.zeros(n_episodes, dtype=bool)
            mask[split_1_idx] = True
            split_2_idx = np.arange(n_episodes)[~mask]
        else:
            split_1_idx = np.arange(start=0, stop=n_episodes - 1, step=2)
            split_2_idx = np.arange(start=1, stop=n_episodes, step=2)

        resp_1 = resp[split_1_idx, :, i_neuron]
        resp_2 = resp[split_2_idx, :, i_neuron]
        reliability[i_neuron], _pval = stats.pearsonr(
            np.mean(resp_1, axis=0), np.mean(resp_2, axis=0)
        )

        shuffled_score = np.zeros(n_shuff)
        for i_shuff in range(n_shuff):
            shuffled_resp = shuffle_activity_single_neuron(resp[:, :, i_neuron])
            resp_1 = shuffled_resp[split_1_idx, :]
            resp_2 = shuffled_resp[split_2_idx, :]
            shuffled_score[i_shuff], _pval = stats.pearsonr(
                np.mean(resp_1, axis=0), np.mean(resp_2, axis=0)
            )
        reliability_thresh[i_neuron] = np.percentile(shuffled_score, percentile)
    return reliability, reliability_thresh


def skaggs_temporal_information(
    resp: np.ndarray,
    ramping_bool: np.ndarray,
    *,
    n_shuff: int = 1000,
    percentile: float = 95.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Skaggs temporal information and shuffle-based thresholds."""
    len_delay = resp.shape[1]
    n_neurons = resp.shape[2]
    info = np.zeros(n_neurons)
    info_thresh = np.zeros(n_neurons)
    p_t = 1 / len_delay
    for i_neuron in tqdm(range(n_neurons)):
        tuning_curve = np.mean(resp[:, :, i_neuron], axis=0)
        if ramping_bool[i_neuron]:
            t = np.arange(len_delay)
            slope, intercept, _r, _p, _std_err = stats.linregress(t, tuning_curve)
            lin_subtracted = tuning_curve - (slope * t + intercept)
            min_val = lin_subtracted.min()
            if min_val < 0:
                lin_subtracted = lin_subtracted - min_val + 1e-10
            tuning_curve = lin_subtracted

        tuning_curve = np.where(tuning_curve <= 0, 1e-10, tuning_curve)
        mean_rate = np.mean(tuning_curve)
        if mean_rate > 0:
            info[i_neuron] = np.sum(tuning_curve * (p_t * np.log2(tuning_curve / mean_rate)))
        else:
            info[i_neuron] = 0.0

        surrogate = np.zeros(n_shuff)
        for i_shuff in range(n_shuff):
            shuffled_resp = shuffle_activity_single_neuron(resp[:, :, i_neuron])
            shuffled_tuning = calculate_tuning_curves_single_neuron(shuffled_resp)
            shuffled_tuning = np.where(shuffled_tuning <= 0, 1e-10, shuffled_tuning)
            mean_rate_shuff = np.mean(shuffled_tuning)
            if mean_rate_shuff > 0:
                surrogate[i_shuff] = np.sum(
                    shuffled_tuning * (p_t * np.log2(shuffled_tuning / mean_rate_shuff))
                )
            else:
                surrogate[i_shuff] = 0.0
        info_thresh[i_neuron] = np.percentile(surrogate, percentile)
    return info, info_thresh


def load_and_sanitize(
    path: str,
    *,
    cast_float64: bool = False,
) -> Dict[str, np.ndarray]:
    """Load full dataset and drop imaginary components."""
    raw = np.load(path, allow_pickle=True)
    if raw.dtype != object:
        raise ValueError("Expected object-dtype npy storing a dict.")
    raw_dict = raw.item()
    sanitized: Dict[str, np.ndarray] = {}
    for key, value in raw_dict.items():
        arr = np.asarray(value)
        if np.iscomplexobj(arr):
            arr = arr.real
        if cast_float64 and arr.dtype.kind in {"f", "c"}:
            arr = arr.astype(np.float64, copy=False)
        sanitized[key] = np.array(arr)
    return sanitized


@dataclass
class DelaySummary:
    p_vals: np.ndarray
    slopes: np.ndarray
    pearson_r: np.ndarray
    ramping_candidate: np.ndarray
    temporal_info: np.ndarray
    temporal_info_thresh: np.ndarray
    time_candidate: np.ndarray
    reliability: np.ndarray
    reliability_thresh: np.ndarray
    ramping_final: np.ndarray
    time_final: np.ndarray


def load_delay_data(
    path: str,
    *,
    sanitized_data: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Load delay responses, keep only real values, then per-trial per-neuron project to [0, 1]."""
    source = sanitized_data if sanitized_data is not None else load_and_sanitize(
        path, cast_float64=True
    )
    delay_data = {}
    # Only process keys that exist in the source data
    available_keys = [key for key in DELAY_KEYS if key in source]
    if not available_keys:
        raise KeyError(f"None of {DELAY_KEYS} found in {path}. Available keys: {list(source.keys())}")
    
    for key in available_keys:
        arr = np.asarray(source[key], dtype=np.float64)  # (episodes, time, neurons)
        # Per-trial, per-neuron min-max to [0,1]
        trial_min = arr.min(axis=1, keepdims=True)   # (episodes, 1, neurons)
        trial_max = arr.max(axis=1, keepdims=True)   # (episodes, 1, neurons)
        denom = np.where(trial_max > trial_min, trial_max - trial_min, 1.0)
        projected = (arr - trial_min) / denom
        projected = np.clip(projected, 0.0, 1.0)
        # Debug: print first 5 neurons' tuning curves after projection
        tuning_curve = projected.mean(axis=0)  # (time, neurons)
        print(f"[{key}] projected tuning curves shape: {tuning_curve.shape}")
        print(f"[{key}] first 5 projected tuning curves:\n{tuning_curve[:, :5]}")
        delay_data[key] = projected
    return delay_data


def analyze_delay(
    resp: np.ndarray,
    *,
    percentile: float,
    n_shuff: int,
) -> DelaySummary:
    """Run analyses using utils_time_ramp methods on pre-projected data."""
    np.random.seed(0)
    # Use utils_time_ramp helpers directly
    p_vals, slopes, _intercepts, pearson_r = lin_reg_ramping(resp)
    ramping_candidate = (p_vals <= 0.05) & (np.abs(pearson_r) >= 0.9)
    reliability, reliability_thresh = trial_reliability_vs_shuffle_score(
        resp, split="odd-even", percentile=percentile, n_shuff=n_shuff
    )
    temporal_info, temporal_info_thresh = skaggs_temporal_information(
        resp, ramping_candidate, n_shuff=n_shuff, percentile=percentile
    )
    time_candidate = temporal_info > temporal_info_thresh
    reliable_bool = reliability > reliability_thresh
    ramping_final = ramping_candidate & reliable_bool
    time_final = time_candidate & reliable_bool

    return DelaySummary(
        p_vals=p_vals,
        slopes=slopes,
        pearson_r=pearson_r,
        ramping_candidate=ramping_candidate,
        temporal_info=temporal_info,
        temporal_info_thresh=temporal_info_thresh,
        time_candidate=time_candidate,
        reliability=reliability,
        reliability_thresh=reliability_thresh,
        ramping_final=ramping_final,
        time_final=time_final,
    )


def summarize(label: str, summary: DelaySummary) -> None:
    """Print concise summary statistics."""
    n_neurons = summary.ramping_candidate.size
    ramp_candidate_count = int(summary.ramping_candidate.sum())
    ramp_final_count = int(summary.ramping_final.sum())
    time_candidate_count = int(summary.time_candidate.sum())
    time_final_count = int(summary.time_final.sum())
    reliable_count = int((summary.reliability > summary.reliability_thresh).sum())

    print(f"=== {label} ===")
    print(f"Neurons analysed      : {n_neurons}")
    print(f"Ramping candidates    : {ramp_candidate_count} ({ramp_candidate_count / n_neurons:.1%})")
    print(f"Ramping (final)       : {ramp_final_count} ({ramp_final_count / n_neurons:.1%})")
    print(f"Time candidates       : {time_candidate_count} ({time_candidate_count / n_neurons:.1%})")
    print(f"Time cells (final)    : {time_final_count} ({time_final_count / n_neurons:.1%})")
    print(f"Reliability > thresh  : {reliable_count} ({reliable_count / n_neurons:.1%})")
    print(f"Median slope          : {np.median(summary.slopes):.4f}")
    print(f"Median temporal info  : {np.nanmedian(summary.temporal_info):.4f}")
    print(f"Median reliability    : {np.median(summary.reliability):.4f}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze delay responses for ramping and time-cell properties."
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--data-path",
        # default=os.path.join(os.path.dirname(script_dir), "3stim_hippo_raw_data.npy"),
        default="/home/lugroup/Documents/Sen_Code/time_cell_submit/capsule-0490840-code/raw_data_storage/3stim_best_model_spiking_raw_data_seed5.npy",
        help="Path to 3stim data file.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Percentile threshold for shuffle-based metrics.",
    )
    parser.add_argument(
        "--n-shuff",
        type=int,
        default=100,
        help="Number of shuffles for RB and reliability controls.",
    )
    parser.add_argument(
        "--real-output",
        help="If provided, save a copy of the dataset with imaginary parts removed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sanitized_full = load_and_sanitize(args.data_path, cast_float64=True)
    if args.real_output:
        np.save(args.real_output, sanitized_full, allow_pickle=True)
        print(f"Saved real-valued data to {args.real_output}")

    delay_data = load_delay_data(args.data_path, sanitized_data=sanitized_full)

    for key in DELAY_KEYS:
        summary = analyze_delay(
            delay_data[key],
            percentile=args.percentile,
            n_shuff=args.n_shuff,
        )
        summarize(key, summary)


if __name__ == "__main__":
    main()

