"""Shared helpers for the response-letter figure pipeline.

Two layers live in this folder:

  extract_*.py  read the ORIGINAL run artifacts under training/ and figures/
                (jsonl, json, npz state caches, slurm logs) and write flat CSVs
                into figures/response/data/.  These need the run tree.
  plot_*.py     read ONLY figures/response/data/*.csv (+ the two heatmap npz
                bundles that live there) and write PNG/PDF into
                figures/response/.  These need no models and no training/ tree.

Everything downstream of data/ is therefore reproducible from this folder alone.
"""
import os
import csv
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESP = os.path.dirname(HERE)
DATA = os.path.join(RESP, "data")
ROOT = os.path.dirname(os.path.dirname(RESP))

CHANCE_3STIM = 100.0 / 3.0

MODE_ORDER = ["hippo", "spectrum_matched", "freq_matched", "perturbed_hippo",
              "s4d_lin", "s4d_inv", "alt_basis", "real_diagonal", "rand_complex"]

MODE_LABEL = {
    "hippo": "HiPPO-LegS",
    "spectrum_matched": "spectrum_matched",
    "freq_matched": "freq_matched",
    "perturbed_hippo": "perturbed_hippo",
    "s4d_lin": "S4D-Lin",
    "s4d_inv": "S4D-Inv",
    "alt_basis": "alt_basis",
    "real_diagonal": "real_diagonal",
    "rand_complex": "rand_complex",
}

MODE_CLASS = {
    "hippo": "learns + cascade",
    "spectrum_matched": "learns + cascade",
    "freq_matched": "learns + cascade",
    "perturbed_hippo": "learns + cascade",
    "s4d_lin": "learns + cascade",
    "s4d_inv": "learns + cascade",
    "alt_basis": "cascade, cannot learn",
    "real_diagonal": "learns, no cascade",
    "rand_complex": "learns, no cascade",
}

def data_path(name):
    os.makedirs(DATA, exist_ok=True)
    return os.path.join(DATA, name)

def fig_path(name):
    os.makedirs(RESP, exist_ok=True)
    return os.path.join(RESP, name)

def write_csv(name, header, rows):
    """Write rows to data/<name> and echo the path."""
    p = data_path(name)
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    print(f"  wrote {os.path.relpath(p, ROOT)}  ({len(rows)} rows)")
    return p

def read_csv(name):
    """Read data/<name> as a list of dicts, numbers coerced to float."""
    p = os.path.join(DATA, name)
    out = []
    with open(p) as f:
        for row in csv.DictReader(f):
            rec = {}
            for k, v in row.items():
                if v is None or v == "":
                    rec[k] = None
                    continue
                try:
                    rec[k] = float(v)
                except (TypeError, ValueError):
                    rec[k] = v
            out.append(rec)
    return out

def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out

def r(x, n=4):
    """Round, tolerating None/NaN."""
    if x is None:
        return ""
    try:
        fx = float(x)
    except (TypeError, ValueError):
        return x
    if fx != fx:
        return ""
    return round(fx, n)

def _count_peaks(sig, prom_frac=0.6):
    """Local maxima including endpoints, with a prominence floor.

    Verbatim from utils.utils_analysis.sort_freq_resp's inner helper, called
    there with prom_frac=0.6.
    """
    if len(sig) == 1:
        return 1
    min_v, max_v = np.min(sig), np.max(sig)
    thr = min_v + prom_frac * (max_v - min_v)
    peaks = 0
    if sig[0] >= sig[1] and sig[0] >= thr:
        peaks += 1
    for j in range(1, len(sig) - 1):
        if sig[j] >= thr and sig[j] >= sig[j - 1] and sig[j] > sig[j + 1]:
            peaks += 1
    if sig[-1] >= sig[-2] and sig[-1] >= thr:
        peaks += 1
    return peaks

def sort_freq_resp(total_resp, norm=True):
    """Verbatim reimplementation of utils.utils_analysis.sort_freq_resp.

    THIS IS THE MANUSCRIPT'S HEATMAP ORDERING.  Units are grouped by frequency
    content -- the number of prominence-filtered peaks in the trial-averaged,
    per-unit normalized profile -- and then ordered by peak time WITHIN the
    one-or-two-peak group (single-peaked time cells and their two-peak
    neighbours are pooled into one group); units with three or more peaks keep
    their original relative order after that block.  ssm_observer_1d.py, which
    produced the paper's sorted-activity panels, calls this and not sort_resp.

    Inlined for the same reason as sort_resp: utils_analysis imports sklearn,
    whose build here is ABI-incompatible with the installed numpy.

    Returns (cell_nums, sorted_matrix, normalized_matrix, sorted_peak_counts).
    """
    np.seterr(divide="ignore", invalid="ignore")
    n_neurons = np.shape(total_resp)[2]
    segments = np.moveaxis(total_resp, 0, 1)
    unsorted_matrix = np.zeros((n_neurons, len(segments)))
    for i in range(len(segments)):
        unsorted_matrix[:, i] = np.transpose(np.mean(segments[i], axis=0))

    scaled = (unsorted_matrix - np.min(unsorted_matrix, axis=1, keepdims=True)) \
        / np.ptp(unsorted_matrix, axis=1, keepdims=True)
    normalized_matrix = scaled * 2 - 1

    peak_counts = np.array([_count_peaks(normalized_matrix[i, :])
                            for i in range(n_neurons)], dtype=int)
    max_indices = np.argmax(normalized_matrix, axis=1)
    keys = np.array([(1, max_indices[i]) if peak_counts[i] in (1, 2)
                     else (peak_counts[i], i) for i in range(n_neurons)],
                    dtype=[("peak_count", int), ("max_idx", int)])
    cell_nums = np.argsort(keys, order=["peak_count", "max_idx"])

    sorted_matrix = np.zeros((n_neurons, len(segments)))
    sorted_peaks = np.zeros(n_neurons, dtype=int)
    for i, i_cell in enumerate(cell_nums):
        sorted_matrix[i] = normalized_matrix[i_cell]
        sorted_peaks[i] = peak_counts[i_cell]
    return cell_nums, sorted_matrix, normalized_matrix, sorted_peaks

