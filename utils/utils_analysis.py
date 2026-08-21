from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy import stats
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.manifold import TSNE
import torch
import numpy as np
from tqdm import tqdm
from matplotlib_venn import venn2
import umap
from statsmodels.formula.api import ols

def sort_freq_resp(total_resp, norm=True):
    """
    Sort neural responses by frequency content (number of peaks detected via Fourier transform),
    then by timing of maximum response within each frequency group.
    
    Steps:
    1. Normalize the activity matrix
    2. Use FFT to detect the number of peaks for each neuron
    3. Group neurons by peak count and sort each group by max response timing
    
    Args:
        total_resp: Response data with shape (episodes, timesteps, neurons)
        norm: Whether to normalize (default True, always applied in this function)
    
    Returns:
        cell_nums: Indices of cells in sorted order
        sorted_matrix: Sorted and normalized response matrix
        normalized_matrix: Normalized response matrix (unsorted)
        peak_counts: Number of peaks detected for each neuron (in sorted order)
    """
    np.seterr(divide='ignore', invalid='ignore')
    n_neurons = np.shape(total_resp)[2]
    segments = np.moveaxis(total_resp, 0, 1)
    unsorted_matrix = np.zeros((n_neurons, len(segments)))
    
    for i in range(len(segments)):
        averages = np.mean(segments[i], axis=0)
        unsorted_matrix[:, i] = np.transpose(averages)
    
    scaled = (unsorted_matrix - np.min(unsorted_matrix, axis=1, keepdims=True)) \
             / np.ptp(unsorted_matrix, axis=1, keepdims=True)
    normalized_matrix = scaled * 2 - 1
    
    def _count_peaks(sig, prom_frac=0.2):
        """Count local maxima including endpoints with a simple prominence filter."""
        if len(sig) == 1:
            return 1
        min_v, max_v = np.min(sig), np.max(sig)
        amp = max_v - min_v
        thr = min_v + prom_frac * amp
        peaks = 0
        if sig[0] >= sig[1] and sig[0] >= thr:
            peaks += 1
        for j in range(1, len(sig) - 1):
            if sig[j] >= thr and sig[j] >= sig[j - 1] and sig[j] > sig[j + 1]:
                peaks += 1
        if sig[-1] >= sig[-2] and sig[-1] >= thr:
            peaks += 1
        return peaks

    peak_counts = np.zeros(n_neurons, dtype=int)
    for i in range(n_neurons):
        signal = normalized_matrix[i, :]
        peak_counts[i] = _count_peaks(signal, prom_frac=0.6)
    
    max_indices = np.argmax(normalized_matrix, axis=1)
    
    sorting_keys = []
    for i in range(n_neurons):
        if peak_counts[i] == 1 or peak_counts[i] == 2:
            sorting_keys.append((1, max_indices[i]))
        else:
            sorting_keys.append((peak_counts[i], i))
    
    sorting_keys = np.array(sorting_keys, dtype=[('peak_count', int), ('max_idx', int)])
    cell_nums = np.argsort(sorting_keys, order=['peak_count', 'max_idx'])
    
    sorted_matrix = np.zeros((n_neurons, len(segments)))
    sorted_peak_counts = np.zeros(n_neurons, dtype=int)
    for i, i_cell in enumerate(cell_nums):
        sorted_matrix[i] = normalized_matrix[i_cell]
        sorted_peak_counts[i] = peak_counts[i_cell]
    
    assert len(sorted_matrix) == n_neurons
    return cell_nums, sorted_matrix, normalized_matrix
    
def sort_resp(total_resp, norm=True):
    """
    Average the responses across episodes, normalize the activity according to the
    maximum and minimum of each cell (optional), and sort cells by when their maximum response happens.
    returns: cell_nums, sorted_matrix
    """
    np.seterr(divide='ignore', invalid='ignore')
    n_neurons = np.shape(total_resp)[2]
    segments = np.moveaxis(total_resp, 0, 1)
    unsorted_matrix = np.zeros((n_neurons, len(segments)))
    sorted_matrix = np.zeros((n_neurons, len(segments)))
    for i in range(len(segments)):
        averages = np.mean(segments[i],
                           axis=0)
        unsorted_matrix[:, i] = np.transpose(
            averages)
        if norm is True:
            scaled = (unsorted_matrix
                      - np.min(unsorted_matrix, axis=1, keepdims=True)) \
                     / np.ptp(unsorted_matrix, axis=1, keepdims=True)

            normalized_matrix = scaled * 2 - 1
            max_indeces = np.argmax(normalized_matrix, axis=1)
            cell_nums = np.argsort(max_indeces)
            for i, i_cell in enumerate(list(cell_nums)):
                sorted_matrix[i] = normalized_matrix[i_cell]
        else:
            max_indeces = np.argmax(unsorted_matrix, axis=1)
            cell_nums = np.argsort(max_indeces)
            for i, i_cell in enumerate(list(cell_nums)):
                sorted_matrix[i] = unsorted_matrix[i_cell]
    assert len(sorted_matrix) == n_neurons
    return cell_nums, sorted_matrix, normalized_matrix

