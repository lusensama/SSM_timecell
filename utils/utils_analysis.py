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


def sort_resp(total_resp, norm=True):
    """
    Average the responses across episodes, normalize the activity according to the
    maximum and minimum of each cell (optional), and sort cells by when their maximum response happens.
    returns: cell_nums, sorted_matrix
    """
    np.seterr(divide='ignore', invalid='ignore')
    n_neurons = np.shape(total_resp)[2]
    segments = np.moveaxis(total_resp, 0, 1)
    unsorted_matrix = np.zeros((n_neurons, len(segments)))  # len(segments) is also len_delay
    sorted_matrix = np.zeros((n_neurons, len(segments)))
    for i in range(len(segments)):  # at timestep i
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

