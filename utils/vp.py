"""
Victor-Purpura spike-train distance, with the fixes from the Exp 5 audit.

The DP in train_and_plot_laps.py / ssm_run_laps.py is a correct Victor-Purpura
edit distance, but its *use* has three problems that matter once VP becomes the
primary metric and the checkpoint-selection criterion:

  1. q was passed inconsistently (q=1 in ssm_run_laps.py, q=0.1 in
     train_and_plot_laps.py) and never recorded, so numbers from the two scripts
     are not comparable. Here you pass a TOLERANCE in timesteps and q is derived
     as q = 2/tol -- the shift/(delete+insert) breakeven is |dt| = 2/q, so `tol`
     is exactly "how many steps of misalignment are still counted as the same
     event". Always report tol.
  2. Unsorted input silently returned a wrong answer. Now asserted.
  3. Both lists empty gave 0/0 -> nan, which poisons np.mean over episodes.
     Now returns 1.0 (two empty trains are identical).

`hard_window=True` additionally forbids matching an event pair further apart than
`tol`, which removes cross-landmark aliasing. Aliasing is only a problem with
(near-)periodic landmarks; under varied lap lengths it is benign, so the default
is False to stay numerically comparable with the existing logs.
"""

import numpy as np

__all__ = ["q_from_tol", "vp_distance", "vp_score"]

def q_from_tol(tol):
    """Shift cost per timestep such that the shift/del+ins breakeven is `tol`."""
    if tol <= 0:
        raise ValueError(f"tol must be positive, got {tol}")
    return 2.0 / float(tol)

def vp_distance(pred, gt, q, hard_window=None):
    """
    Victor-Purpura distance between two sorted 1-D event-time arrays.

    Cost model: delete = 1, insert = 1, shift = q * |dt|. Because the shift branch
    competes against delete+insert, no single event ever costs more than 2.

    hard_window: if not None, a shift of more than `hard_window` steps is
        forbidden outright (forcing delete+insert) instead of merely being
        expensive. Prevents an event from matching a neighbouring landmark.
    """
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    if pred.size > 1 and np.any(np.diff(pred) < 0):
        raise ValueError("vp_distance: `pred` must be sorted ascending")
    if gt.size > 1 and np.any(np.diff(gt) < 0):
        raise ValueError("vp_distance: `gt` must be sorted ascending")

    m, n = len(pred), len(gt)
    D = np.zeros((m + 1, n + 1))
    D[0, :] = np.arange(n + 1)
    D[:, 0] = np.arange(m + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dt = abs(pred[i - 1] - gt[j - 1])
            if hard_window is not None and dt > hard_window:
                cost_shift = np.inf
            else:
                cost_shift = q * dt
            D[i, j] = min(D[i - 1, j - 1] + cost_shift,
                          D[i - 1, j] + 1.0,
                          D[i, j - 1] + 1.0)
    return float(D[m, n])

def vp_score(pred, gt, tol=3.0, hard_window=False):
    """
    Normalized VP alignment score in [0, 1]. 1.0 == identical trains.

    tol: misalignment tolerance in timesteps (q = 2/tol).
    hard_window: forbid matches further apart than `tol` (anti-aliasing).
    """
    m, n = len(pred), len(gt)
    if m == 0 and n == 0:
        return 1.0
    q = q_from_tol(tol)
    dist = vp_distance(pred, gt, q, hard_window=tol if hard_window else None)
    return 1.0 - dist / float(m + n)

