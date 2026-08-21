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

