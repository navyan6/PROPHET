from __future__ import annotations

import numpy as np


def extract_pareto_front(scores: np.ndarray) -> np.ndarray:
    """
    scores: (N,2) with objectives to maximize [wt_binding, mean_escape].
    Returns non-dominated points.
    """
    if scores.ndim != 2 or scores.shape[1] != 2:
        raise ValueError("scores must have shape (N,2)")
    keep = np.ones(scores.shape[0], dtype=bool)
    for i in range(scores.shape[0]):
        if not keep[i]:
            continue
        dominates_i = np.all(scores >= scores[i], axis=1) & np.any(scores > scores[i], axis=1)
        if np.any(dominates_i):
            keep[i] = False
    front = scores[keep]
    order = np.argsort(front[:, 0])
    return front[order]


def hypervolume_indicator(pareto_points: np.ndarray, reference_point: np.ndarray) -> float:
    """
    2D maximization hypervolume (area) relative to reference_point.
    """
    if pareto_points.size == 0:
        return 0.0
    pts = pareto_points[np.argsort(pareto_points[:, 0])]
    hv = 0.0
    prev_x = float(reference_point[0])
    for x, y in pts:
        xx = float(max(x, reference_point[0]))
        yy = float(max(y, reference_point[1]))
        width = max(0.0, xx - prev_x)
        height = max(0.0, yy - float(reference_point[1]))
        hv += width * height
        prev_x = xx
    return float(hv)

