from __future__ import annotations

from collections.abc import Callable

import numpy as np

try:
    from prophet.common import nearest_leaf_edit_distance
except ImportError:  # pragma: no cover
    from common import nearest_leaf_edit_distance  # type: ignore


def _distribution_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.median(a) - np.median(b)))


def calibrate_t_evo(
    sampler_fn: Callable[[float, int], list[str]],
    held_out_leaves: list[str],
    t_candidates: list[float] | None = None,
    n_samples: int = 200,
) -> tuple[float, dict[float, float]]:
    """
    Calibrate t_evo by matching sampled nearest-leaf edit-distance distribution.
    sampler_fn signature: sampler_fn(t_evo, n_samples) -> variants
    """
    if t_candidates is None:
        t_candidates = [0.5, 1.0, 2.0, 5.0]
    target = np.array([nearest_leaf_edit_distance(v, held_out_leaves) for v in held_out_leaves], dtype=np.int32)
    metrics: dict[float, float] = {}
    for t in t_candidates:
        variants = sampler_fn(float(t), int(n_samples))
        d = np.array([nearest_leaf_edit_distance(v, held_out_leaves) for v in variants], dtype=np.int32)
        metrics[float(t)] = _distribution_distance(d, target)
    best_t = min(metrics, key=metrics.get)
    return float(best_t), metrics

