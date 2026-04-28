from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    from prophet.common import dca_energy as _dca_energy
    from prophet.common import hamming_distance, nearest_leaf_edit_distance
except ImportError:  # pragma: no cover
    from common import dca_energy as _dca_energy  # type: ignore
    from common import hamming_distance, nearest_leaf_edit_distance  # type: ignore


def resistance_site_enrichment(variants: list[str], wt_seq: str, resistance_positions: set[int]) -> float:
    """
    Fraction of all mutation events that occur at resistance positions.
    Assumes zero-based resistance positions.
    """
    mut_total = 0
    mut_on_res = 0
    for seq in variants:
        L = min(len(seq), len(wt_seq))
        for i in range(L):
            if seq[i] != wt_seq[i]:
                mut_total += 1
                if i in resistance_positions:
                    mut_on_res += 1
    if mut_total == 0:
        return 0.0
    return float(mut_on_res / mut_total)


def edit_distance_distribution(variants: list[str], held_out_leaves: Iterable[str]) -> np.ndarray:
    return np.array([nearest_leaf_edit_distance(v, held_out_leaves) for v in variants], dtype=np.int32)


def wt_edit_distance_distribution(variants: list[str], wt_seq: str) -> np.ndarray:
    return np.array([hamming_distance(v, wt_seq) for v in variants], dtype=np.int32)


def dca_energy(variant: str, lambda_i: np.ndarray, h: np.ndarray, J: np.ndarray) -> float:
    return _dca_energy(variant, lambda_i, h, J)

