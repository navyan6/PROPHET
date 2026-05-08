from __future__ import annotations

from typing import Iterable

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA)}
GAP = 20


def encode_sequence(seq: str) -> np.ndarray:
    return np.array([AA_TO_IDX.get(a, GAP) for a in seq], dtype=np.int16)


def hamming_distance(a: str, b: str) -> int:
    if len(a) != len(b):
        n = min(len(a), len(b))
        return sum(x != y for x, y in zip(a[:n], b[:n])) + abs(len(a) - len(b))
    return sum(x != y for x, y in zip(a, b))


def nearest_leaf_edit_distance(variant: str, held_out_leaves: Iterable[str]) -> int:
    leaves = list(held_out_leaves)
    if not leaves:
        raise ValueError("held_out_leaves is empty.")
    return min(hamming_distance(variant, leaf) for leaf in leaves)


def dca_energy(variant: str, lambda_i: np.ndarray, h: np.ndarray, J: np.ndarray) -> float:
    x = encode_sequence(variant)
    L = min(len(x), h.shape[0], lambda_i.shape[0], J.shape[0], J.shape[1])
    if L <= 0:
        return float("nan")
    if np.any(x[:L] >= 20):
        raise ValueError("Variant contains unsupported amino acids.")
    unary = 0.0
    pairwise = 0.0
    for i in range(L):
        a = int(x[i])
        unary += float(lambda_i[i] * h[i, a])
    for i in range(L):
        ai = int(x[i])
        for j in range(i + 1, L):
            aj = int(x[j])
            pairwise += float(J[i, j, ai, aj])
    return -(unary + pairwise)

