from __future__ import annotations

from collections.abc import Callable

import numpy as np


def evaluate_peptide_robustness(
    peptides: list[str],
    wt_target: str,
    escape_variants: list[str],
    aff_fn: Callable[[str, str], float],
    tau_bind: float = 0.5,
) -> dict:
    """
    Evaluate each peptide on WT and held-out escape variants.
    """
    rows = []
    for pep in peptides:
        wt_score = float(aff_fn(pep, wt_target))
        esc_scores = np.array([float(aff_fn(pep, v)) for v in escape_variants], dtype=np.float64)
        mean_escape = float(np.mean(esc_scores)) if esc_scores.size else float("nan")
        min_escape = float(np.min(esc_scores)) if esc_scores.size else float("nan")
        retention = float(np.mean(esc_scores >= tau_bind)) if esc_scores.size else float("nan")
        rows.append(
            {
                "peptide": pep,
                "wt_score": wt_score,
                "mean_escape": mean_escape,
                "min_escape": min_escape,
                "retention": retention,
            }
        )

    if not rows:
        return {"per_peptide": [], "aggregate": {}}

    return {
        "per_peptide": rows,
        "aggregate": {
            "mean_wt_score": float(np.mean([r["wt_score"] for r in rows])),
            "mean_mean_escape": float(np.mean([r["mean_escape"] for r in rows])),
            "mean_min_escape": float(np.mean([r["min_escape"] for r in rows])),
            "mean_retention": float(np.mean([r["retention"] for r in rows])),
        },
    }

