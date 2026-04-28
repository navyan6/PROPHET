from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    pts = pareto_points[np.argsort(pareto_points[:, 0])[::-1]]
    hv = 0.0
    prev_y = float(reference_point[1])
    for x, y in pts:
        xx = float(max(x, reference_point[0]))
        yy = float(max(y, reference_point[1]))
        width = max(0.0, xx - float(reference_point[0]))
        height = max(0.0, yy - prev_y)
        hv += width * height
        prev_y = max(prev_y, yy)
    return float(hv)


def _load_stage2_scores(path: Path) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    pairs = []
    for r in rows:
        wt = r.get("wt_score")
        robust = r.get("robust_score")
        if wt is None or robust is None:
            continue
        pairs.append([float(wt), float(robust)])
    if not pairs:
        raise ValueError(f"No (wt_score, robust_score) pairs found in {path}")
    return np.array(pairs, dtype=np.float64)


def main() -> None:
    p = argparse.ArgumentParser(description="Pareto-front and hypervolume metrics for Stage 2 designs")
    p.add_argument("--designs-json", required=True, help="Stage 2 designs JSON")
    p.add_argument("--out-json", default=None, help="Optional metrics output path")
    p.add_argument("--ref-wt", type=float, default=None, help="Reference point WT coordinate")
    p.add_argument("--ref-robust", type=float, default=None, help="Reference point robust coordinate")
    args = p.parse_args()

    scores = _load_stage2_scores(Path(args.designs_json))
    front = extract_pareto_front(scores)
    ref = np.array(
        [
            float(args.ref_wt) if args.ref_wt is not None else float(np.min(scores[:, 0])),
            float(args.ref_robust) if args.ref_robust is not None else float(np.min(scores[:, 1])),
        ],
        dtype=np.float64,
    )
    hv = hypervolume_indicator(front, ref)
    out = {
        "n_designs": int(scores.shape[0]),
        "n_pareto": int(front.shape[0]),
        "reference_point": [float(ref[0]), float(ref[1])],
        "hypervolume": float(hv),
        "pareto_points": [[float(x), float(y)] for x, y in front],
    }
    print(json.dumps(out, indent=2))
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Saved Pareto metrics -> {out_path}")


if __name__ == "__main__":
    main()

