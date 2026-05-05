#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _cvar(scores: np.ndarray, eta: float) -> float:
    """CVaR over the lowest-eta fraction — must match stage2.cvar_robust_score exactly."""
    if scores.size == 0:
        return float("nan")
    eta = float(np.clip(eta, 1e-6, 1.0))
    # Use floor, not ceil, to match prophet/stage2.py::cvar_robust_score
    k = max(1, int(np.floor(eta * scores.size)))
    return float(np.sort(scores)[:k].mean())


def main() -> None:
    p = argparse.ArgumentParser(description="Eta sensitivity over Stage 2 per-variant scores")
    p.add_argument("--designs-json", required=True)
    p.add_argument("--etas", default="0.05,0.1,0.2,0.5,1.0")
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    with open(args.designs_json, "r", encoding="utf-8") as f:
        rows = json.load(f)

    etas = [float(x.strip()) for x in str(args.etas).split(",") if x.strip()]
    if not etas:
        raise ValueError("No valid eta values provided.")

    by_eta: dict[str, dict[str, float]] = {}
    for eta in etas:
        vals = []
        for r in rows:
            per_v = np.array(r.get("per_variant", []), dtype=np.float64)
            if per_v.size == 0:
                continue
            vals.append(_cvar(per_v, eta))
        arr = np.array(vals, dtype=np.float64)
        by_eta[str(eta)] = {
            "n_designs": int(arr.size),
            "mean_cvar": float(np.mean(arr)) if arr.size else float("nan"),
            "median_cvar": float(np.median(arr)) if arr.size else float("nan"),
            "min_cvar": float(np.min(arr)) if arr.size else float("nan"),
            "max_cvar": float(np.max(arr)) if arr.size else float("nan"),
        }

    out = {
        "designs_json": str(Path(args.designs_json)),
        "eta_metrics": by_eta,
    }
    print(json.dumps(out, indent=2))
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Saved eta sensitivity -> {out_path}")


if __name__ == "__main__":
    main()
