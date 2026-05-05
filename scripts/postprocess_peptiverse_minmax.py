#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCORE_KEYS = {
    "wt_score",
    "robust_score",
    "mean_score",
    "min_score",
}


def normalize_score(value: float, score_min: float, score_max: float) -> float:
    scaled = (float(value) - score_min) / (score_max - score_min)
    return float(min(1.0, max(0.0, scaled)))


def normalize_row(row: dict[str, Any], score_min: float, score_max: float) -> dict[str, Any]:
    out = dict(row)
    for key in SCORE_KEYS:
        if key in row and row[key] is not None:
            out[f"raw_{key}"] = row[key]
            out[key] = normalize_score(float(row[key]), score_min, score_max)

    per_variant = row.get("per_variant")
    if isinstance(per_variant, list):
        out["raw_per_variant"] = per_variant
        out["per_variant"] = [
            normalize_score(float(score), score_min, score_max)
            for score in per_variant
        ]

    out["peptiverse_normalization"] = "minmax"
    out["peptiverse_raw_min"] = score_min
    out["peptiverse_raw_max"] = score_max
    return out


def default_output_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Postprocess raw PeptiVerse Stage 2 JSON scores to minmax-normalized copies."
    )
    parser.add_argument("jsons", nargs="+", type=Path)
    parser.add_argument("--peptiverse-min", type=float, default=7.0)
    parser.add_argument("--peptiverse-max", type=float, default=9.0)
    parser.add_argument("--suffix", default="_minmax")
    args = parser.parse_args()

    if args.peptiverse_max <= args.peptiverse_min:
        raise ValueError("--peptiverse-max must be greater than --peptiverse-min")

    for path in args.jsons:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"{path} does not contain a list of design rows")

        normalized = [
            normalize_row(row, args.peptiverse_min, args.peptiverse_max)
            for row in data
            if isinstance(row, dict)
        ]
        out_path = default_output_path(path, args.suffix)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(normalized, handle, indent=2)
            handle.write("\n")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
