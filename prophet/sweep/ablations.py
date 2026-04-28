#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np


def run_ablations(
    base_config: dict,
    runner_fn,
    etas: list[float] | None = None,
) -> dict:
    """
    runner_fn(config_dict) -> dict(metrics/results)
    """
    if etas is None:
        etas = [0.1, 0.5, 1.0]

    configs: dict[str, dict] = {
        "full": base_config,
        "minus_dca": {**copy.deepcopy(base_config), "ablation": {"zero_J": True}},
        "minus_lambda": {**copy.deepcopy(base_config), "ablation": {"unit_lambda": True}},
        "minus_esm_filter": {**copy.deepcopy(base_config), "esm_filter_delta": None},
        "leaves_only": {**copy.deepcopy(base_config), "ablation": {"use_leaf_variants": True}},
    }
    for eta in etas:
        name = f"cvar_eta_{eta}"
        cfg = copy.deepcopy(base_config)
        cfg["eta"] = float(eta)
        configs[name] = cfg

    out = {}
    for name, cfg in configs.items():
        out[name] = runner_fn(cfg)
    return out


def _dummy_runner(cfg: dict) -> dict:
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    return {"config": cfg, "score": float(rng.normal())}


def main() -> None:
    p = argparse.ArgumentParser(description="Run PROPHET ablation matrix from JSON config")
    p.add_argument("--config", required=True)
    p.add_argument("--out-json", default="data/prophet/ablations.json")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        base = json.load(f)

    results = run_ablations(base, _dummy_runner)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved ablations -> {out_path}")


if __name__ == "__main__":
    main()

