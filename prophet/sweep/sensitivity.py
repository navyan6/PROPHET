#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path


def run_sensitivity_sweeps(
    base_config: dict,
    all_trees: list[str],
    runner_fn,
    t_values: list[float] | None = None,
    m_values: list[int] | None = None,
    j_values: list[int] | None = None,
    seed: int = 42,
) -> dict:
    if t_values is None:
        t_values = [0.5, 1.0, 2.0, 5.0]
    if m_values is None:
        m_values = [50, 100, 250, 500, 1000]
    if j_values is None:
        j_values = [25, 50, 100, 200]

    rng = random.Random(seed)
    out: dict[str, dict] = {"t_evo": {}, "M": {}, "J": {}}

    for t in t_values:
        cfg = copy.deepcopy(base_config)
        cfg["t_evo"] = float(t)
        out["t_evo"][str(t)] = runner_fn(cfg)

    for m in m_values:
        cfg = copy.deepcopy(base_config)
        cfg["variant_subset_M"] = int(m)
        out["M"][str(m)] = runner_fn(cfg)

    for j in j_values:
        cfg = copy.deepcopy(base_config)
        j_eff = min(int(j), len(all_trees))
        cfg["tree_subset"] = rng.sample(all_trees, j_eff)
        out["J"][str(j)] = runner_fn(cfg)

    return out


def _dummy_runner(cfg: dict) -> dict:
    # Replace with actual pipeline invocation in your environment.
    return {"ok": True, "config": cfg}


def main() -> None:
    p = argparse.ArgumentParser(description="Run PROPHET sensitivity sweeps")
    p.add_argument("--config", required=True, help="Base JSON config")
    p.add_argument("--trees-file", required=True, help="One tree path per line")
    p.add_argument("--out-json", default="data/prophet/sensitivity.json")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(args.trees_file, "r", encoding="utf-8") as f:
        trees = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

    results = run_sensitivity_sweeps(cfg, trees, _dummy_runner)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved sensitivity sweeps -> {out_path}")


if __name__ == "__main__":
    main()

