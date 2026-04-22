#!/usr/bin/env python3
"""
Run multi-tree holdout experiments for peptide robustness.

For each tree JSON:
1) split variants into train/holdout
2) generate candidate peptides with MOG-DFM solver
3) rank/select peptides by:
   - tree_train_score (tree-aware proxy)
   - wt_score (WT-only baseline)
4) compare holdout performance between selections
5) write per-tree candidates CSV + global summary CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PEPTIVERSE_PATH = REPO_ROOT / "PeptiVerse"
MOGDFM_PATH = REPO_ROOT / "MOG-DFM"
sys.path.insert(0, str(PEPTIVERSE_PATH))
sys.path.insert(0, str(MOGDFM_PATH))

from inference import PeptiVersePredictor  # noqa: E402
from models.peptide_classifiers import load_solver  # noqa: E402


@dataclass
class Variant:
    name: str
    sequence: str
    probability: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-tree holdout evaluation")
    parser.add_argument("--tree-jsons", nargs="+", required=True, help="List of HadSBM tree JSON paths")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional labels for each tree JSON")
    parser.add_argument("--num-candidates", type=int, default=500, help="Candidates sampled per tree")
    parser.add_argument("--select-top-k", type=int, default=100, help="Top-K selected per method")
    parser.add_argument("--holdout-fraction", type=float, default=0.2, help="Fraction of variants held out")
    parser.add_argument("--split-seed", type=int, default=1986, help="Random seed for holdout split")
    parser.add_argument("--length", type=int, default=12, help="Peptide length")
    parser.add_argument("--retention-threshold", type=float, default=5.0, help="Binding threshold for retention")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    parser.add_argument("--out-dir", type=Path, default=Path("data/results/multi_tree_eval"))
    return parser.parse_args()


def load_variants(tree_json: Path) -> tuple[str, list[Variant]]:
    with open(tree_json, encoding="utf-8") as f:
        data = json.load(f)
    wt = data["x_WT"]
    leaves = data["leaf_endpoints_pi"]
    n = len(leaves)
    p = 1.0 / n if n else 0.0
    variants = [
        Variant(
            name=v.get("leaf_id", f"leaf_{i}"),
            sequence=v.get("sequence", ""),
            probability=p,
        )
        for i, v in enumerate(leaves)
        if v.get("sequence", "")
    ]
    return wt, variants


def predict_affinity(predictor: PeptiVersePredictor, target_seq: str, peptide: str) -> float:
    result = predictor.predict_binding_affinity(col="wt", target_seq=target_seq, binder_str=peptide)
    if isinstance(result, dict):
        return float(result.get("affinity", 0.0))
    return float(result)


def generate_peptide(solver, tokenizer, length: int, device: str) -> str:
    x_init = torch.randint(low=4, high=24, size=(1, length), device=device)
    zeros = torch.zeros((1, 1), dtype=x_init.dtype, device=device)
    twos = torch.full((1, 1), 2, dtype=x_init.dtype, device=device)
    x_init = torch.cat([zeros, x_init, twos], dim=1)
    x_t = solver.sample(
        x_init=x_init,
        step_size=1 / 200,
        time_grid=torch.tensor([0.0, 1.0 - 1e-3], device=device),
    )
    peptide = tokenizer.decode(x_t[0]).replace(" ", "")[5:-5]
    return peptide


def evaluate_candidate(
    predictor: PeptiVersePredictor,
    peptide: str,
    wt_seq: str,
    train_variants: list[Variant],
    holdout_variants: list[Variant],
    retention_threshold: float,
) -> dict[str, float]:
    wt_score = predict_affinity(predictor, wt_seq, peptide)
    train_scores = [predict_affinity(predictor, v.sequence, peptide) for v in train_variants]
    holdout_scores = [predict_affinity(predictor, v.sequence, peptide) for v in holdout_variants]

    train_mean = float(np.mean(train_scores)) if train_scores else 0.0
    holdout_mean = float(np.mean(holdout_scores)) if holdout_scores else 0.0
    holdout_min = float(np.min(holdout_scores)) if holdout_scores else 0.0
    holdout_retention = (
        float(np.mean(np.array(holdout_scores) >= retention_threshold)) if holdout_scores else 0.0
    )
    return {
        "wt_score": wt_score,
        "tree_train_score": train_mean,
        "holdout_mean": holdout_mean,
        "holdout_min": holdout_min,
        "holdout_retention": holdout_retention,
    }


def summarize_selection(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"holdout_mean": 0.0, "holdout_min": 0.0, "holdout_retention": 0.0}
    return {
        "holdout_mean": float(np.mean([r["holdout_mean"] for r in rows])),
        "holdout_min": float(np.mean([r["holdout_min"] for r in rows])),
        "holdout_retention": float(np.mean([r["holdout_retention"] for r in rows])),
    }


def run_one_tree(
    predictor: PeptiVersePredictor,
    solver,
    tokenizer,
    label: str,
    tree_json: Path,
    num_candidates: int,
    select_top_k: int,
    holdout_fraction: float,
    split_seed: int,
    length: int,
    retention_threshold: float,
    device: str,
    out_dir: Path,
) -> dict[str, float]:
    wt_seq, variants = load_variants(tree_json)
    if len(variants) < 2:
        raise ValueError(f"Need at least 2 variants for holdout: {tree_json}")

    rng = random.Random(split_seed)
    shuffled = variants[:]
    rng.shuffle(shuffled)
    n_holdout = max(1, int(len(shuffled) * holdout_fraction))
    holdout_variants = shuffled[:n_holdout]
    train_variants = shuffled[n_holdout:]

    candidates = []
    for i in range(num_candidates):
        peptide = generate_peptide(solver, tokenizer, length, device)
        metrics = evaluate_candidate(
            predictor=predictor,
            peptide=peptide,
            wt_seq=wt_seq,
            train_variants=train_variants,
            holdout_variants=holdout_variants,
            retention_threshold=retention_threshold,
        )
        candidates.append({"idx": i + 1, "peptide": peptide, **metrics})

    by_tree = sorted(candidates, key=lambda x: x["tree_train_score"], reverse=True)[:select_top_k]
    by_wt = sorted(candidates, key=lambda x: x["wt_score"], reverse=True)[:select_top_k]
    by_rand = random.Random(split_seed + 7).sample(candidates, k=min(select_top_k, len(candidates)))

    tree_summary = summarize_selection(by_tree)
    wt_summary = summarize_selection(by_wt)
    rand_summary = summarize_selection(by_rand)

    out_dir.mkdir(parents=True, exist_ok=True)
    per_tree_csv = out_dir / f"{label}_candidates.csv"
    with open(per_tree_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "idx",
                "peptide",
                "wt_score",
                "tree_train_score",
                "holdout_mean",
                "holdout_min",
                "holdout_retention",
            ]
        )
        for row in candidates:
            writer.writerow(
                [
                    row["idx"],
                    row["peptide"],
                    f"{row['wt_score']:.6f}",
                    f"{row['tree_train_score']:.6f}",
                    f"{row['holdout_mean']:.6f}",
                    f"{row['holdout_min']:.6f}",
                    f"{row['holdout_retention']:.6f}",
                ]
            )

    print(
        f"[{label}] train={len(train_variants)} holdout={len(holdout_variants)} "
        f"tree_holdout_mean={tree_summary['holdout_mean']:.4f} "
        f"wt_holdout_mean={wt_summary['holdout_mean']:.4f}"
    )

    return {
        "tree": label,
        "tree_json": str(tree_json),
        "n_variants": len(variants),
        "n_train": len(train_variants),
        "n_holdout": len(holdout_variants),
        "num_candidates": num_candidates,
        "top_k": min(select_top_k, len(candidates)),
        "tree_holdout_mean": tree_summary["holdout_mean"],
        "tree_holdout_min": tree_summary["holdout_min"],
        "tree_holdout_retention": tree_summary["holdout_retention"],
        "wt_holdout_mean": wt_summary["holdout_mean"],
        "wt_holdout_min": wt_summary["holdout_min"],
        "wt_holdout_retention": wt_summary["holdout_retention"],
        "rand_holdout_mean": rand_summary["holdout_mean"],
        "rand_holdout_min": rand_summary["holdout_min"],
        "rand_holdout_retention": rand_summary["holdout_retention"],
    }


def main() -> int:
    args = parse_args()
    np.random.seed(args.split_seed)
    random.seed(args.split_seed)
    torch.manual_seed(args.split_seed)

    tree_jsons = [Path(p) for p in args.tree_jsons]
    labels = args.labels if args.labels else [p.stem for p in tree_jsons]
    if len(labels) != len(tree_jsons):
        raise ValueError("--labels must have same length as --tree-jsons")

    print("Loading PeptiVerse...")
    predictor = PeptiVersePredictor(
        manifest_path=str(PEPTIVERSE_PATH / "best_models.txt"),
        classifier_weight_root=str(PEPTIVERSE_PATH),
        device=args.device,
        only_properties=["binding_affinity"],
    )
    print("Loading MOG-DFM solver...")
    solver = load_solver(
        str(MOGDFM_PATH / "ckpt" / "peptide" / "cnn_epoch200_lr0.0001_embed512_hidden256_loss3.1051.ckpt"),
        vocab_size=24,
        device=args.device,
    )
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    summaries = []
    for label, tree_json in zip(labels, tree_jsons):
        summaries.append(
            run_one_tree(
                predictor=predictor,
                solver=solver,
                tokenizer=tokenizer,
                label=label,
                tree_json=tree_json,
                num_candidates=args.num_candidates,
                select_top_k=args.select_top_k,
                holdout_fraction=args.holdout_fraction,
                split_seed=args.split_seed,
                length=args.length,
                retention_threshold=args.retention_threshold,
                device=args.device,
                out_dir=args.out_dir,
            )
        )

    summary_csv = args.out_dir / "summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print(f"\n[OK] wrote summary: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

