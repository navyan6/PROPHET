#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
import sys

import numpy as np
from Bio import SeqIO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prophet.common import dca_energy, hamming_distance, nearest_leaf_edit_distance


WT_SEQ = "PQVTLWQKPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"


def _load_fasta(path: Path) -> list[str]:
    return [
        str(rec.seq).strip().upper().replace("-", "")
        for rec in SeqIO.parse(str(path), "fasta")
        if str(rec.seq).strip()
    ]


def _summary(values: list[float | int]) -> dict[str, float | int]:
    arr = [float(v) for v in values]
    if not arr:
        return {"mean": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": mean(arr),
        "median": median(arr),
        "min": min(arr),
        "max": max(arr),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Build HIV Stage 1 table-ready metrics")
    p.add_argument("--stage1-dir", default="results/all_trees_stage1_train_only")
    p.add_argument("--prefix", default="hiv_train")
    p.add_argument("--test-fasta", default="data/pre_stage1_split/alignments/test/hiv_test_aligned.fasta")
    p.add_argument("--out-dir", default="results/hiv_stage2/tables")
    p.add_argument("--wt-seq", default=WT_SEQ)
    args = p.parse_args()

    stage1_dir = Path(args.stage1_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = _load_fasta(stage1_dir / f"{args.prefix}_gibbs_variants.fasta")
    heldout = _load_fasta(Path(args.test_fasta))
    wt_seq = args.wt_seq.strip().upper().replace("-", "")

    lambda_i = np.load(stage1_dir / f"{args.prefix}_lambda.npy")
    h = np.load(stage1_dir / f"{args.prefix}_h.npy")
    J = np.load(stage1_dir / f"{args.prefix}_J.npz")["J"]

    wt_dist = [hamming_distance(v, wt_seq) for v in variants]
    heldout_dist = [nearest_leaf_edit_distance(v, heldout) for v in variants]
    energies = [dca_energy(v, lambda_i, h, J) for v in variants]
    lambda_nonzero = int(np.sum(lambda_i > 0))

    summary = {
        "n_variants": len(variants),
        "n_heldout": len(heldout),
        "sequence_length": len(wt_seq),
        "lambda_nonzero_sites": lambda_nonzero,
        "lambda_nonzero_fraction": float(lambda_nonzero / len(lambda_i)) if len(lambda_i) else float("nan"),
        "lambda_mean": float(np.mean(lambda_i)) if lambda_i.size else float("nan"),
        "lambda_max": float(np.max(lambda_i)) if lambda_i.size else float("nan"),
    }
    for prefix, vals in [
        ("wt_edit_distance", wt_dist),
        ("nearest_heldout_edit_distance", heldout_dist),
        ("dca_energy", energies),
    ]:
        for key, val in _summary(vals).items():
            summary[f"{prefix}_{key}"] = val

    json_path = out_dir / "hiv_stage1_variant_quality.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = out_dir / "hiv_stage1_variant_quality.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
