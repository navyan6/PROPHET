#!/usr/bin/env python3
"""
Export arbitrary Newick + FASTA to HadSBM tree JSON format.

This is useful for non-HIV trees (e.g., COVID, influenza) that already have:
  - a Newick tree file
  - a FASTA file with sequence IDs matching tree leaf labels
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from Bio import Phylo

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tree_analysis" / "pipelines"))

from hadsbm_export import build_hadsbm_bundle, read_fasta_as_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export generic HadSBM tree JSON")
    parser.add_argument("--nwk", type=Path, required=True, help="Input Newick tree path")
    parser.add_argument("--fasta", type=Path, required=True, help="Input FASTA with leaf sequences")
    parser.add_argument("--out", type=Path, required=True, help="Output HadSBM JSON path")
    parser.add_argument(
        "--prob-mode",
        choices=("length", "uniform"),
        default="length",
        help="How to set split probabilities at each binary node",
    )
    parser.add_argument(
        "--wt-id",
        type=str,
        default=None,
        help="Optional FASTA ID to use as wildtype anchor (x_WT). Defaults to first FASTA record.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.nwk.is_file():
        raise FileNotFoundError(f"Newick not found: {args.nwk}")
    if not args.fasta.is_file():
        raise FileNotFoundError(f"FASTA not found: {args.fasta}")

    tree = Phylo.read(str(args.nwk), "newick")
    fasta_by_id = read_fasta_as_dict(args.fasta)
    if not fasta_by_id:
        raise ValueError(f"No sequences found in FASTA: {args.fasta}")

    if args.wt_id:
        if args.wt_id not in fasta_by_id:
            raise KeyError(f"--wt-id {args.wt_id!r} not present in FASTA")
        wildtype = fasta_by_id[args.wt_id]
    else:
        # Fallback: first sequence in FASTA dictionary order
        wildtype = next(iter(fasta_by_id.values()))

    bundle = build_hadsbm_bundle(
        tree=tree,
        fasta_by_id=fasta_by_id,
        wildtype_sequence=wildtype,
        prob_mode=args.prob_mode,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=2)

    print(f"[OK] Wrote {args.out}")
    print(f"     n_nodes={bundle['n_nodes']} n_leaves={bundle['n_leaves']} splits={len(bundle['splits'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

