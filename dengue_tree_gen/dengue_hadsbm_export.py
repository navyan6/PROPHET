#!/usr/bin/env python3
"""
Export DENV3 phylogenetic tree to hadsbm_tree_v1 JSON format.

Reads:
  - DENV3_tree.nwk  (339-leaf FastTree Newick)
  - cluster2and6.obs.csv  (sequences keyed by GenBank Accession)

Writes:
  - ../../data/trees/DENV3_hadsbm_tree.json

Usage:
  python dengue_hadsbm_export.py [--out PATH] [--prob-mode {length,uniform}]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add tree_analysis/src to path so we can reuse build_hadsbm_bundle
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "tree_analysis" / "src"))
sys.path.insert(0, str(REPO_ROOT / "tree_analysis" / "pipelines"))

from hadsbm_export import build_hadsbm_bundle, read_fasta_as_dict
from compute_probabilities import load_sequences, accession_from_leaf

try:
    from Bio import Phylo
except ImportError:
    print("error: pip install biopython", file=sys.stderr)
    raise SystemExit(1)


NWK_FILE  = SCRIPT_DIR / "DENV3_tree.nwk"
CSV_FILE  = SCRIPT_DIR / "cluster2and6.obs.csv"
OUT_DEFAULT = REPO_ROOT / "data" / "trees" / "DENV3_hadsbm_tree.json"


def build_fasta_dict_from_tree(nwk_path: Path, csv_path: Path) -> tuple[dict[str, str], str]:
    """
    Return (fasta_by_id, wildtype_sequence).

    fasta_by_id: leaf_label → amino-acid sequence
    wildtype_sequence: first sequence from CSV (reference anchor for HadSBM)
    """
    tree = Phylo.read(str(nwk_path), "newick")
    leaf_names = [c.name for c in tree.get_terminals() if c.name]

    # GenBank Accession → sequence
    accession_to_seq = load_sequences(csv_path)

    fasta_by_id: dict[str, str] = {}
    missing: list[str] = []

    for leaf in leaf_names:
        acc = accession_from_leaf(leaf)
        seq = accession_to_seq.get(acc, "")
        if seq:
            fasta_by_id[leaf] = seq
        else:
            missing.append(leaf)

    if missing:
        print(f"[WARNING] {len(missing)} leaves have no sequence in CSV — they will be skipped.", file=sys.stderr)
        print(f"  First 5: {missing[:5]}", file=sys.stderr)

    # Use the first CSV sequence as the wildtype anchor
    wildtype = next(iter(accession_to_seq.values()))
    print(f"[INFO] Wildtype anchor: first CSV sequence (len={len(wildtype)})")

    return fasta_by_id, wildtype


def main() -> int:
    parser = argparse.ArgumentParser(description="Export DENV3 HadSBM tree bundle")
    parser.add_argument("--nwk",        type=Path, default=NWK_FILE)
    parser.add_argument("--csv",        type=Path, default=CSV_FILE)
    parser.add_argument("--out",        type=Path, default=OUT_DEFAULT)
    parser.add_argument("--prob-mode",  choices=("length", "uniform"), default="length")
    args = parser.parse_args()

    if not args.nwk.is_file():
        print(f"error: Newick not found: {args.nwk}", file=sys.stderr)
        return 1
    if not args.csv.is_file():
        print(f"error: CSV not found: {args.csv}", file=sys.stderr)
        return 1

    print(f"Reading tree  : {args.nwk}")
    print(f"Reading CSV   : {args.csv}")

    fasta_by_id, wildtype = build_fasta_dict_from_tree(args.nwk, args.csv)
    print(f"[INFO] {len(fasta_by_id)} leaves matched to sequences")

    tree = Phylo.read(str(args.nwk), "newick")

    bundle = build_hadsbm_bundle(
        tree,
        fasta_by_id,
        wildtype,
        prob_mode=args.prob_mode,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=2)

    print(f"\n[OK] Wrote: {args.out}")
    print(f"     n_nodes  : {bundle['n_nodes']}")
    print(f"     n_leaves : {bundle['n_leaves']}")
    print(f"     splits   : {len(bundle['splits'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
