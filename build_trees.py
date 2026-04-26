#!/usr/bin/env python3
"""
Align protein FASTA files with MAFFT and build phylogenetic trees with FastTree.

For each input FASTA file:
  1. Run MAFFT (auto mode) to produce a multiple sequence alignment
  2. Run FastTree (protein mode, LG model) to infer a Newick tree
  3. Save aligned FASTA and .nwk tree to --out-dir (default: data/trees/)

Usage:
  python build_trees.py sequences/*.fasta
  python build_trees.py hiv.fasta spike.fasta --out-dir data/trees
  python build_trees.py seqs/*.fasta --n-bootstraps 200
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], desc: str) -> str:
    """Run a shell command, return stdout. Exit on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR in {desc}:\n{result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return result.stdout


def align(fasta_path: Path, out_dir: Path) -> Path:
    """Run MAFFT on fasta_path, write aligned FASTA, return its path."""
    aligned_path = out_dir / f"{fasta_path.stem}_aligned.fasta"
    print(f"  Aligning {fasta_path.name} → {aligned_path.name} ...")
    stdout = run(
        ["mafft", "--auto", "--quiet", "--thread", "-1", str(fasta_path)],
        desc=f"mafft {fasta_path.name}",
    )
    aligned_path.write_text(stdout)
    return aligned_path


def build_tree(aligned_path: Path, out_dir: Path) -> Path:
    """Run FastTree on aligned protein FASTA, write .nwk, return its path."""
    tree_path = out_dir / f"{aligned_path.stem.replace('_aligned', '')}_tree.nwk"
    print(f"  Building tree → {tree_path.name} ...")
    stdout = run(
        ["FastTree", "-quiet", "-lg", str(aligned_path)],
        desc=f"FastTree {aligned_path.name}",
    )
    tree_path.write_text(stdout)
    return tree_path


def build_bootstrap_trees(
    aligned_path: Path, out_dir: Path, n_bootstraps: int
) -> list[Path]:
    """
    Build n_bootstraps trees from bootstrap-resampled columns of the alignment.
    Used by PROPHET to average λᵢ and Qᵢ across a tree ensemble.

    Each bootstrap tree is saved as:  stem_boot{k}_tree.nwk
    """
    from Bio import SeqIO
    import random

    print(f"  Building {n_bootstraps} bootstrap trees ...")

    records = list(SeqIO.parse(str(aligned_path), "fasta"))
    L = len(records[0].seq)
    N = len(records)

    boot_dir = out_dir / f"{aligned_path.stem.replace('_aligned', '')}_bootstrap"
    boot_dir.mkdir(exist_ok=True)

    tree_paths = []
    for k in range(n_bootstraps):
        # Resample alignment columns with replacement
        cols = sorted(random.choices(range(L), k=L))
        boot_fasta = boot_dir / f"boot{k}.fasta"
        with open(boot_fasta, "w") as f:
            for rec in records:
                resampled_seq = "".join(str(rec.seq)[c] for c in cols)
                f.write(f">{rec.id}\n{resampled_seq}\n")

        tree_path = boot_dir / f"boot{k}_tree.nwk"
        stdout = run(
            ["FastTree", "-quiet", "-lg", str(boot_fasta)],
            desc=f"FastTree bootstrap {k}",
        )
        tree_path.write_text(stdout)
        tree_paths.append(tree_path)

        if (k + 1) % 10 == 0:
            print(f"    {k+1}/{n_bootstraps} done")

    return tree_paths


def main():
    p = argparse.ArgumentParser(
        description="Align FASTA files and build FastTree phylogenies"
    )
    p.add_argument(
        "fastas",
        nargs="+",
        type=Path,
        help="One or more protein FASTA files",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/trees"),
        help="Output directory for aligned FASTAs and .nwk trees (default: data/trees)",
    )
    p.add_argument(
        "--n-bootstraps",
        type=int,
        default=0,
        help="Number of bootstrap trees to build per input file (default: 0 = single tree only)",
    )
    p.add_argument(
        "--skip-align",
        action="store_true",
        help="Treat input files as already-aligned FASTAs, skip MAFFT",
    )
    args, _ = p.parse_known_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for fasta_path in args.fastas:
        if not fasta_path.exists():
            print(f"WARNING: {fasta_path} not found, skipping", file=sys.stderr)
            continue

        print(f"\n[{fasta_path.name}]")

        if args.skip_align:
            aligned_path = fasta_path
        else:
            aligned_path = align(fasta_path, args.out_dir)

        build_tree(aligned_path, args.out_dir)

        if args.n_bootstraps > 0:
            build_bootstrap_trees(aligned_path, args.out_dir, args.n_bootstraps)

    print(f"\nDone. Trees written to: {args.out_dir}/")


if __name__ == "__main__":
    main()
