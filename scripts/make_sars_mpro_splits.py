#!/usr/bin/env python3
"""
Build PROPHET-ready train/test splits for betacoronavirus Mpro (nsp5 / 3CLpro).

Uses phylogenetic clade holdout (~20% of sequences as test set).
The test clade will typically contain a related cluster (e.g., bat coronaviruses
or MERS-related sequences), testing generalization across the tree.

Usage (after download_sars_mpro.py):
    python scripts/make_sars_mpro_splits.py --in-dir data/sars_mpro

Outputs:
    data/sars_mpro/alignments/train/sars_mpro_train_aligned.fasta
    data/sars_mpro/alignments/test/sars_mpro_test_clade_holdout.fasta
    data/sars_mpro/trees/train/sars_mpro_train_tree.nwk
    data/sars_mpro/trees/train/sars_mpro_bootstrap_trees.txt
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path

from Bio import Phylo, SeqIO
from Bio.SeqRecord import SeqRecord


def run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"ERROR: {cmd[0]}\n{r.stderr.strip()}")
    return r.stdout


def mafft_align(fasta: Path, out: Path) -> Path:
    print(f"    MAFFT aligning {fasta.name} ...")
    out.write_text(run(["mafft", "--auto", "--quiet", "--thread", "-1", str(fasta)]))
    return out


def fasttree(aligned: Path, tree: Path) -> Path:
    print(f"    FastTree {tree.name} ...")
    tree.write_text(run(["FastTree", "-quiet", "-lg", str(aligned)]))
    return tree


def clade_holdout(
    records: list[SeqRecord],
    tree_path: Path,
    target_frac: float = 0.20,
    seed: int = 42,
) -> tuple[list[SeqRecord], list[SeqRecord]]:
    id_set = {r.id for r in records}
    tree = Phylo.read(str(tree_path), "newick")
    tree.root_at_midpoint()

    n = len(records)
    target_n = int(n * target_frac)
    min_c = max(10, target_n // 2)
    max_c = int(target_n * 1.5)

    candidates = []
    for clade in tree.find_clades(order="level"):
        leaves = [c.name for c in clade.get_terminals() if c.name in id_set]
        if min_c <= len(leaves) <= max_c:
            candidates.append((abs(len(leaves) - target_n), leaves))

    if not candidates:
        sys.exit(
            f"ERROR: no clade found in [{min_c}, {max_c}] leaves "
            f"(target {target_n} of {n}). "
            "Increase dataset size or relax target_frac bounds."
        )

    candidates.sort()
    test_ids = set(candidates[0][1])
    print(f"    Clade holdout: {len(test_ids)} test / {n - len(test_ids)} train")

    id_to_rec = {r.id: r for r in records}
    train = [id_to_rec[i] for i in id_set - test_ids if i in id_to_rec]
    test  = [id_to_rec[i] for i in test_ids if i in id_to_rec]
    return train, test


def build_bootstrap_trees(aligned: Path, boot_dir: Path, n: int, seed: int = 42) -> list[Path]:
    random.seed(seed)
    records = list(SeqIO.parse(str(aligned), "fasta"))
    L = len(records[0].seq)
    boot_dir.mkdir(exist_ok=True)
    paths = []
    print(f"    Building {n} bootstrap trees ...")
    for k in range(n):
        cols = sorted(random.choices(range(L), k=L))
        bf = boot_dir / f"boot{k}.fasta"
        with open(bf, "w") as f:
            for rec in records:
                f.write(f">{rec.id}\n{''.join(str(rec.seq)[c] for c in cols)}\n")
        tp = boot_dir / f"boot{k}_tree.nwk"
        tp.write_text(run(["FastTree", "-quiet", "-lg", str(bf)]))
        paths.append(tp)
        if (k + 1) % 20 == 0:
            print(f"      {k + 1}/{n}")
    return paths


def setup_target(
    name: str,
    train_seqs: list[SeqRecord],
    test_seqs: list[SeqRecord],
    out_root: Path,
    n_boot: int = 100,
    seed: int = 42,
) -> None:
    out = out_root / name
    aln_train = out / "alignments" / "train"; aln_train.mkdir(parents=True, exist_ok=True)
    aln_test  = out / "alignments" / "test";  aln_test.mkdir(parents=True, exist_ok=True)
    tr_train  = out / "trees" / "train";      tr_train.mkdir(parents=True, exist_ok=True)
    boot_dir  = tr_train / f"{name}_bootstrap"

    raw = aln_train / f"{name}_train.fasta"
    SeqIO.write(train_seqs, str(raw), "fasta")
    aligned = aln_train / f"{name}_train_aligned.fasta"
    mafft_align(raw, aligned)

    train_tree = tr_train / f"{name}_train_tree.nwk"
    fasttree(aligned, train_tree)

    boot_paths = build_bootstrap_trees(aligned, boot_dir, n=n_boot, seed=seed)
    trees_list = tr_train / f"{name}_bootstrap_trees.txt"
    with open(trees_list, "w") as f:
        for p in boot_paths:
            f.write(str(p) + "\n")

    test_out = aln_test / f"{name}_test_clade_holdout.fasta"
    SeqIO.write(test_seqs, str(test_out), "fasta")

    print(f"  Done: {len(train_seqs)} train, {len(test_seqs)} test")
    print(f"    --tree       {train_tree}")
    print(f"    --trees-file {trees_list}")
    print(f"    --fasta      {aligned}")
    print(f"    holdout      {test_out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir",  default="data/sars_mpro", type=Path)
    ap.add_argument("--out-dir", default="data", type=Path)
    ap.add_argument("--n-boot",  type=int, default=100)
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    in_dir = args.in_dir
    raw_fasta = in_dir / "sars_mpro_raw.fasta"

    print("\n=== Betacoronavirus Mpro (3CLpro / nsp5) — clade holdout ===")
    seqs = list(SeqIO.parse(str(raw_fasta), "fasta"))
    if len(seqs) < 30:
        sys.exit(f"ERROR: only {len(seqs)} sequences — need at least 30.")

    print(f"  {len(seqs)} Mpro sequences")

    tmp_raw  = in_dir / "sars_mpro_tmp.fasta"
    tmp_aln  = in_dir / "sars_mpro_tmp_aligned.fasta"
    tmp_tree = in_dir / "sars_mpro_tmp_tree.nwk"
    SeqIO.write(seqs, str(tmp_raw), "fasta")
    mafft_align(tmp_raw, tmp_aln)
    fasttree(tmp_aln, tmp_tree)

    aligned_recs = list(SeqIO.parse(str(tmp_aln), "fasta"))
    train_aln, test_aln = clade_holdout(aligned_recs, tmp_tree,
                                        target_frac=0.20, seed=args.seed)

    id_map    = {r.id: r for r in seqs}
    train_raw = [id_map[r.id] for r in train_aln if r.id in id_map]
    test_raw  = [id_map[r.id] for r in test_aln  if r.id in id_map]

    for f in [tmp_raw, tmp_aln, tmp_tree]:
        f.unlink(missing_ok=True)

    setup_target("sars_mpro", train_raw, test_raw, args.out_dir, args.n_boot, args.seed)

    print("\nNext:")
    print("  rsync -avz data/sars_mpro/ \\")
    print("    nnori@login.betty.parcc.upenn.edu:/vast/projects/pranam/lab/nnori/hadsbm-hiv/data/sars_mpro/")
    print("  sbatch run_sars_mpro_stage1.slurm")


if __name__ == "__main__":
    main()
