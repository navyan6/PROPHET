#!/usr/bin/env python3
"""
Translate nucleotide alignment to protein, hold out a clade, and set up
PROPHET data structure (train alignment + bootstrap trees + test holdout).

Usage:
    python scripts/prep_virus_protein.py \
        --nt-fasta  alignments/flu_ha_aligned.fasta \
        --name      flu_ha \
        --out-dir   data/flu_ha

Creates:
    data/flu_ha/sequences/flu_ha_protein.fasta         (all translated seqs)
    data/flu_ha/alignments/train/flu_ha_train_aligned.fasta
    data/flu_ha/alignments/test/flu_ha_test_clade_holdout.fasta
    data/flu_ha/trees/train/flu_ha_train_tree.nwk
    data/flu_ha/trees/train/flu_ha_bootstrap/          (100 bootstrap trees)
    data/flu_ha/trees/train/flu_ha_bootstrap_trees.txt
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from Bio import Phylo, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def translate_nt_fasta(nt_fasta: Path, min_length: int = 50) -> list[SeqRecord]:
    records = []
    seen = set()
    for rec in SeqIO.parse(str(nt_fasta), "fasta"):
        nt = str(rec.seq).replace("-", "").upper()
        nt = nt[: len(nt) - len(nt) % 3]
        if not nt:
            continue
        aa = str(Seq(nt).translate(to_stop=True))
        if len(aa) < min_length or aa in seen:
            continue
        seen.add(aa)
        records.append(SeqRecord(Seq(aa), id=rec.id, description=""))
    return records


def run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"ERROR running {cmd[0]}:\n{r.stderr.strip()}")
    return r.stdout


def mafft_align(fasta_path: Path, out_path: Path) -> Path:
    print(f"  Aligning {fasta_path.name} ...")
    out_path.write_text(
        run(["mafft", "--auto", "--quiet", "--thread", "-1", str(fasta_path)])
    )
    return out_path


def fasttree(aligned_path: Path, tree_path: Path) -> Path:
    print(f"  Building tree {tree_path.name} ...")
    tree_path.write_text(
        run(["FastTree", "-quiet", "-lg", str(aligned_path)])
    )
    return tree_path


def clade_holdout(
    records: list[SeqRecord],
    aligned_path: Path,
    tree_path: Path,
    target_frac: float = 0.20,
    seed: int = 42,
) -> tuple[list[SeqRecord], list[SeqRecord]]:
    """
    Build a tree, find a coherent clade of ~target_frac of sequences,
    return (train_records, test_records).
    """
    n_total = len(records)
    min_clade = max(10, int(n_total * target_frac * 0.5))
    max_clade = int(n_total * target_frac * 1.5)

    seq_lookup = {r.id: str(r.seq).replace("-", "") for r in records}

    tree = Phylo.read(str(tree_path), "newick")
    tree.root_at_midpoint()

    target_n = int(n_total * target_frac)
    candidates = []
    for clade in tree.find_clades(order="level"):
        leaves = clade.get_terminals()
        n = len(leaves)
        if not (min_clade <= n <= max_clade):
            continue
        leaf_ids = [c.name for c in leaves if c.name in seq_lookup]
        if len(leaf_ids) < min_clade:
            continue
        candidates.append((abs(n - target_n), n, leaf_ids))

    if not candidates:
        print(f"  WARNING: no clade of size {min_clade}–{max_clade} found. Falling back to random split.")
        random.seed(seed)
        ids = [r.id for r in records]
        random.shuffle(ids)
        n_test = int(n_total * target_frac)
        test_ids  = set(ids[:n_test])
        train_ids = set(ids[n_test:])
    else:
        candidates.sort()
        _, best_n, test_ids_list = candidates[0]
        test_ids  = set(test_ids_list)
        train_ids = {r.id for r in records} - test_ids
        print(f"  Clade holdout: {best_n} test seqs (target {target_n})")

    id_to_rec = {r.id: r for r in records}
    train_recs = [id_to_rec[i] for i in train_ids if i in id_to_rec]
    test_recs  = [id_to_rec[i] for i in test_ids  if i in id_to_rec]
    return train_recs, test_recs


def build_bootstrap_trees(
    aligned_path: Path, boot_dir: Path, n: int, seed: int = 42
) -> list[Path]:
    random.seed(seed)
    records = list(SeqIO.parse(str(aligned_path), "fasta"))
    L = len(records[0].seq)
    boot_dir.mkdir(exist_ok=True)
    paths = []
    print(f"  Building {n} bootstrap trees ...")
    for k in range(n):
        cols = sorted(random.choices(range(L), k=L))
        boot_fasta = boot_dir / f"boot{k}.fasta"
        with open(boot_fasta, "w") as f:
            for rec in records:
                resampled = "".join(str(rec.seq)[c] for c in cols)
                f.write(f">{rec.id}\n{resampled}\n")
        tree_path = boot_dir / f"boot{k}_tree.nwk"
        tree_path.write_text(run(["FastTree", "-quiet", "-lg", str(boot_fasta)]))
        paths.append(tree_path)
        if (k + 1) % 10 == 0:
            print(f"    {k+1}/{n} done")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nt-fasta",     required=True, type=Path)
    ap.add_argument("--name",         required=True)
    ap.add_argument("--out-dir",      required=True, type=Path)
    ap.add_argument("--holdout-frac", type=float, default=0.20)
    ap.add_argument("--n-bootstraps", type=int,   default=100)
    ap.add_argument("--min-length",   type=int,   default=50)
    ap.add_argument("--seed",         type=int,   default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out  = args.out_dir
    name = args.name

    seq_dir   = out / "sequences";               seq_dir.mkdir(parents=True, exist_ok=True)
    aln_all   = out / "alignments" / "all";      aln_all.mkdir(parents=True, exist_ok=True)
    aln_train = out / "alignments" / "train";    aln_train.mkdir(parents=True, exist_ok=True)
    aln_test  = out / "alignments" / "test";     aln_test.mkdir(parents=True, exist_ok=True)
    tr_train  = out / "trees" / "train";         tr_train.mkdir(parents=True, exist_ok=True)

    # 1. Translate
    print(f"[1/5] Translating {args.nt_fasta.name} ...")
    protein_records = translate_nt_fasta(args.nt_fasta, min_length=args.min_length)
    print(f"  {len(protein_records)} unique sequences translated")
    protein_fasta = seq_dir / f"{name}_protein.fasta"
    SeqIO.write(protein_records, str(protein_fasta), "fasta")

    # 2. Align all sequences + build tree for clade detection
    print(f"[2/5] Aligning all sequences and building tree for clade holdout ...")
    all_raw     = aln_all / f"{name}_all.fasta"
    all_aligned = aln_all / f"{name}_all_aligned.fasta"
    all_tree    = aln_all / f"{name}_all_tree.nwk"
    SeqIO.write(protein_records, str(all_raw), "fasta")
    mafft_align(all_raw, all_aligned)
    fasttree(all_aligned, all_tree)

    # Update records with aligned sequences (needed for clade lookup by ID)
    aligned_records = list(SeqIO.parse(str(all_aligned), "fasta"))

    # 3. Clade holdout
    print(f"[3/5] Finding clade holdout (~{args.holdout_frac:.0%} of sequences) ...")
    train_recs, test_recs = clade_holdout(
        aligned_records, all_aligned, all_tree,
        target_frac=args.holdout_frac, seed=args.seed
    )
    print(f"  train={len(train_recs)}, test={len(test_recs)}")

    # Save test holdout (unaligned protein seqs)
    test_protein = {r.id: r for r in protein_records}
    test_out = [test_protein[r.id] for r in test_recs if r.id in test_protein]
    SeqIO.write(test_out, str(aln_test / f"{name}_test_clade_holdout.fasta"), "fasta")

    # Save train (unaligned, for re-alignment)
    train_protein = [test_protein[r.id] for r in train_recs if r.id in test_protein]
    train_raw = aln_train / f"{name}_train.fasta"
    SeqIO.write(train_protein, str(train_raw), "fasta")

    # 4. Align train + build main tree
    print(f"[4/5] Aligning train set and building main tree ...")
    train_aligned = aln_train / f"{name}_train_aligned.fasta"
    train_tree    = tr_train  / f"{name}_train_tree.nwk"
    mafft_align(train_raw, train_aligned)
    fasttree(train_aligned, train_tree)

    # 5. Bootstrap trees on train alignment
    print(f"[5/5] Building {args.n_bootstraps} bootstrap trees ...")
    boot_dir   = tr_train / f"{name}_bootstrap"
    boot_paths = build_bootstrap_trees(train_aligned, boot_dir, args.n_bootstraps, seed=args.seed)

    trees_list = tr_train / f"{name}_bootstrap_trees.txt"
    with open(trees_list, "w") as f:
        f.write(str(train_tree) + "\n")
        for p in boot_paths:
            f.write(str(p) + "\n")

    print(f"\nDone. Stage 1 args:")
    print(f"  --tree       {train_tree}")
    print(f"  --trees-file {trees_list}")
    print(f"  --fasta      {train_aligned}")
    print(f"Test holdout:  {aln_test}/{name}_test_clade_holdout.fasta  ({len(test_out)} seqs)")


if __name__ == "__main__":
    main()
