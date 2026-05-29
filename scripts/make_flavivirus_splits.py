#!/usr/bin/env python3
"""
Build PROPHET-ready train/test splits for flavivirus NS3 targets.

For dengue: DENV3 train, DENV1 holdout (cross-serotype).
For Zika and WNV: clade holdout via prep_virus_protein logic.

Runs MAFFT + FastTree + 100 bootstrap trees for each target.

Usage (after download_flavivirus_ns3.py):
    python scripts/make_flavivirus_splits.py --in-dir data/flavivirus_ns3

Outputs (per virus):
    data/{virus}/alignments/train/{virus}_train_aligned.fasta
    data/{virus}/alignments/test/{virus}_test_holdout.fasta
    data/{virus}/trees/train/{virus}_train_tree.nwk
    data/{virus}/trees/train/{virus}_bootstrap_trees.txt
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
    min_c, max_c = max(10, target_n // 2), int(target_n * 1.5)

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
    else:
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
            print(f"      {k+1}/{n}")
    return paths


def setup_virus(
    name: str,
    train_seqs: list[SeqRecord],
    test_seqs: list[SeqRecord],
    out_root: Path,
    n_boot: int = 100,
    seed: int = 42,
) -> None:
    out = out_root / name
    aln_train = out / "alignments" / "train";  aln_train.mkdir(parents=True, exist_ok=True)
    aln_test  = out / "alignments" / "test";   aln_test.mkdir(parents=True, exist_ok=True)
    tr_train  = out / "trees" / "train";       tr_train.mkdir(parents=True, exist_ok=True)

    print(f"  train={len(train_seqs)}, holdout={len(test_seqs)}")

    # Write holdout (unaligned)
    SeqIO.write(test_seqs, str(aln_test / f"{name}_test_holdout.fasta"), "fasta")

    # Align train
    raw = aln_train / f"{name}_train.fasta"
    SeqIO.write(train_seqs, str(raw), "fasta")
    aligned = mafft_align(raw, aln_train / f"{name}_train_aligned.fasta")

    # Main tree
    train_tree = fasttree(aligned, tr_train / f"{name}_train_tree.nwk")

    # Bootstrap trees
    boot_dir = tr_train / f"{name}_bootstrap"
    boot_paths = build_bootstrap_trees(aligned, boot_dir, n_boot, seed)
    trees_list = tr_train / f"{name}_bootstrap_trees.txt"
    with open(trees_list, "w") as f:
        f.write(str(train_tree) + "\n")
        for p in boot_paths:
            f.write(str(p) + "\n")

    print(f"  Stage 1 args:")
    print(f"    --tree       {train_tree}")
    print(f"    --trees-file {trees_list}")
    print(f"    --fasta      {aligned}")
    print(f"  Holdout:       {aln_test}/{name}_test_holdout.fasta ({len(test_seqs)} seqs)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir",    default="data/flavivirus_ns3", type=Path,
                    help="Directory with downloaded FASTA files from download_flavivirus_ns3.py")
    ap.add_argument("--out-dir",   default="data", type=Path)
    ap.add_argument("--n-boot",    type=int, default=100)
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()

    in_dir = args.in_dir

    # ── Dengue: DENV3 train, DENV1 holdout (cross-serotype) ──────────────────
    print("\n=== Dengue NS3 (DENV3 train / DENV1 holdout) ===")
    denv3 = list(SeqIO.parse(str(in_dir / "denv3_ns3_protease.fasta"), "fasta"))
    denv1 = list(SeqIO.parse(str(in_dir / "denv1_ns3_protease.fasta"), "fasta"))
    if not denv3:
        print("  SKIP: no DENV3 sequences found")
    elif not denv1:
        print("  SKIP: no DENV1 sequences found")
    else:
        setup_virus("dengue_ns3", denv3, denv1, args.out_dir, args.n_boot, args.seed)

    # ── Zika: clade holdout ───────────────────────────────────────────────────
    print("\n=== Zika NS3 (clade holdout) ===")
    zika_seqs = list(SeqIO.parse(str(in_dir / "zika_ns3_protease.fasta"), "fasta"))
    if len(zika_seqs) < 20:
        print(f"  SKIP: only {len(zika_seqs)} Zika sequences")
    else:
        # Build quick tree for clade detection
        tmp_raw = in_dir / "zika_tmp.fasta"
        tmp_aln = in_dir / "zika_tmp_aligned.fasta"
        tmp_tree = in_dir / "zika_tmp_tree.nwk"
        SeqIO.write(zika_seqs, str(tmp_raw), "fasta")
        mafft_align(tmp_raw, tmp_aln)
        fasttree(tmp_aln, tmp_tree)
        aligned_recs = list(SeqIO.parse(str(tmp_aln), "fasta"))
        train, test = clade_holdout(aligned_recs, tmp_tree, target_frac=0.20, seed=args.seed)
        # Map back to unaligned
        id_map = {r.id: r for r in zika_seqs}
        train_u = [id_map[r.id] for r in train if r.id in id_map]
        test_u  = [id_map[r.id] for r in test  if r.id in id_map]
        setup_virus("zika_ns3", train_u, test_u, args.out_dir, args.n_boot, args.seed)
        for f in [tmp_raw, tmp_aln, tmp_tree]:
            f.unlink(missing_ok=True)

    # ── West Nile: clade holdout ──────────────────────────────────────────────
    print("\n=== West Nile NS3 (clade holdout) ===")
    wnv_seqs = list(SeqIO.parse(str(in_dir / "wnv_ns3_protease.fasta"), "fasta"))
    if len(wnv_seqs) < 20:
        print(f"  SKIP: only {len(wnv_seqs)} WNV sequences")
    else:
        tmp_raw  = in_dir / "wnv_tmp.fasta"
        tmp_aln  = in_dir / "wnv_tmp_aligned.fasta"
        tmp_tree = in_dir / "wnv_tmp_tree.nwk"
        SeqIO.write(wnv_seqs, str(tmp_raw), "fasta")
        mafft_align(tmp_raw, tmp_aln)
        fasttree(tmp_aln, tmp_tree)
        aligned_recs = list(SeqIO.parse(str(tmp_aln), "fasta"))
        train, test = clade_holdout(aligned_recs, tmp_tree, target_frac=0.20, seed=args.seed)
        id_map = {r.id: r for r in wnv_seqs}
        train_u = [id_map[r.id] for r in train if r.id in id_map]
        test_u  = [id_map[r.id] for r in test  if r.id in id_map]
        setup_virus("wnv_ns3", train_u, test_u, args.out_dir, args.n_boot, args.seed)
        for f in [tmp_raw, tmp_aln, tmp_tree]:
            f.unlink(missing_ok=True)

    print("\n✓ All splits ready.")


if __name__ == "__main__":
    main()
