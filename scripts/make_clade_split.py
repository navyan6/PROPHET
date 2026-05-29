#!/usr/bin/env python3
"""
Universal clade-based train/test split for any viral protein FASTA.

Steps:
  1. Parse sequences; clean UniProt/NCBI headers to bare accession IDs
  2. Deduplicate by exact sequence; length-filter
  3. Optionally subsample to --max-seqs (random, stratified by nothing — just for size control)
  4. Align all with MAFFT → build tree with FastTree
  5. Select a phylogenetically coherent holdout clade (~20% by default)
  6. Build N bootstrap trees on the training alignment
  7. Save everything in data/{target}/ ready for run_prophet.slurm

Usage:
  python scripts/make_clade_split.py \
      --fasta   denv2_E_uniprot.fasta \
      --target  denv2_e \
      --min-len 486 --max-len 510

  python scripts/make_clade_split.py \
      --fasta   rsv_F_uniprot.fasta \
      --target  rsv_f \
      --min-len 560 --max-len 585
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path

from Bio import Phylo, SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

REPO_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"ERROR running {cmd[0]}:\n{r.stderr.strip()}")
    return r.stdout


def mafft_align(fasta: Path, out: Path) -> Path:
    print(f"  MAFFT aligning {fasta.name} ({sum(1 for _ in open(fasta) if _.startswith('>'))} seqs)...")
    out.write_text(run(["mafft", "--auto", "--quiet", "--thread", "-1", str(fasta)]))
    return out


def fasttree(aligned: Path, tree: Path) -> Path:
    print(f"  FastTree → {tree.name} ...")
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
    min_c = max(5, target_n // 2)
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
            "Try --holdout-frac or check that your sequences are diverse enough."
        )

    candidates.sort()
    test_ids = set(candidates[0][1])
    print(f"  Clade holdout: {len(test_ids)} test / {n - len(test_ids)} train")

    id_to_rec = {r.id: r for r in records}
    train = [id_to_rec[i] for i in sorted(id_set - test_ids) if i in id_to_rec]
    test  = [id_to_rec[i] for i in sorted(test_ids) if i in id_to_rec]
    return train, test


def build_bootstrap_trees(aligned: Path, boot_dir: Path, n: int, seed: int = 42) -> list[Path]:
    random.seed(seed)
    records = list(SeqIO.parse(str(aligned), "fasta"))
    L = len(records[0].seq)
    boot_dir.mkdir(exist_ok=True)
    paths = []
    print(f"  Building {n} bootstrap trees ...")
    for k in range(n):
        cols = sorted(random.choices(range(L), k=L))
        bf = boot_dir / f"boot{k}.fasta"
        with open(bf, "w") as f:
            for rec in records:
                f.write(f">{rec.id}\n{''.join(str(rec.seq)[c] for c in cols)}\n")
        tp = boot_dir / f"boot{k}_tree.nwk"
        tp.write_text(run(["FastTree", "-quiet", "-lg", str(bf)]))
        paths.append(tp)
        if (k + 1) % 10 == 0:
            print(f"    {k + 1}/{n}")
    return paths


def clean_id(header: str) -> str:
    """Extract a clean accession ID from UniProt or NCBI headers."""
    # UniProt: >sp|P12568|FUS_HRSV ... or >tr|A0A3G9DX56|A0A3G9DX56_DENV2 ...
    if header.startswith("sp|") or header.startswith("tr|"):
        return header.split("|")[1]
    # NCBI: >ACC.version description
    return header.split()[0]


def load_and_filter(
    fasta: Path,
    min_len: int,
    max_len: int,
    max_seqs: int | None,
    seed: int,
) -> list[SeqRecord]:
    raw = list(SeqIO.parse(str(fasta), "fasta"))
    print(f"  Loaded {len(raw)} sequences from {fasta.name}")

    # Clean IDs
    cleaned = []
    for r in raw:
        new_id = clean_id(r.id)
        cleaned.append(SeqRecord(Seq(str(r.seq).upper().replace("-", "")),
                                  id=new_id, description=""))

    # Length filter
    before = len(cleaned)
    cleaned = [r for r in cleaned if min_len <= len(r.seq) <= max_len]
    print(f"  Length filter [{min_len}, {max_len}]: {before} → {len(cleaned)} seqs")

    # Deduplicate by exact sequence (keep first occurrence)
    seen: dict[str, SeqRecord] = {}
    for r in cleaned:
        s = str(r.seq)
        if s not in seen:
            seen[s] = r
    deduped = list(seen.values())
    print(f"  Dedup by sequence: {len(cleaned)} → {len(deduped)} unique seqs")

    # Subsample if requested
    if max_seqs is not None and len(deduped) > max_seqs:
        rng = random.Random(seed)
        deduped = rng.sample(deduped, max_seqs)
        print(f"  Subsampled to {max_seqs} sequences")

    if len(deduped) < 20:
        sys.exit(f"ERROR: only {len(deduped)} sequences after filtering — need at least 20.")

    return deduped


def main() -> None:
    ap = argparse.ArgumentParser(description="Clade-based train/test split for any viral protein")
    ap.add_argument("--fasta",        required=True, type=Path, help="Input protein FASTA")
    ap.add_argument("--target",       required=True,            help="Output name (used as directory and file prefix)")
    ap.add_argument("--min-len",      type=int, default=1,      help="Minimum sequence length to keep")
    ap.add_argument("--max-len",      type=int, default=99999,  help="Maximum sequence length to keep")
    ap.add_argument("--max-seqs",     type=int, default=None,   help="Subsample to at most this many sequences")
    ap.add_argument("--holdout-frac", type=float, default=0.20, help="Fraction of sequences for holdout clade (default: 0.20)")
    ap.add_argument("--n-boot",       type=int, default=100,    help="Number of bootstrap trees (default: 100)")
    ap.add_argument("--out-dir",      type=Path, default=REPO_ROOT / "data", help="Root output directory (default: data/)")
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    print(f"\n=== make_clade_split: {args.target} ===")

    seqs = load_and_filter(args.fasta, args.min_len, args.max_len, args.max_seqs, args.seed)

    # Temp files for full-set alignment + tree (used only for clade selection)
    tmp_dir = REPO_ROOT / "data" / args.target
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_raw  = tmp_dir / "_tmp_all.fasta"
    tmp_aln  = tmp_dir / "_tmp_all_aligned.fasta"
    tmp_tree = tmp_dir / "_tmp_all_tree.nwk"

    SeqIO.write(seqs, str(tmp_raw), "fasta")
    mafft_align(tmp_raw, tmp_aln)
    fasttree(tmp_aln, tmp_tree)

    aligned_recs = list(SeqIO.parse(str(tmp_aln), "fasta"))
    train_aln, test_aln = clade_holdout(aligned_recs, tmp_tree,
                                        target_frac=args.holdout_frac, seed=args.seed)

    # Map back to original (unaligned) sequences
    id_map    = {r.id: r for r in seqs}
    train_raw = [id_map[r.id] for r in train_aln if r.id in id_map]
    test_raw  = [id_map[r.id] for r in test_aln  if r.id in id_map]

    for f in [tmp_raw, tmp_aln, tmp_tree]:
        f.unlink(missing_ok=True)

    # Build final data structure
    t = args.target
    out = args.out_dir / t
    aln_train = out / "alignments" / "train"; aln_train.mkdir(parents=True, exist_ok=True)
    aln_test  = out / "alignments" / "test";  aln_test.mkdir(parents=True, exist_ok=True)
    tr_train  = out / "trees" / "train";      tr_train.mkdir(parents=True, exist_ok=True)
    boot_dir  = tr_train / f"{t}_bootstrap"

    # Align training set
    raw_train_path = aln_train / f"{t}_train.fasta"
    SeqIO.write(train_raw, str(raw_train_path), "fasta")
    aligned_train = aln_train / f"{t}_train_aligned.fasta"
    mafft_align(raw_train_path, aligned_train)

    # Main tree on training set
    train_tree = tr_train / f"{t}_train_tree.nwk"
    fasttree(aligned_train, train_tree)

    # Bootstrap trees
    boot_paths = build_bootstrap_trees(aligned_train, boot_dir, n=args.n_boot, seed=args.seed)
    trees_list = tr_train / f"{t}_bootstrap_trees.txt"
    with open(trees_list, "w") as f:
        for p in boot_paths:
            f.write(str(p.relative_to(tr_train)) + "\n")

    # Holdout (unaligned)
    test_out = aln_test / f"{t}_test_clade_holdout.fasta"
    SeqIO.write(test_raw, str(test_out), "fasta")

    print(f"\n{'='*50}")
    print(f"Done: {args.target}")
    print(f"  Train: {len(train_raw)} sequences")
    print(f"  Test:  {len(test_raw)} sequences (clade holdout)")
    print(f"\n  Add to configs/targets.py:")
    print(f'    "{t}": {{')
    print(f'        "alignment":  "{aligned_train.relative_to(REPO_ROOT)}",')
    print(f'        "tree":       "{train_tree.relative_to(REPO_ROOT)}",')
    print(f'        "trees_file": "{trees_list.relative_to(REPO_ROOT)}",')
    print(f'        "holdout":    "{test_out.relative_to(REPO_ROOT)}",')
    print(f'        "t_evo":      0.15,')
    print(f'        "protein":    True,')
    print(f'    }}')


if __name__ == "__main__":
    main()
