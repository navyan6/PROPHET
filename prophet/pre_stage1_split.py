#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

from Bio import Phylo, SeqIO


def read_tree_list(path: Path) -> list[Path]:
    out: list[Path] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line)
            out.append(p if p.is_absolute() else (path.parent / p))
    return out


def pick_internal_subtree(tree, min_size: int, max_frac: float, rng: random.Random):
    total = len(tree.get_terminals())
    max_size = max(1, int(total * max_frac))
    candidates = []
    for clade in tree.get_nonterminals():
        size = len(clade.get_terminals())
        if min_size <= size <= max_size:
            candidates.append((clade, size))
    if not candidates:
        # Fallback: if no "balanced" subtree exists, still pick any non-trivial
        # internal subtree so we can produce a valid pre-Stage-1 split.
        fallback = []
        for clade in tree.get_nonterminals():
            size = len(clade.get_terminals())
            if 1 < size < total:
                fallback.append((clade, size))
        if not fallback:
            return None
        return rng.choice([clade for clade, _ in fallback])
    sizes = sorted({size for _, size in candidates})
    target = sizes[len(sizes) // 2]
    near_median = [clade for clade, size in candidates if size == target]
    return rng.choice(near_median)


def prune_to_leaves(tree, keep: set[str]):
    out = deepcopy(tree)
    for leaf in list(out.get_terminals()):
        if leaf.name and leaf.name not in keep:
            out.prune(target=leaf)
    return out


def find_matching_fasta(virus_name: str, align_dir: Path) -> Path | None:
    direct = align_dir / f"{virus_name}_aligned.fasta"
    if direct.exists():
        return direct
    pref = sorted(align_dir.glob(f"{virus_name}*.fasta"))
    if len(pref) == 1:
        return pref[0]
    return None


def read_fasta_map(path: Path) -> dict[str, str]:
    return {rec.id: str(rec.seq) for rec in SeqIO.parse(str(path), "fasta")}


def write_fasta(ids: list[str], seqs: dict[str, str], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for sid in ids:
            seq = seqs.get(sid)
            if seq:
                f.write(f">{sid}\n{seq}\n")
                count += 1
    return count


def main() -> None:
    p = argparse.ArgumentParser(description="Pre-Stage-1 train/test tree+FASTA split")
    p.add_argument("--tree-list", default="trees/all_trees_paths.txt")
    p.add_argument("--alignments-dir", default="alignments")
    p.add_argument("--out-dir", default="data/pre_stage1_split")
    p.add_argument("--min-test-leaves", type=int, default=20)
    p.add_argument("--max-test-frac", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    tree_list_path = (repo_root / args.tree_list).resolve()
    align_dir = (repo_root / args.alignments_dir).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    train_tree_dir = out_dir / "trees" / "train"
    test_tree_dir = out_dir / "trees" / "test"
    train_fasta_dir = out_dir / "alignments" / "train"
    test_fasta_dir = out_dir / "alignments" / "test"
    for d in [train_tree_dir, test_tree_dir, train_fasta_dir, test_fasta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    trees = read_tree_list(tree_list_path)
    summary: dict[str, dict] = {}
    train_tree_paths: list[str] = []
    test_tree_paths: list[str] = []

    for tree_path in trees:
        tree_path = tree_path.resolve()
        virus = tree_path.stem
        if virus.endswith("_tree"):
            virus = virus[:-5]
        fasta_path = find_matching_fasta(virus, align_dir)
        if fasta_path is None:
            summary[virus] = {"status": "skipped", "reason": "missing_matching_fasta"}
            continue

        tree = Phylo.read(str(tree_path), "newick")
        node = pick_internal_subtree(tree, args.min_test_leaves, args.max_test_frac, rng)
        if node is None:
            summary[virus] = {"status": "skipped", "reason": "no_valid_internal_subtree"}
            continue

        test_leaves = [c.name for c in node.get_terminals() if c.name]
        test_set = set(test_leaves)
        all_leaves = {c.name for c in tree.get_terminals() if c.name}
        train_set = all_leaves - test_set
        if not train_set or not test_set:
            summary[virus] = {"status": "skipped", "reason": "empty_train_or_test"}
            continue

        seqs = read_fasta_map(fasta_path)
        train_ids = sorted([sid for sid in train_set if sid in seqs])
        test_ids = sorted([sid for sid in test_set if sid in seqs])
        if not train_ids or not test_ids:
            summary[virus] = {"status": "skipped", "reason": "no_tree_leaf_overlap_with_fasta"}
            continue

        train_tree = prune_to_leaves(tree, set(train_ids))
        test_tree = deepcopy(node)

        train_tree_path = train_tree_dir / f"{virus}_train_tree.nwk"
        test_tree_path = test_tree_dir / f"{virus}_test_tree.nwk"
        train_fasta_path = train_fasta_dir / f"{virus}_train_aligned.fasta"
        test_fasta_path = test_fasta_dir / f"{virus}_test_aligned.fasta"

        Phylo.write(train_tree, str(train_tree_path), "newick")
        Phylo.write(test_tree, str(test_tree_path), "newick")
        n_train = write_fasta(train_ids, seqs, train_fasta_path)
        n_test = write_fasta(test_ids, seqs, test_fasta_path)

        train_tree_paths.append(str(train_tree_path))
        test_tree_paths.append(str(test_tree_path))
        summary[virus] = {
            "status": "ok",
            "tree": str(tree_path),
            "fasta": str(fasta_path),
            "train_tree": str(train_tree_path),
            "test_tree": str(test_tree_path),
            "train_fasta": str(train_fasta_path),
            "test_fasta": str(test_fasta_path),
            "n_tree_train": len(train_set),
            "n_tree_test": len(test_set),
            "n_fasta_train": n_train,
            "n_fasta_test": n_test,
        }

    with open(out_dir / "train_trees_paths.txt", "w", encoding="utf-8") as f:
        for pth in train_tree_paths:
            f.write(f"{pth}\n")
    with open(out_dir / "test_trees_paths.txt", "w", encoding="utf-8") as f:
        for pth in test_tree_paths:
            f.write(f"{pth}\n")
    with open(out_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    ok = sum(1 for v in summary.values() if v.get("status") == "ok")
    skipped = sum(1 for v in summary.values() if v.get("status") != "ok")
    print(f"Done. split ok={ok}, skipped={skipped}")
    print(f"Outputs under: {out_dir}")


if __name__ == "__main__":
    main()
