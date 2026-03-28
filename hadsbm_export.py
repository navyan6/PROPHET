#!/usr/bin/env python3
"""
Export a phylogenetic tree + sequences to JSON for HadSBM / BranchSBM experiments.

The math object we save is roughly:
  T = (graph G, split times t_k, split probabilities p_k, leaf endpoints pi_1,k)

This file reads:
  - a Newick tree (from FastTree),
  - the variant FASTA (same names as leaf labels),
  - wild-type protease from the same UniProt JSON (or wildtype FASTA) as tree.py.

FastTree "confidence" on a node is SH-like local support — not the same as the
paper's p_k. We store it as field sh_support on each split for optional use.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from Bio import Phylo
except ImportError:
    print("error: pip install biopython", file=sys.stderr)
    raise SystemExit(1)

from pipeline_paths import PipelinePaths, default_paths


@dataclass
class SplitEvent:
    """
    One binary split in the tree (one internal node with exactly two children).

    time_tau is in [0, 1] after we normalize by the longest root-to-tip path.
    p_left + p_right should be 1.0 (we use uniform or length-based heuristic).
    """

    parent_index: int
    left_child_index: int
    right_child_index: int
    time_tau: float
    p_left: float
    p_right: float
    sh_support: float | None
    branch_len_left: float
    branch_len_right: float


@dataclass
class LeafEndpoint:
    """One leaf: where in the node list it is, its FASTA id, and its sequence."""

    node_index: int
    leaf_id: str
    sequence: str


def branch_length_or_zero(clade) -> float:
    """Bio.Phylo sometimes sets branch_length to None."""
    value = clade.branch_length
    if value is None:
        return 0.0
    return float(value)


def read_fasta_as_dict(fasta_path: Path) -> dict[str, str]:
    """
    Read FASTA into a dictionary: sequence id -> sequence string.

    Only the first word after '>' is used as the id (same as other scripts).
    """
    sequences: dict[str, str] = {}
    current_id: str | None = None
    chunks: list[str] = []

    with open(fasta_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line == "":
                continue
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(chunks)
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)

    if current_id is not None:
        sequences[current_id] = "".join(chunks)

    return sequences


def list_nodes_preorder(root) -> list:
    """
    Return all clades in preorder (parent before children).

    We use this order to assign integer indices 0..n-1 to nodes.
    """
    result: list = []

    def visit(clade):
        result.append(clade)
        for child in clade.clades:
            visit(child)

    visit(root)
    return result


def max_depth_root_to_any_leaf(clade, depth_so_far: float = 0.0) -> float:
    """
    Longest path length from the current clade down to any tip, in branch units.

    depth_so_far is the distance from the tree root to the *parent side* of this
    clade (we add this clade's own branch length when recursing).
    """
    my_branch = branch_length_or_zero(clade)

    if clade.is_terminal():
        return depth_so_far + my_branch

    depth_here = depth_so_far + my_branch
    if len(clade.clades) == 0:
        return depth_here

    best = 0.0
    for child in clade.clades:
        child_best = max_depth_root_to_any_leaf(child, depth_here)
        if child_best > best:
            best = child_best
    return best


def depth_from_root_to_clade(root, target, depth_so_far: float = 0.0) -> float | None:
    """
    Total branch length along the path from tree root down to `target` (inclusive).

    Returns None if target is not found under root.
    """
    if root is target:
        return depth_so_far + branch_length_or_zero(root)

    my_branch = branch_length_or_zero(root)
    next_depth = depth_so_far + my_branch

    for child in root.clades:
        found = depth_from_root_to_clade(child, target, next_depth)
        if found is not None:
            return found

    return None


def build_hadsbm_bundle(
    tree: Phylo.BaseTree.Tree,
    fasta_by_id: dict[str, str],
    wildtype_sequence: str,
    *,
    prob_mode: str = "length",
) -> dict[str, Any]:
    """
    Build the big dictionary we will json.dump.

    prob_mode:
      - "uniform": p_left = p_right = 0.5 at every split
      - "length":  split probability proportional to child branch lengths
    """
    root = tree.root
    nodes = list_nodes_preorder(root)

    # Map each clade object's id() to its index in `nodes` (preorder index).
    index_of: dict[int, int] = {}
    for index, clade in enumerate(nodes):
        index_of[id(clade)] = index

    deepest_path = max_depth_root_to_any_leaf(root)
    if deepest_path <= 0:
        deepest_path = 1.0

    num_nodes = len(nodes)

    # Build undirected adjacency matrix (1 = edge).
    adjacency = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    parent_of: list[int | None] = [None for _ in range(num_nodes)]

    for clade in nodes:
        parent_index = index_of[id(clade)]
        for child in clade.clades:
            child_index = index_of[id(child)]
            adjacency[parent_index][child_index] = 1
            adjacency[child_index][parent_index] = 1
            parent_of[child_index] = parent_index

    root_index = index_of[id(root)]
    parent_of[root_index] = None

    split_list: list[SplitEvent] = []

    for clade in nodes:
        if clade.is_terminal():
            continue
        if len(clade.clades) != 2:
            # This exporter only records clean binary splits.
            continue

        parent_index = index_of[id(clade)]
        left_child = clade.clades[0]
        right_child = clade.clades[1]
        left_index = index_of[id(left_child)]
        right_index = index_of[id(right_child)]

        depth_at_split = depth_from_root_to_clade(root, clade)
        if depth_at_split is None:
            depth_at_split = 0.0

        tau = depth_at_split / deepest_path
        if tau < 0.0:
            tau = 0.0
        if tau > 1.0:
            tau = 1.0

        left_len = branch_length_or_zero(left_child)
        right_len = branch_length_or_zero(right_child)

        confidence = getattr(clade, "confidence", None)
        if confidence is None:
            sh_support = None
        else:
            sh_support = float(confidence)

        if prob_mode == "uniform":
            p_left = 0.5
            p_right = 0.5
        else:
            total = left_len + right_len
            if total <= 0.0:
                p_left = 0.5
                p_right = 0.5
            else:
                p_left = left_len / total
                p_right = right_len / total

        split_list.append(
            SplitEvent(
                parent_index=parent_index,
                left_child_index=left_index,
                right_child_index=right_index,
                time_tau=tau,
                p_left=p_left,
                p_right=p_right,
                sh_support=sh_support,
                branch_len_left=left_len,
                branch_len_right=right_len,
            )
        )

    leaf_list: list[LeafEndpoint] = []
    for clade in nodes:
        if not clade.is_terminal():
            continue
        node_index = index_of[id(clade)]
        label = clade.name
        if label is None or label == "":
            continue
        if label not in fasta_by_id:
            raise KeyError(f"Leaf {label!r} is in the tree but not in the FASTA file")
        seq = fasta_by_id[label]
        leaf_list.append(
            LeafEndpoint(node_index=node_index, leaf_id=label, sequence=seq)
        )

    leaf_list.sort(key=lambda item: item.node_index)

    parent_for_json: list[int] = []
    for p in parent_of:
        if p is None:
            parent_for_json.append(-1)
        else:
            parent_for_json.append(p)

    bundle: dict[str, Any] = {
        "format": "hadsbm_tree_v1",
        "description": {
            "G": "Symmetric adjacency; node order = preorder from Bio.Phylo root.",
            "tau": "Split times in [0,1] = depth to split / max root-to-tip depth.",
            "p_k": f"At each binary split, p_left + p_right = 1; mode={prob_mode}.",
            "pi_1_k": "One sequence per leaf (empirical endpoint).",
            "x_WT": "Wild-type protease anchor for BranchSBM.",
        },
        "x_WT": wildtype_sequence,
        "n_nodes": num_nodes,
        "n_leaves": len(leaf_list),
        "node_order": "preorder from Newick root",
        "adjacency_G": adjacency,
        "parent_index": parent_for_json,
        "splits": [asdict(item) for item in split_list],
        "leaf_endpoints_pi": [asdict(item) for item in leaf_list],
        "leaf_ids_in_order": [item.leaf_id for item in leaf_list],
    }
    return bundle


def load_wildtype_protease(
    json_path: Path,
    wildtype_fasta: Path | None = None,
) -> str:
    """Same 99-aa protease WT string that tree.py uses when building variants."""
    import tree as tree_module

    full_polyprotein = tree_module.read_wildtype_polyprotein(
        json_path, wildtype_fasta
    )
    start = tree_module.protease_start_index(full_polyprotein)
    if start < 0:
        raise ValueError("Could not find protease motif in reference sequence")
    end = start + tree_module.PROTEASE_LEN
    return full_polyprotein[start:end]


def run_export(
    paths: PipelinePaths | None = None,
    *,
    prob_mode: str = "length",
) -> int:
    """Main entry: read files, build bundle, write hadsbm_tree.json."""
    if paths is None:
        paths = default_paths()

    if not paths.newick.is_file():
        print(f"error: missing {paths.newick}", file=sys.stderr)
        return 1
    if not paths.hiv_sequences_fasta.is_file():
        print(f"error: missing {paths.hiv_sequences_fasta}", file=sys.stderr)
        return 1

    tree = Phylo.read(str(paths.newick), "newick")
    fasta_by_id = read_fasta_as_dict(paths.hiv_sequences_fasta)
    wildtype_pr = load_wildtype_protease(
        paths.variants_json,
        paths.wildtype_fasta,
    )

    bundle = build_hadsbm_bundle(
        tree,
        fasta_by_id,
        wildtype_pr,
        prob_mode=prob_mode,
    )

    with open(paths.hadsbm_tree_json, "w", encoding="utf-8") as handle:
        json.dump(bundle, handle, indent=2)

    print("Wrote:", paths.hadsbm_tree_json)
    print(
        "Note: this fixes the tree T from data. A generative model would sample "
        "(G, t_k, p_k, pi) instead."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export HadSBM tree bundle from Newick + FASTA + UniProt WT"
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--nwk", type=Path, default=None)
    parser.add_argument("--fasta", type=Path, default=None)
    parser.add_argument("--variants-json", type=Path, default=None)
    parser.add_argument(
        "--wt-fasta",
        type=Path,
        default=None,
        help="Optional polyprotein WT FASTA (must match variant build)",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--prob-mode",
        choices=("uniform", "length"),
        default="length",
        help="How to set p_left and p_right at each split",
    )
    args = parser.parse_args()

    base = default_paths(args.root)
    paths = PipelinePaths(
        root=base.root,
        variants_json=args.variants_json or base.variants_json,
        wildtype_fasta=args.wt_fasta or base.wildtype_fasta,
        hiv_sequences_fasta=args.fasta or base.hiv_sequences_fasta,
        aligned_fasta=base.aligned_fasta,
        newick=args.nwk or base.newick,
        hiv_tree_png=base.hiv_tree_png,
        leaf_paths_csv=base.leaf_paths_csv,
        hadsbm_tree_json=args.out or base.hadsbm_tree_json,
    )
    return run_export(paths, prob_mode=args.prob_mode)


if __name__ == "__main__":
    raise SystemExit(main())
