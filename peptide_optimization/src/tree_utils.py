"""
Minimal tree utilities for extracting variant probabilities.
Used by peptide optimization to weight binding affinities.

Probabilities are computed from tree structure: each leaf's probability is
the product of split probabilities (p_left or p_right) along the path from
root to that leaf. Used as weight vector in MOG-DFM optimization.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class VariantWithProbability:
    """HIV variant with sequence and probability from tree."""
    name: str
    sequence: str
    probability: float


def _build_tree_structure(tree_data: Dict) -> Dict[int, Dict]:
    """
    Build tree as dict: node_index -> {left_child, right_child, p_left, p_right}
    
    Used to trace paths from root to leaves and compute probabilities.
    """
    tree = {}
    splits = tree_data.get("splits", [])
    
    for split in splits:
        parent_idx = split["parent_index"]
        tree[parent_idx] = {
            "left_child": split["left_child_index"],
            "right_child": split["right_child_index"],
            "p_left": split["p_left"],
            "p_right": split["p_right"],
        }
    
    return tree


def _find_parent(node_index: int, tree: Dict[int, Dict]) -> Optional[int]:
    """Find parent of a node in the tree structure."""
    for parent_idx, children in tree.items():
        if node_index == children["left_child"] or node_index == children["right_child"]:
            return parent_idx
    return None


def _compute_leaf_probability(leaf_node_index: int, tree: Dict[int, Dict]) -> float:
    """
    Compute probability of a leaf as product of split probabilities along path.
    
    Traces backwards from leaf to root, then multiplies p_left/p_right for each
    split encountered on the path down.
    """
    # Trace backwards: leaf -> ... -> root (node 0)
    path = [leaf_node_index]
    current = leaf_node_index
    
    while current != 0:
        parent = _find_parent(current, tree)
        if parent is None:
            break
        path.insert(0, parent)
        current = parent
    
    # Forward pass: accumulate split probabilities
    prob = 1.0
    for i in range(len(path) - 1):
        parent_idx = path[i]
        child_idx = path[i + 1]
        
        if parent_idx not in tree:
            continue  # Node 0 (root) has no parent split
        
        if child_idx == tree[parent_idx]["left_child"]:
            prob *= tree[parent_idx]["p_left"]
        else:
            prob *= tree[parent_idx]["p_right"]
    
    return prob


def load_tree_probabilities(
    tree_json_path: Path,
    top_k: Optional[int] = None,
    min_prob: float = 0.0,
) -> List[VariantWithProbability]:
    """
    Load variants with tree-derived probabilities.

    Args:
        tree_json_path: Path to hadsbm_tree.json
        top_k: If set, keep only the top-k variants by probability (after
               computing all probs). Useful for large trees where path-product
               underflow leaves most leaves at near-zero probability.
        min_prob: Drop any variant whose raw probability is below this threshold
                  before renormalisation. Ignored when top_k is also set.

    Returns:
        List of VariantWithProbability, renormalised so probabilities sum to 1.
    """
    with open(tree_json_path, "r") as f:
        tree_data = json.load(f)

    leaves = tree_data.get("leaf_endpoints_pi", [])
    n_leaves = len(leaves)

    if n_leaves == 0:
        raise ValueError(f"No leaf sequences found in {tree_json_path}")

    tree = _build_tree_structure(tree_data)

    variants = []
    for leaf in leaves:
        node_idx = leaf.get("node_index", -1)
        if node_idx >= 0 and tree:
            prob = _compute_leaf_probability(node_idx, tree)
        else:
            prob = 1.0 / n_leaves

        variants.append(
            VariantWithProbability(
                name=leaf.get("leaf_id", f"leaf_{len(variants)}"),
                sequence=leaf.get("sequence", ""),
                probability=prob,
            )
        )

    # Optional: keep top-k by probability
    if top_k is not None:
        variants.sort(key=lambda v: v.probability, reverse=True)
        variants = variants[:top_k]
    elif min_prob > 0.0:
        variants = [v for v in variants if v.probability >= min_prob]

    # Renormalise
    total_prob = sum(v.probability for v in variants)
    if total_prob > 0:
        variants = [
            VariantWithProbability(name=v.name, sequence=v.sequence,
                                   probability=v.probability / total_prob)
            for v in variants
        ]

    return variants


def load_wildtype_sequence(tree_json_path: Path) -> str:
    """Load wildtype reference sequence from tree JSON."""
    with open(tree_json_path, "r") as f:
        tree_data = json.load(f)
    
    wt_seq = tree_data.get("x_WT", "")
    if not wt_seq:
        raise ValueError(f"No wildtype sequence (x_WT) found in {tree_json_path}")
    
    return wt_seq

    if not wt_seq:
        raise ValueError(f"No wildtype sequence (x_WT) found in {tree_json_path}")
    
    return wt_seq
