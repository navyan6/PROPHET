from __future__ import annotations

import json
import math
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from Bio import Phylo
from Bio.Phylo.BaseTree import Tree, Clade


#parse
def parse_newick(source: str | Path) -> Tree:
    if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source
                                    and Path(source).exists()):
        tree = Phylo.read(str(source), "newick")
    else:
        tree = Phylo.read(StringIO(source), "newick")

    return tree


def root_tree(tree: Tree, root_name: Optional[str] = None) -> Tree:
    if root_name is not None:
        hits = list(tree.find_clades(name=root_name))
        if not hits:
            raise ValueError(
                f"Root node '{root_name}' not found in tree. "
                "Check the leaf names with list_leaves()."
            )
        tree.root_with_outgroup({"name": root_name})
        print(f"[root_tree] Tree re-rooted at '{root_name}'.")
    else:
        # Sanity-check: Bio.Phylo trees from Newick are always rooted,
        n_children = len(tree.root.clades)
        if n_children > 2:
            print(
                f"[root_tree] WARNING: root has {n_children} children — "
                "tree may be unrooted (trifurcating root). "
                "Consider passing --root-name to specify an outgroup."
            )
        else:
            print(f"[root_tree] Tree is rooted (root has {n_children} children).")

    return tree


#computing distance
def root_to_tip_distances(tree: Tree) -> Dict[str, float]:
    distances: Dict[str, float] = {}
    missing_warned = False

    def _walk(clade: Clade, accumulated: float) -> None:
        nonlocal missing_warned

        branch = clade.branch_length
        if branch is None:
            if not missing_warned:
                print(
                    "[root_to_tip_distances] WARNING: one or more branches have "
                    "no length annotation; treating them as 0."
                )
                missing_warned = True
            branch = 0.0

        total = accumulated + branch

        if clade.is_terminal():
            name = clade.name or f"leaf_{len(distances)}"
            distances[name] = total
        else:
            for child in clade.clades:
                _walk(child, total)

    _walk(tree.root, 0.0)
    return distances


#norm probability
def distances_to_probabilities(
    distances: Dict[str, float],
    lam: float = 1.0,
) -> Dict[str, float]:
    if not distances:
        raise ValueError("distances dict is empty — no leaves found.")
    if lam <= 0:
        raise ValueError(f"lam must be > 0, got {lam}.")

    # Raw weights
    raw: Dict[str, float] = {
        name: math.exp(-lam * d) for name, d in distances.items()
    }

    total = sum(raw.values())
    if total == 0.0:
        raise ValueError(
            "All raw weights are 0 (distances may be extremely large for this lam). "
            "Try a smaller lam value."
        )

    # Normalise
    probs: Dict[str, float] = {name: w / total for name, w in raw.items()}

    # Floating-point sanity check
    prob_sum = sum(probs.values())
    if abs(prob_sum - 1.0) > 1e-6:
        raise RuntimeError(
            f"Probabilities do not sum to 1 after normalisation (sum={prob_sum})."
        )

    return probs


#sequence lookup
def load_sequences(csv_path: str | Path) -> Dict[str, str]:
    """Return a dict mapping GenBank accession -> sequence from the obs CSV."""
    df = pd.read_csv(csv_path, usecols=["GenBank Accession", "seq"])
    return dict(zip(df["GenBank Accession"].astype(str), df["seq"].astype(str)))


def accession_from_leaf(leaf_name: str) -> str:
    """Extract accession from a leaf name like 'FJ639779|2003|Venezuela'."""
    return leaf_name.split("|")[0]


#sort/print/save
def sorted_probabilities(probs: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(probs.items(), key=lambda x: x[1], reverse=True)


def print_top_k(sorted_pairs: List[Tuple[str, float]], k: int = 10) -> None:
    print(f"\nTop {k} most probable variants:")
    print("-" * 60)
    print(f"  {'Rank':<5} {'Leaf':<45} {'Probability':>12}")
    print("-" * 60)
    for i, (name, prob) in enumerate(sorted_pairs[:k], 1):
        print(f"  {i:<5} {name:<45} {prob:>12.8f}")
    print("-" * 60)
    print(f"  Total leaves: {len(sorted_pairs)}")
    print(f"  Sum of all p: {sum(p for _, p in sorted_pairs):.8f}")


def save_top_k(
    sorted_pairs: List[Tuple[str, float]],
    path: Path,
    k: int = 10,
    sequences: Optional[Dict[str, str]] = None,
) -> None:
    top = {}
    for name, prob in sorted_pairs[:k]:
        entry: Dict = {"probability": prob}
        if sequences is not None:
            acc = accession_from_leaf(name)
            entry["sequence"] = sequences.get(acc, "")
        top[name] = entry
    path.write_text(json.dumps(top, indent=2))
    print(f"\nTop-{k} variants saved to: {path}")


def compute_leaf_probabilities(
    source: str | Path,
    lam: float = 1.0,
    root_name: Optional[str] = None,
) -> Tuple[Dict[str, float], List[Tuple[str, float]]]:

    tree = parse_newick(source)
    tree = root_tree(tree, root_name=root_name)
    distances = root_to_tip_distances(tree)
    probs = distances_to_probabilities(distances, lam=lam)
    sorted_pairs = sorted_probabilities(probs)
    return probs, sorted_pairs

if __name__ == "__main__":
    NWK = Path(__file__).parent / "DENV3_tree.nwk"
    CSV = Path(__file__).parent / "cluster2and6.obs.csv"

    probs, sorted_pairs = compute_leaf_probabilities(source=NWK)
    sequences = load_sequences(CSV)

    print_top_k(sorted_pairs, k=10)

    top10: Dict[str, Dict] = {
        name: {
            "probability": prob,
            "sequence": sequences.get(accession_from_leaf(name), ""),
        }
        for name, prob in sorted_pairs[:10]
    }
    print(top10)
