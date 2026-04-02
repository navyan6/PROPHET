#!/usr/bin/env python3
"""
Draw the tree (PNG) and write a CSV table of paths from root to each leaf.

FastTree puts approximate support values on internal nodes (SH-like local
supports). Those are not the same as bootstrap percentages from other tools.

Needs: pip install biopython matplotlib
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    from Bio import Phylo
except ImportError:
    print(
        "error: install dependencies: pip install biopython matplotlib",
        file=sys.stderr,
    )
    raise SystemExit(1)

from pipeline_paths import PipelinePaths, default_paths


def safe_branch_length(clade) -> float:
    """Bio.Phylo may leave branch_length as None; treat that as 0.0."""
    length = clade.branch_length
    if length is None:
        return 0.0
    return float(length)


def collect_supports_and_lengths_on_path(
    tree: Phylo.BaseTree.Tree,
    leaf,
) -> tuple[list[float], list[float]]:
    """
    Follow the path from the root down to one leaf.

    Returns:
      supports: FastTree confidence at each internal node on that path.
      lengths:  Branch length on each clade along the path (root to leaf).
    """
    path = tree.get_path(leaf)
    support_values: list[float] = []
    branch_lengths: list[float] = []

    for clade in path:
        branch_lengths.append(safe_branch_length(clade))

        if clade.is_terminal():
            continue

        confidence = getattr(clade, "confidence", None)
        if confidence is None:
            continue
        support_values.append(float(confidence))

    return support_values, branch_lengths


def write_leaf_path_csv(output_csv: Path, tree: Phylo.BaseTree.Tree) -> None:
    """One CSV row per leaf: supports and branch lengths from root to that leaf."""
    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "leaf",
                "n_supports_on_path",
                "supports_root_to_leaf",
                "sum_branch_length",
                "branch_lengths_root_to_leaf",
            ]
        )

        leaves = list(tree.get_terminals())
        leaves.sort(key=lambda node: node.name or "")

        for leaf in leaves:
            leaf_name = leaf.name or ""
            supports, lengths = collect_supports_and_lengths_on_path(tree, leaf)
            n_supports = len(supports)
            supports_str = ";".join(f"{value:.4f}" for value in supports)
            total_length = sum(lengths)
            lengths_str = ";".join(f"{value:.6f}" for value in lengths)

            writer.writerow(
                [
                    leaf_name,
                    n_supports,
                    supports_str,
                    f"{total_length:.6f}",
                    lengths_str,
                ]
            )


def draw_tree_to_png(
    tree: Phylo.BaseTree.Tree,
    png_path: Path,
    svg_path: Path | None,
    figsize_inches: tuple[float, float],
) -> None:
    """Use matplotlib to save the tree figure (no GUI window)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=figsize_inches, dpi=120)
    axes = figure.add_subplot(1, 1, 1)

    Phylo.draw(
        tree,
        axes=axes,
        do_show=False,
        show_confidence=True,
    )
    axes.set_title(
        "FastTree SH-like support on internal nodes (0–1); "
        "edge lengths = substitutions/site"
    )
    figure.tight_layout()
    figure.savefig(png_path, bbox_inches="tight")
    if svg_path is not None:
        figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)


def run_visualize(
    paths: PipelinePaths | None = None,
    *,
    print_ascii: bool = False,
    figsize: tuple[float, float] = (14.0, 20.0),
    svg: Path | None = None,
) -> int:
    """
    Read Newick, write PNG (+ optional SVG), write leaf_paths.csv.

    Returns 0 on success, 1 if the Newick file is missing.
    """
    if paths is None:
        paths = default_paths()

    if not paths.newick.is_file():
        print(f"error: missing tree file: {paths.newick}", file=sys.stderr)
        return 1

    tree = Phylo.read(str(paths.newick), "newick")

    draw_tree_to_png(tree, paths.hiv_tree_png, svg, figsize)
    print("Wrote figure:", paths.hiv_tree_png)
    if svg is not None:
        print("Wrote figure:", svg)

    write_leaf_path_csv(paths.leaf_paths_csv, tree)
    print("Wrote table:", paths.leaf_paths_csv)

    if print_ascii:
        Phylo.draw_ascii(tree)

    print(
        "\nNote: numbers on internal nodes are FastTree SH-like local supports "
        "(not bootstrap). For bootstrap support, use IQ-TREE (-B) or similar."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot Newick tree and export per-leaf path CSV"
    )
    parser.add_argument("--root", type=Path, default=None, help="Project root")
    parser.add_argument("--nwk", type=Path, default=None, help="Input Newick")
    parser.add_argument("--png", type=Path, default=None, help="Output PNG")
    parser.add_argument(
        "--svg",
        action="store_true",
        help="Also write <root>/hiv_tree.svg",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Output CSV path")
    parser.add_argument(
        "--ascii",
        action="store_true",
        dest="print_ascii",
        help="Print ASCII tree to stdout",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="14,20",
        help='Figure size as "width,height" in inches',
    )
    args = parser.parse_args()

    base = default_paths(args.root)
    svg_output: Path | None
    if args.svg:
        svg_output = base.root / "hiv_tree.svg"
    else:
        svg_output = None

    paths = PipelinePaths(
        root=base.root,
        variants_json=base.variants_json,
        wildtype_fasta=base.wildtype_fasta,
        hiv_sequences_fasta=base.hiv_sequences_fasta,
        aligned_fasta=base.aligned_fasta,
        newick=args.nwk or base.newick,
        hiv_tree_png=args.png or base.hiv_tree_png,
        leaf_paths_csv=args.csv or base.leaf_paths_csv,
        hadsbm_tree_json=base.hadsbm_tree_json,
    )

    width_str, height_str = args.figsize.split(",")
    width = float(width_str.strip())
    height = float(height_str.strip())

    return run_visualize(
        paths,
        print_ascii=args.print_ascii,
        figsize=(width, height),
        svg=svg_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
