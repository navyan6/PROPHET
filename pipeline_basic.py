#!/usr/bin/env python3
"""
Run the whole pipeline in order:

  1. phylogeny: JSON -> FASTA (optional), MAFFT, FastTree
  2. visualize_tree: PNG + leaf_paths.csv
  3. hadsbm_export: hadsbm_tree.json

Other code (training, notebooks) can import default_paths() and call the same
steps with the same file names.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_paths import PipelinePaths, default_paths


def run_basic(
    paths: PipelinePaths | None = None,
    *,
    skip_fasta: bool = False,
    no_viz: bool = False,
    fasttree_bin: str | None = None,
    prob_mode: str = "length",
    viz_ascii: bool = False,
) -> int:
    """
    Run the three stages. Return 0 if all succeeded, else first non-zero code.

    We import heavy modules inside this function so `python -c "import ..."`
    stays light when you only need paths.
    """
    if paths is None:
        paths = default_paths()

    import hadsbm_export
    import phylogeny
    import visualize_tree

    exit_code = phylogeny.run_phylogeny(
        paths,
        skip_fasta=skip_fasta,
        fasttree_bin=fasttree_bin,
    )
    if exit_code != 0:
        return exit_code

    if not no_viz:
        exit_code = visualize_tree.run_visualize(paths, print_ascii=viz_ascii)
        if exit_code != 0:
            return exit_code

    return hadsbm_export.run_export(paths, prob_mode=prob_mode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Full pipeline: phylogeny, optional plot, HadSBM JSON export"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root (default: folder containing this script)",
    )
    parser.add_argument(
        "--skip-fasta",
        action="store_true",
        help="Do not rebuild hiv_sequences.fasta from JSON",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip matplotlib figure and leaf_paths.csv",
    )
    parser.add_argument(
        "--fasttree-bin",
        default=None,
        help="FastTree executable if not on PATH",
    )
    parser.add_argument(
        "--prob-mode",
        choices=("uniform", "length"),
        default="length",
        help="Split masses in hadsbm_tree.json",
    )
    parser.add_argument(
        "--ascii-tree",
        action="store_true",
        help="Print ASCII tree to the terminal during the visualize step",
    )
    parser.add_argument(
        "--wt-fasta",
        type=Path,
        default=None,
        help="Optional polyprotein WT FASTA (first record)",
    )
    args = parser.parse_args()

    base = default_paths(args.root)
    paths = PipelinePaths(
        root=base.root,
        variants_json=base.variants_json,
        wildtype_fasta=args.wt_fasta or base.wildtype_fasta,
        hiv_sequences_fasta=base.hiv_sequences_fasta,
        aligned_fasta=base.aligned_fasta,
        newick=base.newick,
        hiv_tree_png=base.hiv_tree_png,
        leaf_paths_csv=base.leaf_paths_csv,
        hadsbm_tree_json=base.hadsbm_tree_json,
    )

    return run_basic(
        paths,
        skip_fasta=args.skip_fasta,
        no_viz=args.no_viz,
        fasttree_bin=args.fasttree_bin,
        prob_mode=args.prob_mode,
        viz_ascii=args.ascii_tree,
    )


if __name__ == "__main__":
    raise SystemExit(main())
