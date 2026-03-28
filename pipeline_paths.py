"""
Central place for file paths used by the HIV / HadSBM pipeline.

Why this exists: every script (phylogeny, plot, export) needs to read and write
the same files. One dataclass avoids typos like different default names in each
script.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelinePaths:
    """
    Holds absolute paths under one project folder (root).

    frozen=True means: after you create a PipelinePaths object, you cannot
    change its fields (helps catch bugs).
    """

    root: Path
    variants_json: Path
    wildtype_fasta: Path
    hiv_sequences_fasta: Path
    aligned_fasta: Path
    newick: Path
    hiv_tree_png: Path
    leaf_paths_csv: Path
    hadsbm_tree_json: Path


def default_paths(root: Path | None = None) -> PipelinePaths:
    """
    Build a PipelinePaths using standard filenames in the project directory.

    If root is None, we use the folder that contains this file (pipeline_paths.py).
    """
    if root is None:
        project_root = Path(__file__).resolve().parent
    else:
        project_root = root
    project_root = project_root.resolve()

    return PipelinePaths(
        root=project_root,
        variants_json=project_root / "hiv-variants.json",
        wildtype_fasta=project_root / "wildtype.fasta",
        hiv_sequences_fasta=project_root / "hiv_sequences.fasta",
        aligned_fasta=project_root / "hiv_sequences_aligned.fasta",
        newick=project_root / "hiv_tree.nwk",
        hiv_tree_png=project_root / "hiv_tree.png",
        leaf_paths_csv=project_root / "leaf_paths.csv",
        hadsbm_tree_json=project_root / "hadsbm_tree.json",
    )
