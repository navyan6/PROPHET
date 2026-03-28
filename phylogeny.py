#!/usr/bin/env python3
"""
Build a phylogenetic tree from the variant FASTA.

Steps:
  1. (Optional) Rebuild hiv_sequences.fasta from JSON using tree.py.
  2. Run MAFFT to align sequences.
  3. Run FastTree on the alignment to get a Newick tree file.

You need MAFFT and FastTree installed (for example: conda install -c bioconda mafft fasttree).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import tree as variant_fasta

from pipeline_paths import PipelinePaths, default_paths


def _candidate_conda_fasttree_paths() -> list[Path]:
    """
    Common install locations for conda/mamba so we can find FastTree
    even when it is not on PATH.
    """
    home = Path.home()
    install_roots = [
        home / "miniconda3",
        home / "anaconda3",
        home / "miniforge3",
        home / "mambaforge",
        home / "micromamba",
    ]
    paths: list[Path] = []
    for root in install_roots:
        paths.append(root / "bin" / "FastTree")
    return paths


def find_fasttree_executable(user_path: str | None) -> str | None:
    """
    Return a string you can pass to subprocess (command name or full path).

    If user_path is set, try that first. Otherwise search PATH, then conda bins.
    """
    if user_path is not None:
        as_path = Path(user_path)
        if as_path.is_file():
            return str(as_path.resolve())
        found = shutil.which(user_path)
        if found is not None:
            return found
        return None

    for command_name in ("FastTree", "fasttree"):
        found = shutil.which(command_name)
        if found is not None:
            return command_name

    for candidate in _candidate_conda_fasttree_paths():
        if candidate.is_file():
            return str(candidate)

    return None


def run_mafft(mafft_command: str, input_fasta: Path, output_aligned: Path) -> None:
    """Call MAFFT and save stdout to output_aligned."""
    with open(output_aligned, "wb") as out_file:
        subprocess.run(
            [mafft_command, "--auto", str(input_fasta)],
            stdout=out_file,
            check=True,
        )


def run_fasttree(
    fasttree_command: str,
    aligned_fasta: Path,
    output_newick: Path,
) -> None:
    """Call FastTree with the LG model (-lg) and save Newick to a file."""
    with open(output_newick, "w", encoding="utf-8") as out_file:
        subprocess.run(
            [fasttree_command, "-lg", str(aligned_fasta)],
            stdout=out_file,
            check=True,
        )


def run_phylogeny(
    paths: PipelinePaths | None = None,
    *,
    skip_fasta: bool = False,
    fasttree_bin: str | None = None,
) -> int:
    """
    Run the full alignment + tree step.

    Returns 0 if everything worked, 1 if a required tool or file is missing.
    """
    if paths is None:
        paths = default_paths()

    mafft_command = shutil.which("mafft")
    if mafft_command is None:
        print(
            "error: mafft not found on PATH. Install e.g. "
            "`brew install mafft` or `conda install -c bioconda mafft`",
            file=sys.stderr,
        )
        return 1

    fasttree_command = find_fasttree_executable(fasttree_bin)
    if fasttree_command is None:
        print(
            "error: FastTree not found. Install e.g. "
            "`conda install -c bioconda fasttree`, or pass --fasttree-bin PATH_TO_FastTree",
            file=sys.stderr,
        )
        return 1

    if not skip_fasta:
        variant_fasta.generate_variants_from_json(
            str(paths.variants_json),
            str(paths.hiv_sequences_fasta),
            wildtype_fasta=paths.wildtype_fasta,
        )

    if not paths.hiv_sequences_fasta.is_file():
        print(
            f"error: missing input FASTA: {paths.hiv_sequences_fasta}",
            file=sys.stderr,
        )
        return 1

    print("Running MAFFT...")
    run_mafft(mafft_command, paths.hiv_sequences_fasta, paths.aligned_fasta)
    print("Wrote alignment:", paths.aligned_fasta)

    print("Running FastTree:", fasttree_command)
    run_fasttree(fasttree_command, paths.aligned_fasta, paths.newick)
    print("Wrote Newick tree:", paths.newick)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MAFFT alignment + FastTree Newick for HIV variant FASTA"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root (default: folder of this script)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="UniProt-style JSON (default: <root>/hiv-variants.json)",
    )
    parser.add_argument("--fasta", type=Path, default=None, help="Unaligned FASTA")
    parser.add_argument("--aligned", type=Path, default=None, help="MAFFT output")
    parser.add_argument("--newick", type=Path, default=None, help="Newick output")
    parser.add_argument(
        "--wt-fasta",
        type=Path,
        default=None,
        help="Optional polyprotein WT FASTA (first record); else JSON sequence",
    )
    parser.add_argument(
        "--skip-fasta",
        action="store_true",
        help="Do not rebuild FASTA from JSON; use existing --fasta",
    )
    parser.add_argument(
        "--fasttree-bin",
        metavar="CMD",
        default=None,
        help="FastTree executable name or path",
    )
    args = parser.parse_args()

    base = default_paths(args.root)
    paths = PipelinePaths(
        root=base.root,
        variants_json=args.json or base.variants_json,
        wildtype_fasta=args.wt_fasta or base.wildtype_fasta,
        hiv_sequences_fasta=args.fasta or base.hiv_sequences_fasta,
        aligned_fasta=args.aligned or base.aligned_fasta,
        newick=args.newick or base.newick,
        hiv_tree_png=base.hiv_tree_png,
        leaf_paths_csv=base.leaf_paths_csv,
        hadsbm_tree_json=base.hadsbm_tree_json,
    )

    return run_phylogeny(
        paths,
        skip_fasta=args.skip_fasta,
        fasttree_bin=args.fasttree_bin,
    )


if __name__ == "__main__":
    raise SystemExit(main())
