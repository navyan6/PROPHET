#!/usr/bin/env python3
"""
Batch-render phylogenetic tree files to PNG.

Default behavior:
  - scans the script directory for .nwk/.newick/.tree/.tre files
  - writes one PNG per tree to ./pngs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Bio import Phylo

TREE_EXTENSIONS = (".nwk", ".newick", ".tree", ".tre")


def _iter_tree_files(input_dir: Path, recursive: bool) -> list[Path]:
    walker = input_dir.rglob("*") if recursive else input_dir.glob("*")
    tree_paths = [
        p for p in walker
        if p.is_file() and p.suffix.lower() in TREE_EXTENSIONS
    ]
    return sorted(tree_paths)


def _draw_single_tree(
    tree_path: Path,
    out_path: Path,
    dpi: int,
    show_labels: bool,
    width: float,
    height: float,
) -> None:
    tree = Phylo.read(str(tree_path), "newick")

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(1, 1, 1)
    Phylo.draw(
        tree,
        axes=ax,
        do_show=False,
        show_confidence=False,
        label_func=(lambda clade: clade.name if show_labels else None),
    )
    ax.set_title(tree_path.stem, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    default_input_dir = Path(__file__).resolve().parent

    p = argparse.ArgumentParser(description="Render all tree files in a directory to PNG.")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir,
        help="Directory containing tree files.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=default_input_dir / "pngs",
        help="Directory for rendered PNG files.",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan input directory for tree files.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output PNG resolution.",
    )
    p.add_argument(
        "--width",
        type=float,
        default=14.0,
        help="Figure width in inches.",
    )
    p.add_argument(
        "--height",
        type=float,
        default=8.0,
        help="Figure height in inches.",
    )
    p.add_argument(
        "--no-labels",
        action="store_true",
        help="Hide tip labels (useful for very large trees).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tree_files = _iter_tree_files(input_dir, recursive=args.recursive)
    if not tree_files:
        print(f"[warn] No tree files found in {input_dir}")
        return 0

    print(f"[info] Found {len(tree_files)} tree files in {input_dir}")
    success = 0
    failed = 0
    for idx, tree_path in enumerate(tree_files, start=1):
        out_path = out_dir / f"{tree_path.stem}.png"
        try:
            _draw_single_tree(
                tree_path=tree_path,
                out_path=out_path,
                dpi=args.dpi,
                show_labels=not args.no_labels,
                width=args.width,
                height=args.height,
            )
            success += 1
            print(f"[{idx}/{len(tree_files)}] ok   {tree_path.name} -> {out_path.name}")
        except Exception as exc:
            failed += 1
            print(f"[{idx}/{len(tree_files)}] fail {tree_path.name} ({exc})")

    print(f"[done] rendered={success} failed={failed} output_dir={out_dir}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
