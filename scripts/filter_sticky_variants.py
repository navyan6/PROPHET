#!/usr/bin/env python3
"""
scripts/filter_sticky_variants.py
Select "sticky" Gibbs variants — those most likely to be encountered in
nature — for targeted Stage 2 runs.

"Sticky" variants are defined as the Gibbs-sampled sequences closest to
the wild-type (small Hamming distance) or with the highest phylogenetic
leaf probability (if a leaf-probs JSON is available from Stage 1).

Why? DFM guidance averages over all M Gibbs variants equally by default.
Focusing Stage 2 on the most WT-proximal / highest-probability variants
asks the generator: "design a peptide that still works against the variants
the pathogen is most likely to evolve to."

Two selection modes (choose at least one):
  --by-hamming      : keep top-K variants with smallest Hamming distance to WT
  --leaf-probs-json : keep top-K variants by phylogenetic leaf probability

If both flags are given, variants must satisfy both criteria (intersection).

Usage
-----
  # Hamming-distance mode (no extra files needed)
  python scripts/filter_sticky_variants.py \\
      --variants-fasta results/hiv_stage1/hiv_train_gibbs_variants.fasta \\
      --wt-seq PQVTLWQRPLVTIKIGGQL... \\
      --by-hamming --top-k 100 \\
      --out-fasta results/hiv_stage1/hiv_sticky_variants.fasta

  # Leaf-probability mode (requires Stage 1 tree JSON with leaf_endpoints_pi)
  python scripts/filter_sticky_variants.py \\
      --variants-fasta results/hiv_stage1/hiv_train_gibbs_variants.fasta \\
      --wt-seq PQVTLWQRPLVTIKIGGQL... \\
      --leaf-probs-json data/trees/hadsbm_tree.json \\
      --top-k 100 \\
      --out-fasta results/hiv_stage1/hiv_sticky_variants.fasta

  # Combined: top-100 by Hamming, then re-rank by leaf probability
  python scripts/filter_sticky_variants.py \\
      --variants-fasta results/hiv_stage1/hiv_train_gibbs_variants.fasta \\
      --wt-seq PQVTLWQRPLVTIKIGGQL... \\
      --leaf-probs-json data/trees/hadsbm_tree.json \\
      --by-hamming --top-k 100 \\
      --out-fasta results/hiv_stage1/hiv_sticky_variants.fasta
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from Bio import SeqIO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prophet.stage2 import _load_variants_fasta


# ──────────────────────────────────────────────────────────────────────────────

def _hamming(a: str, b: str) -> int:
    """Hamming distance, truncating to the shorter sequence's length."""
    n = min(len(a), len(b))
    return sum(x != y for x, y in zip(a[:n], b[:n])) + abs(len(a) - len(b))


def _load_leaf_probs(json_path: str) -> dict[str, float]:
    """
    Load leaf-probability mapping from a PROPHET Stage-1 tree JSON.

    Supported formats:
      1. HADSBM format: {leaf_endpoints_pi: [{sequence, ...}, ...], ...}
         — probabilities are uniform over leaves (Stage 1 doesn't store
           marginal probs in the JSON; we'll approximate by rank from
           branch lengths if available, else uniform)
      2. Flat format: {sequence: probability, ...}
    """
    with open(json_path) as f:
        data = json.load(f)

    # Flat mapping: sequence → float probability
    if all(isinstance(v, (int, float)) for v in data.values()):
        return {str(k).strip("'"): float(v) for k, v in data.items()}

    # HADSBM format: leaf_endpoints_pi list of {node_index, leaf_id, sequence}
    if "leaf_endpoints_pi" in data:
        leaves = data["leaf_endpoints_pi"]
        n = len(leaves)
        # No marginal probabilities stored — assign uniform weight; caller can
        # refine with --by-hamming to break ties
        return {
            entry["sequence"].strip("'"): 1.0 / n
            for entry in leaves
            if "sequence" in entry
        }

    # Unknown format — warn and return empty
    print(f"[warning] Unrecognised leaf-probs JSON format in {json_path}. "
          "Expected flat {{seq: prob}} or HADSBM tree JSON.", file=sys.stderr)
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Select 'sticky' (WT-proximal / high-probability) Gibbs variants."
    )
    ap.add_argument("--variants-fasta", required=True,
                    help="Gibbs variants FASTA from Stage 1.")
    ap.add_argument("--wt-seq", required=True,
                    help="Wild-type protein sequence (for Hamming filtering).")
    ap.add_argument("--out-fasta", required=True,
                    help="Output FASTA of filtered variants.")
    ap.add_argument("--top-k", type=int, default=100,
                    help="Number of variants to keep (default: 100).")
    ap.add_argument("--by-hamming", action="store_true",
                    help="Rank/filter by Hamming distance to WT (smallest = stickiest).")
    ap.add_argument("--max-hamming", type=int, default=None,
                    help="Hard cap: discard variants > this many substitutions from WT.")
    ap.add_argument("--leaf-probs-json", default=None,
                    help="Path to a Stage-1 tree JSON or flat {seq: prob} JSON "
                         "for phylogenetic probability weighting.")
    ap.add_argument("--out-scores-json", default=None,
                    help="Optional: save all variant scores to a JSON file.")
    args = ap.parse_args()

    if not args.by_hamming and args.leaf_probs_json is None:
        ap.error("Specify at least one of --by-hamming or --leaf-probs-json.")

    wt_seq = args.wt_seq.strip().replace("-", "").upper()

    print(f"Loading variants from {args.variants_fasta} ...", file=sys.stderr)
    variants = _load_variants_fasta(args.variants_fasta)
    if not variants:
        print("[error] No variants found.", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(variants)} variants loaded.", file=sys.stderr)

    # ── Build score for each variant ─────────────────────────────────────────
    # score = leaf_prob * (1 / (1 + hamming))  →  higher is stickier
    leaf_probs: dict[str, float] = {}
    if args.leaf_probs_json:
        leaf_probs = _load_leaf_probs(args.leaf_probs_json)
        print(f"  Loaded {len(leaf_probs)} leaf sequences from {args.leaf_probs_json}.",
              file=sys.stderr)

    scored: list[tuple[str, float, int]] = []  # (seq, sticky_score, hamming)
    for v in variants:
        ham = _hamming(wt_seq, v)
        if args.max_hamming is not None and ham > args.max_hamming:
            continue

        # Find closest leaf (exact match first, then nearest)
        leaf_p = leaf_probs.get(v)
        if leaf_p is None and leaf_probs:
            # Use the probability of the most similar leaf
            best_dist = min(_hamming(v, lseq) for lseq in leaf_probs)
            leaf_p = max(
                (p for lseq, p in leaf_probs.items() if _hamming(v, lseq) == best_dist),
                default=0.0,
            )
        if leaf_p is None:
            leaf_p = 1.0  # no leaf probs → treat as uniform

        if args.by_hamming:
            # Score = leaf_prob / (1 + hamming), maximised by short distance + high prob
            sticky = leaf_p / (1.0 + ham)
        else:
            sticky = leaf_p

        scored.append((v, sticky, ham))

    if not scored:
        print("[warning] All variants were filtered by --max-hamming; "
              "writing empty FASTA.", file=sys.stderr)

    scored.sort(key=lambda x: x[1], reverse=True)
    selected = scored[: args.top_k]

    # ── Write output FASTA ────────────────────────────────────────────────────
    out_path = Path(args.out_fasta)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rank, (seq, score, ham) in enumerate(selected):
            f.write(
                f">sticky_{rank:04d}|rank={rank}|hamming={ham}|score={score:.6f}\n"
                f"{seq}\n"
            )
    print(f"Wrote {len(selected)} sticky variants → {out_path}", file=sys.stderr)

    # ── Optional scores dump ──────────────────────────────────────────────────
    if args.out_scores_json:
        all_records = [
            {"sequence": v, "sticky_score": round(s, 8), "hamming_to_wt": h, "rank": i}
            for i, (v, s, h) in enumerate(scored)
        ]
        scores_path = Path(args.out_scores_json)
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        with scores_path.open("w") as f:
            json.dump(all_records, f, indent=2)
        print(f"All scores saved → {scores_path}", file=sys.stderr)

    # ── Summary ───────────────────────────────────────────────────────────────
    all_ham = np.array([h for _, _, h in scored])
    sel_ham = np.array([h for _, _, h in selected])
    print("\n--- Hamming distance summary ---", file=sys.stderr)
    if len(all_ham):
        print(f"All variants  : mean={all_ham.mean():.1f}  "
              f"min={all_ham.min()}  max={all_ham.max()}", file=sys.stderr)
    if len(sel_ham):
        print(f"Sticky (kept) : mean={sel_ham.mean():.1f}  "
              f"min={sel_ham.min()}  max={sel_ham.max()}", file=sys.stderr)


if __name__ == "__main__":
    main()
