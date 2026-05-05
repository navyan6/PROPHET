#!/usr/bin/env python3
"""
scripts/filter_sticky_variants.py
Select Gibbs-sampled variants with the highest PeptiVerse binding scores
("sticky" variants) and write them to a sub-FASTA for targeted Stage 2 runs.

The resulting FASTA can be passed directly as --variants-fasta to stage2.py,
focusing the DFM guidance on variants that are already predicted to bind well.

Usage
-----
  python scripts/filter_sticky_variants.py \
      --variants-fasta results/hiv_stage1/hiv_train_gibbs_variants.fasta \
      --wt-seq PQVTLWQRPLVTIKIGGQL... \
      --top-k 100 \
      --out-fasta results/hiv_stage1/hiv_sticky_variants.fasta \
      --device cuda:0

  # Or filter by absolute score threshold instead of top-K
  python scripts/filter_sticky_variants.py \
      --variants-fasta results/hiv_stage1/hiv_train_gibbs_variants.fasta \
      --wt-seq PQVTLWQRPLVTIKIGGQL... \
      --min-score 8.0 \
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
from prophet.stage2 import AffinityScorer, _load_variants_fasta


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Filter Gibbs variants by PeptiVerse binding score."
    )
    ap.add_argument("--variants-fasta", required=True,
                    help="Stage-1 Gibbs variants FASTA.")
    ap.add_argument("--wt-seq", required=True,
                    help="Wild-type protein sequence (used as PeptiVerse target).")
    ap.add_argument("--top-k", type=int, default=None,
                    help="Keep top-K variants by binding score.")
    ap.add_argument("--min-score", type=float, default=None,
                    help="Keep variants with PeptiVerse score >= this threshold.")
    ap.add_argument("--out-fasta", required=True,
                    help="Output FASTA path for filtered variants.")
    ap.add_argument("--out-scores-json", default=None,
                    help="Optional: save all variant scores to a JSON file.")
    ap.add_argument("--peptiverse-normalization",
                    choices=["minmax", "raw"], default="raw")
    ap.add_argument("--peptiverse-min", type=float, default=7.0)
    ap.add_argument("--peptiverse-max", type=float, default=9.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.top_k is None and args.min_score is None:
        ap.error("Provide at least one of --top-k or --min-score.")

    wt_seq = args.wt_seq.strip().replace("-", "").upper()

    print(f"Loading variants from {args.variants_fasta} ...", file=sys.stderr)
    variants = _load_variants_fasta(args.variants_fasta)
    print(f"  {len(variants)} variants loaded.", file=sys.stderr)

    print(f"Building AffinityScorer on {args.device} ...", file=sys.stderr)
    scorer = AffinityScorer(
        wt_seq,
        mode="peptiverse",
        device=args.device,
        peptiverse_normalization=args.peptiverse_normalization,
        peptiverse_min=args.peptiverse_min,
        peptiverse_max=args.peptiverse_max,
    )

    print("Scoring variants ...", file=sys.stderr)
    scored: list[tuple[str, float]] = []
    for i, v in enumerate(variants):
        s = float(scorer(v))
        scored.append((v, s))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(variants)} scored", file=sys.stderr)

    scored.sort(key=lambda x: x[1], reverse=True)

    # Apply filters
    filtered = scored
    if args.min_score is not None:
        filtered = [(v, s) for v, s in filtered if s >= args.min_score]
        print(f"After min-score={args.min_score}: {len(filtered)} variants.",
              file=sys.stderr)
    if args.top_k is not None:
        filtered = filtered[: args.top_k]
        print(f"After top-k={args.top_k}: {len(filtered)} variants.",
              file=sys.stderr)

    if not filtered:
        print("[warning] No variants passed filters — writing empty FASTA.",
              file=sys.stderr)

    # Write FASTA
    out_path = Path(args.out_fasta)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rank, (seq, score) in enumerate(filtered):
            f.write(f">sticky_variant_{rank:04d}|rank={rank}|score={score:.4f}\n{seq}\n")
    print(f"Wrote {len(filtered)} sticky variants → {out_path}", file=sys.stderr)

    # Optionally dump all scores
    if args.out_scores_json:
        all_scores = [
            {"sequence": v, "score": round(s, 6), "rank": i}
            for i, (v, s) in enumerate(scored)
        ]
        scores_path = Path(args.out_scores_json)
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        with scores_path.open("w") as f:
            json.dump(all_scores, f, indent=2)
        print(f"All scores saved → {scores_path}", file=sys.stderr)

    # Print summary stats
    all_s = np.array([s for _, s in scored])
    filt_s = np.array([s for _, s in filtered])
    print("\n--- Score summary ---", file=sys.stderr)
    print(f"All variants   : n={len(all_s)}, mean={all_s.mean():.3f}, "
          f"max={all_s.max():.3f}, min={all_s.min():.3f}", file=sys.stderr)
    if len(filt_s):
        print(f"Sticky filtered: n={len(filt_s)}, mean={filt_s.mean():.3f}, "
              f"max={filt_s.max():.3f}, min={filt_s.min():.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
