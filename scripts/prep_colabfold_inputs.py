#!/usr/bin/env python3
"""
scripts/prep_colabfold_inputs.py
Prepare a ColabFold-compatible FASTA of peptide–protein complexes from
PROPHET / PepTune Stage 2 output, then optionally launch ColabFold.

Each entry is formatted as:
  >design_<N>_rb<robust_score>_wt<wt_score>
  <peptide>:<target_protein>

The colon-delimited format tells ColabFold to model a hetero-complex.
The --calc-extra-ptm flag (ColabFold ≥ 1.5) outputs per-chain pTM scores
(ipTM / interface quality), which are the best proxy for binding pose quality
without running expensive free-energy calculations.

Usage
-----
  # 1. Prepare input FASTA (top 20 designs, min PeptiVerse score 7.5)
  python scripts/prep_colabfold_inputs.py \\
      --designs-json results/ablations/t2_prophet.json \\
      --target-seq PQVTLWQRPLVTIKIGGQL... \\
      --out-fasta results/ablations/colabfold_inputs.fasta \\
      --top-n 20 --min-wt-score 7.5

  # 2. Run ColabFold (requires colabfold_batch installed)
  colabfold_batch \\
      results/ablations/colabfold_inputs.fasta \\
      results/ablations/colabfold_out/ \\
      --calc-extra-ptm \\
      --num-recycle 3 \\
      --model-type alphafold2_multimer_v3

  # 3. Parse ipTM scores from ColabFold JSON outputs
  python scripts/prep_colabfold_inputs.py \\
      --parse-results results/ablations/colabfold_out/ \\
      --designs-json results/ablations/t2_prophet.json \\
      --out-json results/ablations/t2_prophet_iptm.json

Note: --calc-extra-ptm is a ColabFold ≥ 1.5 flag that records chain-pair pTM
scores (interface pTM, ipTM) in each prediction's score JSON under the key
"iptm". Higher ipTM → better predicted complex quality.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-.]", "_", s)[:80]


def _prepare_fasta(
    designs: list[dict],
    target_seq: str,
    top_n: int | None,
    min_wt_score: float | None,
    sort_by: str,
    out_fasta: Path,
) -> list[dict]:
    """Filter, sort and write the ColabFold FASTA. Returns selected designs."""
    candidates = list(designs)

    if min_wt_score is not None:
        candidates = [d for d in candidates
                      if (d.get("wt_score") or 0.0) >= min_wt_score]

    # Sort
    rev = sort_by in ("robust_score", "wt_score", "mean_score")
    candidates.sort(key=lambda d: d.get(sort_by) or 0.0, reverse=rev)

    if top_n is not None:
        candidates = candidates[:top_n]

    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    with out_fasta.open("w") as f:
        for i, d in enumerate(candidates):
            pep    = d["peptide"]
            rb     = d.get("robust_score", 0.0) or 0.0
            wt     = d.get("wt_score",    0.0) or 0.0
            method = d.get("method", "design")
            name   = _safe_name(f"{method}_{i:04d}_rb{rb:.2f}_wt{wt:.2f}")
            # ColabFold complex: peptide chain A, target chain B, separated by ":"
            f.write(f">{name}\n{pep}:{target_seq}\n")

    print(f"Wrote {len(candidates)} complexes → {out_fasta}", file=sys.stderr)
    return candidates


def _parse_results(
    results_dir: Path,
    designs: list[dict],
    out_json: Path,
) -> None:
    """
    Parse ColabFold output directory and attach ipTM / pTM scores to designs.
    ColabFold writes one JSON per prediction named <query>_scores_rank_*.json.
    With --calc-extra-ptm the JSON contains:
      - "iptm"          : float  (interface pTM, chain-pair quality)
      - "ptm"           : float  (overall pTM)
      - "plddt"         : list[float]
    """
    # Build peptide → best ipTM mapping
    score_files = sorted(results_dir.rglob("*scores_rank_001*.json"))
    if not score_files:
        score_files = sorted(results_dir.rglob("*scores*.json"))

    iptm_map: dict[str, dict] = {}  # design name prefix → scores
    for sf in score_files:
        try:
            with sf.open() as f:
                data = json.load(f)
        except Exception:
            continue
        iptm = data.get("iptm")
        ptm  = data.get("ptm")
        # Extract plddt mean
        plddt = data.get("plddt")
        plddt_mean = float(sum(plddt) / len(plddt)) if plddt else None

        # The name encoded in the filename: everything before "_scores"
        stem = sf.stem  # e.g. design_0001_rb7.50_wt8.20_scores_rank_001_...
        key  = stem.split("_scores")[0]
        iptm_map[key] = {
            "colabfold_iptm":       iptm,
            "colabfold_ptm":        ptm,
            "colabfold_plddt_mean": plddt_mean,
        }

    # Match back to designs
    augmented = []
    matched = 0
    for i, d in enumerate(designs):
        rb     = d.get("robust_score", 0.0) or 0.0
        wt     = d.get("wt_score",    0.0) or 0.0
        method = d.get("method", "design")
        name   = _safe_name(f"{method}_{i:04d}_rb{rb:.2f}_wt{wt:.2f}")
        entry  = dict(d)
        if name in iptm_map:
            entry.update(iptm_map[name])
            matched += 1
        else:
            entry["colabfold_iptm"]       = None
            entry["colabfold_ptm"]        = None
            entry["colabfold_plddt_mean"] = None
        augmented.append(entry)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(augmented, f, indent=2)
    print(f"Matched {matched}/{len(designs)} designs with ColabFold scores.",
          file=sys.stderr)
    print(f"Saved → {out_json}", file=sys.stderr)

    # Quick summary
    iptms = [d["colabfold_iptm"] for d in augmented
             if d.get("colabfold_iptm") is not None]
    if iptms:
        import statistics
        print(f"\n=== ColabFold ipTM summary (n={len(iptms)}) ===",
              file=sys.stderr)
        print(f"  mean   = {statistics.mean(iptms):.3f}", file=sys.stderr)
        print(f"  median = {statistics.median(iptms):.3f}", file=sys.stderr)
        print(f"  max    = {max(iptms):.3f}", file=sys.stderr)
        print(f"  n(ipTM > 0.6) = {sum(1 for v in iptms if v > 0.6)}/{len(iptms)}",
              file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare ColabFold inputs / parse ColabFold results."
    )
    # ── Prepare mode ──────────────────────────────────────────────────────────
    prep = ap.add_argument_group("Prepare FASTA (default mode)")
    prep.add_argument("--designs-json",
                      help="PROPHET/PepTune output JSON.")
    prep.add_argument("--target-seq",
                      help="Full target protein sequence.")
    prep.add_argument("--out-fasta",
                      help="Output FASTA path for ColabFold.")
    prep.add_argument("--top-n", type=int, default=None,
                      help="Keep top-N designs (default: all).")
    prep.add_argument("--min-wt-score", type=float, default=None,
                      help="Filter by minimum PeptiVerse score.")
    prep.add_argument("--sort-by", default="robust_score",
                      choices=["robust_score", "wt_score", "mean_score"],
                      help="Sort metric for selecting top-N (default: robust_score).")
    # ── Parse mode ────────────────────────────────────────────────────────────
    parse = ap.add_argument_group("Parse ColabFold results")
    parse.add_argument("--parse-results",
                       help="ColabFold output directory to parse.")
    parse.add_argument("--out-json",
                       help="Output JSON path for augmented designs.")

    args = ap.parse_args()

    if args.parse_results:
        # Parse mode
        if not args.designs_json or not args.out_json:
            ap.error("--parse-results requires --designs-json and --out-json")
        with open(args.designs_json) as f:
            designs = json.load(f)
        _parse_results(Path(args.parse_results), designs, Path(args.out_json))
    else:
        # Prepare mode
        if not args.designs_json or not args.target_seq or not args.out_fasta:
            ap.error("Prepare mode requires --designs-json, --target-seq, --out-fasta")
        with open(args.designs_json) as f:
            designs = json.load(f)
        target_seq = args.target_seq.strip().replace("-", "").upper()
        _prepare_fasta(
            designs, target_seq,
            top_n=args.top_n,
            min_wt_score=args.min_wt_score,
            sort_by=args.sort_by,
            out_fasta=Path(args.out_fasta),
        )

        # Print ColabFold run command as a convenience
        out_fasta = Path(args.out_fasta)
        out_dir   = out_fasta.parent / (out_fasta.stem + "_cf_out")
        print(f"\n# Run ColabFold with:", file=sys.stderr)
        print(f"colabfold_batch \\", file=sys.stderr)
        print(f"    {out_fasta} \\", file=sys.stderr)
        print(f"    {out_dir} \\", file=sys.stderr)
        print(f"    --calc-extra-ptm \\", file=sys.stderr)
        print(f"    --num-recycle 3 \\", file=sys.stderr)
        print(f"    --model-type alphafold2_multimer_v3", file=sys.stderr)
        print(f"\n# Then parse results with:", file=sys.stderr)
        augmented_json = out_fasta.parent / (out_fasta.stem + "_iptm.json")
        print(f"python {Path(__file__).name} \\", file=sys.stderr)
        print(f"    --parse-results {out_dir} \\", file=sys.stderr)
        print(f"    --designs-json {args.designs_json} \\", file=sys.stderr)
        print(f"    --out-json {augmented_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
