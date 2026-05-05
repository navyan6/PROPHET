#!/usr/bin/env python3
"""
scripts/random_mutations_baseline.py
Generate random peptide sequences and score them with PeptiVerse, then compare
against PROPHET/PepTune designed peptides to verify that evolutionary-guided
designs achieve higher binding affinity than random sequences of the same length.

Three generation modes:
  random_aa   — uniformly sample amino acids (most stringent baseline)
  random_mut  — start from a random variant in the Gibbs FASTA, apply N random
                single-residue substitutions (tests robustness of evolutionary
                variants to local perturbation)
  scramble    — take PROPHET peptides and scramble their residues (same
                composition, different order)

Usage
-----
  python scripts/random_mutations_baseline.py \
      --wt-seq PQVTLWQRPLVTIKIGGQL... \
      --prophet-json results/ablations/t2_prophet.json \
      --variants-fasta results/hiv_stage1/hiv_train_gibbs_variants.fasta \
      --out-json results/ablations/random_mutations_baseline.json \
      --n-random 1000 --n-mutations 3 --device cuda:0

Output JSON has the same schema as stage2.py output so it can be fed into
evaluate_generated_peptides.py for unified comparison.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from Bio import SeqIO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prophet.stage2 import AffinityScorer, cvar_robust_score, _load_variants_fasta

AAS = list("ACDEFGHIKLMNPQRSTVWY")


def _random_peptide(length: int) -> str:
    return "".join(random.choices(AAS, k=length))


def _random_mutate(seq: str, n_mut: int) -> str:
    seq = list(seq)
    positions = random.sample(range(len(seq)), min(n_mut, len(seq)))
    for pos in positions:
        seq[pos] = random.choice(AAS)
    return "".join(seq)


def _scramble(seq: str) -> str:
    chars = list(seq)
    random.shuffle(chars)
    return "".join(chars)


def _score_batch(
    seqs: list[str],
    wt_scorer: AffinityScorer,
    variants: list[str],
    eta: float,
) -> list[dict]:
    results = []
    for seq in seqs:
        wt_score = float(wt_scorer(seq))
        per_variant = np.array([float(wt_scorer(v)) for v in variants[:200]])
        robust = float(cvar_robust_score(per_variant, eta))
        results.append({
            "peptide": seq,
            "wt_score": wt_score,
            "robust_score": robust,
            "mean_score": float(per_variant.mean()),
            "min_score": float(per_variant.min()),
        })
    return results


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Random peptide baseline for binding affinity comparison."
    )
    ap.add_argument("--wt-seq", required=True)
    ap.add_argument("--prophet-json", required=True,
                    help="PROPHET stage2 output JSON for comparison.")
    ap.add_argument("--variants-fasta", required=True,
                    help="Gibbs variants FASTA (used for robust score computation).")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--n-random", type=int, default=1000,
                    help="Number of random sequences per mode.")
    ap.add_argument("--n-mutations", type=int, default=3,
                    help="Mutations per sequence for random_mut mode.")
    ap.add_argument("--eta", type=float, default=0.1,
                    help="CVaR eta for robust score.")
    ap.add_argument("--peptiverse-normalization",
                    choices=["minmax", "raw"], default="raw")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    wt_seq = args.wt_seq.strip().replace("-", "").upper()

    with open(args.prophet_json) as f:
        prophet_designs = json.load(f)
    if not prophet_designs:
        print("[error] Empty prophet JSON.", file=sys.stderr)
        sys.exit(1)

    pep_len = len(prophet_designs[0]["peptide"])
    print(f"Peptide length: {pep_len}", file=sys.stderr)

    print("Loading Gibbs variants ...", file=sys.stderr)
    variants = _load_variants_fasta(args.variants_fasta)
    print(f"  {len(variants)} variants loaded.", file=sys.stderr)

    print(f"Building AffinityScorer on {args.device} ...", file=sys.stderr)
    scorer = AffinityScorer(
        wt_seq,
        mode="peptiverse",
        device=args.device,
        peptiverse_normalization=args.peptiverse_normalization,
    )

    all_results: list[dict] = []

    # ── Mode 1: random_aa ────────────────────────────────────────────────────
    print(f"Generating {args.n_random} random_aa peptides ...", file=sys.stderr)
    ra_seqs = [_random_peptide(pep_len) for _ in range(args.n_random)]
    for i, seq in enumerate(ra_seqs):
        wt_score = float(scorer(seq))
        all_results.append({
            "method": "random_aa",
            "peptide": seq,
            "wt_score": wt_score,
            "robust_score": None,
        })
        if (i + 1) % 100 == 0:
            print(f"  random_aa {i+1}/{args.n_random}", file=sys.stderr)

    # ── Mode 2: random_mut (perturb Gibbs variants) ───────────────────────────
    print(f"Generating {args.n_random} random_mut peptides "
          f"(n_mut={args.n_mutations}) ...", file=sys.stderr)
    for i in range(args.n_random):
        base = random.choice(variants)
        # Trim or pad to pep_len
        if len(base) > pep_len:
            start = random.randint(0, len(base) - pep_len)
            base = base[start: start + pep_len]
        elif len(base) < pep_len:
            base = base + _random_peptide(pep_len - len(base))
        mutated = _random_mutate(base, args.n_mutations)
        wt_score = float(scorer(mutated))
        all_results.append({
            "method": "random_mut",
            "peptide": mutated,
            "wt_score": wt_score,
            "robust_score": None,
        })
        if (i + 1) % 100 == 0:
            print(f"  random_mut {i+1}/{args.n_random}", file=sys.stderr)

    # ── Mode 3: scramble (shuffle PROPHET peptides) ───────────────────────────
    print(f"Generating {min(args.n_random, len(prophet_designs))} "
          "scrambled prophet peptides ...", file=sys.stderr)
    prophet_seqs = [d["peptide"] for d in prophet_designs]
    n_scramble = min(args.n_random, len(prophet_seqs))
    for i in range(n_scramble):
        scrambled = _scramble(prophet_seqs[i % len(prophet_seqs)])
        wt_score = float(scorer(scrambled))
        all_results.append({
            "method": "scramble",
            "peptide": scrambled,
            "wt_score": wt_score,
            "robust_score": None,
        })
        if (i + 1) % 100 == 0:
            print(f"  scramble {i+1}/{n_scramble}", file=sys.stderr)

    # ── PROPHET reference ────────────────────────────────────────────────────
    for d in prophet_designs:
        all_results.append({
            "method": "prophet",
            "peptide": d["peptide"],
            "wt_score": d.get("wt_score"),
            "robust_score": d.get("robust_score"),
        })

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} entries → {out_path}", file=sys.stderr)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n=== Binding score comparison (PeptiVerse) ===", file=sys.stderr)
    for method in ["prophet", "random_aa", "random_mut", "scramble"]:
        scores = [r["wt_score"] for r in all_results
                  if r["method"] == method and r["wt_score"] is not None]
        if scores:
            arr = np.array(scores, dtype=float)
            print(f"  {method:20s}: n={len(arr):4d}  mean={arr.mean():.3f}  "
                  f"median={np.median(arr):.3f}  max={arr.max():.3f}",
                  file=sys.stderr)

    prophet_scores = np.array(
        [r["wt_score"] for r in all_results
         if r["method"] == "prophet" and r["wt_score"] is not None], dtype=float)
    random_scores = np.array(
        [r["wt_score"] for r in all_results
         if r["method"] == "random_aa" and r["wt_score"] is not None], dtype=float)
    if len(prophet_scores) and len(random_scores):
        better = float(np.mean(prophet_scores)) > float(np.mean(random_scores))
        print(f"\nPROPHET mean > random_aa mean: {better}", file=sys.stderr)


if __name__ == "__main__":
    main()
