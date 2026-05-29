#!/usr/bin/env python3
"""
Re-run Gibbs sampling from already-fitted Stage 1 DCA params.

Useful when Gibbs variants were overwritten or produced 0 variants due to a
too-strict ESM filter, but h/J/lambda/Qi files are still intact.

Usage:
  python scripts/regibbs_from_params.py \
    --params-dir   results/flu_ha_prophet_final \
    --prefix       flu_ha_prophet \
    --fasta        data/flu_ha/alignments/train/flu_ha_train_aligned.fasta \
    --t-evo        5.0 \
    --sample-variants 500 \
    --burn-in      200 \
    --out-fasta    results/flu_ha_prophet_final/flu_ha_prophet_gibbs_variants.fasta \
    --seed         42
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
from Bio import SeqIO

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from prophet.stage1 import (
    AA, AA_TO_IDX, GAP,
    gibbs_sample_variants,
    build_consensus_wt,
    normalize_protein_alignment,
)


def _load_protein_seqs(fasta_path: str) -> dict[str, str]:
    recs = [(r.id, str(r.seq)) for r in SeqIO.parse(fasta_path, "fasta")]
    if not recs:
        raise ValueError(f"Empty FASTA: {fasta_path}")
    aln_len = len(recs[0][1])
    N = len(recs)
    keep = [
        i for i in range(aln_len)
        if sum(1 for _, s in recs if s[i] in "-X*.") / N <= 0.5
    ]
    if len(keep) < aln_len:
        print(f"  Gap filter: {aln_len} → {len(keep)} columns kept")
    seqs = {sid: "".join(s[i] for i in keep) for sid, s in recs}
    return normalize_protein_alignment(seqs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params-dir",      required=True)
    p.add_argument("--prefix",          required=True)
    p.add_argument("--fasta",           required=True)
    p.add_argument("--t-evo",           type=float, default=5.0)
    p.add_argument("--sample-variants", type=int, default=500)
    p.add_argument("--burn-in",         type=int, default=200)
    p.add_argument("--energy-mode",     default="paper_dca")
    p.add_argument("--conserv-weight",  type=float, default=0.05)
    p.add_argument("--esm-filter-delta", type=float, default=None)
    p.add_argument("--esm-model",       default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--esm-device",      default="cpu")
    p.add_argument("--out-fasta",       required=True)
    p.add_argument("--seed",            type=int, default=42)
    args = p.parse_args()

    d = Path(args.params_dir)
    px = args.prefix

    h          = np.load(d / f"{px}_h.npy")
    J          = np.load(d / f"{px}_J.npz")["J"]
    lambda_i   = np.load(d / f"{px}_lambda.npy")
    qi         = np.load(d / f"{px}_Qi.npz")["Qi"]
    conservation = None
    cons_path = d / f"{px}_conservation.npy"
    if cons_path.exists():
        conservation = np.load(cons_path)

    L = h.shape[0]
    print(f"Loaded params: L={L}, h{h.shape}, J{J.shape}, λ{lambda_i.shape}")

    protein_seqs = _load_protein_seqs(args.fasta)
    print(f"Training sequences: {len(protein_seqs)}, length {len(next(iter(protein_seqs.values())))}")

    wt_seq = build_consensus_wt(protein_seqs)
    assert len(wt_seq) == L, f"WT length {len(wt_seq)} ≠ L={L}"
    wt_x = np.array([AA_TO_IDX.get(a, GAP) for a in wt_seq], dtype=np.int8)

    print(f"\nGibbs sampling: M={args.sample_variants}, burn_in={args.burn_in}, T={args.t_evo}")
    if args.esm_filter_delta is not None:
        print(f"  ESM filter: delta={args.esm_filter_delta}, model={args.esm_model}, device={args.esm_device}")
    else:
        print("  ESM filter: none")

    variants, plls = gibbs_sample_variants(
        wt_seq=wt_seq,
        lambda_i=lambda_i,
        qi=qi,
        h=h,
        J=J,
        n_samples=args.sample_variants,
        burn_in=args.burn_in,
        t_evo=args.t_evo,
        energy_mode=args.energy_mode,
        seed=args.seed,
        esm_filter_delta=args.esm_filter_delta,
        esm_model_name=args.esm_model,
        esm_device=args.esm_device,
        conservation=conservation,
        conserv_weight=args.conserv_weight,
        wt_x=wt_x,
    )

    out = Path(args.out_fasta)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for i, (seq, pll) in enumerate(zip(variants, plls)):
            f.write(f">variant_{i}\n{seq}\n")

    edits = [sum(a != b for a, b in zip(v, wt_seq)) for v in variants]
    print(f"\nSaved {len(variants)} variants → {out}")
    if variants:
        print(f"  Edit dist: mean={np.mean(edits):.1f}, std={np.std(edits):.1f}, "
              f"min={min(edits)}, max={max(edits)}")


if __name__ == "__main__":
    main()
