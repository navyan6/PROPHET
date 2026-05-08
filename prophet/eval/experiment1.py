#!/usr/bin/env python3
"""
Experiment 1: Variant Quality Validation (Table 1 from PROPHET paper).

Compares 4 variant generation strategies on HIV-1 protease:
  - PROPHET:      Gibbs samples from p_evo (output of Stage 1)
  - Random mut.:  uniform random mutations matched to PROPHET edit distance
  - ESM-only:     iterative masked prediction with ESM-2 (optional, skip with --no-esm)
  - Held-out:     20% of alignment sequences held out before Stage 1

Metrics:
  (a) Resistance site enrichment  -- fraction of mutations at Stanford DB positions
  (b) ESM pseudo-log-likelihood   -- structural plausibility proxy
  (c) Edit distance to WT         -- mean Hamming distance from wildtype
  (d) DCA energy                  -- E_evo(x) under learned coevo landscape

Usage:
  python prophet/eval/experiment1.py \
    --variants    data/prophet/hiv_gibbs_variants.fasta \
    --alignment   alignments/hiv_protease_aligned.fasta \
    --lambda-npy  data/prophet/hiv_lambda.npy \
    --h-npy       data/prophet/hiv_h.npy \
    --J-npz       data/prophet/hiv_J.npz \
    --resistance  data/hiv_resistance_positions.csv \
    --out-json    data/prophet/experiment1_results.json \
    --no-esm
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from Bio import SeqIO

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA)}
GAP = 20


# ── helpers ──────────────────────────────────────────────────────────────────

def _encode(seq: str) -> np.ndarray:
    return np.array([AA_TO_IDX.get(a, GAP) for a in seq], dtype=np.int8)


def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def _dca_energy(seq: str, lambda_i: np.ndarray, h: np.ndarray, J: np.ndarray) -> float:
    x = _encode(seq)
    L = min(len(x), h.shape[0])
    e = float(np.sum(lambda_i[:L] * h[np.arange(L), x[:L].clip(0, 19)]))
    for i in range(L):
        if x[i] >= 20:
            continue
        for j in range(i + 1, L):
            if x[j] < 20:
                e += float(J[i, j, int(x[i]), int(x[j])])
    return -e


def _resistance_enrichment(variants: list[str], wt: str, res_pos: set[int]) -> float:
    mut_total = mut_on_res = 0
    for seq in variants:
        for i, (a, b) in enumerate(zip(seq, wt)):
            if a != b:
                mut_total += 1
                if i in res_pos:
                    mut_on_res += 1
    return float(mut_on_res / mut_total) if mut_total else 0.0


def _pll_esm2(seqs: list[str], model_name: str, device: str, batch_size: int = 64) -> list[float]:
    """
    Batched ESM-2 pseudo-log-likelihood. Masks position i across ALL sequences
    simultaneously, so total forward passes = L (not N*L).
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    print(f"  [esm-local] loading {model_name} ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()
    aa_ids = {aa: tok.convert_tokens_to_ids(aa) for aa in AA}

    N = len(seqs)
    L = len(seqs[0])
    plls = [0.0] * N

    for pos in range(L):
        masked = [s[:pos] + tok.mask_token + s[pos + 1:] for s in seqs]
        orig_ids = [aa_ids.get(s[pos]) for s in seqs]

        for b_start in range(0, N, batch_size):
            b_end = min(b_start + batch_size, N)
            enc = tok(masked[b_start:b_end], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                logits = model(**enc).logits  # (B, seq_len+2, vocab)
            mask_pos = pos + 1  # +1 for [CLS]
            log_probs = torch.log_softmax(logits[:, mask_pos, :], dim=-1)
            for k, (seq_idx, tid) in enumerate(zip(range(b_start, b_end), orig_ids[b_start:b_end])):
                if tid is not None:
                    plls[seq_idx] += float(log_probs[k, tid].item())

        if (pos + 1) % 10 == 0:
            print(f"  [esm-local] pLL: {pos+1}/{L} positions done")

    return plls


def _pll_esm2_hf(seqs: list[str], client, model: str) -> list[float]:
    """
    Compute ESM-2 pseudo-log-likelihood via HF Inference API fill_mask.
    For each sequence, masks each position, gets P(original_aa | context),
    sums log-probs across all positions.
    """
    import time

    plls = []
    for k, seq in enumerate(seqs):
        total = 0.0
        for i, aa in enumerate(seq):
            masked = seq[:i] + "<mask>" + seq[i + 1:]
            for attempt in range(5):
                try:
                    result = client.fill_mask(masked, model=model)
                    break
                except Exception as e:
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)

            # result is list of {token_str, score, ...}
            score = next(
                (r["score"] for r in result if r["token_str"].strip() == aa),
                1e-9
            )
            total += float(np.log(max(score, 1e-9)))
        plls.append(total)
        if (k + 1) % 10 == 0:
            print(f"  [esm-hf] {k+1}/{len(seqs)} sequences scored")
    return plls


def _make_esm_variants_hf(wt: str, n: int, client, model: str, seed: int) -> list[str]:
    """
    Generate ESM-only variants via HF fill_mask iterative masking.
    For each variant: sweep all positions in random order, mask each,
    sample from the returned probability distribution.
    """
    import time

    rng = random.Random(seed)
    out = []
    for v in range(n):
        seq = list(wt)
        for pos in rng.sample(range(len(wt)), len(wt)):
            masked = "".join(seq[:pos]) + "<mask>" + "".join(seq[pos + 1:])
            for attempt in range(5):
                try:
                    result = client.fill_mask(masked, model=model)
                    break
                except Exception as e:
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)

            # Build prob distribution over standard AAs only
            scores = {r["token_str"].strip(): r["score"] for r in result}
            aa_probs = np.array([scores.get(aa, 1e-9) for aa in AA], dtype=np.float64)
            aa_probs /= aa_probs.sum()
            seq[pos] = AA[int(np.random.default_rng(rng.randint(0, 2**31)).choice(20, p=aa_probs))]

        out.append("".join(seq))
        if (v + 1) % 10 == 0:
            print(f"  [esm-hf] {v+1}/{n} variants generated")
    return out


# ── variant generators ────────────────────────────────────────────────────────

def _make_random_variants(wt: str, n: int, target_edits: float, seed: int) -> list[str]:
    """Uniform random mutations matched to a target mean edit distance."""
    rng = random.Random(seed)
    out = []
    k = max(1, round(target_edits))
    positions = list(range(len(wt)))
    for _ in range(n):
        x = list(wt)
        chosen = rng.sample(positions, min(k, len(positions)))
        for pos in chosen:
            alts = [a for a in AA if a != wt[pos]]
            x[pos] = rng.choice(alts)
        out.append("".join(x))
    return out


def _make_esm_variants(wt: str, n: int, model_name: str, device: str, seed: int, batch_size: int = 64) -> list[str]:
    """
    Batched ESM-only variant generation. All n variants are evolved in parallel:
    at each sweep step, mask the same position in all in-progress variants,
    run one batched forward pass, sample each independently.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    rng = random.Random(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()
    aa_token_ids = [tok.convert_tokens_to_ids(aa) for aa in AA]
    torch_rng = torch.Generator().manual_seed(rng.randint(0, 2**31))

    seqs = [list(wt) for _ in range(n)]
    sweep_order = rng.sample(range(len(wt)), len(wt))

    for step, pos in enumerate(sweep_order):
        masked = ["".join(s[:pos]) + tok.mask_token + "".join(s[pos + 1:]) for s in seqs]
        for b_start in range(0, n, batch_size):
            b_end = min(b_start + batch_size, n)
            enc = tok(masked[b_start:b_end], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                logits = model(**enc).logits  # (B, seq_len+2, vocab)
            mask_pos = pos + 1
            probs = torch.softmax(logits[:, mask_pos, :], dim=-1)
            aa_probs = probs[:, aa_token_ids]  # (B, 20)
            aa_probs = aa_probs / aa_probs.sum(dim=-1, keepdim=True)
            sampled = torch.multinomial(aa_probs, 1, generator=torch_rng).squeeze(1)
            for k, seq_idx in enumerate(range(b_start, b_end)):
                seqs[seq_idx][pos] = AA[int(sampled[k].item())]

        if (step + 1) % 10 == 0:
            print(f"  [esm-local] variants: {step+1}/{len(wt)} positions swept")

    return ["".join(s) for s in seqs]


# ── scoring ───────────────────────────────────────────────────────────────────

def score_variants(
    label: str,
    variants: list[str],
    wt: str,
    lambda_i: np.ndarray,
    h: np.ndarray,
    J: np.ndarray,
    res_pos: set[int],
    pll_vals: list[float] | None = None,
) -> dict:
    enrichment = _resistance_enrichment(variants, wt, res_pos)
    edits = [_hamming(v, wt) for v in variants]
    energies = [_dca_energy(v, lambda_i, h, J) for v in variants]
    mean_pll = float(np.mean(pll_vals)) if pll_vals else None

    print(f"\n  [{label}]")
    print(f"    n={len(variants)}")
    print(f"    Resistance enrichment : {enrichment:.4f}")
    print(f"    Edit dist (mean/std)  : {np.mean(edits):.2f} ± {np.std(edits):.2f}")
    print(f"    DCA energy (mean/std) : {np.mean(energies):.3f} ± {np.std(energies):.3f}")
    if mean_pll is not None:
        print(f"    ESM pLL (mean)        : {mean_pll:.3f}")

    return {
        "label": label,
        "n": len(variants),
        "resistance_enrichment": enrichment,
        "edit_dist_mean": float(np.mean(edits)),
        "edit_dist_std": float(np.std(edits)),
        "dca_energy_mean": float(np.mean(energies)),
        "dca_energy_std": float(np.std(energies)),
        "esm_pll_mean": mean_pll,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="PROPHET Experiment 1: Variant Quality")
    p.add_argument("--variants",    required=True,  help="PROPHET Gibbs variants FASTA (Stage 1 output)")
    p.add_argument("--alignment",   required=True,  help="Full protein alignment FASTA (used for held-out split)")
    p.add_argument("--lambda-npy",  required=True,  help="hiv_lambda.npy from Stage 1")
    p.add_argument("--h-npy",       required=True,  help="hiv_h.npy from Stage 1")
    p.add_argument("--J-npz",       required=True,  help="hiv_J.npz from Stage 1")
    p.add_argument("--resistance",  required=True,  help="CSV of resistance positions")
    p.add_argument("--out-json",    default="data/prophet/experiment1_results.json")
    p.add_argument("--n-random",    type=int, default=500)
    p.add_argument("--held-out-frac", type=float, default=0.2,
                   help="Fraction of alignment to use as held-out leaves")
    p.add_argument("--no-esm",      action="store_true", help="Skip ESM pLL scoring and ESM-only variants")
    p.add_argument("--hf-esm",      action="store_true", help="Use HF Inference API instead of local ESM-2")
    p.add_argument("--hf-token",    default=None, help="HF API token (default: $HF_TOKEN env var)")
    p.add_argument("--esm-model",   default="facebook/esm2_t12_35M_UR50D",
                   help="ESM-2 model name (local or HF API)")
    p.add_argument("--esm-device",  default="cpu", help="Device for local ESM-2 inference")
    p.add_argument("--hf-n-score",  type=int, default=100,
                   help="Number of sequences per method to score with ESM pLL via HF API (default: 100)")
    p.add_argument("--hf-n-variants", type=int, default=100,
                   help="Number of ESM-only variants to generate via HF API (default: 100)")
    p.add_argument("--esm-n-score", type=int, default=20,
                   help="Number of sequences per method to score with local ESM pLL (default: 20)")
    p.add_argument("--esm-n-variants", type=int, default=20,
                   help="Number of ESM-only variants to generate locally (default: 20)")
    p.add_argument("--seed",        type=int, default=42)
    args, _ = p.parse_known_args()

    print("=" * 60)
    print("PROPHET Experiment 1: Variant Quality")
    print("=" * 60)

    # Load Stage 1 outputs
    lambda_i = np.load(args.lambda_npy)
    h        = np.load(args.h_npy)
    J        = np.load(args.J_npz)["J"]
    print(f"Loaded λ ({lambda_i.shape}), h ({h.shape}), J ({J.shape})")

    # Load resistance positions (zero-indexed)
    from prophet.utils.hiv_resistance import load_resistance_positions
    res_pos = load_resistance_positions(args.resistance, one_indexed=True)
    print(f"Loaded {len(res_pos)} resistance positions: {sorted(res_pos)}")

    # Load full alignment, split held-out
    all_seqs = [(r.id, str(r.seq).replace("-","")) for r in SeqIO.parse(args.alignment, "fasta")]
    all_seqs = [(sid, s) for sid, s in all_seqs if len(s) == 99 and "X" not in s]
    rng = random.Random(args.seed)
    rng.shuffle(all_seqs)
    n_held = max(10, int(len(all_seqs) * args.held_out_frac))
    held_out = [s for _, s in all_seqs[:n_held]]
    print(f"Alignment: {len(all_seqs)} seqs → {n_held} held-out, {len(all_seqs)-n_held} train")

    # Build consensus WT from training sequences
    train_seqs = [s for _, s in all_seqs[n_held:]]
    L = 99
    wt = ""
    for i in range(L):
        counts = {}
        for s in train_seqs:
            a = s[i] if i < len(s) else "-"
            if a in AA:
                counts[a] = counts.get(a, 0) + 1
        wt += max(counts, key=counts.get) if counts else "A"
    print(f"Consensus WT: {wt[:30]}...")

    # Load PROPHET variants
    prophet_variants = [str(r.seq) for r in SeqIO.parse(args.variants, "fasta")]
    prophet_variants = [v for v in prophet_variants if len(v) == L and "X" not in v]
    print(f"PROPHET variants: {len(prophet_variants)}")

    # Mean edit distance of PROPHET variants (for matching random)
    prophet_edits = np.mean([_hamming(v, wt) for v in prophet_variants])

    # Generate random variants matched to PROPHET edit distance
    random_variants = _make_random_variants(wt, args.n_random, prophet_edits, args.seed)
    print(f"Random variants generated: {len(random_variants)}")

    # ESM-only variants + pLL scoring (optional)
    esm_variants = None
    prophet_pll = random_pll = held_pll = esm_pll = None

    if not args.no_esm:
        if args.hf_esm:
            import os
            from huggingface_hub import InferenceClient
            token = args.hf_token or os.environ.get("HF_TOKEN")
            if not token:
                raise ValueError("--hf-esm requires HF_TOKEN env var or --hf-token")
            client = InferenceClient(provider="hf-inference", api_key=token)
            n_score = args.hf_n_score
            n_var   = args.hf_n_variants
            model   = args.esm_model

            print(f"Generating {n_var} ESM-only variants via HF API ({model})...")
            esm_variants = _make_esm_variants_hf(wt, n_var, client, model, args.seed)

            print(f"\nScoring ESM pLL via HF API (first {n_score} per method)...")
            prophet_pll = _pll_esm2_hf(prophet_variants[:n_score], client, model)
            random_pll  = _pll_esm2_hf(random_variants[:n_score],  client, model)
            held_pll    = _pll_esm2_hf(held_out[:n_score],          client, model)
            esm_pll     = _pll_esm2_hf(esm_variants[:n_score],      client, model)
        else:
            n_score = args.esm_n_score  # 0 means all
            n_esm_var = args.esm_n_variants
            print(f"Generating {n_esm_var} ESM-only variants locally ({args.esm_model})...")
            esm_variants = _make_esm_variants(
                wt, n_esm_var, args.esm_model, args.esm_device, args.seed
            )
            def _cap(lst): return lst if n_score <= 0 else lst[:n_score]
            print(f"\nScoring ESM pLL locally (batched, {'all' if n_score <= 0 else n_score} per method)...")
            prophet_pll = _pll_esm2(_cap(prophet_variants), args.esm_model, args.esm_device)
            random_pll  = _pll_esm2(_cap(random_variants),  args.esm_model, args.esm_device)
            held_pll    = _pll_esm2(_cap(held_out),          args.esm_model, args.esm_device)
            esm_pll     = _pll_esm2(_cap(esm_variants),      args.esm_model, args.esm_device)

    # Score all groups
    print("\n" + "=" * 60)
    print("Table 1: Variant Quality")
    print("=" * 60)

    results = []
    results.append(score_variants("Held-out leaves", held_out,         wt, lambda_i, h, J, res_pos, held_pll))
    results.append(score_variants("PROPHET",         prophet_variants,  wt, lambda_i, h, J, res_pos, prophet_pll))
    if esm_variants:
        results.append(score_variants("ESM-only",    esm_variants,      wt, lambda_i, h, J, res_pos, esm_pll))
    results.append(score_variants("Random mut.",     random_variants,   wt, lambda_i, h, J, res_pos, random_pll))

    # Print table
    print("\n")
    print(f"{'Source':<18} {'Enr.↑':>8} {'pLL↑':>10} {'Edit':>8} {'DCA↓':>10}")
    print("-" * 58)
    for r in results:
        pll_str = f"{r['esm_pll_mean']:10.2f}" if r["esm_pll_mean"] is not None else f"{'(skip)':>10}"
        print(f"{r['label']:<18} {r['resistance_enrichment']:8.4f} {pll_str} "
              f"{r['edit_dist_mean']:8.2f} {r['dca_energy_mean']:10.3f}")

    # Save
    out_path = REPO_ROOT / args.out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
