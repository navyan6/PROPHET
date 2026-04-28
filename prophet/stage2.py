#!/usr/bin/env python3
"""
PROPHET Stage 2: escape-robust peptide design via multi-objective guided sampling.

Implements Algorithm 2 from the paper as a surrogate-guided discrete sampler with:
  - Two-objective structure: s1 = Aff(y, x_WT), s2 = CVaR robustness over variants
  - Pareto weight vector omega blending s1 and s2
  - Rank-normalised ΔS guidance signal (Eq. 2 approximation)
  - Hypercone filtering on the Pareto improvement direction
  - MOG-DFM-style guided transition rates (plug-in: replace _base_logits with a
    real pre-trained flow model's position logits when available)
"""
from __future__ import annotations

import argparse
import json
import math
from types import SimpleNamespace
import warnings
from dataclasses import asdict, dataclass
import torch
from pathlib import Path
from typing import Callable

import numpy as np
from Bio import SeqIO

try:
    from prophet.common import AA
except ImportError:
    AA = "ACDEFGHIKLMNPQRSTVWY"

REPO_ROOT = Path(__file__).resolve().parent.parent
AA_TO_IDX = {aa: i for i, aa in enumerate(AA)}


def _tokens_to_peptide(tokens: list[int] | np.ndarray) -> str:
    seq: list[str] = []
    for tok in tokens:
        t = int(tok)
        # Support both direct AA indices (0..19) and ESM-like peptide ids (4..23).
        if 0 <= t < len(AA):
            seq.append(AA[t])
        elif 4 <= t < 4 + len(AA):
            seq.append(AA[t - 4])
    return "".join(seq)


@dataclass
class DesignResult:
    peptide: str
    wt_score: float
    robust_score: float
    mean_score: float
    min_score: float
    omega: list[float]
    per_variant: list[float]


def cvar_robust_score(scores: np.ndarray, eta: float) -> float:
    """
    Mean over the bottom-eta fraction of variant binding scores.
    """
    if scores.size == 0:
        return float("nan")
    eta = float(np.clip(eta, 1e-6, 1.0))
    k = max(1, int(math.floor(eta * scores.size)))
    return float(np.sort(scores)[:k].mean())


class AffinityScorer:
    """
    Callable wrapper: Aff(peptide, target) -> float in [0, 1].
    """

    def __init__(self, mode: str = "surrogate", device: str = "cpu"):
        self.mode = mode
        self.device = device
        self._predict: Callable[[str, str], float] = self._surrogate
        if mode == "peptiverse":
            self._try_load_peptiverse()

    def _try_load_peptiverse(self) -> None:
        try:
            import sys
            peptiverse = REPO_ROOT / "PeptiVerse"
            if peptiverse.exists():
                sys.path.insert(0, str(peptiverse))
            from inference import WTEmbedder, load_binding_model  # type: ignore

            model_pt = (
                peptiverse
                / "training_classifiers"
                / "binding_affinity"
                / "wt_wt_unpooled"
                / "best_model.pt"
            )
            embedder = WTEmbedder(device=self.device)
            model = load_binding_model(
                model_pt, pooled_or_unpooled="unpooled", device=self.device
            )

            def _peptiverse(peptide: str, target: str) -> float:
                import torch
                T, Mt = embedder.unpooled(target)
                B, Mb = embedder.unpooled(peptide)
                with torch.no_grad():
                    reg, _ = model(T, Mt, B, Mb)
                return float(reg.squeeze().cpu().item())

            self._predict = _peptiverse
            print("  [scorer] PeptiVerse loaded.")
        except Exception as exc:
            warnings.warn(
                f"PeptiVerse unavailable ({exc}); falling back to surrogate scorer.",
                stacklevel=2,
            )
            self._predict = self._surrogate

    @staticmethod
    def _surrogate(peptide: str, target: str) -> float:
        pos = set("KRH")
        neg = set("DE")
        hyd = set("VILMFYW")

        def props(seq: str) -> tuple[int, int, int]:
            return (
                sum(1 for a in seq if a in pos),
                sum(1 for a in seq if a in neg),
                sum(1 for a in seq if a in hyd),
            )

        pp, pn, ph = props(peptide)
        tp, tn, th = props(target)
        n_p = max(len(peptide), 1)
        n_t = max(len(target), 1)

        charge_comp = (
            min(pp / n_p, tn / n_t) + min(pn / n_p, tp / n_t)
        ) / 2.0
        hyd_overlap = min(ph / n_p, th / n_t)
        score = (charge_comp + hyd_overlap) / 2.0
        return float(np.clip(score, 0.0, 1.0))

    def __call__(self, peptide: str, target: str) -> float:
        return self._predict(peptide, target)


def score_peptide_against_variants(
    peptide: str,
    variants: list[str],
    wt_seq: str,
    aff_fn: AffinityScorer,
) -> tuple[float, np.ndarray]:
    wt_score = aff_fn(peptide, wt_seq)
    var_scores = np.array([aff_fn(peptide, v) for v in variants], dtype=np.float64)
    return wt_score, var_scores


def _rank_normalise(values: np.ndarray) -> np.ndarray:
    n = values.size
    if n == 1:
        return np.array([0.5])
    order = np.argsort(values)
    ranks = np.empty(n)
    ranks[order] = np.arange(n)
    return ranks / (n - 1)


# New: Modular score model for CVaR robustness
class RobustnessScoreModel:
    def __init__(self, variants, eta, aff_fn, wt_seq):
        self.variants = variants
        self.eta = eta
        self.aff_fn = aff_fn
        self.wt_seq = wt_seq
    def __call__(self, x, t=None):
        # x: (batch, seq_len) integer tokens
        # Decode to string, score against variants
        device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        seqs = [_tokens_to_peptide(row) for row in x]
        scores = []
        for seq in seqs:
            var_scores = np.array([self.aff_fn(seq, v) for v in self.variants], dtype=np.float64)
            scores.append(cvar_robust_score(var_scores, self.eta))
        return torch.tensor(scores, dtype=torch.float32, device=device)

def mog_dfm_guided_design(
    wt_seq: str,
    variants: list[str],
    aff_fn: AffinityScorer,
    peptide_length: int,
    n_designs: int,
    eta: float,
    dfm_model=None,
    dfm_tokenizer=None,
    dfm_device="cpu",
    **kwargs
) -> list[DesignResult]:
    if dfm_model is None:
        raise ValueError(
            "DFM model is not loaded. Provide a valid --dfm-ckpt and ensure "
            "MOG-DFM dependencies/device are available."
        )

    # Determine valid AA token indices
    valid_aa_tokens = None
    if dfm_tokenizer is not None and hasattr(dfm_tokenizer, "get_vocab"):
        vocab = dfm_tokenizer.get_vocab()
        # Try to find tokens that map to AA
        valid_aa_tokens = [vocab[aa] for aa in AA if aa in vocab]
        if len(valid_aa_tokens) != len(AA):
            # Fallback: use indices 4-23 as before
            valid_aa_tokens = list(range(4, 24))
    else:
        valid_aa_tokens = list(range(4, 24))

    # Build guidance args expected by this MOG-DFM solver implementation.
    guidance_args = SimpleNamespace(
        num_div=64,
        lambda_=float(kwargs.get("delta_alpha", 1.0)),
        beta=float(kwargs.get("beta", 5.0)),
        alpha_r=0.5,
        eta=1.0,
        Phi_init=float(kwargs.get("hypercone_angle", math.radians(45.0))),
        Phi_min=math.radians(15.0),
        Phi_max=math.radians(75.0),
        tau=0.3,
        is_peptide=True,
    )

    # Omega sweep for Pareto front
    n_grid = 10
    weight_grid = [[float(w), 1.0 - float(w)] for w in np.linspace(0, 1, n_grid)]
    all_results = []
    for omega in weight_grid:
        # Prepare initial batch for each omega
        n_samples = n_designs // n_grid if n_grid > 0 else n_designs
        x_init = torch.tensor(
            np.random.choice(valid_aa_tokens, size=(n_samples, peptide_length)),
            dtype=torch.long, device=dfm_device
        )
        zeros = torch.zeros((n_samples, 1), dtype=x_init.dtype, device=dfm_device)
        twos = torch.full((n_samples, 1), 2, dtype=x_init.dtype, device=dfm_device)
        x_init = torch.cat([zeros, x_init, twos], dim=1)

        # Score models: WT affinity and CVaR robustness
        score_models = []
        class WTScoreModel:
            def __init__(self, aff_fn, wt_seq):
                self.aff_fn = aff_fn
                self.wt_seq = wt_seq
            def __call__(self, x, t=None):
                device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
                if isinstance(x, torch.Tensor):
                    x = x.cpu().numpy()
                seqs = [_tokens_to_peptide(row) for row in x]
                return torch.tensor(
                    [self.aff_fn(seq, self.wt_seq) for seq in seqs],
                    dtype=torch.float32,
                    device=device,
                )
        score_models.append(WTScoreModel(aff_fn, wt_seq))
        score_models.append(RobustnessScoreModel(variants, eta, aff_fn, wt_seq))

        # Run DFM sampling for this omega
        x_samples = dfm_model.multi_guidance_sample(
            args=guidance_args,
            x_init=x_init,
            step_size=1/200,
            verbose=True,
            time_grid=torch.tensor([0.0, 1.0-1e-3], device=dfm_device),
            score_models=score_models,
            importance=omega,
        )

        # Decode and score
        decoded = []
        for seq in x_samples.tolist():
            # Use tokenizer if available, else fallback to AA mapping
            if dfm_tokenizer is not None:
                d = dfm_tokenizer.decode(seq, skip_special_tokens=True)
                d = d.replace(" ", "")
            else:
                d = _tokens_to_peptide(seq)
            decoded.append(d)
        for seq in decoded:
            wt_score, var_scores = score_peptide_against_variants(seq, variants, wt_seq, aff_fn)
            robust = cvar_robust_score(var_scores, eta)
            all_results.append(DesignResult(
                peptide=seq,
                wt_score=wt_score,
                robust_score=robust,
                mean_score=float(np.mean(var_scores)),
                min_score=float(np.min(var_scores)),
                omega=omega,
                per_variant=var_scores.tolist(),
            ))
    all_results.sort(key=lambda r: (r.robust_score, r.wt_score), reverse=True)
    return all_results


def _resolve_user_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    candidates = [REPO_ROOT / p, Path.cwd() / p, Path(__file__).resolve().parent / p]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    return (REPO_ROOT / p).resolve()


def _load_variants_fasta(
    path: Path,
    limit: int | None = None,
    seed: int = 42,
) -> list[str]:
    out = [
        str(rec.seq).strip().upper().replace("-", "")
        for rec in SeqIO.parse(str(path), "fasta")
        if str(rec.seq).strip()
    ]
    if limit is not None and limit < len(out):
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(out), size=limit, replace=False)
        return [out[i] for i in sorted(idxs)]
    return out


def main() -> None:
    import sys
    sys.path.insert(0, str(REPO_ROOT / "MOG-DFM"))
    from models.peptide_classifiers import load_solver
    from transformers import AutoTokenizer
    p = argparse.ArgumentParser(
        description="PROPHET Stage 2 — escape-robust peptide design with CVaR guidance"
    )
    p.add_argument(
        "--variants-fasta", required=True,
        help="FASTA of Gibbs-sampled variants from Stage 1"
    )
    p.add_argument(
        "--wt-seq", required=True,
        help="Wildtype viral target sequence (amino acids, no gaps)"
    )
    p.add_argument("--out-json",       default="data/prophet/stage2_designs.json")
    p.add_argument("--n-designs",      type=int,   default=500,
                   help="Total peptides to generate (covers Pareto front)")
    p.add_argument("--n-steps",        type=int,   default=200,
                   help="Guided sampling steps per peptide (Algorithm 2 inner loop)")
    p.add_argument("--peptide-length", type=int,   default=10)
    p.add_argument("--eta",            type=float, default=0.1,
                   help="CVaR tail fraction in (0,1]; 0.1 = worst 10%% of variants")
    p.add_argument("--beta",           type=float, default=5.0,
                   help="Guidance strength scaling ΔS (Eq. 2)")
    p.add_argument("--delta-alpha",    type=float, default=1.0,
                   help="Weight on directional improvement term in composite ΔS")
    p.add_argument("--hypercone-angle",type=float, default=45.0,
                   help="Half-angle of Pareto hypercone filter (degrees)")
    p.add_argument("--omega-grid",     type=int,   default=None,
                   help="If set, sweep omega over this many evenly-spaced grid points")
    p.add_argument("--variant-limit",  type=int,   default=None,
                   help="Randomly subsample this many variants (None = use all)")
    p.add_argument("--affinity-mode",  choices=["surrogate", "peptiverse"],
                   default="surrogate")
    p.add_argument("--device",         default="cpu")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--dfm-ckpt", type=str, default=None, help="Path to DFM model checkpoint (MOG-DFM)")
    p.add_argument("--dfm-device", type=str, default=None, help="Device for DFM model (default: same as --device)")
    args = p.parse_args()

    variants_path = _resolve_user_path(args.variants_fasta)
    variants = _load_variants_fasta(variants_path, args.variant_limit, args.seed)
    if not variants:
        raise ValueError(f"No variants loaded from {variants_path}.")
    print(f"Loaded {len(variants)} variants from {variants_path}")

    wt_seq = args.wt_seq.strip().upper().replace("-", "")
    if not wt_seq:
        raise ValueError("--wt-seq is empty after stripping gaps.")
    print(f"WT sequence: {wt_seq[:30]}{'...' if len(wt_seq) > 30 else ''} (len={len(wt_seq)})")

    scorer = AffinityScorer(mode=args.affinity_mode, device=args.device)
    hypercone_rad = math.radians(args.hypercone_angle)

    # Load DFM model if requested
    dfm_model = None
    dfm_tokenizer = None
    dfm_device = args.dfm_device if args.dfm_device else args.device
    if not args.dfm_ckpt:
        raise ValueError(
            "--dfm-ckpt is required for MOG-DFM sampling. "
            "Pass a valid checkpoint path."
        )

    print(f"Loading DFM model from {args.dfm_ckpt} on device {dfm_device}...")
    try:
        dfm_model = load_solver(args.dfm_ckpt, vocab_size=24, device=dfm_device)
        dfm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        print("✓ DFM model loaded.")
    except Exception as e:
        raise RuntimeError(
            "Failed to load DFM model. Check --dfm-ckpt path, checkpoint "
            "compatibility, and CUDA/device availability."
        ) from e

    print(
        f"\nRunning MOG-DFM guided design: "
        f"n_designs={args.n_designs}, n_steps={args.n_steps}, "
        f"eta={args.eta}, beta={args.beta}"
    )
    designs = mog_dfm_guided_design(
        wt_seq=wt_seq,
        variants=variants,
        aff_fn=scorer,
        peptide_length=args.peptide_length,
        n_steps=args.n_steps,
        n_designs=args.n_designs,
        eta=args.eta,
        beta=args.beta,
        delta_alpha=args.delta_alpha,
        hypercone_angle=hypercone_rad,
        omega_samples=args.omega_grid,
        seed=args.seed,
        dfm_model=dfm_model,
        dfm_tokenizer=dfm_tokenizer,
        dfm_device=dfm_device,
    )

    out_path = _resolve_user_path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(d) for d in designs], f, indent=2)
    print(f"\nSaved {len(designs)} designs -> {out_path}")

    if designs:
        top = designs[0]
        print("Top design (by robust score):")
        print(f"  Peptide : {top.peptide}")
        print(f"  WT aff  : {top.wt_score:.4f}")
        print(f"  Robust  : {top.robust_score:.4f}  (CVaR eta={args.eta})")
        print(f"  Mean    : {top.mean_score:.4f}")
        print(f"  Min     : {top.min_score:.4f}")
        print(f"  omega   : ({top.omega[0]:.2f}, {top.omega[1]:.2f})")

        pareto_wt = [d.wt_score for d in designs]
        pareto_rb = [d.robust_score for d in designs]
        print(
            f"\nPareto front range:"
            f"  WT [{min(pareto_wt):.3f}, {max(pareto_wt):.3f}]"
            f"  Robust [{min(pareto_rb):.3f}, {max(pareto_rb):.3f}]"
        )


if __name__ == "__main__":
    main()

