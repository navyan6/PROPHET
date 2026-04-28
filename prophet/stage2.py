#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from Bio import SeqIO

try:
    from prophet.common import AA
except ImportError:  # pragma: no cover - script execution fallback
    from common import AA  # type: ignore

REPO_ROOT = Path(__file__).resolve().parent.parent


def cvar_robust_score(scores: np.ndarray, eta: float) -> float:
    """
    Eq. 8-style robust objective: mean over bottom eta mass (worst-case tail).

    NOTE:
    We intentionally take the BOTTOM-k values (worst binders). Maximizing this
    score pushes the weak tail upward, improving robustness against hard variants.
    eta=1.0 reduces to the mean over all variants.
    """
    if scores.size == 0:
        return float("nan")
    eta = float(np.clip(eta, 1e-6, 1.0))
    k = max(1, int(np.ceil(eta * scores.size)))
    return float(np.sort(scores)[:k].mean())


@dataclass
class DesignResult:
    peptide: str
    robust_score: float
    wt_score: float
    blended_score: float
    omega_wt: float
    mean_score: float
    min_score: float
    per_variant: list[float]


class AffinityScorer:
    """
    Callable Aff(peptide, target). Prefers PeptiVerse when configured;
    falls back to a deterministic surrogate for portable experimentation.
    """

    def __init__(self, mode: str = "surrogate", device: str = "cpu"):
        self.mode = mode
        self.device = device
        self._predict = self._surrogate
        if mode == "peptiverse":
            self._try_load_peptiverse()

    def _try_load_peptiverse(self) -> None:
        try:
            import sys
            from pathlib import Path as _Path

            peptiverse = REPO_ROOT / "PeptiVerse"
            if peptiverse.exists():
                sys.path.insert(0, str(peptiverse))
            from inference import WTEmbedder, load_binding_model  # type: ignore

            model_pt = (
                peptiverse / "training_classifiers" / "binding_affinity" / "wt_wt_unpooled" / "best_model.pt"
            )
            embedder = WTEmbedder(device=self.device)
            model = load_binding_model(model_pt, pooled_or_unpooled="unpooled", device=self.device)

            def _peptiverse(peptide: str, target: str) -> float:
                import torch

                T, Mt = embedder.unpooled(target)
                B, Mb = embedder.unpooled(peptide)
                with torch.no_grad():
                    reg, _ = model(T, Mt, B, Mb)
                return float(reg.squeeze().cpu().item())

            self._predict = _peptiverse
        except Exception:
            self._predict = self._surrogate

    @staticmethod
    def _surrogate(peptide: str, target: str) -> float:
        # Heuristic surrogate (not biophysical truth):
        # favors coarse physicochemical complementarity between peptide and target.
        n = min(len(peptide), len(target))
        if n == 0:
            return 0.0
        charge = set("KRH")
        acidic = set("DE")
        hydrophobic = set("AILMFWVY")

        comp = 0.0
        for a, b in zip(peptide[:n], target[:n]):
            if (a in charge and b in acidic) or (a in acidic and b in charge):
                comp += 1.0
            elif a in hydrophobic and b in hydrophobic:
                comp += 0.35
            elif a == b:
                comp += 0.15
            else:
                comp -= 0.25

        length_penalty = 0.03 * abs(len(peptide) - len(target))
        return float(comp / n - length_penalty)

    def __call__(self, peptide: str, target: str) -> float:
        return self._predict(peptide, target)


def mog_dfm_guided_design(
    variants: list[str],
    wt_seq: str,
    aff_fn: Callable[[str, str], float],
    peptide_length: int,
    n_steps: int,
    n_designs: int,
    eta: float,
    beta: float = 6.0,
    temp: float = 0.25,
    seed: int = 42,
) -> list[DesignResult]:
    """
    Surrogate-guided discrete design with two objectives:
      s1(y) = Aff(y, x_WT)
      s2(y) = CVaR_eta over variant affinities
      S_omega(y) = omega * s1(y) + (1-omega) * s2(y)

    This approximates guided transition preference with a Metropolis-style
    acceptance ratio over the blended objective.
    """
    rng = np.random.default_rng(seed)
    results: list[DesignResult] = []

    for _ in range(n_designs):
        # Pareto-style exploration over objective blend weights.
        omega_wt = float(rng.uniform(0.05, 0.95))
        pep = [AA[int(rng.integers(0, len(AA)))] for _ in range(peptide_length)]
        seq = "".join(pep)
        per_variant = np.array([aff_fn(seq, v) for v in variants], dtype=np.float64)
        robust = cvar_robust_score(per_variant, eta)
        wt = float(aff_fn(seq, wt_seq))
        blended = float(omega_wt * wt + (1.0 - omega_wt) * robust)

        for _step in range(n_steps):
            cand = pep.copy()
            i = int(rng.integers(0, peptide_length))
            cand[i] = AA[int(rng.integers(0, len(AA)))]
            cand_seq = "".join(cand)
            cand_per_variant = np.array([aff_fn(cand_seq, v) for v in variants], dtype=np.float64)
            cand_robust = cvar_robust_score(cand_per_variant, eta)
            cand_wt = float(aff_fn(cand_seq, wt_seq))
            cand_blended = float(omega_wt * cand_wt + (1.0 - omega_wt) * cand_robust)

            delta = cand_blended - blended
            accept_prob = min(1.0, float(np.exp(beta * delta / max(temp, 1e-8))))
            if delta >= 0.0 or float(rng.random()) < accept_prob:
                pep = cand
                seq = cand_seq
                per_variant = cand_per_variant
                robust = cand_robust
                wt = cand_wt
                blended = cand_blended

        results.append(
            DesignResult(
                peptide=seq,
                robust_score=float(robust),
                wt_score=float(wt),
                blended_score=float(blended),
                omega_wt=float(omega_wt),
                mean_score=float(np.mean(per_variant)),
                min_score=float(np.min(per_variant)),
                per_variant=[float(x) for x in per_variant],
            )
        )

    results.sort(key=lambda r: r.blended_score, reverse=True)
    return results


def _load_variants_fasta(path: Path, limit: int | None = None, seed: int = 42) -> list[str]:
    out = [str(rec.seq).strip().upper() for rec in SeqIO.parse(str(path), "fasta") if str(rec.seq).strip()]
    if limit is not None and limit < len(out):
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(out), size=limit, replace=False)
        out = [out[int(i)] for i in sorted(idxs)]
    return out

def main() -> None:
    p = argparse.ArgumentParser(description="PROPHET Stage 2: robust peptide design with surrogate-guided multi-objective sampling")
    p.add_argument("--variants-fasta", required=True, help="Input variant FASTA from Stage 1 Gibbs or leaves")
    p.add_argument("--out-json", default="data/prophet/stage2_designs.json")
    p.add_argument("--n-designs", type=int, default=50)
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--peptide-length", type=int, default=12)
    p.add_argument("--eta", type=float, default=0.1, help="CVaR tail fraction in (0,1]")
    p.add_argument("--variant-limit", type=int, default=None)
    p.add_argument("--wt-seq", default=None, help="Optional wildtype sequence for s1(y)=Aff(y, x_WT)")
    p.add_argument("--beta", type=float, default=6.0, help="Acceptance sharpness for guided sampling")
    p.add_argument("--temp", type=float, default=0.25, help="Sampling temperature for guided acceptance")
    p.add_argument("--affinity-mode", choices=["surrogate", "peptiverse"], default="surrogate")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    variants = _load_variants_fasta(REPO_ROOT / args.variants_fasta, args.variant_limit, args.seed)
    if not variants:
        raise ValueError("No variants loaded.")

    wt_seq = str(args.wt_seq).strip().upper() if args.wt_seq else variants[0]

    scorer = AffinityScorer(mode=args.affinity_mode, device=args.device)
    designs = mog_dfm_guided_design(
        variants=variants,
        wt_seq=wt_seq,
        aff_fn=scorer,
        peptide_length=args.peptide_length,
        n_steps=args.n_steps,
        n_designs=args.n_designs,
        eta=args.eta,
        beta=args.beta,
        temp=args.temp,
        seed=args.seed,
    )

    out_path = REPO_ROOT / args.out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [d.__dict__ for d in designs]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {len(designs)} designs -> {out_path}")
    if designs:
        print(
            f"Top design: {designs[0].peptide} | "
            f"blended={designs[0].blended_score:.4f} "
            f"(wt={designs[0].wt_score:.4f}, robust={designs[0].robust_score:.4f}, "
            f"omega={designs[0].omega_wt:.2f})"
        )


if __name__ == "__main__":
    main()

