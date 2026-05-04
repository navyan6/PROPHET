#!/usr/bin/env python3
"""
PROPHET Stage 2 – PepTune backend.

Drop-in replacement for stage2.py that uses PepTune's masked discrete
diffusion model (MDLM) + Pareto-MCTS as the peptide generator instead
of MOG-DFM's guided flow-matching sampler.

Generation pipeline
-------------------
1.  Load pretrained MDLM checkpoint (PepTune/checkpoints/peptune-pretrained.ckpt).
2.  Monkey-patch pareto_mcts.ScoringFunctions with a proxy that replaces
    PepTune's built-in binding_affinity1 with PROPHET's CVaR-robustness
    score (computed over the Gibbs-sampled viral variants from Stage 1).
3.  Run MCTS-guided generation targeting the viral WT sequence.
4.  Convert Pareto-front SMILES outputs → standard AA sequences via
    PeptideAnalyzer.
5.  Score each AA sequence with PROPHET's AffinityScorer and emit
    DesignResult objects.

The CLI mirrors stage2.py so downstream scripts (run_ablations.py) can
substitute `stage2_peptune.py` for `stage2.py` without changes.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

# ─── PepTune source path ──────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PEPTUNE_SRC = _REPO_ROOT / "PepTune" / "src"
_PEPTUNE_CKPT_DEFAULT = _REPO_ROOT / "PepTune" / "checkpoints" / "peptune-pretrained.ckpt"

if str(_PEPTUNE_SRC) not in sys.path:
    sys.path.insert(0, str(_PEPTUNE_SRC))

# ─── PROPHET imports ──────────────────────────────────────────────────────────
_PROPHET_DIR = Path(__file__).parent
if str(_PROPHET_DIR) not in sys.path:
    sys.path.insert(1, str(_PROPHET_DIR))

from stage2 import (  # noqa: E402
    AffinityScorer,
    DesignResult,
    cvar_robust_score,
    _load_variants_fasta,
    _resolve_user_path,
    _set_global_seed,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config builder
# ─────────────────────────────────────────────────────────────────────────────

def _ns(**kw) -> SimpleNamespace:
    for k, v in kw.items():
        if isinstance(v, dict):
            kw[k] = _ns(**v)
    return SimpleNamespace(**kw)


def build_peptune_config(
    seq_length: int = 200,
    sampling_steps: int = 128,
    num_iter: int = 128,
    num_children: int = 50,
    num_objectives: int = 5,
) -> SimpleNamespace:
    """Minimal config namespace matching PepTune's YAML structure."""
    return _ns(
        noise=dict(type="loglinear", sigma_min=1e-4, sigma_max=20.0, state_dependent=True),
        backbone="roformer",
        parameterization="subs",
        time_conditioning=False,
        T=0,
        subs_masking=False,
        mcts=dict(
            num_children=num_children,
            num_objectives=num_objectives,
            topk=100,
            mask_token=4,
            num_iter=num_iter,
            sampling=0,
            invalid_penalty=0.5,
            sample_prob=1.0,
            perm=True,
            dual=False,
            single=False,
            time_dependent=True,
        ),
        sampling=dict(
            predictor="ddpm_cache",
            num_sequences=100,
            sampling_eps=1e-3,
            steps=sampling_steps,
            seq_length=seq_length,
            noise_removal=True,
        ),
        training=dict(antithetic_sampling=True, sampling_eps=1e-3),
        eval=dict(gen_ppl_eval_model_name_or_path="gpt2-large"),
        optim=dict(lr=3e-4, weight_decay=0.075, beta1=0.9, beta2=0.999, eps=1e-8),
        roformer=dict(hidden_size=768, n_layers=8, n_heads=8, max_position_embeddings=1035),
        model=dict(
            type="ddit", hidden_size=768, cond_dim=128, length=512,
            n_blocks=12, n_heads=12, scale_by_sigma=True, dropout=0.1,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SMILES → AA conversion helpers
# ─────────────────────────────────────────────────────────────────────────────

_THREE_TO_ONE = {
    "Ala": "A", "Cys": "C", "Asp": "D", "Glu": "E",
    "Phe": "F", "Gly": "G", "His": "H", "Ile": "I",
    "Lys": "K", "Leu": "L", "Met": "M", "Asn": "N",
    "Pro": "P", "Gln": "Q", "Arg": "R", "Ser": "S",
    "Thr": "T", "Val": "V", "Trp": "W", "Tyr": "Y",
}


@contextlib.contextmanager
def _quiet():
    """Suppress stdout/stderr (PeptideAnalyzer is very verbose)."""
    with open(os.devnull, "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def smiles_to_aa(smiles: str, analyzer) -> Optional[str]:
    """
    Convert a SMILES peptide string to a standard one-letter AA sequence.
    Returns None for cyclic, non-peptide, or non-standard-residue sequences.
    """
    try:
        with _quiet():
            three_letter, _length = analyzer.analyze_structure(smiles)
        if not three_letter or three_letter.startswith("cyclo("):
            return None
        residues = three_letter.split("-")
        aa_seq = "".join(_THREE_TO_ONE.get(r.split("(")[0], "X") for r in residues)
        if "X" in aa_seq or not aa_seq:
            return None
        return aa_seq
    except Exception:
        return None


def _prefer_peptune_imports() -> None:
    """Ensure PepTune package names win over similarly named PROPHET modules."""
    peptune_src = str(_PEPTUNE_SRC)
    if peptune_src in sys.path:
        sys.path.remove(peptune_src)
    sys.path.insert(0, peptune_src)

    mod = sys.modules.get("utils")
    mod_file = getattr(mod, "__file__", "") if mod is not None else ""
    if mod is not None and str(_PEPTUNE_SRC) not in str(mod_file):
        sys.modules.pop("utils", None)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring proxy (replaces PepTune's ScoringFunctions inside MCTS)
# ─────────────────────────────────────────────────────────────────────────────

class _ProphetScoringProxy:
    """
    Satisfies PepTune's ScoringFunctions interface.
    Replaces 'binding_affinity1' with PROPHET's CVaR-robustness score;
    all other objectives are delegated to PepTune's native scorers.
    """

    def __init__(
        self,
        score_func_names: list[str],
        prot_seqs: list[str],
        device: str = "cuda:0",
        *,
        aff_scorer: AffinityScorer,
        variants: list[str],
        eta: float,
        guidance_var_limit: Optional[int],
        rng: np.random.Generator,
        peptune_base_path: str,
    ):
        _prefer_peptune_imports()
        os.environ.setdefault("PEPTUNE_BASE_PATH", peptune_base_path)

        from utils.app import PeptideAnalyzer  # noqa: PLC0415
        from scoring.scoring_functions import ScoringFunctions as _NativeSF  # noqa: PLC0415

        self.score_func_names = score_func_names
        self.aff_scorer = aff_scorer
        self.variants = variants
        self.eta = eta
        self.guidance_var_limit = guidance_var_limit
        self.rng = rng
        self.analyzer = PeptideAnalyzer()

        # Load PepTune's native scorers only for non-binding objectives
        self._binding_idx: Optional[int] = None
        non_binding_names = []
        self._non_binding_idxs: list[int] = []
        for i, name in enumerate(score_func_names):
            if name == "binding_affinity1":
                self._binding_idx = i
            else:
                non_binding_names.append(name)
                self._non_binding_idxs.append(i)

        if non_binding_names:
            self._native_sf = _NativeSF(
                score_func_names=non_binding_names,
                prot_seqs=prot_seqs,
                device=device,
            )
        else:
            self._native_sf = None

    def _binding_score(self, aa: Optional[str]) -> float:
        if not aa:
            return 0.0
        vs = self.variants
        if self.guidance_var_limit and len(vs) > self.guidance_var_limit:
            idx = self.rng.choice(len(vs), size=self.guidance_var_limit, replace=False)
            vs = [vs[i] for i in idx]
        try:
            if hasattr(self.aff_scorer, "score_variants_batched"):
                var_scores = self.aff_scorer.score_variants_batched(aa, vs)
            else:
                var_scores = np.array(
                    [self.aff_scorer(aa, v) for v in vs], dtype=np.float64
                )
            return float(cvar_robust_score(var_scores, self.eta))
        except Exception:
            return 0.0

    def forward(self, input_seqs: list[str]) -> np.ndarray:
        """Returns (N, K) float32 score array, one column per objective."""
        N = len(input_seqs)
        K = len(self.score_func_names)
        scores = np.zeros((N, K), dtype=np.float32)

        if self._binding_idx is not None:
            aa_seqs = [smiles_to_aa(s, self.analyzer) for s in input_seqs]
            for j, aa in enumerate(aa_seqs):
                scores[j, self._binding_idx] = self._binding_score(aa)

        if self._native_sf is not None and self._non_binding_idxs:
            native = self._native_sf.forward(input_seqs)  # (N, len(non_binding_names))
            for local_i, global_i in enumerate(self._non_binding_idxs):
                scores[:, global_i] = native[:, local_i]

        return scores

    def __call__(self, input_seqs: list[str]) -> np.ndarray:
        return self.forward(input_seqs)


# ─────────────────────────────────────────────────────────────────────────────
# MDLM loader
# ─────────────────────────────────────────────────────────────────────────────

def load_mdlm(ckpt_path: Path, config: SimpleNamespace, device: str):
    """Load PepTune's pretrained Diffusion model from checkpoint."""
    _prefer_peptune_imports()
    os.environ.setdefault("PEPTUNE_BASE_PATH", str(_REPO_ROOT / "PepTune"))
    from diffusion import Diffusion  # noqa: PLC0415
    from tokenizer.my_tokenizers import SMILES_SPE_Tokenizer  # noqa: PLC0415

    tokenizer = SMILES_SPE_Tokenizer(
        str(_PEPTUNE_SRC / "tokenizer" / "new_vocab.txt"),
        str(_PEPTUNE_SRC / "tokenizer" / "new_splits.txt"),
    )
    mdlm = Diffusion.load_from_checkpoint(
        str(ckpt_path),
        config=config,
        tokenizer=tokenizer,
        strict=False,
        map_location=device,
        weights_only=False,
    )
    mdlm = mdlm.to(device).eval()
    return mdlm


# ─────────────────────────────────────────────────────────────────────────────
# Main design function
# ─────────────────────────────────────────────────────────────────────────────

def peptune_guided_design(
    wt_seq: str,
    eval_variants: list[str],
    aff_scorer: AffinityScorer,
    eta: float,
    tau_bind: float = float("nan"),
    ckpt_path: Path = _PEPTUNE_CKPT_DEFAULT,
    device: str = "cuda:0",
    seq_length: int = 200,
    sampling_steps: int = 128,
    num_iter: int = 128,
    num_children: int = 50,
    guidance_var_limit: Optional[int] = None,
    score_func_names: Optional[list[str]] = None,
    seed: int = 42,
) -> list[DesignResult]:
    """
    Generate peptides with PepTune MCTS, score them with PROPHET's CVaR
    robustness metric, and return a sorted list of DesignResults.

    Parameters
    ----------
    wt_seq            : Wildtype viral target sequence (plain AA string).
    eval_variants     : Gibbs-sampled escape variants from Stage 1.
    aff_scorer        : Loaded AffinityScorer (PeptiVerse).
    eta               : CVaR tail fraction (0–1].
    tau_bind          : Binding-score threshold for retention_score.
    ckpt_path         : PepTune pretrained checkpoint.
    device            : Torch device string.
    seq_length        : SMILES token sequence length for MCTS expansion.
    sampling_steps    : Diffusion denoising steps (MCTS timesteps).
    num_iter          : MCTS iterations.
    num_children      : MCTS branching factor.
    guidance_var_limit: Subsample variants during MCTS scoring (speed-up).
    score_func_names  : PepTune scoring objectives.  Defaults to all 5.
    seed              : RNG seed.
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"PepTune checkpoint not found: {ckpt_path}\n"
            "Download it from "
            "https://drive.google.com/file/d/1oXGDpKLNF0KX0ZdOcl1NZj5Czk2lSFUn"
            " and place it in PepTune/checkpoints/"
        )

    if score_func_names is None:
        score_func_names = [
            "binding_affinity1",
            "solubility",
            "hemolysis",
            "nonfouling",
            "permeability",
        ]

    rng = np.random.default_rng(seed)
    config = build_peptune_config(
        seq_length=seq_length,
        sampling_steps=sampling_steps,
        num_iter=num_iter,
        num_children=num_children,
        num_objectives=len(score_func_names),
    )

    print("[peptune] Loading MDLM checkpoint …", flush=True)
    mdlm = load_mdlm(ckpt_path, config, device)
    print("[peptune] MDLM loaded.", flush=True)

    # ── Inject PROPHET scoring proxy ──────────────────────────────────────────
    proxy_cls = _make_proxy_cls(
        aff_scorer=aff_scorer,
        variants=eval_variants,
        eta=eta,
        guidance_var_limit=guidance_var_limit,
        rng=rng,
        peptune_base_path=str(_REPO_ROOT / "PepTune"),
        device=device,
    )

    import pareto_mcts as _pm  # noqa: PLC0415
    _original_sf = _pm.ScoringFunctions
    _pm.ScoringFunctions = proxy_cls

    try:
        from pareto_mcts import Node, MCTS  # noqa: PLC0415
        from utils.generate_utils import mask_for_de_novo  # noqa: PLC0415

        # Match PepTune's generate_mcts.py initialization path exactly.
        masked_array = mask_for_de_novo(config, seq_length)
        root_tokens = mdlm.tokenizer.encode(masked_array)
        root_tokens = {key: value.to(device) for key, value in root_tokens.items()}
        root_node = Node(
            config=config,
            tokens=root_tokens,
            parentNode=None,
            childNodes=[],
            timestep=0,
        )

        mcts = MCTS(
            config=config,
            max_sequence_length=seq_length,
            mdlm=mdlm,
            score_func_names=score_func_names,
            prot_seqs=[wt_seq],
            num_func=[
                int(num_iter * 0.0),
                int(num_iter * 0.2),
                int(num_iter * 0.4),
                int(num_iter * 0.6),
                int(num_iter * 0.8),
            ],
        )

        print(f"[peptune] Running MCTS ({num_iter} iterations) …", flush=True)
        t0 = time.time()
        pareto_front = mcts.forward(root_node)
        print(
            f"[peptune] MCTS done in {time.time() - t0:.1f}s  "
            f"({len(pareto_front)} Pareto sequences).",
            flush=True,
        )
    finally:
        _pm.ScoringFunctions = _original_sf  # restore

    # ── Post-process Pareto front ──────────────────────────────────────────────
    from utils.app import PeptideAnalyzer  # noqa: PLC0415
    analyzer = PeptideAnalyzer()
    results: list[DesignResult] = []

    for smiles, info in pareto_front.items():
        aa_seq = smiles_to_aa(smiles, analyzer)
        if not aa_seq:
            continue  # cyclic or non-standard residues – skip

        try:
            if hasattr(aff_scorer, "score_variants_batched"):
                var_scores = aff_scorer.score_variants_batched(aa_seq, eval_variants)
            else:
                var_scores = np.array(
                    [aff_scorer(aa_seq, v) for v in eval_variants], dtype=np.float64
                )
            wt_score = float(aff_scorer(aa_seq, wt_seq))
        except Exception as exc:
            print(f"[peptune] scoring failed for {aa_seq[:20]}…: {exc}", flush=True)
            continue

        robust_score = float(cvar_robust_score(var_scores, eta))
        mean_score = float(np.mean(var_scores)) if var_scores.size else float("nan")
        min_score = float(np.min(var_scores)) if var_scores.size else float("nan")
        retention = (
            float(np.mean(var_scores >= tau_bind))
            if (not math.isnan(tau_bind) and var_scores.size)
            else float("nan")
        )

        # omega comes from the Pareto-front score vector (binding_affinity1 weight)
        score_vec = np.array(info["scores"], dtype=float)
        binding_weight = float(score_vec[0]) if score_vec.size else float("nan")

        results.append(DesignResult(
            method="peptune",
            peptide=aa_seq,
            wt_score=wt_score,
            robust_score=robust_score,
            mean_score=mean_score,
            min_score=min_score,
            omega=[binding_weight],
            per_variant=var_scores.tolist(),
        ))

    results.sort(key=lambda r: (r.robust_score, r.wt_score), reverse=True)
    return results


def _make_proxy_cls(
    aff_scorer: AffinityScorer,
    variants: list[str],
    eta: float,
    guidance_var_limit: Optional[int],
    rng: np.random.Generator,
    peptune_base_path: str,
    device: str,
):
    """Return a class (not instance) whose __init__ matches ScoringFunctions."""

    class _Proxy(_ProphetScoringProxy):
        def __init__(self, score_func_names, prot_seqs, device_inner=None):
            super().__init__(
                score_func_names=score_func_names,
                prot_seqs=prot_seqs,
                device=device_inner or device,
                aff_scorer=aff_scorer,
                variants=variants,
                eta=eta,
                guidance_var_limit=guidance_var_limit,
                rng=rng,
                peptune_base_path=peptune_base_path,
            )

    return _Proxy


# ─────────────────────────────────────────────────────────────────────────────
# CLI (mirrors stage2.py for drop-in use)
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="PROPHET Stage 2 (PepTune backend) — escape-robust peptide design"
    )
    # ── shared with stage2.py ─────────────────────────────────────────────────
    p.add_argument("--variants-fasta", required=True,
                   help="FASTA of Gibbs-sampled variants from Stage 1")
    p.add_argument("--wt-seq", required=True,
                   help="Wildtype viral target sequence (AAs, no gaps)")
    p.add_argument("--out-json", default="data/prophet/stage2_peptune_designs.json")
    p.add_argument("--eta", type=float, default=0.1,
                   help="CVaR tail fraction (0,1]; 0.1 = worst 10%% of variants")
    p.add_argument("--tau-bind", type=float, default=float("nan"),
                   help="Binding-score threshold for retention_score metric")
    p.add_argument("--variant-limit", type=int, default=None,
                   help="Subsample this many variants for evaluation (None=all)")
    p.add_argument("--guidance-var-limit", type=int, default=None,
                   help="Subsample this many variants during MCTS guidance (speed-up)")
    p.add_argument("--peptiverse-normalization", choices=["minmax", "raw"], default="raw")
    p.add_argument("--peptiverse-min", type=float, default=7.0)
    p.add_argument("--peptiverse-max", type=float, default=9.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    # ── PepTune-specific ──────────────────────────────────────────────────────
    p.add_argument("--ckpt", default=str(_PEPTUNE_CKPT_DEFAULT),
                   help="Path to PepTune pretrained MDLM checkpoint")
    p.add_argument("--seq-length", type=int, default=200,
                   help="SMILES sequence length used in MCTS expansion")
    p.add_argument("--sampling-steps", type=int, default=128,
                   help="Diffusion denoising / MCTS timesteps")
    p.add_argument("--num-iter", type=int, default=128,
                   help="Number of MCTS iterations")
    p.add_argument("--num-children", type=int, default=50,
                   help="MCTS branching factor")
    p.add_argument("--score-funcs", nargs="+",
                   default=["binding_affinity1", "solubility",
                            "hemolysis", "nonfouling", "permeability"],
                   help="PepTune scoring objectives (binding_affinity1 is PROPHET-replaced)")
    args = p.parse_args()

    _set_global_seed(args.seed)

    variants_path = _resolve_user_path(args.variants_fasta)
    variants = _load_variants_fasta(variants_path, args.variant_limit, args.seed)
    if not variants:
        raise ValueError(f"No variants loaded from {variants_path}.")
    print(f"[peptune] Loaded {len(variants)} variants from {variants_path}", flush=True)

    wt_seq = args.wt_seq.strip().upper().replace("-", "")
    if not wt_seq:
        raise ValueError("--wt-seq is empty after stripping gaps.")
    print(f"[peptune] WT: {wt_seq[:30]}{'...' if len(wt_seq) > 30 else ''} (len={len(wt_seq)})",
          flush=True)

    scorer = AffinityScorer(
        device=args.device,
        peptiverse_normalization=args.peptiverse_normalization,
        peptiverse_min=args.peptiverse_min,
        peptiverse_max=args.peptiverse_max,
    )

    results = peptune_guided_design(
        wt_seq=wt_seq,
        eval_variants=variants,
        aff_scorer=scorer,
        eta=args.eta,
        tau_bind=args.tau_bind,
        ckpt_path=Path(args.ckpt),
        device=args.device,
        seq_length=args.seq_length,
        sampling_steps=args.sampling_steps,
        num_iter=args.num_iter,
        num_children=args.num_children,
        guidance_var_limit=args.guidance_var_limit,
        score_func_names=args.score_funcs,
        seed=args.seed,
    )

    print(f"[peptune] {len(results)} valid AA designs generated.", flush=True)
    if results:
        best = results[0]
        print(
            f"[peptune] Best: peptide={best.peptide[:20]}…  "
            f"robust_score={best.robust_score:.4f}  wt_score={best.wt_score:.4f}",
            flush=True,
        )

    out_path = _resolve_user_path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from dataclasses import asdict
    payload = [asdict(r) for r in results]
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[peptune] Results written → {out_path}", flush=True)


if __name__ == "__main__":
    main()
