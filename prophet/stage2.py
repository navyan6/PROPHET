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
import time
from types import SimpleNamespace
from dataclasses import asdict, dataclass
import torch
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from Bio import SeqIO

try:
    from prophet.common import AA
except ImportError:
    AA = "ACDEFGHIKLMNPQRSTVWY"

REPO_ROOT = Path(__file__).resolve().parent.parent
AA_TO_IDX = {aa: i for i, aa in enumerate(AA)}


def diagnose_scoring_speed(
    scorer: "AffinityScorer",
    wt_seq: str,
    variants: list[str],
    n_probe: int = 5,
) -> None:
    """Print a timing breakdown for the PeptiVerse scorer.

    Run this before the main sampling loop to understand where time goes.
    Pass --guidance-var-limit K to use only K variants during DFM guidance.
    """
    import time as _time

    probe_peptide = "ACDEFGHIKL"[:10]  # dummy 10-AA peptide

    # 1. Single-pair call (baseline)
    t0 = _time.perf_counter()
    for _ in range(n_probe):
        scorer(probe_peptide, wt_seq)
    t_single = (_time.perf_counter() - t0) / n_probe

    # 2. Sequential variant loop (old path)
    t0 = _time.perf_counter()
    for _ in range(n_probe):
        _ = [scorer(probe_peptide, v) for v in variants[:50]]
    t_seq_50 = (_time.perf_counter() - t0) / n_probe

    # 3. Batched variant scoring (new path)
    if hasattr(scorer, "score_variants_batched"):
        t0 = _time.perf_counter()
        for _ in range(n_probe):
            scorer.score_variants_batched(probe_peptide, variants[:50])
        t_batch_50 = (_time.perf_counter() - t0) / n_probe
    else:
        t_batch_50 = float("nan")

    V = len(variants)
    steps = 200
    seqs_per_step = 24  # B * (vocab_size-1) + B ≈ 24 for B=1
    print(
        f"\n[timing] PeptiVerse scoring diagnosis"
        f"\n  single call              : {t_single*1000:.2f} ms"
        f"\n  sequential 50 variants   : {t_seq_50*1000:.1f} ms"
        f"\n  batched    50 variants   : {t_batch_50*1000:.1f} ms"
        f"\n  speedup (seq→batch)      : {t_seq_50/t_batch_50:.1f}x"
        f"\n"
        f"\n  Projected total (no opts): {t_single*steps*seqs_per_step*V/60:.0f} min  "
        f"({steps} steps × {seqs_per_step} seqs/step × {V} variants)"
        f"\n  With batching only       : {t_batch_50/50*V*steps*seqs_per_step/60:.0f} min"
        f"\n  With batch + 50-var limit: {t_batch_50*steps*seqs_per_step/60:.0f} min",
        flush=True,
    )


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


def _decode_tokens_to_peptide(tokens: list[int] | np.ndarray, tokenizer=None) -> str:
    if tokenizer is not None:
        token_list = [int(tok) for tok in tokens]
        return tokenizer.decode(token_list, skip_special_tokens=True).replace(" ", "")
    return _tokens_to_peptide(tokens)


def _set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class DesignResult:
    method: str
    peptide: str
    wt_score: float
    robust_score: float
    mean_score: float
    min_score: float
    omega: list[float]
    per_variant: list[float]
    retention_score: float = float("nan")


def cvar_robust_score(scores: np.ndarray, eta: float) -> float:
    """
    Mean over the bottom-eta fraction of variant binding scores.
    """
    if scores.size == 0:
        return float("nan")
    eta = float(np.clip(eta, 1e-6, 1.0))
    k = max(1, int(math.floor(eta * scores.size)))
    return float(np.sort(scores)[:k].mean())



# PeptiVerse-only AffinityScorer
class AffinityScorer:
    def __init__(
        self,
        device: str = "cuda:0",
        peptiverse_normalization: Literal["minmax", "raw"] = "raw",
        peptiverse_min: float = 7.0,
        peptiverse_max: float = 9.0,
        eval_batch_size: int = 64,
    ):
        self.device = device
        self.peptiverse_normalization = peptiverse_normalization
        self.peptiverse_min = float(peptiverse_min)
        self.peptiverse_max = float(peptiverse_max)
        self.eval_batch_size = eval_batch_size
        if self.peptiverse_max <= self.peptiverse_min:
            raise ValueError("--peptiverse-max must be greater than --peptiverse-min")
        # Set by _try_load_peptiverse for batched scoring
        self._raw_model = None
        self._embed_fn = None
        self._try_load_peptiverse()

    def _normalize_peptiverse_score(self, raw_score: float) -> float:
        if self.peptiverse_normalization == "raw":
            return raw_score
        if self.peptiverse_normalization != "minmax":
            raise ValueError(
                f"Unknown PeptiVerse normalization: {self.peptiverse_normalization}"
            )
        score = (raw_score - self.peptiverse_min) / (
            self.peptiverse_max - self.peptiverse_min
        )
        return float(np.clip(score, 0.0, 1.0))

    def _normalize_array(self, scores: np.ndarray) -> np.ndarray:
        if self.peptiverse_normalization == "raw":
            return scores
        return np.clip(
            (scores - self.peptiverse_min) / (self.peptiverse_max - self.peptiverse_min),
            0.0, 1.0,
        )

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
            embedding_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

            def _embed(seq: str) -> tuple[torch.Tensor, torch.Tensor]:
                cached = embedding_cache.get(seq)
                if cached is None:
                    cached = embedder.unpooled(seq)
                    embedding_cache[seq] = cached
                return cached

            def _peptiverse(peptide: str, target: str) -> float:
                T, Mt = _embed(target)
                B_emb, Mb = _embed(peptide)
                with torch.no_grad():
                    reg, _ = model(T, Mt, B_emb, Mb)
                raw_score = float(reg.squeeze().cpu().item())
                return self._normalize_peptiverse_score(raw_score)

            self._predict = _peptiverse
            self._raw_model = model
            self._embed_fn = _embed
            print(
                "  [scorer] PeptiVerse loaded "
                f"(normalization={self.peptiverse_normalization}, "
                f"min={self.peptiverse_min:g}, max={self.peptiverse_max:g})."
                f" Embeddings will be cached per unique sequence.",
                flush=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "PeptiVerse scoring was requested but could not be loaded. "
                "Install/fix PeptiVerse dependencies and checkpoint paths."
            ) from exc

    def score_variants_batched(self, peptide: str, variants: list[str]) -> np.ndarray:
        """Score one peptide against all variants in batched forward passes.

        Embeds the peptide once, then groups variants into chunks of
        eval_batch_size and runs one model(T_batch, Mt_batch, B_exp, Mb_exp)
        call per chunk instead of one call per variant.  Falls back to the
        sequential path if the model can't handle batched input (e.g. mixed
        sequence lengths that would require padding the embeddings).
        """
        if self._raw_model is None or self._embed_fn is None:
            return np.array([self._predict(peptide, v) for v in variants], dtype=np.float64)

        B_emb, Mb_emb = self._embed_fn(peptide)
        all_scores: list[np.ndarray] = []

        for start in range(0, len(variants), self.eval_batch_size):
            chunk = variants[start : start + self.eval_batch_size]
            try:
                var_embeds = [self._embed_fn(v) for v in chunk]
                # Stack variant embeddings: (chunk_size, L_t, d) and (chunk_size, L_t)
                T_batch  = torch.cat([e[0] for e in var_embeds], dim=0)
                Mt_batch = torch.cat([e[1] for e in var_embeds], dim=0)
                V = T_batch.shape[0]
                # Expand peptide embedding to match chunk size
                B_exp  = B_emb.expand(V, *B_emb.shape[1:])   if B_emb.dim()  >= 2 else B_emb.unsqueeze(0).expand(V, -1)
                Mb_exp = Mb_emb.expand(V, *Mb_emb.shape[1:]) if Mb_emb.dim() >= 2 else Mb_emb.unsqueeze(0).expand(V, -1)
                with torch.no_grad():
                    reg, _ = self._raw_model(T_batch, Mt_batch, B_exp, Mb_exp)
                chunk_scores = np.atleast_1d(reg.squeeze(-1).cpu().float().numpy())
            except Exception:
                # Fallback for variable-length variants or unexpected model API
                chunk_scores = np.atleast_1d(np.array(
                    [self._predict(peptide, v) for v in chunk], dtype=np.float64
                ))
                all_scores.append(chunk_scores)
                continue
            all_scores.append(np.atleast_1d(self._normalize_array(chunk_scores)))

        return np.concatenate(all_scores) if all_scores else np.array([], dtype=np.float64)

    def __call__(self, peptide: str, target: str) -> float:
        return self._predict(peptide, target)


def score_peptide_against_variants(
    peptide: str,
    variants: list[str],
    wt_seq: str,
    aff_fn: AffinityScorer,
) -> tuple[float, np.ndarray]:
    if hasattr(aff_fn, "score_variants_batched"):
        wt_score = float(aff_fn.score_variants_batched(peptide, [wt_seq])[0])
        var_scores = aff_fn.score_variants_batched(peptide, variants)
    else:
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
    def __init__(
        self,
        variants,
        eta,
        aff_fn,
        wt_seq,
        tokenizer=None,
        guidance_var_limit: int | None = None,
        seed: int = 42,
    ):
        self.variants = variants
        self.eta = eta
        self.aff_fn = aff_fn
        self.wt_seq = wt_seq
        self.tokenizer = tokenizer
        self.guidance_var_limit = guidance_var_limit
        self._rng = np.random.default_rng(seed)

    def _score_seq(self, seq: str, variants: list[str]) -> float:
        if hasattr(self.aff_fn, "score_variants_batched"):
            var_scores = self.aff_fn.score_variants_batched(seq, variants)
        else:
            var_scores = np.array([self.aff_fn(seq, v) for v in variants], dtype=np.float64)
        return cvar_robust_score(var_scores, self.eta)

    def __call__(self, x, t=None):
        device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        seqs = [_decode_tokens_to_peptide(row, self.tokenizer) for row in x]
        # Subsample variants for cheaper guidance scoring (if configured)
        variants = self.variants
        if self.guidance_var_limit is not None and self.guidance_var_limit < len(variants):
            idxs = self._rng.choice(len(variants), size=self.guidance_var_limit, replace=False)
            variants = [variants[i] for i in sorted(idxs)]
        scores = [self._score_seq(seq, variants) for seq in seqs]
        return torch.tensor(scores, dtype=torch.float32, device=device)


class WTScoreModel:
    def __init__(self, aff_fn, wt_seq, tokenizer=None):
        self.aff_fn = aff_fn
        self.wt_seq = wt_seq
        self.tokenizer = tokenizer

    def __call__(self, x, t=None):
        device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        seqs = [_decode_tokens_to_peptide(row, self.tokenizer) for row in x]
        # Batch score all seqs against wt_seq in one call when possible
        if hasattr(self.aff_fn, "score_variants_batched"):
            scores = [float(self.aff_fn.score_variants_batched(seq, [self.wt_seq])[0]) for seq in seqs]
        else:
            scores = [self.aff_fn(seq, self.wt_seq) for seq in seqs]
        return torch.tensor(scores, dtype=torch.float32, device=device)


def mog_dfm_guided_design(
    wt_seq: str,
    eval_variants: list[str],
    aff_fn: AffinityScorer,
    peptide_length: int,
    n_designs: int,
    eta: float,
    design_mode: str = "prophet",
    guidance_variants: list[str] | None = None,
    dfm_model=None,
    dfm_tokenizer=None,
    dfm_device="cuda:0",
    tau_bind: float | None = None,
    guidance_var_limit: int | None = None,
    **kwargs
) -> list[DesignResult]:
    if dfm_model is None:
        raise ValueError(
            "DFM model is not loaded. Provide a valid --dfm-ckpt and ensure "
            "MOG-DFM dependencies/device are available."
        )
    seed = kwargs.get("seed")
    rng = np.random.default_rng(seed)

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

    if guidance_variants is None:
        guidance_variants = eval_variants

    # Omega sweep for Pareto front
    # fixed_omega: if provided, skip the sweep and use this single weight vector
    fixed_omega = kwargs.get("fixed_omega")
    if fixed_omega is not None:
        weight_grid = [[float(fixed_omega[0]), float(fixed_omega[1])]]
        n_grid = 1
    else:
        n_grid = int(kwargs.get("omega_samples") or 10)
        n_grid = max(1, n_grid)
        weight_grid = [[float(w), 1.0 - float(w)] for w in np.linspace(0, 1, n_grid)]
    all_results = []
    produced = 0
    total_started = 0
    run_start = time.time()
    verbose = bool(kwargs.get("verbose", False))
    for grid_idx, omega in enumerate(weight_grid):
        # Prepare initial batch for each omega
        remaining = n_designs - produced
        if remaining <= 0:
            break
        slots_left = n_grid - grid_idx
        n_samples = max(1, math.ceil(remaining / slots_left))
        produced += n_samples
        total_started += n_samples
        omega_start = time.time()
        print(
            "[progress] "
            f"omega {grid_idx + 1}/{n_grid} "
            f"weights=({omega[0]:.3f},{omega[1]:.3f}) "
            f"batch={n_samples} "
            f"started={min(total_started, n_designs)}/{n_designs} "
            f"elapsed={time.time() - run_start:.1f}s",
            flush=True,
        )
        x_init = torch.tensor(
            rng.choice(valid_aa_tokens, size=(n_samples, peptide_length)),
            dtype=torch.long, device=dfm_device
        )
        zeros = torch.zeros((n_samples, 1), dtype=x_init.dtype, device=dfm_device)
        twos = torch.full((n_samples, 1), 2, dtype=x_init.dtype, device=dfm_device)
        x_init = torch.cat([zeros, x_init, twos], dim=1)

        if design_mode == "wt_only":
            score_models = [WTScoreModel(aff_fn, wt_seq, tokenizer=dfm_tokenizer)]
            importance = [1.0]
            result_omega = [1.0, 0.0]
            guidance_weight = [1.0]
        else:
            score_models = [
                WTScoreModel(aff_fn, wt_seq, tokenizer=dfm_tokenizer),
                RobustnessScoreModel(
                    guidance_variants, eta, aff_fn, wt_seq,
                    tokenizer=dfm_tokenizer,
                    guidance_var_limit=guidance_var_limit,
                    seed=kwargs.get("seed") or 42,
                ),
            ]
            importance = omega
            result_omega = omega
            guidance_weight = omega

        # Run DFM sampling for this omega
        print(
            "[progress] "
            f"sampling omega {grid_idx + 1}/{n_grid} "
            f"steps={int(kwargs.get('n_steps', 200))} "
            f"device={dfm_device}",
            flush=True,
        )
        x_samples = dfm_model.multi_guidance_sample(
            args=guidance_args,
            x_init=x_init,
            step_size=(1.0 - 1e-3) / (max(int(kwargs.get("n_steps", 200)), 1) + 1e-6),
            verbose=verbose,
            time_grid=torch.tensor([0.0, 1.0-1e-3], device=dfm_device),
            score_models=score_models,
            importance=importance,
            guidance_weight=guidance_weight,
        )
        print(
            "[progress] "
            f"scoring omega {grid_idx + 1}/{n_grid} "
            f"sampled={len(x_samples)}",
            flush=True,
        )

        # Decode and score
        decoded = []
        for seq in x_samples.tolist():
            # Use tokenizer if available, else fallback to AA mapping
            if dfm_tokenizer is not None:
                d = dfm_tokenizer.decode(seq, skip_special_tokens=True)
                d = d.replace(" ", "")
            else:
                d = _decode_tokens_to_peptide(seq)
            decoded.append(d)
        for seq in decoded:
            wt_score, var_scores = score_peptide_against_variants(seq, eval_variants, wt_seq, aff_fn)
            robust = cvar_robust_score(var_scores, eta)
            if tau_bind is not None and var_scores.size > 0:
                ret = float(np.mean(var_scores >= tau_bind))
            else:
                ret = float("nan")
            all_results.append(DesignResult(
                method=design_mode,
                peptide=seq,
                wt_score=wt_score,
                robust_score=robust,
                mean_score=float(np.mean(var_scores)),
                min_score=float(np.min(var_scores)),
                omega=result_omega,
                per_variant=var_scores.tolist(),
                retention_score=ret,
            ))
        print(
            "[progress] "
            f"finished omega {grid_idx + 1}/{n_grid} "
            f"total_scored={len(all_results)}/{n_designs} "
            f"omega_elapsed={time.time() - omega_start:.1f}s "
            f"total_elapsed={time.time() - run_start:.1f}s",
            flush=True,
        )
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


def _hamming(a: str, b: str) -> int:
    n = min(len(a), len(b))
    return sum(x != y for x, y in zip(a[:n], b[:n])) + abs(len(a) - len(b))


def _random_variants_matched_to_edit_distance(
    wt_seq: str,
    reference_variants: list[str],
    seed: int,
) -> list[str]:
    rng = np.random.default_rng(seed)
    wt_arr = np.array(list(wt_seq))
    out: list[str] = []
    for ref in reference_variants:
        k = min(_hamming(wt_seq, ref), len(wt_seq))
        arr = wt_arr.copy()
        if k > 0:
            positions = rng.choice(len(wt_seq), size=k, replace=False)
            for pos in positions:
                choices = [aa for aa in AA if aa != arr[pos]]
                arr[pos] = rng.choice(choices)
        out.append("".join(arr.tolist()))
    return out


def _esm_variants_matched_to_edit_distance(
    wt_seq: str,
    reference_variants: list[str],
    seed: int,
    model_name: str,
    device: str,
    temperature: float = 1.0,
) -> list[str]:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    rng = np.random.default_rng(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()

    aa_token_ids = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in AA}
    if any(tok is None or tok < 0 for tok in aa_token_ids.values()):
        raise ValueError(f"Tokenizer for {model_name} does not expose all amino-acid tokens.")
    if tokenizer.mask_token is None or tokenizer.mask_token_id is None:
        raise ValueError(f"Tokenizer for {model_name} does not define a mask token.")

    out: list[str] = []
    for ref in reference_variants:
        k = min(_hamming(wt_seq, ref), len(wt_seq))
        seq = list(wt_seq)
        if k > 0:
            positions = rng.choice(len(wt_seq), size=k, replace=False)
            for pos in positions:
                masked = "".join(seq[:pos]) + tokenizer.mask_token + "".join(seq[pos + 1 :])
                enc = tokenizer(masked, return_tensors="pt")
                input_ids = enc["input_ids"].to(device)
                attn = enc.get("attention_mask")
                if attn is not None:
                    attn = attn.to(device)
                with torch.no_grad():
                    logits = model(input_ids=input_ids, attention_mask=attn).logits[0]
                mask_pos = (input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
                if len(mask_pos) == 0:
                    continue
                aa_logits = torch.tensor(
                    [float(logits[int(mask_pos[0].item()), aa_token_ids[aa]].item()) for aa in AA],
                    dtype=torch.float64,
                )
                current_idx = AA.find(seq[pos])
                if current_idx >= 0:
                    aa_logits[current_idx] = float("-inf")
                if temperature <= 0:
                    next_idx = int(torch.argmax(aa_logits).item())
                else:
                    probs = torch.softmax(aa_logits / float(temperature), dim=0).cpu().numpy()
                    next_idx = int(rng.choice(len(AA), p=probs))
                seq[pos] = AA[next_idx]
        out.append("".join(seq))
    return out


def _write_fasta(path: Path, seqs: list[str], prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, seq in enumerate(seqs, start=1):
            f.write(f">{prefix}_{i}\n{seq}\n")


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
    p.add_argument("--guidance-variants-fasta", default=None,
                   help="Optional FASTA for guidance variants; defaults to --variants-fasta")
    p.add_argument("--edit-distance-reference-fasta", default=None,
                   help="FASTA whose WT edit-distance distribution is used by matched random/ESM baselines")
    p.add_argument("--guidance-out-fasta", default=None,
                   help="Optional path to save the final guidance variants used by the run")
    p.add_argument("--design-mode", choices=["prophet", "wt_only", "uniform_leaves", "random_variants", "esm_only_variants"],
                   default="prophet",
                   help="Design objective to run. uniform_leaves uses guidance variants as a leaf baseline.")
    p.add_argument("--esm-variant-model", default="facebook/esm2_t6_8M_UR50D",
                   help="Masked-LM model used to make ESM-only matched guidance variants")
    p.add_argument("--esm-variant-device", default=None,
                   help="Device for ESM-only variant generation (default: same as --device)")
    p.add_argument("--esm-variant-temperature", type=float, default=1.0,
                   help="Sampling temperature for ESM-only variant mutations; <=0 uses argmax")
    # Only PeptiVerse mode is supported now
    p.add_argument("--peptiverse-normalization", choices=["minmax", "raw"],
                   default="raw",
                   help=(
                       "PeptiVerse score normalization. minmax maps raw regression "
                       "scores to [0,1] using --peptiverse-min/--peptiverse-max; "
                       "raw uses the regression score directly."
                   ))
    p.add_argument("--peptiverse-min", type=float, default=7.0,
                   help="Raw PeptiVerse score mapped to 0.0 when using minmax normalization.")
    p.add_argument("--peptiverse-max", type=float, default=9.0,
                   help="Raw PeptiVerse score mapped to 1.0 when using minmax normalization.")
    p.add_argument("--device",         default="cuda:0")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--dfm-ckpt", type=str, default=None, help="Path to DFM model checkpoint (MOG-DFM)")
    p.add_argument("--dfm-device", type=str, default=None, help="Device for DFM model (default: same as --device)")
    p.add_argument("--verbose-sampling", action="store_true",
                   help="Enable tqdm progress inside MOG-DFM sampling")
    p.add_argument("--tau-bind", type=float, default=None,
                   help="Binding score threshold for retention metric (fraction of variants >= tau). "
                        "Used for Tables 2/4/5/6/7 'Ret.' column. If not set, retention_score=nan.")
    p.add_argument("--diagnose-speed", action="store_true",
                   help="Print a PeptiVerse timing breakdown before sampling and exit.")
    p.add_argument("--guidance-var-limit", type=int, default=None,
                   help="Subsample this many variants during DFM guidance scoring (not final eval). "
                        "Dramatically speeds up sampling, e.g. --guidance-var-limit 50. "
                        "Final DesignResult scores always use all variants.")
    args = p.parse_args()
    _set_global_seed(args.seed)

    variants_path = _resolve_user_path(args.variants_fasta)
    variants = _load_variants_fasta(variants_path, args.variant_limit, args.seed)
    if not variants:
        raise ValueError(f"No variants loaded from {variants_path}.")
    print(f"Loaded {len(variants)} variants from {variants_path}")

    guidance_variants = variants
    if args.guidance_variants_fasta:
        guidance_path = _resolve_user_path(args.guidance_variants_fasta)
        guidance_variants = _load_variants_fasta(guidance_path, args.variant_limit, args.seed)
        print(f"Loaded {len(guidance_variants)} guidance variants from {guidance_path}")

    edit_distance_reference = guidance_variants
    if args.edit_distance_reference_fasta:
        ref_path = _resolve_user_path(args.edit_distance_reference_fasta)
        edit_distance_reference = _load_variants_fasta(ref_path, args.variant_limit, args.seed)
        print(f"Loaded {len(edit_distance_reference)} edit-distance reference variants from {ref_path}")

    wt_seq = args.wt_seq.strip().upper().replace("-", "")
    if not wt_seq:
        raise ValueError("--wt-seq is empty after stripping gaps.")
    print(f"WT sequence: {wt_seq[:30]}{'...' if len(wt_seq) > 30 else ''} (len={len(wt_seq)})")

    if args.design_mode == "random_variants":
        guidance_variants = _random_variants_matched_to_edit_distance(
            wt_seq, edit_distance_reference, seed=args.seed + 17
        )
        print("Using random variants matched to edit-distance reference distribution.")
    elif args.design_mode == "esm_only_variants":
        esm_device = args.esm_variant_device if args.esm_variant_device else args.device
        guidance_variants = _esm_variants_matched_to_edit_distance(
            wt_seq=wt_seq,
            reference_variants=edit_distance_reference,
            seed=args.seed + 29,
            model_name=args.esm_variant_model,
            device=esm_device,
            temperature=args.esm_variant_temperature,
        )
        print(
            "Using ESM-only variants matched to edit-distance reference distribution "
            f"(model={args.esm_variant_model}, device={esm_device})."
        )
    elif args.design_mode == "wt_only":
        print("Using WT-only guidance; variants are retained for post-hoc scoring.")
    elif args.design_mode == "uniform_leaves":
        print("Using supplied guidance variants as uniformly weighted leaves.")

    if args.guidance_out_fasta:
        guidance_out = _resolve_user_path(args.guidance_out_fasta)
        _write_fasta(guidance_out, guidance_variants, args.design_mode)
        print(f"Saved final guidance variants -> {guidance_out}")

    scorer = AffinityScorer(
        device=args.device,
        peptiverse_normalization=args.peptiverse_normalization,
        peptiverse_min=args.peptiverse_min,
        peptiverse_max=args.peptiverse_max,
    )
    if getattr(args, "diagnose_speed", False):
        diagnose_scoring_speed(scorer, wt_seq, variants)
        import sys as _sys; _sys.exit(0)
    hypercone_rad = math.radians(args.hypercone_angle)

    # Load DFM model if requested
    dfm_model = None
    dfm_tokenizer = None
    dfm_device = args.dfm_device if args.dfm_device else args.device
    if not args.dfm_ckpt:
        default_ckpt = (
            REPO_ROOT / "MOG-DFM" / "ckpt" / "peptide"
            / "cnn_epoch200_lr0.0001_embed512_hidden256_loss3.1051.ckpt"
        )
        if default_ckpt.exists():
            args.dfm_ckpt = str(default_ckpt)
        else:
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
        f"mode={args.design_mode}, "
        f"n_designs={args.n_designs}, n_steps={args.n_steps}, "
        f"eta={args.eta}, beta={args.beta}"
    )
    designs = mog_dfm_guided_design(
        wt_seq=wt_seq,
        eval_variants=variants,
        guidance_variants=guidance_variants,
        aff_fn=scorer,
        peptide_length=args.peptide_length,
        design_mode=args.design_mode,
        n_steps=args.n_steps,
        n_designs=args.n_designs,
        eta=args.eta,
        beta=args.beta,
        delta_alpha=args.delta_alpha,
        hypercone_angle=hypercone_rad,
        omega_samples=args.omega_grid,
        seed=args.seed,
        verbose=args.verbose_sampling,
        dfm_model=dfm_model,
        dfm_tokenizer=dfm_tokenizer,
        dfm_device=dfm_device,
        tau_bind=args.tau_bind,
        guidance_var_limit=args.guidance_var_limit,
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
        if not math.isnan(top.retention_score):
            print(f"  Ret.    : {top.retention_score:.4f}  (tau_bind={args.tau_bind})")

        pareto_wt = [d.wt_score for d in designs]
        pareto_rb = [d.robust_score for d in designs]
        print(
            f"\nPareto front range:"
            f"  WT [{min(pareto_wt):.3f}, {max(pareto_wt):.3f}]"
            f"  Robust [{min(pareto_rb):.3f}, {max(pareto_rb):.3f}]"
        )


if __name__ == "__main__":
    main()
