#!/usr/bin/env python3
"""
Standalone MOG-DFM baselines for PROPHET Stage 2 inputs.

This file intentionally lives next to, but separate from, stage1.py/stage2.py so
baseline experiments can be changed without touching the main PROPHET scripts.
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
        embed_batch_size: int = 64,
        score_batch_size: int = 256,
    ):
        self.device = device
        self.peptiverse_normalization = peptiverse_normalization
        self.peptiverse_min = float(peptiverse_min)
        self.peptiverse_max = float(peptiverse_max)
        self.embed_batch_size = int(embed_batch_size)
        self.score_batch_size = int(score_batch_size)
        if self.peptiverse_max <= self.peptiverse_min:
            raise ValueError("--peptiverse-max must be greater than --peptiverse-min")
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
            self._embedder = embedder
            self._binding_model = model
            self._embedding_cache = embedding_cache

            def _embed(seq: str) -> tuple[torch.Tensor, torch.Tensor]:
                cached = embedding_cache.get(seq)
                if cached is None:
                    cached = embedder.unpooled(seq)
                    embedding_cache[seq] = cached
                return cached

            def _peptiverse(peptide: str, target: str) -> float:
                import torch
                T, Mt = _embed(target)
                B, Mb = _embed(peptide)
                with torch.no_grad():
                    reg, _ = model(T, Mt, B, Mb)
                raw_score = float(reg.squeeze().cpu().item())
                return self._normalize_peptiverse_score(raw_score)

            self._predict = _peptiverse
            print(
                "  [scorer] PeptiVerse loaded "
                f"(normalization={self.peptiverse_normalization}, "
                f"min={self.peptiverse_min:g}, max={self.peptiverse_max:g})."
                f" Embeddings will be cached per unique sequence "
                f"(embed_batch={self.embed_batch_size}, score_batch={self.score_batch_size}).",
                flush=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "PeptiVerse scoring was requested but could not be loaded. "
                "Install/fix PeptiVerse dependencies and checkpoint paths."
            ) from exc

    def __call__(self, peptide: str, target: str) -> float:
        return self._predict(peptide, target)

    def _embed_many_unpooled(self, seqs: list[str]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        cleaned = [s.strip() for s in seqs]
        missing = list(dict.fromkeys(s for s in cleaned if s not in self._embedding_cache))
        for start in range(0, len(missing), self.embed_batch_size):
            batch = missing[start : start + self.embed_batch_size]
            tok = self._embedder._tokenize(batch)
            with torch.no_grad():
                h = self._embedder.model(**tok).last_hidden_state
            valid = self._embedder._valid_mask(tok["input_ids"], tok["attention_mask"])
            for idx, seq in enumerate(batch):
                X = h[idx : idx + 1, valid[idx], :]
                M = torch.ones((1, X.shape[1]), dtype=torch.bool, device=self.device)
                self._embedding_cache[seq] = (X, M)
        return [self._embedding_cache[s] for s in cleaned]

    def _stack_unpooled(
        self,
        embs: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(int(x.shape[1]) for x, _ in embs)
        hidden = int(embs[0][0].shape[-1])
        X_out = torch.zeros((len(embs), max_len, hidden), dtype=embs[0][0].dtype, device=self.device)
        M_out = torch.zeros((len(embs), max_len), dtype=torch.bool, device=self.device)
        for idx, (X, M) in enumerate(embs):
            length = int(X.shape[1])
            X_out[idx, :length, :] = X[0, :length, :]
            M_out[idx, :length] = M[0, :length]
        return X_out, M_out

    def predict_many(self, peptides: list[str], targets: list[str]) -> np.ndarray:
        if len(peptides) != len(targets):
            raise ValueError("predict_many requires equal-length peptide and target lists")
        if not peptides:
            return np.array([], dtype=np.float64)

        # Batch ESM embedding generation for all new unique sequences first.
        self._embed_many_unpooled(list(dict.fromkeys(peptides + targets)))

        preds: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(peptides), self.score_batch_size):
                pep_batch = [p.strip() for p in peptides[start : start + self.score_batch_size]]
                tgt_batch = [t.strip() for t in targets[start : start + self.score_batch_size]]
                T, Mt = self._stack_unpooled([self._embedding_cache[t] for t in tgt_batch])
                B, Mb = self._stack_unpooled([self._embedding_cache[p] for p in pep_batch])
                reg, _ = self._binding_model(T, Mt, B, Mb)
                raw = reg.detach().float().cpu().numpy().reshape(-1)
                if self.peptiverse_normalization == "minmax":
                    raw = np.clip(
                        (raw - self.peptiverse_min) / (self.peptiverse_max - self.peptiverse_min),
                        0.0,
                        1.0,
                    )
                preds.append(raw.astype(np.float64))
        return np.concatenate(preds, axis=0)

    def predict_matrix(self, peptides: list[str], targets: list[str]) -> np.ndarray:
        if not peptides or not targets:
            return np.empty((len(peptides), len(targets)), dtype=np.float64)
        total = len(peptides) * len(targets)
        out = np.empty(total, dtype=np.float64)
        flat_peptides: list[str] = []
        flat_targets: list[str] = []
        flat_indices: list[int] = []
        for i, pep in enumerate(peptides):
            for j, tgt in enumerate(targets):
                flat_peptides.append(pep)
                flat_targets.append(tgt)
                flat_indices.append(i * len(targets) + j)
                if len(flat_peptides) >= self.score_batch_size:
                    out[np.array(flat_indices, dtype=np.int64)] = self.predict_many(flat_peptides, flat_targets)
                    flat_peptides.clear()
                    flat_targets.clear()
                    flat_indices.clear()
        if flat_peptides:
            out[np.array(flat_indices, dtype=np.int64)] = self.predict_many(flat_peptides, flat_targets)
        return out.reshape(len(peptides), len(targets))


def score_peptide_against_variants(
    peptide: str,
    variants: list[str],
    wt_seq: str,
    aff_fn: AffinityScorer,
) -> tuple[float, np.ndarray]:
    wt_score = float(aff_fn.predict_many([peptide], [wt_seq])[0])
    var_scores = aff_fn.predict_matrix([peptide], variants)[0]
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

    def _guidance_subset(self) -> list[str]:
        if self.guidance_var_limit is None or self.guidance_var_limit >= len(self.variants):
            return self.variants
        idxs = self._rng.choice(len(self.variants), size=self.guidance_var_limit, replace=False)
        return [self.variants[i] for i in sorted(idxs)]

    def __call__(self, x, t=None):
        # x: (batch, seq_len) integer tokens
        # Decode to string, score against variants
        device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        seqs = [_decode_tokens_to_peptide(row, self.tokenizer) for row in x]
        score_matrix = self.aff_fn.predict_matrix(seqs, self._guidance_subset())
        scores = [cvar_robust_score(row, self.eta) for row in score_matrix]
        return torch.tensor(scores, dtype=torch.float32, device=device)


class MeanVariantScoreModel:
    def __init__(
        self,
        variants,
        aff_fn,
        weights=None,
        tokenizer=None,
        guidance_var_limit: int | None = None,
        seed: int = 42,
    ):
        self.variants = variants
        self.aff_fn = aff_fn
        self.tokenizer = tokenizer
        self.guidance_var_limit = guidance_var_limit
        self._rng = np.random.default_rng(seed)
        if weights is None:
            self.weights = np.full(len(variants), 1.0 / max(len(variants), 1), dtype=np.float64)
        else:
            self.weights = np.asarray(weights, dtype=np.float64)
            self.weights = self.weights / max(float(self.weights.sum()), 1e-12)

    def _guidance_subset(self) -> tuple[list[str], np.ndarray]:
        if self.guidance_var_limit is None or self.guidance_var_limit >= len(self.variants):
            return self.variants, self.weights
        idxs = np.sort(
            self._rng.choice(len(self.variants), size=self.guidance_var_limit, replace=False)
        )
        variants = [self.variants[int(i)] for i in idxs]
        weights = self.weights[idxs]
        weights = weights / max(float(weights.sum()), 1e-12)
        return variants, weights

    def __call__(self, x, t=None):
        device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        seqs = [_decode_tokens_to_peptide(row, self.tokenizer) for row in x]
        variants, weights = self._guidance_subset()
        score_matrix = self.aff_fn.predict_matrix(seqs, variants)
        scores = score_matrix @ weights
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
        scores = self.aff_fn.predict_many(seqs, [self.wt_seq] * len(seqs))
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
    guidance_weights: np.ndarray | None = None,
    dfm_model=None,
    dfm_tokenizer=None,
    dfm_device="cuda:0",
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
    guidance_var_limit = kwargs.get("guidance_var_limit")
    if guidance_var_limit is not None:
        guidance_var_limit = int(guidance_var_limit)

    # Omega sweep for Pareto front
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
        elif design_mode in {"uniform_leaves", "random_variants", "esm_only_variants"}:
            score_models = [
                WTScoreModel(aff_fn, wt_seq, tokenizer=dfm_tokenizer),
                MeanVariantScoreModel(
                    guidance_variants,
                    aff_fn,
                    tokenizer=dfm_tokenizer,
                    guidance_var_limit=guidance_var_limit,
                    seed=int(seed or 42) + grid_idx,
                ),
            ]
            importance = omega
            result_omega = omega
            guidance_weight = omega
        elif design_mode == "prob_weighted_variants":
            score_models = [
                WTScoreModel(aff_fn, wt_seq, tokenizer=dfm_tokenizer),
                MeanVariantScoreModel(
                    guidance_variants,
                    aff_fn,
                    weights=guidance_weights,
                    tokenizer=dfm_tokenizer,
                    guidance_var_limit=guidance_var_limit,
                    seed=int(seed or 42) + grid_idx,
                ),
            ]
            importance = omega
            result_omega = omega
            guidance_weight = omega
        else:
            score_models = [
                WTScoreModel(aff_fn, wt_seq, tokenizer=dfm_tokenizer),
                RobustnessScoreModel(
                    guidance_variants,
                    eta,
                    aff_fn,
                    wt_seq,
                    tokenizer=dfm_tokenizer,
                    guidance_var_limit=guidance_var_limit,
                    seed=int(seed or 42) + grid_idx,
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
            if design_mode == "prob_weighted_variants" and guidance_weights is not None:
                # guidance_weights aligns with guidance_variants (used for DFM guidance),
                # while var_scores aligns with eval_variants (used for final scoring).
                # When they differ in length we must re-score against guidance_variants.
                if len(guidance_weights) != len(var_scores):
                    _, guided_scores = score_peptide_against_variants(
                        seq, guidance_variants, wt_seq, aff_fn
                    )
                else:
                    guided_scores = var_scores
                robust = float(np.dot(guidance_weights, guided_scores))
            elif design_mode in {"uniform_leaves", "random_variants", "esm_only_variants"}:
                robust = float(np.mean(var_scores))
            else:
                robust = cvar_robust_score(var_scores, eta)
            tau_bind = kwargs.get("tau_bind")
            ret = float("nan")
            if tau_bind is not None and var_scores.size > 0:
                ret = float(np.mean(var_scores >= float(tau_bind)))
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


def _load_npz_array(path: str | Path, key: str) -> np.ndarray:
    data = np.load(str(_resolve_user_path(path)))
    if key not in data:
        raise KeyError(f"{path} does not contain key {key!r}; available={list(data.files)}")
    return data[key]


def _sequence_energy(
    seq: str,
    lambda_i: np.ndarray,
    qi: np.ndarray,
    h: np.ndarray,
    J: np.ndarray,
    wt_seq: str,
    t_evo: float,
    energy_mode: str,
) -> float:
    x = np.array([AA_TO_IDX.get(a, len(AA)) for a in seq], dtype=np.int16)
    wt = np.array([AA_TO_IDX.get(a, len(AA)) for a in wt_seq], dtype=np.int16)
    if np.any(x >= len(AA)) or x.shape[0] != lambda_i.shape[0]:
        return float("inf")

    e = 0.0
    for i, a in enumerate(x):
        aa = int(a)
        e += float(lambda_i[i] * h[i, aa])
        if energy_mode == "dca_plus_qi":
            wt_aa = int(wt[i]) if i < wt.shape[0] and wt[i] < len(AA) else aa
            e -= float(lambda_i[i] * np.log(qi[i, wt_aa, aa] + 1e-9))
        elif energy_mode != "paper_dca":
            raise ValueError(f"Unsupported energy mode for probability weights: {energy_mode}")
        for j in range(i + 1, x.shape[0]):
            b = int(x[j])
            e += float(lambda_i[i] * J[i, j, aa, b])
    return e / max(float(t_evo), 1e-8)


def _probability_weights_from_stage1(
    variants: list[str],
    wt_seq: str,
    lambda_path: str | Path,
    qi_path: str | Path,
    h_path: str | Path,
    j_path: str | Path,
    t_evo: float,
    energy_mode: str,
) -> np.ndarray:
    lambda_i = np.load(str(_resolve_user_path(lambda_path)))
    qi = _load_npz_array(qi_path, "Qi")
    h = np.load(str(_resolve_user_path(h_path)))
    J = _load_npz_array(j_path, "J")
    energies = np.array(
        [_sequence_energy(v, lambda_i, qi, h, J, wt_seq, t_evo, energy_mode) for v in variants],
        dtype=np.float64,
    )
    finite = np.isfinite(energies)
    if not finite.any():
        raise ValueError("Could not compute finite probability weights for any variant.")
    logits = -energies
    logits[~finite] = float("-inf")
    logits -= np.max(logits[finite])
    weights = np.exp(logits)
    weights /= max(float(weights.sum()), 1e-12)
    entropy = -float(np.sum(weights * np.log(weights + 1e-12)))
    print(
        "Loaded probability weights from Stage 1 energy files: "
        f"min={weights.min():.6g}, max={weights.max():.6g}, entropy={entropy:.3f}",
        flush=True,
    )
    return weights


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
        description="Standalone MOG-DFM baselines for PROPHET Stage 2 variant sets"
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
    p.add_argument(
        "--design-mode",
        choices=[
            "wt_only",
            "uniform_leaves",
            "random_variants",
            "esm_only_variants",
            "prob_weighted_variants",
        ],
        default="uniform_leaves",
        help="MOG-DFM baseline objective to run."
    )
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
    p.add_argument("--lambda-path", default=None, help="Stage 1 lambda.npy for probability-weighted variants")
    p.add_argument("--qi-path", default=None, help="Stage 1 Qi.npz for probability-weighted variants")
    p.add_argument("--h-path", default=None, help="Stage 1 h.npy for probability-weighted variants")
    p.add_argument("--j-path", default=None, help="Stage 1 J.npz for probability-weighted variants")
    p.add_argument("--t-evo", type=float, default=1.0, help="Stage 1 Gibbs temperature for probability weights")
    p.add_argument("--energy-mode", choices=["paper_dca", "dca_plus_qi"], default="dca_plus_qi")
    p.add_argument("--verbose-sampling", action="store_true",
                   help="Enable tqdm progress inside MOG-DFM sampling")
    p.add_argument("--tau-bind", type=float, default=None,
                   help="Binding score threshold for retention metric. "
                        "If not set, retention_score=nan.")
    p.add_argument("--guidance-var-limit", type=int, default=None,
                   help="Subsample this many variants during DFM guidance scoring. "
                        "Final DesignResult scores still use all variants.")
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
    elif args.design_mode == "prob_weighted_variants":
        print("Using supplied guidance variants weighted by Stage 1 probability.")

    guidance_weights = None
    if args.design_mode == "prob_weighted_variants":
        missing = [
            name for name, value in [
                ("--lambda-path", args.lambda_path),
                ("--qi-path", args.qi_path),
                ("--h-path", args.h_path),
                ("--j-path", args.j_path),
            ]
            if value is None
        ]
        if missing:
            raise ValueError(
                "prob_weighted_variants requires Stage 1 energy files: "
                + ", ".join(missing)
            )
        guidance_weights = _probability_weights_from_stage1(
            guidance_variants,
            wt_seq=wt_seq,
            lambda_path=args.lambda_path,
            qi_path=args.qi_path,
            h_path=args.h_path,
            j_path=args.j_path,
            t_evo=args.t_evo,
            energy_mode=args.energy_mode,
        )

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
        guidance_weights=guidance_weights,
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
        objective_name = "weighted/mean variant score" if args.design_mode != "wt_only" else "WT score"
        print(f"Top design (by {objective_name}):")
        print(f"  Peptide : {top.peptide}")
        print(f"  WT aff  : {top.wt_score:.4f}")
        print(f"  Obj     : {top.robust_score:.4f}")
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
