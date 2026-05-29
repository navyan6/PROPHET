#!/usr/bin/env python3
"""
PROPHET Stage 1: per-site evolutionary rates (λ_i, Q_i) from a phylogenetic
tree via Fitch parsimony, plus DCA coupling parameters (h, J) via
pseudolikelihood maximization. Outputs are used to guide Gibbs sampling of
escape variants and ESM-2 pLL filtering in Stage 1, and passed to Stage 2
for CVaR-robust peptide design.
"""

import sys
import warnings
import argparse
from pathlib import Path
import random

import numpy as np
from Bio import SeqIO, Phylo
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed

REPO_ROOT = Path(__file__).resolve().parent.parent

# 20 standard amino acids, alphabetical order
AA        = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA)}
GAP       = 20   # index used for gap / ambiguous / unknown


def resolve_user_path(path_like: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve a path relative to base_dir, REPO_ROOT, cwd, or script dir (in that order)."""
    p = Path(path_like)
    if p.is_absolute():
        return p

    candidates: list[Path] = []
    if base_dir is not None:
        candidates.append(Path(base_dir) / p)
    candidates.extend(
        [
            REPO_ROOT / p,
            Path.cwd() / p,
            Path(__file__).resolve().parent / p,
        ]
    )
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    # Fall back to repo-root-relative path for deterministic output locations.
    return (REPO_ROOT / p).resolve()


def normalize_protein_alignment(protein_seqs: dict[str, str]) -> dict[str, str]:
    """Enforce uniform alignment length: truncate overlength seqs, pad underlength with '-'."""
    cleaned = {k: v.upper() for k, v in protein_seqs.items() if v}
    if not cleaned:
        raise ValueError("No non-empty protein sequences were provided.")

    lengths = np.array([len(s) for s in cleaned.values()], dtype=int)
    uniq, counts = np.unique(lengths, return_counts=True)
    target_len = int(uniq[np.argmax(counts)])

    truncated = 0
    padded = 0
    normalized: dict[str, str] = {}
    for sid, seq in cleaned.items():
        if len(seq) > target_len:
            normalized[sid] = seq[:target_len]
            truncated += 1
        elif len(seq) < target_len:
            normalized[sid] = seq + ("-" * (target_len - len(seq)))
            padded += 1
        else:
            normalized[sid] = seq

    if len(uniq) > 1:
        print(
            f"  [align] mixed lengths {uniq.tolist()} detected; "
            f"target={target_len}, truncated={truncated}, padded={padded}"
        )

    return normalized


def compute_lambda_and_qi(
    tree_nwk: str | Path,
    protein_seqs: dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-site evolutionary rates λ_i and substitution matrices Q_i via Fitch parsimony on the tree."""

    tree = Phylo.read(str(tree_nwk), "newick")
    tree.root_at_midpoint()
    n_terminals = len(tree.get_terminals())
    # Fitch traversals are recursive below; scale guard with tree size.
    sys.setrecursionlimit(max(10_000, 10 * n_terminals))

    L = len(next(iter(protein_seqs.values())))

    # Encode leaf protein sequences as int8 arrays (0-19 = amino acid, 20 = gap)
    leaf_enc: dict = {}
    n_missing = 0
    for clade in tree.get_terminals():
        seq = protein_seqs.get(clade.name)
        if seq:
            leaf_enc[clade.name] = np.array(
                [AA_TO_IDX.get(c, GAP) for c in seq], dtype=np.int8
            )
        else:
            n_missing += 1

    print(f"  [lambda] matched {len(leaf_enc)} / {len(tree.get_terminals())} leaves "
          f"({n_missing} missing)")

    total_bl = sum(
        c.branch_length for c in tree.find_clades()
        if c.branch_length and c.branch_length > 0
    )
    print(f"  [lambda] total branch length: {total_bl:.6f}")

    # Parent map for O(1) lookup when counting substitutions
    parent_map: dict = {}
    for clade in tree.find_clades():
        for child in clade.clades:
            parent_map[child] = clade

    # NOTE: These dicts are intentionally local to this function.
    # If tree-level parallelism is added later, keep state per task/process.
    # fitch_sets[id(node)]: bool array (L, 21)
    # True at [i, a] means amino acid a is in the parsimony set at position i
    fitch_sets: dict = {}

    def _bottom_up(node):
        for child in node.clades:
            _bottom_up(child)

        if node.is_terminal():
            mask = np.zeros((L, 21), dtype=bool)
            enc = leaf_enc.get(node.name)
            if enc is not None:
                mask[np.arange(L), enc] = True
            else:
                mask[:] = True          # unknown leaf: all states possible
            fitch_sets[id(node)] = mask
        else:
            child_masks = [fitch_sets[id(c)] for c in node.clades]

            inter = child_masks[0].copy()
            union = child_masks[0].copy()
            for m in child_masks[1:]:
                inter &= m
                union |= m

            # Where intersection is non-empty use it, else fall back to union
            has_inter = inter.any(axis=1)           # (L,)
            fitch_sets[id(node)] = np.where(has_inter[:, None], inter, union)

    _bottom_up(tree.root)

    # assigned[id(node)]: int8 array (L,) — the single assigned amino acid
    assigned: dict = {}

    def _top_down(node, parent_state=None):
        mask = fitch_sets[id(node)]   # (L, 21)

        if parent_state is None:
            state = np.zeros(L, dtype=np.int8)
            for i in range(L):
                cands = np.where(mask[i, :20])[0]
                state[i] = cands[0] if len(cands) else GAP
        else:
            parent_in_set = mask[np.arange(L), parent_state]   # (L,) bool
            state = parent_state.copy()
            # Only loop over positions that need a substitution
            for i in np.where(~parent_in_set)[0]:
                cands = np.where(mask[i, :20])[0]
                state[i] = cands[0] if len(cands) else GAP

        assigned[id(node)] = state
        for child in node.clades:
            _top_down(child, state)

    _top_down(tree.root)
    sub_counts = np.zeros(L, dtype=np.float64)
    # qi_counts[i, a, b] counts substitutions at site i from aa a -> aa b
    qi_counts = np.zeros((L, 20, 20), dtype=np.float64)

    for clade in tree.find_clades():
        parent = parent_map.get(clade)
        if parent is None:
            continue   # root has no incoming branch
        bl = clade.branch_length or 0.0
        if bl <= 0:
            continue

        ps = assigned[id(parent)]
        cs = assigned[id(clade)]
        changed = (ps != cs) & (ps != GAP) & (cs != GAP)
        sub_counts += changed.astype(np.float64)
        changed_idx = np.where(changed)[0]
        for i in changed_idx:
            a = int(ps[i])
            b = int(cs[i])
            if 0 <= a < 20 and 0 <= b < 20:
                qi_counts[i, a, b] += bl  # branch length weighting

    lambda_i = sub_counts / max(total_bl, 1e-9)
    # Row-normalized conditional substitution probabilities with smoothing
    qi = np.zeros((L, 20, 20), dtype=np.float64)
    alpha = 1e-6
    for i in range(L):
        row_sums = qi_counts[i].sum(axis=1, keepdims=True)
        qi[i] = (qi_counts[i] + alpha) / (row_sums + 20.0 * alpha)

    print(f"  [lambda] variable sites (λ > 0): {(lambda_i > 0).sum()} / {L}")
    return lambda_i, qi


def load_tree_list(single_tree: str, trees_file: str | None) -> list[str]:
    """Combine the main tree with any additional bootstrap trees listed in trees_file."""
    trees = [str(single_tree)]
    if trees_file:
        trees_file_path = resolve_user_path(trees_file)
        with open(trees_file_path, encoding="utf-8") as f:
            extra = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        trees.extend(
            str(resolve_user_path(line, base_dir=trees_file_path.parent))
            for line in extra
        )
    # preserve submission order, deduplicate
    seen = set()
    ordered = []
    for t in trees:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered



def _sequence_weights(X: np.ndarray, threshold: float = 0.9) -> np.ndarray:
    """Phylogenetic reweighting: downweight sequences sharing ≥threshold identity with any other."""
    N, L = X.shape
    valid = (X < GAP).astype(np.float32)
    weights = np.ones(N)

    for n in range(N):
        both     = valid[n] * valid                                  # (N, L)
        n_both   = both.sum(axis=1)                                  # (N,)
        n_match  = ((X[n] == X).astype(np.float32) * both).sum(axis=1)
        identity = np.divide(n_match, n_both, out=np.zeros(N), where=n_both > 0)
        weights[n] = 1.0 / max((identity >= threshold).sum(), 1)

    return weights


def _fit_position(i: int, X: np.ndarray, weights: np.ndarray, l2_reg: float) -> tuple:
    """Fit one PLM position via weighted L2-regularized logistic regression; returns (i, h_i, J_row)."""
    N, L = X.shape

    y = X[:, i].astype(int)
    valid_rows = y < GAP
    if valid_rows.sum() < 5 or len(np.unique(y[valid_rows])) < 2:
        return i, np.zeros(20), np.zeros((L, 20, 20))

    y_v = y[valid_rows]
    w_v = weights[valid_rows]
    X_v = X[valid_rows]

    F = np.zeros((valid_rows.sum(), (L - 1) * 20), dtype=np.float32)
    col = 0
    for j in range(L):
        if j == i:
            continue
        nongap = X_v[:, j] < GAP
        aa_idx = X_v[nongap, j].astype(np.int32, copy=False)
        F[nongap, col + aa_idx] = 1.0
        col += 20

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(
            C=1.0 / l2_reg,
            solver="lbfgs",
            max_iter=200,
            fit_intercept=True,
        )
        clf.fit(F, y_v, sample_weight=w_v)

    # Extract h[i] and J_row from fitted coefficients.
    # sklearn >= 1.5 returns 1 coef row for binary (2-class) problems and
    # n_classes rows for multinomial — handle both.
    h_i   = np.zeros(20)
    J_row = np.zeros((L, 20, 20))

    n_rows = len(clf.intercept_)   # 1 for binary, n_classes for multinomial

    def _extract(k, cls_a):
        if cls_a >= 20:
            return
        h_i[cls_a] = clf.intercept_[k]
        col = 0
        for j in range(L):
            if j == i:
                continue
            for b in range(20):
                J_row[j, cls_a, b] = clf.coef_[k, col + b]
            col += 20

    if n_rows == 1:
        # Binary: the single row encodes P(classes_[1]) vs P(classes_[0])
        _extract(0, clf.classes_[1])
    else:
        for k, cls_a in enumerate(clf.classes_):
            _extract(k, int(cls_a))

    return i, h_i, J_row


def compute_dca(
    protein_seqs: dict,
    n_bootstraps: int = 1,
    l2_reg: float = 0.01,
    n_jobs: int = -1,
    seed: int = 42,
    adaptive_l2_reg_base: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit DCA fields h (L,20) and couplings J (L,L,20,20) via PLM; returns (h, J, h_std, J_std)."""
    seqs = list(protein_seqs.values())
    N, L = len(seqs), len(seqs[0])

    X = np.full((N, L), GAP, dtype=np.int8)
    for n, seq in enumerate(seqs):
        for i, aa in enumerate(seq):
            X[n, i] = AA_TO_IDX.get(aa, GAP)

    print(f"  [dca] {N} sequences, {L} positions — computing sequence weights...")
    weights = _sequence_weights(X)
    n_eff = weights.sum()
    print(f"  [dca] N_eff ≈ {n_eff:.1f}  (effective non-redundant sequences)")

    if adaptive_l2_reg_base is not None:
        l2_reg = adaptive_l2_reg_base * (L ** 2) / n_eff
        print(f"  [dca] adaptive l2_reg = {adaptive_l2_reg_base:.2e} × {L}² / {n_eff:.1f} = {l2_reg:.4f}")
    else:
        print(f"  [dca] l2_reg = {l2_reg:.4f} (fixed)")

    rng    = np.random.default_rng(seed)
    n_runs = max(1, n_bootstraps)

    h_mean = np.zeros((L, 20),          dtype=np.float32)
    J_mean = np.zeros((L, L, 20, 20),   dtype=np.float32)
    h_M2   = np.zeros((L, 20),          dtype=np.float32)
    J_M2   = np.zeros((L, L, 20, 20),   dtype=np.float32)

    for b in range(n_runs):
        if n_bootstraps > 1:
            idx  = rng.integers(0, N, size=N)
            X_b  = X[idx]
            w_b  = weights[idx]
            print(f"  [dca] bootstrap {b+1}/{n_bootstraps} ...")
        else:
            X_b, w_b = X, weights
            print(f"  [dca] fitting {L} positions (n_jobs={n_jobs}) ...")

        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_position)(i, X_b, w_b, l2_reg)
            for i in range(L)
        )

        h_b = np.zeros((L, 20),        dtype=np.float32)
        J_b = np.zeros((L, L, 20, 20), dtype=np.float32)
        for i, h_i, J_row_i in results:
            h_b[i]    = h_i
            J_b[i]    = J_row_i

        # Symmetrize: average the two directed estimates of each coupling
        J_b = (J_b + J_b.transpose(1, 0, 3, 2)) / 2

        # Zero-sum gauge: subtract row/column means so couplings are
        # identifiable (removes redundancy between h and J)
        J_b -= J_b.mean(axis=2, keepdims=True)
        J_b -= J_b.mean(axis=3, keepdims=True)
        h_b -= h_b.mean(axis=1, keepdims=True)

        # Welford update for mean and variance
        delta_h = h_b - h_mean
        h_mean += delta_h / (b + 1)
        h_M2   += delta_h * (h_b - h_mean)

        delta_J = J_b - J_mean
        J_mean += delta_J / (b + 1)
        J_M2   += delta_J * (J_b - J_mean)

    h_std = np.sqrt(h_M2 / max(n_runs, 1))
    J_std = np.sqrt(J_M2 / max(n_runs, 1))

    return h_mean, J_mean, h_std, J_std


def _sample_site_conditional(
    i: int,
    x: np.ndarray,
    lambda_i: np.ndarray,
    qi: np.ndarray,
    h: np.ndarray,
    J: np.ndarray,
    t_evo: float,
    energy_mode: str,
    rng: np.random.Generator,
    conservation: np.ndarray | None = None,
    conserv_weight: float = 0.0,
    wt_x: np.ndarray | None = None,
) -> int:
    """Sample position i from p(x_i | x_{-i}) ∝ exp(E_i / T) where E_i combines DCA fields, couplings, and conservation penalty."""
    e = lambda_i[i] * h[i].astype(np.float64)
    if energy_mode == "dca_plus_qi":
        wt_aa = int(wt_x[i]) if wt_x is not None else int(x[i])
        if 0 <= wt_aa < 20:
            e += lambda_i[i] * np.log(qi[i, wt_aa, :].astype(np.float64) + 1e-9)
    elif energy_mode != "paper_dca":
        raise ValueError(f"Unsupported energy_mode: {energy_mode}")
    # pairwise DCA term — J is NOT scaled by lambda_i per the paper's energy function
    for j in range(x.shape[0]):
        if j == i:
            continue
        b = int(x[j])
        if 0 <= b < 20:
            e += J[i, j, :, b].astype(np.float64)
    if conservation is not None and wt_x is not None and conserv_weight > 0.0:
        penalty = conservation[i] * conserv_weight
        wt_aa = int(wt_x[i])
        for a in range(20):
            if a != wt_aa:
                e[a] -= penalty
    # sample proportional to exp(+e/T)
    logits = e / max(t_evo, 1e-8)
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    return int(rng.choice(20, p=p))


def _pll_esm2(
    seq: str,
    tokenizer,
    model,
    device: str,
) -> float:
    """ESM-2 pseudo-log-likelihood: sum of masked-position log-probs across all residues."""
    import torch

    seq_len = len(seq)
    if seq_len == 0:
        return float("-inf")

    total = 0.0
    aa_to_tok = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in AA}
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("Tokenizer has no mask token id.")

    for i, aa in enumerate(seq):
        tok_id = aa_to_tok.get(aa, None)
        if tok_id is None:
            continue
        masked = seq[:i] + tokenizer.mask_token + seq[i + 1 :]
        enc = tokenizer(masked, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits[0]
        mask_pos = (input_ids[0] == mask_id).nonzero(as_tuple=True)[0]
        if len(mask_pos) == 0:
            continue
        m = int(mask_pos[0].item())
        logp = torch.log_softmax(logits[m], dim=-1)[tok_id].item()
        total += float(logp)
    return total


def gibbs_sample_variants(
    wt_seq: str,
    lambda_i: np.ndarray,
    qi: np.ndarray,
    h: np.ndarray,
    J: np.ndarray,
    n_samples: int,
    burn_in: int,
    t_evo: float,
    energy_mode: str = "dca_plus_qi",
    seed: int = 42,
    esm_filter_delta: float | None = None,
    esm_model_name: str = "facebook/esm2_t6_8M_UR50D",
    esm_device: str = "cpu",
    lambda_ensemble: list[np.ndarray] | None = None,
    qi_ensemble: list[np.ndarray] | None = None,
    conservation: np.ndarray | None = None,
    conserv_weight: float = 0.0,
    wt_x: np.ndarray | None = None,
) -> tuple[list[str], list[float]]:
    """Gibbs-sample escape variants from p_evo(x) with per-residue-scaled ESM-2 pLL filtering. Returns (variants, plls)."""
    rng = np.random.default_rng(seed)
    x = np.array([AA_TO_IDX.get(a, GAP) for a in wt_seq], dtype=np.int8)
    if np.any(x >= 20):
        raise ValueError("WT sequence contains unsupported amino acids for Gibbs sampling.")

    use_ensemble = lambda_ensemble is not None and qi_ensemble is not None

    pll_model = None
    pll_tokenizer = None
    wt_pll = None
    if esm_filter_delta is not None:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        print(f"  [gibbs] loading ESM model for pLL filter: {esm_model_name}")
        pll_tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
        pll_model = AutoModelForMaskedLM.from_pretrained(esm_model_name).to(esm_device)
        pll_model.eval()
        wt_pll = _pll_esm2(wt_seq, pll_tokenizer, pll_model, esm_device)
        print(f"  [gibbs] WT pLL = {wt_pll:.3f}, delta = {esm_filter_delta}")

    out_variants: list[str] = []
    out_pll: list[float] = []
    total_steps = burn_in + n_samples
    L = len(wt_seq)
    for t in range(total_steps):
        if use_ensemble:
            t_idx = rng.integers(len(lambda_ensemble))
            lam_t = lambda_ensemble[t_idx]
            qi_t  = qi_ensemble[t_idx]
        else:
            lam_t, qi_t = lambda_i, qi
        for i in rng.permutation(L):  # random scan Gibbs
            x[i] = _sample_site_conditional(
                i, x, lam_t, qi_t, h, J, t_evo, energy_mode, rng,
                conservation=conservation, conserv_weight=conserv_weight, wt_x=wt_x,
            )

        if t >= burn_in:
            seq = "".join(AA[int(a)] for a in x)
            if pll_model is not None and wt_pll is not None:
                pll = _pll_esm2(seq, pll_tokenizer, pll_model, esm_device)
                if pll < (wt_pll - float(esm_filter_delta)):
                    continue
            else:
                pll = float("nan")
            out_variants.append(seq)
            out_pll.append(pll)
            if len(out_variants) >= n_samples:
                break

    return out_variants, out_pll


def build_consensus_wt(protein_seqs: dict[str, str]) -> str:
    """Majority-vote consensus sequence over the alignment; used as the Gibbs sampling starting point."""
    seqs = list(protein_seqs.values())
    L = len(seqs[0])
    consensus = []
    for i in range(L):
        counts = {aa: 0 for aa in AA}
        for s in seqs:
            a = s[i]
            if a in counts:
                counts[a] += 1
        best = max(counts, key=counts.get)
        if counts[best] == 0:
            best = "A"
        consensus.append(best)
    return "".join(consensus)

def compute_conservation_scores(protein_seqs: dict[str, str]) -> np.ndarray:
    """Per-site conservation score in [0,1] based on normalized Shannon entropy (1 = fully conserved)."""
    seqs = list(protein_seqs.values())
    L = len(seqs[0])
    conservation = np.zeros(L)
    for i in range(L):
        counts = np.zeros(20)
        for seq in seqs:
            idx = AA_TO_IDX.get(seq[i], None)
            if idx is not None:
                counts[idx] += 1
        total = counts.sum()
        if total == 0:
            continue
        freqs = counts / total 
        shannon_entropy = -np.sum(freqs * np.log(freqs + 1e-9))
        conservation[i] = 1 - (shannon_entropy / np.log(20))
    
    return conservation




def _eval_metrics(
    variants: list[str],
    alignment_seqs: list[str],
    lambda_i: np.ndarray,
    h: np.ndarray,
    J: np.ndarray,
    wt_seq: str,
) -> None:
    """Print basic variant quality metrics after Gibbs sampling."""
    if not variants:
        print("  [eval] No variants to evaluate.")
        return
    edits = [sum(a != b for a, b in zip(v, wt_seq)) for v in variants]
    energies = []
    L = len(wt_seq)
    for v in variants:
        x = np.array([AA_TO_IDX.get(a, GAP) for a in v], dtype=np.int8)
        e = float(np.sum(lambda_i[:L] * h[np.arange(L), x[:L].clip(0, 19)]))
        for ii in range(L):
            for jj in range(ii + 1, L):
                if x[ii] < 20 and x[jj] < 20:
                    e += float(J[ii, jj, int(x[ii]), int(x[jj])])
        energies.append(-e)
    print(f"  [eval] Edit distance to WT: mean={np.mean(edits):.1f}  "
          f"min={np.min(edits)}  max={np.max(edits)}")
    print(f"  [eval] DCA energy:          mean={np.mean(energies):.3f}  "
          f"std={np.std(energies):.3f}")
    print(f"  [eval] Sample variants:")
    for v in variants[:3]:
        ed = sum(a != b for a, b in zip(v, wt_seq))
        print(f"    {v}  edit={ed}")


def main():
    p = argparse.ArgumentParser(
        description="PROPHET Stage 1 — λᵢ and DCA from a phylogenetic tree"
    )
    p.add_argument("--tree",         default="flu_tree/ha_tree.nwk",
                   help="Newick tree (default: flu HA)")
    p.add_argument("--trees-file",   default=None,
                   help="Optional file containing additional tree paths (one per line)")
    p.add_argument("--fasta",        default="flu_tree/ha_aligned.fasta",
                   help="Aligned protein FASTA")
    p.add_argument("--out-dir",      default="data/prophet",
                   help="Output directory for .npy files")
    p.add_argument("--prefix",       default="flu",
                   help="Filename prefix for outputs (default: flu)")
    p.add_argument("--n-bootstraps", type=int, default=1,
                   help="DCA bootstrap replicates (1 = single run, no std)")
    p.add_argument("--l2-reg",       type=float, default=0.01,
                   help="L2 regularisation strength for PLM (default: 0.01)")
    p.add_argument("--n-jobs",       type=int, default=-1,
                   help="Parallel jobs for DCA position fitting (-1 = all cores)")
    p.add_argument("--skip-dca",     action="store_true",
                   help="Only compute λᵢ, skip DCA (fast sanity check)")
    p.add_argument("--sample-variants", type=int, default=0,
                   help="If >0, run Gibbs sampling and save sampled variants")
    p.add_argument("--burn-in",      type=int, default=200,
                   help="Gibbs burn-in sweeps before collecting samples")
    p.add_argument("--t-evo",        type=float, default=1.0,
                   help="Evolutionary temperature for Gibbs sampling (calibrate to held-out edit distances)")
    p.add_argument("--energy-mode",  choices=["paper_dca", "dca_plus_qi"], default="paper_dca",
                   help="Energy model for Gibbs sampling: paper DCA only or DCA+Qi extension (default: paper_dca)")
    p.add_argument("--ensemble-mode", action="store_true",
                   help="Draw a random tree per Gibbs sweep instead of using averaged λ/Qi")
    p.add_argument("--conserv-weight", type=float, default=0.0,
                   help="Conservation regularization strength (0=off, try 0.5–2.0)")
    p.add_argument("--seed",         type=int, default=42,
                   help="Random seed for DCA bootstraps and Gibbs sampling")
    p.add_argument("--wt-seq-id",    default=None,
                   help="Optional FASTA ID to use as WT for Gibbs initialization")
    p.add_argument("--esm-filter-delta", type=float, default=None,
                   help="Enable ESM pLL filter with threshold pLL(x) >= pLL(wt)-delta (absolute)")
    p.add_argument("--esm-filter-delta-per-residue", type=float, default=None,
                   help="ESM pLL filter threshold per residue: effective delta = value * L. "
                        "Takes precedence over --esm-filter-delta when set. "
                        "Recommended: 0.20 (equivalent to delta=20 for L=99)")
    p.add_argument("--adaptive-l2",  action="store_true", default=False,
                   help="Use N_eff + L²-adaptive l2_reg: l2_reg = l2_reg_base * L² / N_eff")
    p.add_argument("--l2-reg-base",  type=float, default=1e-4,
                   help="Base for adaptive l2_reg (used only when --adaptive-l2 is set, default: 1e-4)")
    p.add_argument("--esm-model",    default="facebook/esm2_t6_8M_UR50D",
                   help="ESM masked LM model name for pLL filtering")
    p.add_argument("--esm-device",   default="cpu",
                   help="Device for ESM pLL filtering")
    p.add_argument("--dca-max-edit", type=int, default=None,
                   help="If set, restrict DCA training to sequences with at most this many edits from WT. "
                        "Tree/lambda/Qi still use all sequences. Keeps DCA landscape near WT.")
    p.add_argument("--ablate-zero-dca-couplings", action="store_true",
                   help="Ablation: zero out DCA h and J before Gibbs sampling (removes DCA guidance)")
    p.add_argument("--ablate-flatten-lambda", action="store_true",
                   help="Ablation: set all lambda_i to 1 before Gibbs sampling (removes per-site rate weighting)")
    args, _ = p.parse_known_args()

    fasta_path = resolve_user_path(args.fasta)
    out_dir    = resolve_user_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tree_list  = load_tree_list(args.tree, args.trees_file)
    tree_paths = [resolve_user_path(t) for t in tree_list]

    print("=" * 60)
    print("PROPHET Stage 1")
    if len(tree_paths) == 1:
        print(f"  tree : {tree_paths[0]}")
    else:
        print(f"  trees: {len(tree_paths)} (first: {tree_paths[0]})")
    print(f"  fasta: {fasta_path}")
    print("=" * 60)

    # ── Step 1: load protein sequences ───────────────────────────────────────
    print("\n[1/3] Loading protein alignment...")
    protein_seqs = {
        rec.id: str(rec.seq)
        for rec in SeqIO.parse(str(fasta_path), "fasta")
    }
    raw_len = len(next(iter(protein_seqs.values())))
    print(f"  Loaded {len(protein_seqs)} sequences, raw alignment length {raw_len}")
    seqs = list(protein_seqs.values())
    N = len(seqs)
    keep = [
        i for i in range(raw_len)
        if sum(1 for s in seqs if s[i] in "-X*.") / N <= 0.5
    ]
    if len(keep) < raw_len:
        print(f"  Gap filter: {raw_len} → {len(keep)} columns kept (≤50% gaps)")
        protein_seqs = {sid: "".join(seq[i] for i in keep)
                        for sid, seq in protein_seqs.items()}

    protein_seqs = normalize_protein_alignment(protein_seqs)

    L = len(next(iter(protein_seqs.values())))

    # ── Step 2: per-site mutation rate ────────────────────────────────────────
    print(f"\n[2/3] Computing per-site mutation rates (L={L})...")
    lambda_runs = []
    qi_runs = []
    for idx, tree_path in enumerate(tree_paths, start=1):
        if len(tree_paths) > 1:
            print(f"  [lambda] tree {idx}/{len(tree_paths)}: {tree_path}")
        lam_i, qi_i = compute_lambda_and_qi(tree_path, protein_seqs)
        lambda_runs.append(lam_i)
        qi_runs.append(qi_i)
    lambda_ensemble = lambda_runs
    qi_ensemble = qi_runs

    lambda_i = np.mean(np.stack(lambda_ensemble), axis=0)
    qi = np.mean(np.stack(qi_ensemble), axis=0)

    lam_path = out_dir / f"{args.prefix}_lambda.npy"
    np.save(lam_path, lambda_i)
    qi_path = out_dir / f"{args.prefix}_Qi.npz"
    np.savez_compressed(qi_path, Qi=qi)
    print(f"  Saved → {lam_path}")
    print(f"  Saved → {qi_path}  ({qi.shape[0]} positions × 20×20)")
    print(f"  λ range: [{lambda_i.min():.4f}, {lambda_i.max():.4f}]")

    top10 = np.argsort(lambda_i)[-10:][::-1]
    print(f"  Top 10 most variable positions: {top10.tolist()}")
    print(f"  Top 10 λ values:                {lambda_i[top10].round(4).tolist()}")

    # ── Step 3: DCA ───────────────────────────────────────────────────────────
    if args.skip_dca:
        print("\n[3/3] DCA skipped (--skip-dca).")
    else:
        # Optionally restrict DCA to sequences close to WT so the energy
        # landscape is centered in the WT neighborhood, not across all variants.
        dca_seqs = protein_seqs
        if args.dca_max_edit is not None:
            _wt_for_filter = build_consensus_wt(protein_seqs)
            dca_seqs = {k: v for k, v in protein_seqs.items()
                        if sum(a != b for a, b in zip(v, _wt_for_filter)) <= args.dca_max_edit}
            print(f"\n[3/3] DCA restricted to {len(dca_seqs)}/{len(protein_seqs)} sequences "
                  f"(edit ≤ {args.dca_max_edit} from WT); "
                  f"n_bootstraps={args.n_bootstraps}, l2_reg={args.l2_reg}, n_jobs={args.n_jobs}...")
        else:
            print(f"\n[3/3] Fitting DCA  (n_bootstraps={args.n_bootstraps}, "
                  f"l2_reg={args.l2_reg}, n_jobs={args.n_jobs})...")

        h, J, h_std, J_std = compute_dca(
            dca_seqs,
            n_bootstraps=args.n_bootstraps,
            l2_reg=args.l2_reg,
            n_jobs=args.n_jobs,
            seed=args.seed,
            adaptive_l2_reg_base=args.l2_reg_base if args.adaptive_l2 else None,
        )

        np.save(out_dir / f"{args.prefix}_h.npy",     h)
        np.save(out_dir / f"{args.prefix}_h_std.npy", h_std)
        np.savez_compressed(out_dir / f"{args.prefix}_J.npz",     J=J)
        np.savez_compressed(out_dir / f"{args.prefix}_J_std.npz", J=J_std)

        print(f"  Saved → {out_dir}/{args.prefix}_h.npy  "
              f"({h.shape[0]} positions × 20 amino acids)")
        print(f"  Saved → {out_dir}/{args.prefix}_J.npz  "
              f"({J.shape[0]}×{J.shape[1]}×20×20, "
              f"{J.nbytes / 1e6:.0f} MB uncompressed)")

        # Sanity: top 5 coupled pairs by norm of J[i,j]
        J_frob = np.sqrt((J ** 2).sum(axis=(2, 3)))   # (L, L)
        np.fill_diagonal(J_frob, 0)
        flat_top = np.argsort(J_frob.ravel())[-5:][::-1]
        print("  Top 5 coupled pairs by ||J[i,j]||_F:")
        for idx in flat_top:
            i, j = np.unravel_index(idx, J_frob.shape)
            print(f"    positions ({i:4d}, {j:4d}):  ||J|| = {J_frob[i,j]:.4f}")

        if args.sample_variants > 0:
            print(
                f"\n[4/4] Gibbs sampling variants "
                f"(M={args.sample_variants}, burn_in={args.burn_in}, "
                f"T={args.t_evo}, mode={args.energy_mode})..."
            )
            if args.wt_seq_id is not None and args.wt_seq_id in protein_seqs:
                wt_seq = protein_seqs[args.wt_seq_id]
            else:
                wt_seq = build_consensus_wt(protein_seqs)

            conservation = compute_conservation_scores(protein_seqs)
            wt_x = np.array([AA_TO_IDX.get(a, GAP) for a in wt_seq], dtype=np.int8)
            np.save(out_dir / f"{args.prefix}_conservation.npy", conservation)
            print(f"  Conservation range: [{conservation.min():.3f}, {conservation.max():.3f}]")

            # Controlled ablations: only affect Gibbs, not saved files
            h_gibbs = np.zeros_like(h) if args.ablate_zero_dca_couplings else h
            J_gibbs = np.zeros_like(J) if args.ablate_zero_dca_couplings else J
            lam_gibbs = np.ones_like(lambda_i) if args.ablate_flatten_lambda else lambda_i
            lam_ens_gibbs = (
                [np.ones_like(l) for l in lambda_ensemble]
                if args.ablate_flatten_lambda else lambda_ensemble
            )

            # Compute effective ESM filter delta (per-residue takes precedence).
            esm_delta_effective = args.esm_filter_delta
            if args.esm_filter_delta_per_residue is not None:
                L_eff = h.shape[0]
                esm_delta_effective = args.esm_filter_delta_per_residue * L_eff
                print(f"  [esm-filter] per-residue delta: {args.esm_filter_delta_per_residue} × L={L_eff} = {esm_delta_effective:.1f}")
            variants, pll_vals = gibbs_sample_variants(
                wt_seq=wt_seq,
                lambda_i=lam_gibbs,
                qi=qi,
                h=h_gibbs,
                J=J_gibbs,
                n_samples=args.sample_variants,
                burn_in=args.burn_in,
                t_evo=args.t_evo,
                energy_mode=args.energy_mode,
                seed=args.seed,
                esm_filter_delta=esm_delta_effective,
                esm_model_name=args.esm_model,
                esm_device=args.esm_device,
                lambda_ensemble=lam_ens_gibbs if args.ensemble_mode else None,
                qi_ensemble=qi_ensemble if args.ensemble_mode else None,
                conservation=conservation,
                conserv_weight=args.conserv_weight,
                wt_x=wt_x,
            )
            v_path = out_dir / f"{args.prefix}_gibbs_variants.fasta"
            with open(v_path, "w", encoding="utf-8") as f:
                for i, seq in enumerate(variants, start=1):
                    pll = pll_vals[i - 1]
                    if np.isnan(pll):
                        f.write(f">variant_{i}\n{seq}\n")
                    else:
                        f.write(f">variant_{i} pll={pll:.4f}\n{seq}\n")
            print(f"  Saved → {v_path}  ({len(variants)} accepted variants)")
            # --- Evaluation metrics ---
            _eval_metrics(variants, list(protein_seqs.values()), lambda_i, h, J, wt_seq=wt_seq)



if __name__ == "__main__":
    main()
