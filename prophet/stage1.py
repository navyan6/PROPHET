#!/usr/bin/env python3
"""
PROPHET Stage 1: per-site evolutionary parameters from a phylogenetic tree.

Two functions:
  compute_lambda(tree_nwk, protein_seqs) -> lambda_i  shape (L,)

  compute_dca(protein_seqs, n_bootstraps) -> h (L,20), J (L,L,20,20)
      DCA parameters via pseudolikelihood maximization.
"""

import sys
import warnings
import argparse
from pathlib import Path
import random

import numpy as np
from Bio import SeqIO, Phylo
from Bio.Data import CodonTable
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed

REPO_ROOT = Path(__file__).resolve().parent.parent

# 20 standard amino acids, alphabetical order
AA        = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA)}
GAP       = 20   # index used for gap / ambiguous / unknown


def resolve_user_path(path_like: str | Path, base_dir: str | Path | None = None) -> Path:
    """
    Resolve user-provided paths robustly after script moves.

    Resolution order for relative paths:
      1) base_dir (if provided)
      2) REPO_ROOT
      3) current working directory
      4) directory containing this script
    """
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
    """
    Ensure all sequences share one alignment length.

    For mixed lengths, use modal length as target, truncate longer sequences,
    and right-pad shorter ones with '-' gaps.
    """
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

def translate_alignment(fasta_path: str | Path) -> dict[str, str]:
    """
    Translate a gapped nucleotide MSA to a protein MSA.
    """
    fwd   = CodonTable.standard_dna_table.forward_table
    stops = set(CodonTable.standard_dna_table.stop_codons)

    raw = {}
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        nt = str(rec.seq).upper()
        aa = []
        for k in range(0, len(nt) - 2, 3):
            codon = nt[k : k + 3]
            if codon == "---":
                aa.append("-")
            elif "-" in codon or "N" in codon:
                aa.append("X")
            elif codon in stops:
                aa.append("*")
            elif codon in fwd:
                aa.append(fwd[codon])
            else:
                aa.append("X")
        raw[rec.id] = "".join(aa)

    if not raw:
        return raw

    seqs = list(raw.values())
    L, N = len(seqs[0]), len(seqs)

    keep = [
        i for i in range(L)
        if sum(1 for s in seqs if s[i] in "-X*") / N <= 0.5
    ]
    print(f"  [translate] {N} seqs | {L} raw aa positions → {len(keep)} after gap filter")

    return {sid: "".join(seq[i] for i in keep) for sid, seq in raw.items()}


def compute_lambda_and_qi(
    tree_nwk: str | Path,
    protein_seqs: dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-site amino acid mutation rate lambda and substitution matrices Qi
    from a phylogenetic tree.
    """

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
                qi_counts[i, a, b] += 1.0

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
    trees = [str(single_tree)]
    if trees_file:
        trees_file_path = resolve_user_path(trees_file)
        with open(trees_file_path, encoding="utf-8") as f:
            extra = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        trees.extend(
            str(resolve_user_path(line, base_dir=trees_file_path.parent))
            for line in extra
        )
    # preserve order, remove duplicates
    seen = set()
    ordered = []
    for t in trees:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def maybe_subsample_trees(trees: list[str], j: int | None, seed: int) -> list[str]:
    if j is None:
        return trees
    if j <= 0:
        raise ValueError("--tree-subsample-j must be positive when provided.")
    if j >= len(trees):
        return trees
    rng = random.Random(seed)
    return rng.sample(trees, j)


# DCA via pseudolikelihood maximization (PLM)

def _sequence_weights(X: np.ndarray, threshold: float = 0.9) -> np.ndarray:

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

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
    rng: np.random.Generator,
) -> int:
    """
    Sample x_i from p_evo(x_i | x_-i) proportional to exp(-E_i / T).
    """
    # unary term (DCA field)
    e = lambda_i[i] * h[i].astype(np.float64)
    # incorporate tree-estimated substitution preferences Q_i(a->b)
    current_aa = int(x[i])
    if 0 <= current_aa < 20:
        e += lambda_i[i] * np.log(qi[i, current_aa, :].astype(np.float64) + 1e-9)
    # pairwise term with all other sites
    for j in range(x.shape[0]):
        if j == i:
            continue
        b = int(x[j])
        if 0 <= b < 20:
            e += J[i, j, :, b].astype(np.float64)
    # stable softmax over -E/T
    logits = -e / max(t_evo, 1e-8)
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
    """
    Compute pseudo-log-likelihood by masking each position.
    """
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
    seed: int = 42,
    esm_filter_delta: float | None = None,
    esm_model_name: str = "facebook/esm2_t6_8M_UR50D",
    esm_device: str = "cpu",
) -> tuple[list[str], list[float]]:
    """
    Algorithm 1 variant sampling from p_evo with optional ESM pLL filtering.
    """
    rng = np.random.default_rng(seed)
    x = np.array([AA_TO_IDX.get(a, GAP) for a in wt_seq], dtype=np.int8)
    if np.any(x >= 20):
        raise ValueError("WT sequence contains unsupported amino acids for Gibbs sampling.")

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
        for i in range(L):
            x[i] = _sample_site_conditional(i, x, lambda_i, qi, h, J, t_evo, rng)

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
    """
    Build a valid AA-only WT-like sequence from the alignment by majority vote.
    Non-standard symbols are ignored at each site; fallback is 'A'.
    """
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



def main():
    p = argparse.ArgumentParser(
        description="PROPHET Stage 1 — λᵢ and DCA from a phylogenetic tree"
    )
    p.add_argument("--tree",         default="flu_tree/ha_tree.nwk",
                   help="Newick tree (default: flu HA)")
    p.add_argument("--trees-file",   default=None,
                   help="Optional file containing additional tree paths (one per line)")
    p.add_argument("--tree-subsample-j", type=int, default=None,
                   help="Optional J-sweep support: random subset size of trees to average over")
    p.add_argument("--tree-subsample-seed", type=int, default=42,
                   help="Random seed for tree subsampling when --tree-subsample-j is set")
    p.add_argument("--fasta",        default="flu_tree/ha_aligned.fasta",
                   help="Aligned FASTA (default: flu HA nucleotide MSA)")
    p.add_argument("--nucleotide",   action="store_true", default=True,
                   help="FASTA is nucleotide — translate to protein first (default: True)")
    p.add_argument("--protein",      dest="nucleotide", action="store_false",
                   help="FASTA is already a protein alignment")
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
    p.add_argument("--seed",         type=int, default=42,
                   help="Random seed for DCA bootstraps and Gibbs sampling")
    p.add_argument("--wt-seq-id",    default=None,
                   help="Optional FASTA ID to use as WT for Gibbs initialization")
    p.add_argument("--esm-filter-delta", type=float, default=None,
                   help="Enable ESM pLL filter with threshold pLL(x) >= pLL(wt)-delta")
    p.add_argument("--esm-model",    default="facebook/esm2_t6_8M_UR50D",
                   help="ESM masked LM model name for pLL filtering")
    p.add_argument("--esm-device",   default="cpu",
                   help="Device for ESM pLL filtering")
    args, _ = p.parse_known_args()  

    fasta_path = resolve_user_path(args.fasta)
    out_dir    = resolve_user_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tree_list = load_tree_list(args.tree, args.trees_file)
    tree_list = maybe_subsample_trees(tree_list, args.tree_subsample_j, args.tree_subsample_seed)
    tree_paths = [resolve_user_path(t) for t in tree_list]

    print("=" * 60)
    print("PROPHET Stage 1")
    if len(tree_paths) == 1:
        print(f"  tree : {tree_paths[0]}")
    else:
        print(f"  trees: {len(tree_paths)} (first: {tree_paths[0]})")
    if args.tree_subsample_j is not None:
        print(f"  tree subset J={len(tree_paths)} (seed={args.tree_subsample_seed})")
    print(f"  fasta: {fasta_path}")
    print("=" * 60)

    # ── Step 1: load protein sequences ───────────────────────────────────────
    if args.nucleotide:
        print("\n[1/3] Translating nucleotide alignment → protein...")
        protein_seqs = translate_alignment(fasta_path)
    else:
        print("\n[1/3] Loading protein alignment...")
        protein_seqs = {
            rec.id: str(rec.seq)
            for rec in SeqIO.parse(str(fasta_path), "fasta")
        }
        print(f"  Loaded {len(protein_seqs)} sequences, length {len(next(iter(protein_seqs.values())))}")

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
    lambda_i = np.mean(np.stack(lambda_runs, axis=0), axis=0)
    qi = np.mean(np.stack(qi_runs, axis=0), axis=0)

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
        print(f"\n[3/3] Fitting DCA  (n_bootstraps={args.n_bootstraps}, "
              f"l2_reg={args.l2_reg}, n_jobs={args.n_jobs})...")

        h, J, h_std, J_std = compute_dca(
            protein_seqs,
            n_bootstraps=args.n_bootstraps,
            l2_reg=args.l2_reg,
            n_jobs=args.n_jobs,
            seed=args.seed,
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
                f"(M={args.sample_variants}, burn_in={args.burn_in}, T={args.t_evo})..."
            )
            warnings.warn(
                "t_evo should be calibrated on held-out phylogenies so sampled edit distances "
                "match observed leaf-sequence distances; using the provided value as-is.",
                stacklevel=2,
            )
            if args.wt_seq_id is not None and args.wt_seq_id in protein_seqs:
                wt_seq = protein_seqs[args.wt_seq_id]
            else:
                wt_seq = build_consensus_wt(protein_seqs)
            variants, pll_vals = gibbs_sample_variants(
                wt_seq=wt_seq,
                lambda_i=lambda_i,
                qi=qi,
                h=h,
                J=J,
                n_samples=args.sample_variants,
                burn_in=args.burn_in,
                t_evo=args.t_evo,
                seed=args.seed,
                esm_filter_delta=args.esm_filter_delta,
                esm_model_name=args.esm_model,
                esm_device=args.esm_device,
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

    print("\n" + "=" * 60)
    print("Stage 1 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
