"""
binding_affinity.py
───────────────────
1. Pull the top-10 variant dict from compute_probabilities.py.
2. Generate ONE peptide with PepDFM.
3. Score that peptide against every variant with PeptiVerse.
4. Print results sorted by probability, binding score, and combined score.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import types

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent  
ROOT       = SCRIPT_DIR.parent               

sys.path.insert(0, str(SCRIPT_DIR))          
sys.path.insert(0, str(ROOT / "MOG-DFM"))     
sys.path.insert(0, str(ROOT / "PeptiVerse"))  

for _dep in ("torchdiffeq", "esm"):
    if _dep not in sys.modules:
        _stub = types.ModuleType(_dep)
        _stub.odeint = None          
        sys.modules[_dep] = _stub

from compute_probabilities import (
    compute_leaf_probabilities,
    load_sequences,
    accession_from_leaf,
)

from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver.discrete_solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from models.peptide_models import CNNModel

from inference import WTEmbedder, load_binding_model

NWK_FILE        = SCRIPT_DIR / "DENV3_tree.nwk"
CSV_FILE        = SCRIPT_DIR / "cluster2and6.obs.csv"
PEPDM_CKPT      = (
    ROOT / "MOG-DFM" / "ckpt" / "peptide"
    / "cnn_epoch200_lr0.0001_embed512_hidden256_loss3.1051.ckpt"
)
PEPTIVERSE_ROOT = ROOT / "PeptiVerse"

#  hyper-parameters
TOP_K      = 10
VOCAB_SIZE = 24
STEP_SIZE  = 1 / 100

def get_top10_dict() -> Dict[str, Dict]:
    """Return top-k variants with probability and sequence keyed by leaf name."""
    _, sorted_pairs = compute_leaf_probabilities(NWK_FILE)
    sequences = load_sequences(CSV_FILE)

    top10: Dict[str, Dict] = {}
    for name, prob in sorted_pairs[:TOP_K]:
        seq = sequences.get(accession_from_leaf(name), "")
        top10[name] = {"probability": prob, "sequence": seq}

    print(f"\n[Step 1] Top-{TOP_K} variants:")
    print(f"  {'Rank':<5} {'Variant':<50} {'Probability':>12}  Seq?")
    print("  " + "-" * 75)
    for i, (name, entry) in enumerate(top10.items(), 1):
        has_seq = "yes" if entry["sequence"] else "NO"
        print(f"  {i:<5} {name:<50} {entry['probability']:>12.8f}  {has_seq}")

    return top10

class _WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        return torch.softmax(self.model(x, t), dim=-1)


def generate_peptide(length: int, device: torch.device) -> str:
    if not PEPDM_CKPT.exists():
        raise FileNotFoundError(f"PepDFM checkpoint not found:\n  {PEPDM_CKPT}")

    dfm = CNNModel(alphabet_size=VOCAB_SIZE, embed_dim=512, hidden_dim=256).to(device)
    dfm.load_state_dict(torch.load(str(PEPDM_CKPT), map_location=device))
    dfm.eval()

    scheduler = PolynomialConvexScheduler(n=2.0)
    path      = MixtureDiscreteProbPath(scheduler=scheduler)
    solver    = MixtureDiscreteEulerSolver(
        model=_WrappedModel(dfm), path=path, vocabulary_size=VOCAB_SIZE
    )

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    x_init = torch.randint(low=4, high=VOCAB_SIZE, size=(1, length), device=device)
    zeros  = torch.zeros((1, 1), dtype=x_init.dtype, device=device)
    twos   = torch.full((1, 1), 2, dtype=x_init.dtype, device=device)
    x_init = torch.cat([zeros, x_init, twos], dim=1)

    sol = solver.sample(
        x_init=x_init,
        step_size=STEP_SIZE,
        verbose=False,
        time_grid=torch.tensor([0.0, 1.0 - 1e-3]),
    )
    peptide = tokenizer.decode(sol[0].tolist()).replace(" ", "")[5:-5]
    return peptide


def compute_binding_affinities(
    peptide: str,
    top10: Dict[str, Dict],
    device: torch.device,
) -> List[Dict]:
    model_pt = PEPTIVERSE_ROOT / "training_classifiers" / "binding_affinity" / "wt_wt_unpooled" / "best_model.pt"
    binding_model = load_binding_model(model_pt, pooled_or_unpooled="unpooled", device=device)
    embedder = WTEmbedder(device=device)

    results: List[Dict] = []
    for variant_name, entry in tqdm(top10.items(), desc="Scoring variants"):
        seq = entry["sequence"]
        if not seq:
            print(f"  [WARNING] No sequence for '{variant_name}', skipping.")
            continue
        T, Mt = embedder.unpooled(seq)
        B, Mb = embedder.unpooled(peptide)
        with torch.no_grad():
            reg, _ = binding_model(T, Mt, B, Mb)
        affinity = float(reg.squeeze().cpu().item())
        results.append(
            {
                "variant":       variant_name,
                "probability":   entry["probability"],
                "binding_score": affinity,
            }
        )
    return results


def _print_table(rows: List[Dict], title: str) -> None:
    W = 50
    print(f"\n── {title} ──")
    print(f"  {'Rank':<5} {'Variant':<{W}} {'Prob':>10} {'Affinity':>10} {'Combined':>10}")
    print("  " + "-" * (5 + W + 35))
    for i, r in enumerate(rows, 1):
        combined = r["probability"] * r["binding_score"]
        print(
            f"  {i:<5} {r['variant']:<{W}} "
            f"{r['probability']:>10.6f} "
            f"{r['binding_score']:>10.4f} "
            f"{combined:>10.6f}"
        )


def print_results(peptide: str, results: List[Dict]) -> None:
    print(f"\n{'='*70}")
    print(f"  Generated peptide : {peptide}")
    print(f"  Variants scored   : {len(results)}")
    print(f"{'='*70}")

    _print_table(
        sorted(results, key=lambda x: x["probability"],   reverse=True),
        "Sorted by probability",
    )
    _print_table(
        sorted(results, key=lambda x: x["binding_score"], reverse=True),
        "Sorted by binding score (pKd – higher = tighter)",
    )
    _print_table(
        sorted(results, key=lambda x: x["probability"] * x["binding_score"], reverse=True),
        "Sorted by combined score (probability × affinity)",
    )

def run(peptide_length: int = 8, device_str: str = "cpu") -> List[Dict]:
    device = torch.device(device_str)

    # 1. top-10 dict with probabilities + sequences
    top10 = get_top10_dict()

    # 2. generate ONE peptide – stored once, reused for all variants
    print(f"\n[Step 2] Generating ONE peptide with PepDFM "
          f"(length={peptide_length}, device={device}) ...")
    peptide = generate_peptide(length=peptide_length, device=device)
    print(f"  Generated peptide: {peptide}")

    # 3. score peptide vs each variant
    print(f"\n[Step 3] Computing binding affinities with PeptiVerse ...")
    results = compute_binding_affinities(peptide, top10, device)

    # 5. display
    print_results(peptide, results)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PepDFM + PeptiVerse dengue pipeline")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string, e.g. 'cpu', 'cuda', 'cuda:0'  (default: auto-detect)",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=8,
        help="Peptide length to generate with PepDFM (default: 8)",
    )
    args, _ = parser.parse_known_args()  
    run(peptide_length=args.length, device_str=args.device)
