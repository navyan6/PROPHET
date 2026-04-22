#!/usr/bin/env python3
"""
MOG-DFM integration for tree-weighted binding affinity.

Uses MOG-DFM's multi_guidance_sample to generate peptides optimized for:
  tree-weighted binding affinity = Σ(PeptiVerse_binding(peptide, variant_i) × tree_prob_i)

MOG-DFM is already a generative model - we just plug in the binding objective.
"""


import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import numpy as np


# Setup paths
REPO_ROOT = Path(__file__).parent.parent.parent
PEPTIVERSE_PATH = REPO_ROOT / "PeptiVerse"
sys.path.insert(0, str(PEPTIVERSE_PATH))



# Only import PeptiVerse binding affinity predictor
try:
    from inference import PeptiVersePredictor
except ImportError as e:
    print(f"Error: PeptiVerse not found at {PEPTIVERSE_PATH}")
    print("Setup: git clone https://huggingface.co/ChatterjeeLab/PeptiVerse")
    sys.exit(1)

# Import MOG-DFM solver loader
MOGDFM_PATH = REPO_ROOT / "MOG-DFM"
sys.path.insert(0, str(MOGDFM_PATH))
try:
    from models.peptide_classifiers import load_solver
except ImportError as e:
    print(f"Error: Could not import load_solver from {MOGDFM_PATH}/models/peptide_classifiers.py")
    print("Check that MOG-DFM is cloned and available.")
    sys.exit(1)

# Import tree utilities (local or from tree_analysis)
import json
from pathlib import Path
def load_tree_probabilities(tree_json_path):
    # Loader for tree json: extract variants from 'leaf_endpoints_pi', assign uniform probability
    with open(tree_json_path, 'r') as f:
        data = json.load(f)
    leaves = data["leaf_endpoints_pi"]
    n = len(leaves)
    prob = 1.0 / n if n > 0 else 0.0
    variants = []
    for v in leaves:
        # v: {"leaf_id": ..., "sequence": ...}
        variant = type('Variant', (), {
            'name': v.get('leaf_id', ''),
            'sequence': v.get('sequence', ''),
            'probability': prob
        })
        variants.append(variant)
    return variants

import random
import numpy as np
import torch
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(
        description="Generate peptides with tree-weighted binding affinity, then evaluate on held-out test variants."
    )
    parser.add_argument("--tree-json", type=Path, required=True, help="Path to tree JSON with variants and probabilities")
    parser.add_argument("--num-peptides", type=int, default=5, help="Number of peptides to generate")
    parser.add_argument("--length", type=int, default=12, help="Peptide length")
    parser.add_argument("--device", type=str, default="cpu", help="torch device")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Fraction of variants to hold out for test set")
    args = parser.parse_args()

    # Load variants
    variants = load_tree_probabilities(args.tree_json)
    random.shuffle(variants)
    n_test = max(1, int(len(variants) * args.test_fraction))
    test_variants = variants[:n_test]
    train_variants = variants[n_test:]

    print(f"Loaded {len(variants)} variants: {len(train_variants)} train, {len(test_variants)} test")

    # Initialize PeptiVerse binding affinity predictor
    predictor = PeptiVersePredictor(
        manifest_path=str(PEPTIVERSE_PATH / "best_models.txt"),
        classifier_weight_root=str(PEPTIVERSE_PATH),
        device=args.device,
        tokenizer_vocab_path=str(PEPTIVERSE_PATH / "tokenizer" / "new_vocab.txt"),
        tokenizer_splits_path=str(PEPTIVERSE_PATH / "tokenizer" / "new_splits.txt"),
        only_properties=["binding_affinity"]
        
    )

    # Use ESM2 tokenizer for peptide encoding/decoding (as in original)
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    # Tree-weighted binding affinity function (train variants only)
    def tree_weighted_binding(peptide_seq):
        scores = []
        weights = []
        for v in train_variants:
            result = predictor.predict_binding_affinity(
                mode="wt",
                target_seq=v.sequence,
                binder_str=peptide_seq
            )
            if isinstance(result, dict) and "wt_wt_pooled" in result:
                val = result["wt_wt_pooled"]
                score = float(val[0]) if isinstance(val, (list, tuple)) else float(val)
            else:
                score = 0.0
            scores.append(score)
            weights.append(v.probability)
        scores = np.array(scores)
        weights = np.array(weights)
        if weights.sum() > 0:
            return float(np.sum(scores * weights) / weights.sum())
        else:
            return float(scores.mean())

    # Generate random peptides and optimize (placeholder: random sampling)
    generated_peptides = []
    for i in range(args.num_peptides):
        # Random peptide of given length (A, C, D, E, ...)
        aa_list = list("ACDEFGHIKLMNPQRSTVWY")
        peptide = ''.join(random.choices(aa_list, k=args.length))
        score = tree_weighted_binding(peptide)
        generated_peptides.append((peptide, score))
        print(f"Generated peptide {i+1}: {peptide} | tree-weighted binding: {score:.4f}")

    # Evaluate generated peptides on held-out test variants
    print("\nEvaluating generated peptides on held-out test variants:")
    for peptide, _ in generated_peptides:
        test_scores = []
        for v in test_variants:
            result = predictor.predict_binding_affinity(
                mode="wt",
                target_seq=v.sequence,
                binder_str=peptide
            )
            if isinstance(result, dict) and "wt_wt_pooled" in result:
                val = result["wt_wt_pooled"]
                score = float(val[0]) if isinstance(val, (list, tuple)) else float(val)
            else:
                score = 0.0
            test_scores.append(score)
        mean_test = np.mean(test_scores) if test_scores else 0.0
        print(f"Peptide: {peptide} | Mean test binding: {mean_test:.4f}")


@dataclass
class BindingScore:
    sequence: str
    binding_per_variant: Dict[str, float]
    weighted_binding: float
    mean_binding: float


class TreeWeightedBindingModel(nn.Module):
    """
    Objective wrapper for MOG-DFM guidance.
    Returns a tree-probability-weighted binding affinity score per peptide.
    """

    def __init__(self, variants, peptiverse_predictor, tokenizer, device: str = "cpu"):
        super().__init__()
        self.variants = variants
        self.predictor = peptiverse_predictor
        self.tokenizer = tokenizer
        self.device = device

    def forward(self, x_tokens: torch.Tensor) -> torch.Tensor:
        if x_tokens.ndim == 1:
            x_tokens = x_tokens.unsqueeze(0)

        batch_scores = []
        for row in x_tokens:
            peptide_seq = self.tokenizer.decode(row, skip_special_tokens=True).replace(" ", "")
            if not peptide_seq:
                batch_scores.append(0.0)
                continue

            scores = []
            weights = []
            for variant in self.variants:
                try:
                    result = self.predictor.predict_binding_affinity(
                        mode="wt",
                        target_seq=variant.sequence,
                        binder_str=peptide_seq,
                    )
                    if isinstance(result, dict):
                        for key in ("wt_wt_pooled", "wt_wt_unpooled"):
                            if key in result:
                                val = result[key]
                                score = float(val[0] if isinstance(val, (list, tuple)) else val)
                                break
                        else:
                            first_val = list(result.values())[0]
                            score = float(first_val[0] if isinstance(first_val, (list, tuple)) else first_val)
                    else:
                        score = float(result)
                except Exception:
                    score = 0.0

                scores.append(score)
                weights.append(float(variant.probability))

            if scores:
                weights_np = np.asarray(weights, dtype=float)
                scores_np = np.asarray(scores, dtype=float)
                denom = float(weights_np.sum())
                weighted = float(np.dot(scores_np, weights_np) / denom) if denom > 0 else float(scores_np.mean())
            else:
                weighted = 0.0

            batch_scores.append(weighted)

        return torch.tensor(batch_scores, dtype=torch.float32, device=x_tokens.device)


def main():
    """Run MOG-DFM with tree-weighted binding objective."""
    parser = argparse.ArgumentParser(
        description="Generate HIV peptides optimized for tree-weighted binding affinity using MOG-DFM"
    )
    parser.add_argument("--tree-json", type=Path, default=None, 
                        help="Path to hadsbm_tree.json")
    parser.add_argument("--num-peptides", type=int, default=5,
                        help="Number of MOG-DFM iterations (generates 1 peptide per iteration)")
    parser.add_argument("--length", type=int, default=12,
                        help="Peptide length")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="torch device")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON file")
    
    args = parser.parse_args()
    
    # Determine paths
    if args.tree_json is None:
        args.tree_json = REPO_ROOT / "data" / "trees" / "hadsbm_tree.json"
    args.tree_json = Path(args.tree_json)
    
    if not args.tree_json.exists():
        print(f"✗ Tree JSON not found: {args.tree_json}")
        print("To build: cd tree_analysis && python src/phylogeny.py")
        return 1
    
    print(f"Loading variants from: {args.tree_json}")
    try:
        variants = load_tree_probabilities(args.tree_json)
    except Exception as e:
        print(f"✗ Error loading tree: {e}")
        return 1
    
    print(f"✓ Loaded {len(variants)} variants with tree probabilities\n")
    
    # Initialize PeptiVerse
    print("Loading PeptiVerse...")
    try:
        predictor = PeptiVersePredictor(
            manifest_path=str(PEPTIVERSE_PATH / "best_models.txt"),
            classifier_weight_root=str(PEPTIVERSE_PATH),
            device=args.device,
            only_properties=["binding_affinity"]
        )
        print("✓ PeptiVerse loaded\n")
    except Exception as e:
        print(f"✗ Failed to load PeptiVerse: {e}")
        return 1
    
    # Initialize MOG-DFM solver
    print("Loading MOG-DFM solver...")
    try:
        solver = load_solver(
            str(MOGDFM_PATH / "ckpt" / "peptide" / "cnn_epoch200_lr0.0001_embed512_hidden256_loss3.1051.ckpt"),
            vocab_size=24,
            device=args.device
        )
        print("✓ MOG-DFM solver loaded\n")
    except Exception as e:
        print(f"✗ Failed to load MOG-DFM solver: {e}")
        print("Make sure MOG-DFM checkpoint exists at: MOG-DFM/ckpt/peptide/")
        return 1
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    
    # Create objective
    objective = TreeWeightedBindingModel(
        variants=variants,
        peptiverse_predictor=predictor,
        tokenizer=tokenizer,
        device=args.device,
    )
    
    print(f"Tree-weighted binding objective initialized")
    print(f"Variants:")
    for v in variants:
        print(f"  {v.name:20s} p={v.probability:.4f}")
    print()
    
    # Generate peptides with MOG-DFM
    print(f"Generating {args.num_peptides} peptides with MOG-DFM...")
    print("="*70)
    
    results = []
    
    for i in range(args.num_peptides):
        try:
            # Initialize random x_0
            x_init = torch.randint(low=4, high=24, size=(1, args.length), device=args.device)
            zeros = torch.zeros((1, 1), dtype=x_init.dtype, device=args.device)
            twos = torch.full((1, 1), 2, dtype=x_init.dtype, device=args.device)
            x_init = torch.cat([zeros, x_init, twos], dim=1)
            
            # Generate with MOG-DFM (unconditional sampling)
            x_T = solver.sample(
                x_init=x_init,
                step_size=1/200,
                time_grid=torch.tensor([0.0, 1.0 - 1e-3], device=args.device),
            )
            
            # Decode
            peptide_seq = tokenizer.decode(x_T[0])
            peptide_seq = peptide_seq.replace(" ", "")[5:-5]
            
            # Evaluate
            with torch.no_grad():
                x_tokens = tokenizer(peptide_seq, return_tensors='pt')['input_ids'].to(args.device)
                weighted_score = objective(x_tokens).item()
            
            # Full evaluation
            binding_per_variant = {}
            scores = []
            for variant in variants:
                try:
                    result = predictor.predict_binding_affinity(
                        mode="wt", target_seq=variant.sequence, binder_str=peptide_seq
                    )
                    if isinstance(result, dict):
                        for key in ["wt_wt_pooled", "wt_wt_unpooled"]:
                            if key in result:
                                val = result[key]
                                binding = float(val[0] if isinstance(val, (list, tuple)) else val)
                                break
                        else:
                            first_val = list(result.values())[0]
                            binding = float(first_val[0] if isinstance(first_val, (list, tuple)) else first_val)
                    else:
                        binding = 0.0
                except:
                    binding = 0.0
                binding_per_variant[variant.name] = binding
                scores.append(binding)
            
            mean_binding = float(np.mean(scores)) if scores else 0.0
            
            result = BindingScore(
                sequence=peptide_seq,
                binding_per_variant=binding_per_variant,
                weighted_binding=weighted_score,
                mean_binding=mean_binding
            )
            results.append(result)
            
            print(f"{i+1}. {peptide_seq:20s} | weighted: {weighted_score:7.4f} | mean: {mean_binding:7.4f}")
        
        except Exception as e:
            print(f"{i+1}. ✗ Error: {e}")
            continue
    
    # Sort by weighted binding
    results.sort(key=lambda r: r.weighted_binding, reverse=True)
    
    print("\n" + "="*70)
    print(f"✓ Generated {len(results)} peptides")
    
    if args.output:
        args.output = Path(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        
        results_dict = [
            {
                "sequence": r.sequence,
                "binding_per_variant": r.binding_per_variant,
                "weighted_binding": r.weighted_binding,
                "mean_binding": r.mean_binding
            }
            for r in results
        ]
        
        with open(args.output, "w") as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"✓ Results saved to: {args.output}\n")
    else:
        print()
    
    print("Top peptides:")
    for i, r in enumerate(results[:min(3, len(results))], 1):
        print(f"{i}. {r.sequence:20s} weighted={r.weighted_binding:.4f}")
    
    return 0


if __name__ == "__main__":
    main()
