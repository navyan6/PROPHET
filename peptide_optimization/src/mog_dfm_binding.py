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

import types
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer
import inspect

# Setup paths
REPO_ROOT = Path(__file__).parent.parent.parent
MOGDFM_PATH = REPO_ROOT / "MOG-DFM"
PEPTIVERSE_PATH = REPO_ROOT / "PeptiVerse"

sys.path.insert(0, str(MOGDFM_PATH))
sys.path.insert(0, str(PEPTIVERSE_PATH))

# Stub missing optional deps that MOG-DFM imports at module level but doesn't
# need for the CNN peptide model / solver path we actually use.
# NOTE: only stub packages that are genuinely absent; torchdiffeq is installed.
for _name in ("esm",):
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)
for _name in ("modules", "modules.bindevaluator_modules"):
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)

# Import tree utilities
try:
    from tree_utils import load_tree_probabilities, VariantWithProbability
except ImportError as e:
    print(f"Error importing tree_utils: {e}")
    sys.exit(1)

# Import PeptiVerse (lower-level API: binding affinity only, no sklearn/cuml models)
try:
    from inference import WTEmbedder, load_binding_model
except ImportError as e:
    print(f"Error: PeptiVerse not found at {PEPTIVERSE_PATH}: {e}")
    print("Setup: git clone https://huggingface.co/ChatterjeeLab/PeptiVerse")
    sys.exit(1)

BINDING_MODEL_PT = (
    PEPTIVERSE_PATH / "training_classifiers" / "binding_affinity"
    / "wt_wt_unpooled" / "best_model.pt"
)

# Import MOG-DFM
try:
    from models.peptide_classifiers import load_solver
except ImportError as e:
    print(f"Error: MOG-DFM not found at {MOGDFM_PATH}: {e}")
    sys.exit(1)


@dataclass
class BindingScore:
    """Results for one peptide."""
    sequence: str
    binding_per_variant: Dict[str, float]
    weighted_binding: float
    mean_binding: float


class TreeWeightedBindingModel(nn.Module):
    """
    MOG-DFM objective: tree-weighted binding affinity.
    Uses PeptiVerse wt_wt_unpooled model directly (no full PeptiVersePredictor).
    """

    def __init__(
        self,
        variants: List[VariantWithProbability],
        embedder,          # WTEmbedder instance
        binding_model,     # loaded binding affinity nn.Module
        tokenizer,
        device: str = "cpu"
    ):
        super().__init__()
        self.variants = variants
        self.embedder = embedder
        self.binding_model = binding_model
        self.tokenizer = tokenizer
        self.device = device

        probs = [v.probability for v in variants]
        self.weights = torch.tensor(probs, dtype=torch.float32, device=device)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Peptide token IDs (batch_size, seq_len)
            t: Time (MOG-DFM compat, unused)
        Returns:
            Tensor (batch_size,) of tree-weighted binding scores
        """
        peptide_seqs = self.tokenizer.batch_decode(x)
        peptide_seqs = [seq.replace(" ", "")[5:-5] for seq in peptide_seqs]

        weighted_scores = []
        for peptide_seq in peptide_seqs:
            binding_scores = []
            for variant in self.variants:
                try:
                    T, Mt = self.embedder.unpooled(variant.sequence)
                    B, Mb = self.embedder.unpooled(peptide_seq)
                    with torch.no_grad():
                        reg, _ = self.binding_model(T, Mt, B, Mb)
                    binding_scores.append(float(reg.squeeze().cpu().item()))
                except Exception:
                    binding_scores.append(0.0)

            bs = torch.tensor(binding_scores, dtype=torch.float32, device=self.device)
            weighted_scores.append((bs * self.weights).sum().item())

        return torch.tensor(weighted_scores, dtype=torch.float32, device=self.device)


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
    parser.add_argument("--top-k", type=int, default=None,
                        help="Limit to top-K variants by tree probability (useful for large trees)")

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
        variants = load_tree_probabilities(args.tree_json, top_k=args.top_k)
    except Exception as e:
        print(f"✗ Error loading tree: {e}")
        return 1
    
    print(f"✓ Loaded {len(variants)} variants with tree probabilities\n")
    
    # Initialize PeptiVerse (binding affinity model only)
    print("Loading PeptiVerse binding model...")
    try:
        embedder = WTEmbedder(device=args.device)
        binding_model = load_binding_model(
            BINDING_MODEL_PT,
            pooled_or_unpooled="unpooled",
            device=args.device,
        )
        print("✓ PeptiVerse binding model loaded\n")
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
        embedder=embedder,
        binding_model=binding_model,
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
            
            # Full per-variant evaluation
            binding_per_variant = {}
            scores = []
            for variant in variants:
                try:
                    T, Mt = embedder.unpooled(variant.sequence)
                    B, Mb = embedder.unpooled(peptide_seq)
                    with torch.no_grad():
                        reg, _ = binding_model(T, Mt, B, Mb)
                    binding = float(reg.squeeze().cpu().item())
                except Exception:
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
    sys.exit(main())
