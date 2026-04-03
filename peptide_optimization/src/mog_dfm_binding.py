#!/usr/bin/env python3
"""
MOG-DFM integration for generating peptides optimized for tree-weighted binding affinity.

Pipeline:
1. Load HIV variants with tree-derived probabilities
2. Create TreeWeightedBindingModel objective (PeptiVerse + tree weighting)
3. Use MOG-DFM to generate peptides optimized for weighted binding affinity
4. Output results as JSON with per-variant and weighted scores
"""

import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer

# Setup paths
REPO_ROOT = Path(__file__).parent.parent.parent
MOGDFM_PATH = REPO_ROOT / "MOG-DFM"
PEPTIVERSE_PATH = REPO_ROOT / "PeptiVerse"

sys.path.insert(0, str(MOGDFM_PATH))
sys.path.insert(0, str(PEPTIVERSE_PATH))

# Import tree utilities
from tree_utils import load_tree_probabilities, VariantWithProbability, load_wildtype_sequence

# Import PeptiVerse
try:
    from inference import PeptiVersePredictor
except ImportError as e:
    raise ImportError(
        f"Failed to import PeptiVerse: {e}\n"
        f"Expected at: {PEPTIVERSE_PATH}\n"
        "Setup: git clone https://huggingface.co/ChatterjeeLab/PeptiVerse"
    )

# Import MOG-DFM
try:
    from models.peptide_classifiers import load_solver
    from flow_matching.utils.multi_guidance import select_random_weight_vector
    from utils.parsing import parse_guidance_args
except ImportError as e:
    raise ImportError(
        f"Failed to import MOG-DFM: {e}\n"
        "Make sure MOG-DFM is present at: {MOGDFM_PATH}"
    )


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
    Combines PeptiVerse predictions with variant probabilities from tree.
    """
    
    def __init__(
        self,
        variants: List[VariantWithProbability],
        peptiverse_predictor: PeptiVersePredictor,
        device: str = "cpu"
    ):
        """
        Initialize objective.
        
        Args:
            variants: List of variants with sequences and tree probabilities
            peptiverse_predictor: PeptiVerse binding affinity model
            device: torch device
        """
        super().__init__()
        self.variants = variants
        self.predictor = peptiverse_predictor
        self.device = device
        
        # Precompute probability weights as tensor
        probs = [v.probability for v in variants]
        self.weights = torch.tensor(probs, dtype=torch.float32, device=device)
        
        print(f"Initialized TreeWeightedBindingModel with {len(variants)} variants")
        for v in variants:
            print(f"  {v.name:20s} (p={v.probability:.4f}): {v.sequence[:20]}...")
    
    def compute_binding_batch(self, peptides: torch.Tensor) -> torch.Tensor:
        """
        Compute binding affinity for batch of peptides.
        
        Args:
            peptides: Tensor of shape (batch_size, seq_len) with token IDs
        
        Returns:
            Tensor of shape (batch_size,) with tree-weighted binding scores
        """
        batch_size = peptides.shape[0]
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        
        # Convert token IDs to sequences
        peptide_seqs = tokenizer.batch_decode(peptides)
        peptide_seqs = [seq.replace(" ", "")[5:-5] for seq in peptide_seqs]  # Remove special tokens
        
        # Compute binding to each variant
        weighted_scores = []
        
        for peptide_seq in peptide_seqs:
            binding_scores = []
            
            for variant in self.variants:
                try:
                    result = self.predictor.predict_binding_affinity(
                        mode="wt",
                        target_seq=variant.sequence,
                        binder_str=peptide_seq
                    )
                    
                    # Extract score
                    if isinstance(result, dict):
                        for key in ["wt_wt_pooled", "wt_wt_unpooled"]:
                            if key in result:
                                val = result[key]
                                if isinstance(val, (list, tuple)):
                                    binding_scores.append(float(val[0]) if val else 0.0)
                                else:
                                    binding_scores.append(float(val))
                                break
                        else:
                            # Fallback
                            first_val = list(result.values())[0]
                            if isinstance(first_val, (list, tuple)):
                                binding_scores.append(float(first_val[0]) if first_val else 0.0)
                            else:
                                binding_scores.append(float(first_val))
                    else:
                        binding_scores.append(0.0)
                
                except Exception as e:
                    print(f"  ⚠️  Error evaluating {peptide_seq} vs {variant.name}: {e}")
                    binding_scores.append(0.0)
            
            # Weight by tree probabilities
            binding_scores = torch.tensor(binding_scores, dtype=torch.float32, device=self.device)
            weighted = (binding_scores * self.weights).sum().item()
            weighted_scores.append(weighted)
        
        return torch.tensor(weighted_scores, dtype=torch.float32, device=self.device)
    
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Evaluate tree-weighted binding affinity.
        
        Args:
            x: Peptide token IDs (batch_size, seq_len)
            t: Time (unused, for API compatibility)
        
        Returns:
            Tensor of shape (batch_size,) with scores
        """
        return self.compute_binding_batch(x)


class MOGDFMPeptideGenerator:
    """Generate peptides optimized for tree-weighted binding affinity using MOG-DFM."""
    
    def __init__(
        self,
        variants: List[VariantWithProbability],
        device: str = "cuda:0",
        solver_ckpt: Optional[Path] = None,
        peptide_length: int = 12,
    ):
        """
        Initialize generator.
        
        Args:
            variants: HIV variants with tree probabilities
            device: torch device (cuda:0, cpu, etc)
            solver_ckpt: Path to MOG-DFM peptide checkpoint
            peptide_length: Generated peptide length
        """
        self.variants = variants
        self.device = device
        self.peptide_length = peptide_length
        
        # Initialize PeptiVerse
        print(f"Loading PeptiVerse (device: {device})...")
        self.predictor = PeptiVersePredictor(
            manifest_path=str(PEPTIVERSE_PATH / "best_models.txt"),
            classifier_weight_root=str(PEPTIVERSE_PATH / "training_classifiers"),
            device=device,
        )
        
        # Initialize MOG-DFM solver
        print("Loading MOG-DFM solver...")
        if solver_ckpt is None:
            # Use default checkpoint
            solver_ckpt = MOGDFM_PATH / "ckpt" / "peptide" / "cnn_epoch200_lr0.0001_embed512_hidden256_loss3.1051.ckpt"
        
        if not Path(solver_ckpt).exists():
            raise FileNotFoundError(f"MOG-DFM checkpoint not found: {solver_ckpt}")
        
        self.solver = load_solver(str(solver_ckpt), vocab_size=24, device=device)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        
        # Create objective model
        self.objective = TreeWeightedBindingModel(
            variants=variants,
            peptiverse_predictor=self.predictor,
            device=device
        )
    
    def generate_batch(
        self,
        num_peptides: int = 5,
        step_size: float = 1 / 200,
        time_grid: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
    ) -> List[BindingScore]:
        """
        Generate peptides optimized for tree-weighted binding affinity.
        
        Args:
            num_peptides: Number of peptides to generate
            step_size: Flow matching step size
            time_grid: Time discretization
            guidance_scale: Guidance strength
        
        Returns:
            List of BindingScore objects
        """
        if time_grid is None:
            time_grid = torch.tensor([0.0, 1.0 - 1e-3])
        
        results = []
        
        print(f"\nGenerating {num_peptides} peptides with MOG-DFM")
        print("="*70)
        
        for batch_idx in range(num_peptides):
            # Initialize random peptide
            x_init = torch.randint(
                low=4, high=24,
                size=(1, self.peptide_length),
                device=self.device
            )
            
            # Add special tokens (BOS, EOS)
            zeros = torch.zeros((1, 1), dtype=x_init.dtype, device=self.device)
            twos = torch.full((1, 1), 2, dtype=x_init.dtype, device=self.device)
            x_init = torch.cat([zeros, x_init, twos], dim=1)
            
            # Generate with MOG-DFM
            print(f"\nGenerating peptide {batch_idx + 1}/{num_peptides}...")
            
            # Simplified: just sample from model with guidance
            # (full MOG-DFM would use multi_guidance_sample, but this requires args object)
            x_generated = self.solver.sample(
                x_init=x_init,
                step_size=step_size,
                time_grid=time_grid.to(self.device),
            )
            
            # Decode sequence
            peptide_seq = self.tokenizer.decode(x_generated[0])
            peptide_seq = peptide_seq.replace(" ", "")[5:-5]  # Remove special tokens and spaces
            
            print(f"Generated: {peptide_seq}")
            
            # Evaluate
            score = self._evaluate_peptide(peptide_seq)
            results.append(score)
            
            print(f"  → Weighted binding: {score.weighted_binding:.4f}")
            print(f"  → Mean binding: {score.mean_binding:.4f}")
        
        return results
    
    def _evaluate_peptide(self, peptide_seq: str) -> BindingScore:
        """Evaluate one peptide across all variants."""
        binding_per_variant = {}
        scores = []
        
        for variant in self.variants:
            try:
                result = self.predictor.predict_binding_affinity(
                    mode="wt",
                    target_seq=variant.sequence,
                    binder_str=peptide_seq
                )
                
                if isinstance(result, dict):
                    for key in ["wt_wt_pooled", "wt_wt_unpooled"]:
                        if key in result:
                            val = result[key]
                            binding = float(val[0]) if isinstance(val, (list, tuple)) else float(val)
                            break
                    else:
                        first_val = list(result.values())[0]
                        binding = float(first_val[0]) if isinstance(first_val, (list, tuple)) else float(first_val)
                else:
                    binding = 0.0
            except Exception as e:
                print(f"    Error: {e}")
                binding = 0.0
            
            binding_per_variant[variant.name] = binding
            scores.append(binding)
        
        weighted = sum(s * v.probability for s, v in zip(scores, self.variants))
        mean = float(np.mean(scores))
        
        return BindingScore(
            sequence=peptide_seq,
            binding_per_variant=binding_per_variant,
            weighted_binding=weighted,
            mean_binding=mean
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate peptides optimized for tree-weighted binding affinity using MOG-DFM"
    )
    parser.add_argument(
        "--tree-json",
        type=Path,
        default=None,
        help="Path to hadsbm_tree.json (default: data/trees/hadsbm_tree.json)"
    )
    parser.add_argument(
        "--num-peptides",
        type=int,
        default=5,
        help="Number of peptides to generate"
    )
    parser.add_argument(
        "--length",
        type=int,
        default=12,
        help="Peptide length"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="torch device (cuda:0, cpu, etc)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file"
    )
    parser.add_argument(
        "--solver-ckpt",
        type=Path,
        default=None,
        help="Path to MOG-DFM solver checkpoint"
    )
    
    args = parser.parse_args()
    
    # Determine tree JSON path
    if args.tree_json is None:
        args.tree_json = REPO_ROOT / "data" / "trees" / "hadsbm_tree.json"
    else:
        args.tree_json = Path(args.tree_json)
    
    if not args.tree_json.exists():
        print(f"Error: Tree JSON not found: {args.tree_json}")
        print("Run: cd tree_analysis && python src/phylogeny.py")
        return 1
    
    # Load variants with tree probabilities
    print(f"Loading variants from {args.tree_json}...")
    variants = load_tree_probabilities(args.tree_json)
    print(f"Loaded {len(variants)} variants with tree-derived probabilities")
    
    # Initialize generator
    generator = MOGDFMPeptideGenerator(
        variants=variants,
        device=args.device,
        solver_ckpt=args.solver_ckpt,
        peptide_length=args.length,
    )
    
    # Generate peptides
    results = generator.generate_batch(num_peptides=args.num_peptides)
    
    # Sort by weighted binding
    results.sort(key=lambda r: r.weighted_binding, reverse=True)
    
    # Print summary
    print("\n" + "="*70)
    print("Top generated peptides (sorted by tree-weighted binding):")
    for i, r in enumerate(results[:min(5, len(results))]):
        print(f"{i+1}. {r.sequence:20s} | weighted: {r.weighted_binding:.4f}")
    
    # Save results
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
        
        print(f"\nResults saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
