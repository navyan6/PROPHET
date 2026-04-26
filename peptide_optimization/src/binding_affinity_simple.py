#!/usr/bin/env python3
"""
Binding affinity computation for HIV variants using tree probabilities.

Pipeline:
1. Load HIV variant sequences with probabilities from tree_analysis/hadsbm_tree.json
2. Generate random peptides
3. Compute binding affinity to each variant using PeptiVerse
4. Weight affinities by variant probability from tree
5. Output results as JSON
"""

import sys
import json
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np

# Setup paths
REPO_ROOT = Path(__file__).parent.parent.parent
PEPTIVERSE_PATH = REPO_ROOT / "PeptiVerse"
sys.path.insert(0, str(PEPTIVERSE_PATH))

# Import tree utilities
from tree_utils import load_tree_probabilities, VariantWithProbability

# Import PeptiVerse (lower-level API: binding affinity only, no sklearn/cuml models)
try:
    from inference import WTEmbedder, load_binding_model
except ImportError as e:
    raise ImportError(
        f"Failed to import PeptiVerse: {e}\n"
        f"Expected at: {PEPTIVERSE_PATH}\n"
        "Setup: git clone https://huggingface.co/ChatterjeeLab/PeptiVerse"
    )

BINDING_MODEL_PT = (
    PEPTIVERSE_PATH / "training_classifiers" / "binding_affinity"
    / "wt_wt_unpooled" / "best_model.pt"
)


@dataclass
class BindingScore:
    """Results for one peptide evaluation."""
    sequence: str
    binding_per_variant: Dict[str, float]
    weighted_binding: float  
    mean_binding: float


class VariantBindingPredictor:
    """Compute binding affinity for peptides across HIV variants."""
    
    def __init__(
        self,
        variants: List[VariantWithProbability],
        device: str = "cpu"
    ):
        """
        Initialize predictor.
        
        Args:
            variants: List of variants with sequences and probabilities
            device: "cpu" or "cuda:0" etc.
        """
        self.variants = variants
        self.device = device
        
        # Validate probabilities
        total_prob = sum(v.probability for v in variants)
        if abs(total_prob - 1.0) > 1e-6:
            print(f"Probabilities don't sum to 1.0 ({total_prob:.4f}), normalizing...")
            for v in variants:
                v.probability /= total_prob
        
        # Initialize PeptiVerse (binding affinity model only)
        print(f"Loading PeptiVerse binding model (device: {device})...")
        import torch
        self._torch_device = torch.device(device)
        try:
            self.embedder = WTEmbedder(device=self._torch_device)
            self.binding_model = load_binding_model(
                BINDING_MODEL_PT,
                pooled_or_unpooled="unpooled",
                device=self._torch_device,
            )
            print(f"✓ PeptiVerse binding model loaded")
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize PeptiVerse: {e}\n"
                "Make sure PeptiVerse is cloned: "
                "git clone https://huggingface.co/ChatterjeeLab/PeptiVerse"
            )

        print(f"✓ Initialized with {len(variants)} variants")
        for v in variants:
            print(f"  - {v.name:20s} (p={v.probability:.3f})")
    
    @staticmethod
    def generate_random_peptide(length: int = 12) -> str:
        """Generate random amino acid sequence."""
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        return "".join(random.choice(amino_acids) for _ in range(length))
    
    def compute_binding(self, peptide: str, variant: VariantWithProbability) -> float:
        """Compute binding affinity using PeptiVerse wt_wt_unpooled model."""
        import torch
        try:
            T, Mt = self.embedder.unpooled(variant.sequence)
            B, Mb = self.embedder.unpooled(peptide)
            with torch.no_grad():
                reg, _ = self.binding_model(T, Mt, B, Mb)
            return float(reg.squeeze().cpu().item())
        except Exception as e:
            print(f"Binding error for {variant.name}: {e}")
            return 0.0
    
    def evaluate_peptide(self, peptide: str) -> BindingScore:
        """
        Evaluate one peptide across all variants.
        
        Args:
            peptide: Amino acid sequence
        
        Returns:
            BindingScore with per-variant and weighted scores
        """
        binding_per_variant = {}
        scores = []
        
        for variant in self.variants:
            binding = self.compute_binding(peptide, variant)
            binding_per_variant[variant.name] = binding
            scores.append(binding)
            print(f"  {variant.name:20s} (p={variant.probability:.3f}): {binding:.4f}")
        
        # Weighted average by tree probability
        weighted = sum(s * v.probability for s, v in zip(scores, self.variants))
        mean_binding = float(np.mean(scores))
        
        return BindingScore(
            sequence=peptide,
            binding_per_variant=binding_per_variant,
            weighted_binding=weighted,
            mean_binding=mean_binding
        )
    
    def optimize_batch(
        self,
        num_peptides: int = 5,
        peptide_length: int = 12,
        output_file: Optional[Path] = None
    ) -> List[BindingScore]:
        """
        Generate and evaluate multiple random peptides.
        
        Args:
            num_peptides: Number of peptides to test
            peptide_length: Length of each peptide
            output_file: Optional output JSON file
        
        Returns:
            List of BindingScore objects, sorted by weighted binding
        """
        results = []
        
        print(f"\nEvaluating {num_peptides} peptides (length={peptide_length})")
        print("="*70)
        
        for i in range(num_peptides):
            peptide = self.generate_random_peptide(peptide_length)
            print(f"\nPeptide {i+1}/{num_peptides}: {peptide}")
            
            score = self.evaluate_peptide(peptide)
            results.append(score)
            
            print(f"  → Weighted binding: {score.weighted_binding:.4f}")
            print(f"  → Mean binding: {score.mean_binding:.4f}")
        
        # Sort by weighted binding (descending)
        results.sort(key=lambda r: r.weighted_binding, reverse=True)
        
        # Print summary
        print("\n" + "="*70)
        print("Top results (sorted by tree-weighted binding):")
        for i, r in enumerate(results[:min(3, len(results))]):
            print(f"{i+1}. {r.sequence} | weighted: {r.weighted_binding:.4f}")
        
        # Save results
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            results_dict = [
                {
                    "sequence": r.sequence,
                    "binding_per_variant": r.binding_per_variant,
                    "weighted_binding": r.weighted_binding,
                    "mean_binding": r.mean_binding
                }
                for r in results
            ]
            
            with open(output_file, "w") as f:
                json.dump(results_dict, f, indent=2)
            
            print(f"Results saved to: {output_file}")
        
        return results


def main(
    tree_json: Optional[Path] = None,
    num_peptides: int = 3,
    peptide_length: int = 12,
    output_file: Optional[Path] = None,
    device: str = "cpu",
    seed: int = 42,
    top_k: Optional[int] = None,
):
    """
    Main entry point for binding affinity evaluation.
    
    Args:
        tree_json: Path to hadsbm_tree.json from tree_analysis
        num_peptides: Number of random peptides to generate
        peptide_length: Length of each peptide
        output_file: Path to save results JSON
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    
    print("\n" + "="*70)
    print("HIV Variant Binding Affinity Predictor (Tree-Weighted)")
    print("="*70)
    
    # Load variants
    if tree_json is None:
        tree_json = Path("/Users/navyanori/hadsbm-hiv/data/trees/hadsbm_tree.json")
    
    tree_json = Path(tree_json)
    
    if not tree_json.exists():
        raise FileNotFoundError(
            f"Tree JSON not found: {tree_json}\n"
            "Run tree_analysis pipeline first:\n"
            "  cd tree_analysis && make tree"
        )
    
    print(f"\n📖 Loading variants from: {tree_json}")
    variants = load_tree_probabilities(tree_json, top_k=top_k)
    print(f"✓ Loaded {len(variants)} variants")
    
    # Create predictor and evaluate peptides
    predictor = VariantBindingPredictor(variants, device=device)
    results = predictor.optimize_batch(
        num_peptides=num_peptides,
        peptide_length=peptide_length,
        output_file=output_file
    )
    
    print("\n✓ Done\n")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute binding affinity of peptides to HIV variants (tree-weighted)"
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
        default=3,
        help="Number of random peptides to evaluate (default: 3)"
    )
    parser.add_argument(
        "--length",
        type=int,
        default=12,
        help="Peptide length (default: 12)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for computation (default: cpu, use 'cuda:0' for GPU)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Limit to top-K variants by tree probability (useful for large trees)"
    )

    args = parser.parse_args()
    main(
        tree_json=args.tree_json,
        num_peptides=args.num_peptides,
        peptide_length=args.length,
        output_file=args.output,
        device=args.device,
        seed=args.seed,
        top_k=args.top_k,
    )
