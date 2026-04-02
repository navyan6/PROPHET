"""
Multi-objective peptide optimization for multiple target proteins/variants.

Generates peptides optimized across:
- Binding affinity to each target variant
- Hemolysis (toxicity)
- Non-fouling (biofouling resistance)
- Solubility
- Half-life
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import inspect

import numpy as np
import torch
from transformers import AutoTokenizer

# Add MOG-DFM and PeptiVerse to path
MOG_DFM_PATH = Path(__file__).parent.parent.parent / "MOG-DFM"
PEPTIVERSE_PATH = Path(__file__).parent.parent.parent / "PeptiVerse"
sys.path.insert(0, str(MOG_DFM_PATH))
sys.path.insert(0, str(PEPTIVERSE_PATH))

from models.peptide_classifiers import (
    HemolysisModel, NonfoulingModel,
    SolubilityModelNew, HalfLifeModel, load_solver
)
from inference import PropertyPredictor 


@dataclass
class Config:
    """Configuration for peptide optimization."""
    mog_dfm_dir: Path = field(default_factory=lambda: Path(__file__).parent / "MOG-DFM")
    
    peptide_length: int = 12
    num_samples: int = 1
    num_batches: int = 2
    num_steps: int = 100
    step_size: float = 1 / 200
    
    # Device
    device: str = field(default_factory=lambda: "cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Objectives and their weights (higher = more important)
    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        "affinity": 1.0,
        "hemolysis": 0.5,
        "nonfouling": 0.5,
        "solubility": 0.5,
        "halflife": 0.5,
    })
    
    def get_ckpt_dir(self) -> Path:
        return self.mog_dfm_dir / "classifier_ckpt"
    
    def get_solver_ckpt(self) -> Path:
        return self.mog_dfm_dir / "ckpt" / "peptide" / "cnn_epoch200_lr0.0001_embed512_hidden256_loss3.1051.ckpt"


@dataclass
class PeptideScores:
    """Scores for a single peptide."""
    sequence: str
    affinity_per_target: Dict[str, float]
    hemolysis: float
    nonfouling: float
    solubility: float
    halflife: float
    
    def to_dict(self) -> dict:
        return {
            "sequence": self.sequence,
            "affinity_per_target": self.affinity_per_target,
            "hemolysis": self.hemolysis,
            "nonfouling": self.nonfouling,
            "solubility": self.solubility,
            "halflife": self.halflife,
        }


class TargetVariant:
    """Represents a target protein/variant for optimization."""
    
    def __init__(self, name: str, sequence: str, device: str, tokenizer):
        self.name = name
        self.sequence = sequence
        self.device = device
        self.tokenizer = tokenizer
        
        # Tokenize the target sequence for affinity prediction
        self.input_ids = tokenizer(sequence, return_tensors='pt')['input_ids'].to(device)
    
    def get_name(self) -> str:
        return self.name


class PropertyScorer:
    """Handles scoring of peptides for various properties."""
    
    def __init__(self, config: Config, device: str):
        self.config = config
        self.device = device
        
        print("Loading property prediction models...")
        # Load property prediction models
        self.hemolysis_model = HemolysisModel(device=device)
        self.nonfouling_model = NonfoulingModel(device=device)
        self.solubility_model = SolubilityModelNew(device=device)
        self.halflife_model = HalfLifeModel(device=device)
        print("✓ Property models loaded")
    
    def load_affinity_predictors(self, targets: List[TargetVariant]) -> Dict[str, object]:
        """Load PeptiVerse binding affinity predictor for each target.
        
        Uses PeptiVerse's best_model_wt for WT peptide binding prediction.
        """
        print("Initializing PeptiVerse binding affinity predictor...")
        
        # Initialize PeptiVerse predictor
        try:
            predictor = PropertyPredictor()
            print("✓ PeptiVerse binding affinity predictor loaded")
        except Exception as e:
            print(f"Warning: PeptiVerse initialization may have issues: {e}")
            raise
        
        # Create one predictor instance per target (each target gets same model)
        affinity_models = {}
        for target in targets:
            affinity_models[target.name] = predictor
        
        return affinity_models
    
    def score_properties(self, peptide_ids: torch.Tensor, 
                        affinity_models: Dict[str, object]) -> Tuple[Dict[str, float], Dict]:
        """Score a peptide for all properties using PeptiVerse and MOG-DFM models.
        
        Returns:
            tuple: (property_scores dict, target_affinities dict)
        """
        scores = {}
        target_affinities = {}
        
        # Decode peptide sequence for property prediction
        decoded = self.tokenizer.decode(peptide_ids[0]).replace(' ', '')[5:-5]
        
        # Compute binding affinity for each target using PeptiVerse
        with torch.no_grad():
            for target_name, affinity_predictor in affinity_models.items():
                try:
                    # PeptiVerse binding_affinity prediction
                    affinity = affinity_predictor.predict(
                        sequences=[decoded],
                        property_name="binding_affinity"
                    )
                    target_affinities[target_name] = float(affinity[0] if isinstance(affinity, np.ndarray) else affinity)
                except Exception as e:
                    print(f"Warning: Affinity prediction for {target_name} failed: {e}")
                    target_affinities[target_name] = 0.0
        
        # Average affinity across targets for multi-objective guidance
        avg_affinity = np.mean(list(target_affinities.values())) if target_affinities else 0.0
        scores["affinity"] = float(avg_affinity)
        
        # Compute other properties using MOG-DFM models
        with torch.no_grad():
            hemolysis = self.hemolysis_model(peptide_ids)
            scores["hemolysis"] = float(hemolysis.item() if hemolysis.numel() == 1 else hemolysis[0].item())
            
            nonfouling = self.nonfouling_model(peptide_ids)
            scores["nonfouling"] = float(nonfouling.item() if nonfouling.numel() == 1 else nonfouling[0].item())
            
            solubility = self.solubility_model(peptide_ids)
            scores["solubility"] = float(solubility.item() if solubility.numel() == 1 else solubility[0].item())
            
            halflife = self.halflife_model(peptide_ids)
            scores["halflife"] = float(halflife.item() if halflife.numel() == 1 else halflife[0].item())
        
        return scores, target_affinities


class PeptideOptimizer:
    """Main optimization pipeline."""
    
    def __init__(self, config: Config, targets: List[TargetVariant]):
        self.config = config
        self.targets = targets
        self.device = config.device
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        
        # Load solver
        print(f"Loading solver from {config.get_solver_ckpt()}...")
        self.solver = load_solver(
            str(config.get_solver_ckpt()),
            vocab_size=24,
            device=config.device
        )
        
        # Load scorers
        self.scorer = PropertyScorer(config, config.device)
        self.affinity_models = self.scorer.load_affinity_predictors(targets)
    
    def generate_random_peptide(self, length: int, num_samples: int = 1) -> torch.Tensor:
        """Generate random peptides as starting point.
        
        Args:
            length: Peptide length (amino acids)
            num_samples: Number of peptides to generate
            
        Returns:
            Token tensor of shape (num_samples, length + 2) with BOS/EOS tokens
        """
        vocab_size = 24
        vocab_start = 4  
        random_tokens = torch.randint(
            low=vocab_start,
            high=vocab_size,
            size=(num_samples, length),
            device=self.device
        )
        
        bos = torch.zeros((num_samples, 1), dtype=random_tokens.dtype, device=self.device)
        eos = torch.full((num_samples, 1), 2, dtype=random_tokens.dtype, device=self.device)
        
        peptide_ids = torch.cat([bos, random_tokens, eos], dim=1)
        return peptide_ids
    
    def decode_peptide(self, token_ids: torch.Tensor) -> str:
        """Convert token IDs to peptide sequence."""
        decoded = self.tokenizer.decode(token_ids)
        # Remove special tokens and whitespace
        cleaned = decoded.replace(' ', '')[5:-5]
        return cleaned
    
    def optimize(self) -> List[PeptideScores]:
        """Run multi-objective optimization.
        
        Returns:
            List of optimized PeptideScores
        """
        results = []
        
        for batch_idx in range(self.config.num_batches):
            print(f"\n=== Batch {batch_idx + 1}/{self.config.num_batches} ===")
            
            # Generate random starting peptide
            peptide_ids = self.generate_random_peptide(
                self.config.peptide_length,
                self.config.num_samples
            )
            
            print(f"Initial peptide tokens shape: {peptide_ids.shape}")
            
            # Multi-objective guided sampling
            optimized_ids = self.solver.multi_guidance_sample(
                args=self.config,
                x_init=peptide_ids,
                step_size=self.config.step_size,
                verbose=True,
                time_grid=torch.tensor([0.0, 1.0 - 1e-3], device=self.device),
                score_models=[],
                num_objectives=len(self.config.objective_weights),
                weights=list(self.config.objective_weights.values())
            )
            
            # Decode and score
            for sample_idx in range(optimized_ids.shape[0]):
                trajectory = optimized_ids[sample_idx].cpu()
                sequence = self.decode_peptide(trajectory)
                
                # Score the peptide
                peptide_ids_batch = optimized_ids[sample_idx:sample_idx+1]
                property_scores, target_affinities = self.scorer.score_properties(
                    peptide_ids_batch,
                    self.affinity_models
                )
                
                # Create result
                result = PeptideScores(
                    sequence=sequence,
                    affinity_per_target=target_affinities,
                    hemolysis=property_scores.get("hemolysis", 0.0),
                    nonfouling=property_scores.get("nonfouling", 0.0),
                    solubility=property_scores.get("solubility", 0.0),
                    halflife=property_scores.get("halflife", 0.0),
                )
                results.append(result)
                
                print(f"  Sequence: {sequence}")
                print(f"    Affinity by target: {target_affinities}")
                print(f"    Properties: H={property_scores['hemolysis']:.3f}, "
                      f"NF={property_scores['nonfouling']:.3f}, "
                      f"Sol={property_scores['solubility']:.3f}, "
                      f"HL={property_scores['halflife']:.3f}")
        
        return results


def read_fasta(fasta_path: Path) -> Dict[str, str]:
    """Parse FASTA file to dict."""
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:].split()[0]  # First word as ID
                current_seq = []
            else:
                current_seq.append(line)
        
        if current_id:
            sequences[current_id] = ''.join(current_seq)
    
    return sequences


def main(targets: Optional[List[str]] = None, output_file: Optional[str] = None):
    """Run optimization pipeline.
    
    Args:
        targets: List of target protein sequences. If None, uses HIV WT from wildtype.fasta.
        output_file: Path to save results JSON. Defaults to "optimized_peptides.json".
    """
    config = Config()
    
    print(f"Using device: {config.device}")
    print(f"MOG-DFM path: {config.mog_dfm_dir}")
    
    # Prepare targets
    target_variants = []
    
    if targets is None:
        # Load from FASTA
        fasta_path = Path(__file__).parent / "wildtype.fasta"
        if not fasta_path.exists():
            raise FileNotFoundError(f"Default FASTA not found: {fasta_path}")
        sequences = read_fasta(fasta_path)
        targets = list(sequences.values())
    
    # Create target objects
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    for i, target_seq in enumerate(targets):
        name = f"target_{i}" if len(targets) > 1 else "target"
        target_variants.append(TargetVariant(name, target_seq, config.device, tokenizer))
    
    print(f"\nOptimizing for {len(target_variants)} target(s):")
    for t in target_variants:
        seq_preview = t.sequence[:50] + ('...' if len(t.sequence) > 50 else '')
        print(f"  {t.name}: {seq_preview} (len={len(t.sequence)})")
    
    # Run optimization
    print("\nInitializing optimizer...")
    optimizer = PeptideOptimizer(config, target_variants)
    
    print("\nStarting multi-objective optimization...")
    results = optimizer.optimize()
    
    # Save results
    if output_file is None:
        output_file = "optimized_peptides.json"
    
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    
    print(f"\n✓ Saved {len(results)} results to {output_path}")
    
    # Print summary
    if results:
        print("\n=== Summary ===")
        for i, r in enumerate(results[:5], 1):  # Show first 5
            print(f"{i}. {r.sequence} | Affinity: {list(r.affinity_per_target.values())}")
        if len(results) > 5:
            print(f"... and {len(results) - 5} more")
    
    return results


if __name__ == "__main__":
    main()
