#!/usr/bin/env python3
"""
Example usage of multi-objective peptide optimizer.

Demonstrates:
1. Basic usage with default HIV WT
2. Multi-target optimization
3. Custom configuration
"""

from pathlib import Path
from optimize_peptides_moo import (
    main, Config, PeptideOptimizer, TargetVariant, read_fasta
)
from transformers import AutoTokenizer
import torch


def example_1_basic():
    print("\n" + "="*60)
    print("Example 1: Basic optimization (HIV WT only)")
    print("="*60)
    
    results = main(output_file="results_hiv_wt.json")
    
    print(f"\nGenerated {len(results)} peptide sequences")
    if results:
        print(f"Best affinity: {max([max(r.affinity_per_target.values()) for r in results]):.3f}")


def example_2_multi_targets():
    """Example 2: Optimize across multiple variants."""
    print("\n" + "="*60)
    print("Example 2: Multi-target optimization")
    print("="*60)

    hiv_wt = "MGARASVLSGGELDRWEKIRLRPGGKKKYKLKHIVWASRELERFAVNPGLLETSEGCRQILQQLQPSLQTGSEELRSLYNTVATLYCVHQRIEIKDTKEALDKIEEEQNKSKKKAQQAAA"
    
    variant_list = [hiv_wt[:i] + chr(ord(hiv_wt[i]) + 1) + hiv_wt[i+1:] 
                    for i in range(0, len(hiv_wt), 30)][:2]  # Create 2 "variants"
    
    targets = [hiv_wt] + variant_list
    
    results = main(targets=targets, output_file="results_multi_target.json")
    
    print(f"\nOptimized for {len(targets)} targets")
    print(f"Generated {len(results)} peptide sequences")


def example_3_custom_config():
    """Example 3: Custom configuration."""
    print("\n" + "="*60)
    print("Example 3: Custom configuration")
    print("="*60)
    
    fasta_path = Path(__file__).parent / "wildtype.fasta"
    if fasta_path.exists():
        seqs = read_fasta(fasta_path)
        target_seq = list(seqs.values())[0]
    else:
        target_seq = "MGARASVLSGGELDRWEKIRLRPGGKKKYKLKHIVWASRELERFAVNPGLLETSEGCRQILQQQLQPSLQTGSEELRSLYNTVATLYCVHQ"
    
    config = Config()
    config.peptide_length = 15           
    config.num_batches = 2               
    config.num_steps = 50                
    config.objective_weights = {
        "affinity": 2.0,    
        "hemolysis": 1.0,
        "nonfouling": 0.5,
        "solubility": 1.0,
        "halflife": 0.5,
    }
    
    print(f"Config: peptide_length={config.peptide_length}, "
          f"num_batches={config.num_batches}, "
          f"weights={config.objective_weights}")
    
    # Create optimizer with custom config
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    target_var = TargetVariant("hiv_wt", target_seq, config.device, tokenizer)
    
    print(f"Initializing optimizer for target: {target_var.name}")
    optimizer = PeptideOptimizer(config, [target_var])
    
    print("Running optimization...")
    results = optimizer.optimize()
    
    # Save results to JSON
    import json
    with open("results_custom_config.json", 'w') as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    
    print(f"\n✓ Saved {len(results)} results to results_custom_config.json")


def example_4_inspect_results():
    """Example 4: Analyze and visualize results."""
    print("\n" + "="*60)
    print("Example 4: Analyzing results")
    print("="*60)
    
    import json
    
    result_file = Path("results_hiv_wt.json")
    if not result_file.exists():
        print(f"Run Example 1 first to generate {result_file}")
        return
    
    with open(result_file) as f:
        results = json.load(f)
    
    if not results:
        print("No results to analyze")
        return
    
    print(f"\nAnalyzing {len(results)} peptides...")
    
    # Compute statistics
    affinities = [max(r['affinity_per_target'].values()) for r in results]
    hemolysis = [r['hemolysis'] for r in results]
    nonfouling = [r['nonfouling'] for r in results]
    solubility = [r['solubility'] for r in results]
    halflife = [r['halflife'] for r in results]
    
    print(f"\nBinding Affinity:")
    print(f"  Mean: {sum(affinities)/len(affinities):.3f}")
    print(f"  Range: [{min(affinities):.3f}, {max(affinities):.3f}]")
    
    print(f"\nHemolysis (lower is better):")
    print(f"  Mean: {sum(hemolysis)/len(hemolysis):.3f}")
    
    print(f"\nNon-fouling (higher is better):")
    print(f"  Mean: {sum(nonfouling)/len(nonfouling):.3f}")
    
    print(f"\nSolubility (higher is better):")
    print(f"  Mean: {sum(solubility)/len(solubility):.3f}")
    
    print(f"\nHalf-life (higher is better):")
    print(f"  Mean: {sum(halflife)/len(halflife):.3f}")
    
    # Find best binder
    best_idx = affinities.index(max(affinities))
    best = results[best_idx]
    print(f"\nBest binder: {best['sequence']}")
    print(f"  Affinity: {max(best['affinity_per_target'].values()):.3f}")
    print(f"  Hemolysis: {best['hemolysis']:.3f}")
    print(f"  Non-fouling: {best['nonfouling']:.3f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example = sys.argv[1]
    else:
        example = "1"
    
    if example == "1":
        example_1_basic()
    elif example == "2":
        example_2_multi_targets()
    elif example == "3":
        example_3_custom_config()
    elif example == "4":
        example_4_inspect_results()
    else:
        print("Usage: python examples.py [1|2|3|4]")
        print("  1: Basic optimization (HIV WT)")
        print("  2: Multi-target optimization")
        print("  3: Custom configuration")
        print("  4: Analyze results")
