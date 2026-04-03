#!/usr/bin/env python3
"""
Smoke test for the hadsbm-hiv pipeline.
Tests that all components can be imported and basic functions work.
Does NOT run expensive computations.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("\n" + "="*70)
    print("Testing imports...")
    print("="*70)
    
    try:
        import torch
        print("✓ PyTorch")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        return False
    
    try:
        from transformers import AutoTokenizer
        print("✓ Transformers")
    except ImportError as e:
        print(f"✗ Transformers: {e}")
        return False
    
    # Try loading tree utilities
    repo_root = Path(__file__).parent
    peptide_src = repo_root / "peptide_optimization" / "src"
    
    if str(peptide_src) not in sys.path:
        sys.path.insert(0, str(peptide_src))
    
    try:
        from tree_utils import load_tree_probabilities
        print("✓ tree_utils")
    except ImportError as e:
        print(f"✗ tree_utils: {e}")
        return False
    
    # Try importing PeptiVerse
    peptiverse_path = repo_root / "PeptiVerse"
    if str(peptiverse_path) not in sys.path:
        sys.path.insert(0, str(peptiverse_path))
    
    try:
        from inference import PeptiVersePredictor
        print("✓ PeptiVerse")
    except ImportError as e:
        print(f"✗ PeptiVerse: {e}")
        return False
    
    return True


def test_tree_utils():
    """Test tree utility functions and probability computation."""
    print("\n" + "="*70)
    print("Testing tree utilities and probability computation...")
    print("="*70)
    
    import json
    import tempfile
    
    repo_root = Path(__file__).parent
    peptide_src = repo_root / "peptide_optimization" / "src"
    
    # Add to path if not already there
    if str(peptide_src) not in sys.path:
        sys.path.insert(0, str(peptide_src))
    
    from tree_utils import VariantWithProbability, load_tree_probabilities, _compute_leaf_probability, _build_tree_structure
    
    # Create test variant
    test_var = VariantWithProbability(
        name="test",
        sequence="MFEML",
        probability=1.0
    )
    
    print(f"✓ Created test variant: {test_var.name}")
    
    # Test probability computation with a simple tree
    # Tree structure:
    #     root (0)
    #    /  (0.6)  \  (0.4)
    #   1           2
    #  / (0.5) \ (0.5)
    # 3         4
    # Expected: leaf 3 = 0.6*0.5 = 0.3, leaf 4 = 0.6*0.5 = 0.3, leaf 2 = 0.4
    
    test_tree = {
        "format": "hadsbm_tree_v1",
        "x_WT": "MFEML",
        "n_nodes": 5,
        "n_leaves": 3,
        "splits": [
            {
                "parent_index": 0,
                "left_child_index": 1,
                "right_child_index": 2,
                "time_tau": 0.5,
                "p_left": 0.6,
                "p_right": 0.4,
                "sh_support": None,
                "branch_len_left": 0.3,
                "branch_len_right": 0.2,
            },
            {
                "parent_index": 1,
                "left_child_index": 3,
                "right_child_index": 4,
                "time_tau": 0.25,
                "p_left": 0.5,
                "p_right": 0.5,
                "sh_support": None,
                "branch_len_left": 0.15,
                "branch_len_right": 0.15,
            },
        ],
        "leaf_endpoints_pi": [
            {"node_index": 3, "leaf_id": "leaf_0", "sequence": "MFEML"},
            {"node_index": 4, "leaf_id": "leaf_1", "sequence": "MFEML"},
            {"node_index": 2, "leaf_id": "leaf_2", "sequence": "MFEML"},
        ],
        "leaf_ids_in_order": ["leaf_0", "leaf_1", "leaf_2"],
    }
    
    # Write to temp file and load
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_tree, f)
        temp_path = Path(f.name)
    
    try:
        variants = load_tree_probabilities(temp_path)
        
        # Check that probabilities sum to 1.0
        total_prob = sum(v.probability for v in variants)
        if abs(total_prob - 1.0) < 1e-6:
            print(f"✓ Probabilities sum to 1.0")
        else:
            print(f"✗ Probabilities sum to {total_prob:.4f} (expected 1.0)")
            return False
        
        # Check individual probabilities
        probs_by_name = {v.name: v.probability for v in variants}
        expected = {"leaf_0": 0.3, "leaf_1": 0.3, "leaf_2": 0.4}
        
        all_close = True
        for name, expected_prob in expected.items():
            actual_prob = probs_by_name.get(name, 0.0)
            if abs(actual_prob - expected_prob) < 1e-6:
                print(f"✓ {name}: {actual_prob:.4f} (expected {expected_prob:.4f})")
            else:
                print(f"✗ {name}: {actual_prob:.4f} (expected {expected_prob:.4f})")
                all_close = False
        
        return all_close
        
    finally:
        temp_path.unlink()


def test_generate_peptide():
    """Test peptide generation."""
    print("\n" + "="*70)
    print("Testing peptide generation...")
    print("="*70)
    
    repo_root = Path(__file__).parent
    peptide_src = repo_root / "peptide_optimization" / "src"
    
    if str(peptide_src) not in sys.path:
        sys.path.insert(0, str(peptide_src))
    
    from binding_affinity_simple import VariantBindingPredictor
    
    # Test generation
    pep = VariantBindingPredictor.generate_random_peptide(12)
    print(f"✓ Generated peptide: {pep}")
    
    if len(pep) != 12:
        print("✗ Peptide length mismatch")
        return False
    
    if not all(c in "ACDEFGHIKLMNPQRSTVWY" for c in pep):
        print("✗ Invalid amino acids")
        return False
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("HADSBM-HIV Pipeline Smoke Tests")
    print("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("Tree utilities", test_tree_utils),
        ("Peptide generation", test_generate_peptide),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"{status} {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! Ready to run on GPU.")
        return 0
    else:
        print("\n✗ Some tests failed. Check setup.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
