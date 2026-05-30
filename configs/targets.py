"""
Central registry of PROPHET pipeline targets.

Each entry maps a short target name to all file paths and Stage 1 parameters
needed to run the full pipeline. Shared defaults are defined in DEFAULTS and
can be overridden per target.

Usage in Python:
    from configs.targets import TARGETS, DEFAULTS, get_target
    cfg = get_target("hiv_protease")

Usage from the CLI (used by run_prophet.slurm):
    python -c "from configs.targets import get_target; import json; print(json.dumps(get_target('hiv_protease')))"
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Shared defaults (overridable per target)
# ---------------------------------------------------------------------------
DEFAULTS: dict = {
    "n_bootstraps":                 1,
    "sample_variants":              500,
    "burn_in":                      200,
    "energy_mode":                  "paper_dca",
    "adaptive_l2":                  True,
    "l2_reg_base":                  1e-4,
    "esm_filter_delta_per_residue": 0.20,
    "esm_model":                    "facebook/esm2_t33_650M_UR50D",
    "conserv_weight":               0.05,
    "seed":                         42,
    "ensemble_mode":                True,
}

# ---------------------------------------------------------------------------
# Per-target configuration
# Required keys: alignment, tree, trees_file, holdout, t_evo
# Optional keys: any DEFAULTS key to override, plus:
#   protein (bool)       — True if alignment is already protein (default True)
#   resistance_fasta     — path to resistance/holdout FASTA for experiment 1
#   out_prefix           — prefix for output files (default: target name)
# ---------------------------------------------------------------------------
TARGETS: dict[str, dict] = {
    "hiv_protease": {
        "alignment":         "data/pre_stage1_split/alignments/train/hiv_train_aligned.fasta",
        "tree":              "data/pre_stage1_split/trees/train/hiv_train_tree.nwk",
        "trees_file":        "data/pre_stage1_split/trees/train/hiv_bootstrap_trees.txt",
        "holdout":           "data/pre_stage1_split/alignments/test/hiv_test_clade_holdout.fasta",
        "resistance_fasta":  "data/hiv_resistance_holdout.fasta",
        "t_evo":             0.15,
        "protein":           True,
    },
    "hcv_ns3": {
        "alignment":  "data/hcv_ns3/alignments/train/hcv_ns3_train_aligned.fasta",
        "tree":       "data/hcv_ns3/trees/train/hcv_ns3_train_tree.nwk",
        "trees_file": "data/hcv_ns3/trees/train/hcv_ns3_bootstrap_trees.txt",
        "holdout":    "data/hcv_ns3/alignments/test/hcv_ns3_test_clade_holdout.fasta",
        "t_evo":      0.15,
        "protein":    True,
    },
    "flu_ha": {
        "alignment":  "data/flu_ha/alignments/train/flu_ha_train_aligned.fasta",
        "tree":       "data/flu_ha/trees/train/flu_ha_train_tree.nwk",
        "trees_file": "data/flu_ha/trees/train/flu_ha_bootstrap_trees.txt",
        "holdout":    "data/flu_ha/alignments/test/flu_ha_test_clade_holdout.fasta",
        "t_evo":      0.15,
        "protein":    True,
    },
    "flu_na": {
        "alignment":  "data/flu_na/alignments/train/flu_na_train_aligned.fasta",
        "tree":       "data/flu_na/trees/train/flu_na_train_tree.nwk",
        "trees_file": "data/flu_na/trees/train/flu_na_bootstrap_trees.txt",
        "holdout":    "data/flu_na/alignments/test/flu_na_test_clade_holdout.fasta",
        "t_evo":      0.15,
        "protein":    True,
    },
    "sars_mpro": {
        "alignment":  "data/sars_mpro/alignments/train/sars_mpro_train_aligned.fasta",
        "tree":       "data/sars_mpro/trees/train/sars_mpro_train_tree.nwk",
        "trees_file": "data/sars_mpro/trees/train/sars_mpro_bootstrap_trees.txt",
        "holdout":    "data/sars_mpro/alignments/test/sars_mpro_test_clade_holdout.fasta",
        "t_evo":      0.5,
        "protein":    True,
    },
    "zika_ns3": {
        "alignment":  "data/zika_ns3/alignments/train/zika_ns3_train_aligned.fasta",
        "tree":       "data/zika_ns3/trees/train/zika_ns3_train_tree.nwk",
        "trees_file": "data/zika_ns3/trees/train/zika_ns3_bootstrap_trees.txt",
        "holdout":    "data/zika_ns3/alignments/test/zika_ns3_test_holdout.fasta",
        "t_evo":      0.15,
        "protein":    True,
    },
    "wnv_ns3": {
        "alignment":  "data/wnv_ns3/alignments/train/wnv_ns3_train_aligned.fasta",
        "tree":       "data/wnv_ns3/trees/train/wnv_ns3_train_tree.nwk",
        "trees_file": "data/wnv_ns3/trees/train/wnv_ns3_bootstrap_trees.txt",
        "holdout":    "data/wnv_ns3/alignments/test/wnv_ns3_test_holdout.fasta",
        "t_evo":      0.15,
        "protein":    True,
    },
    "denv2_e": {
        "alignment":  "data/denv2_e/alignments/train/denv2_e_train_aligned.fasta",
        "tree":       "data/denv2_e/trees/train/denv2_e_train_tree.nwk",
        "trees_file": "data/denv2_e/trees/train/denv2_e_bootstrap_trees.txt",
        "holdout":    "data/denv2_e/alignments/test/denv2_e_test_clade_holdout.fasta",
        "t_evo":      0.15,
        "protein":    True,
        "esm_filter_delta_per_residue": None,  # disable ESM filter; 150 train seqs, strict delta rejects all
    },
    "rsv_f": {
        "alignment":  "data/rsv_f/alignments/train/rsv_f_train_aligned.fasta",
        "tree":       "data/rsv_f/trees/train/rsv_f_train_tree.nwk",
        "trees_file": "data/rsv_f/trees/train/rsv_f_bootstrap_trees.txt",
        "holdout":    "data/rsv_f/alignments/test/rsv_f_test_clade_holdout.fasta",
        "t_evo":      0.15,
        "protein":    True,
        "esm_filter_delta_per_residue": None,  # disable ESM filter; strict delta rejects all
    },
}


def get_target(name: str) -> dict:
    """Return merged config for a target (DEFAULTS + target-specific overrides)."""
    if name not in TARGETS:
        raise KeyError(f"Unknown target '{name}'. Valid targets: {sorted(TARGETS)}")
    cfg = {**DEFAULTS, **TARGETS[name]}
    cfg.setdefault("out_prefix", name)
    cfg.setdefault("protein", True)
    return cfg


def list_targets() -> list[str]:
    return sorted(TARGETS.keys())


if __name__ == "__main__":
    import argparse
    import json
    p = argparse.ArgumentParser(description="Print config for a PROPHET target as JSON")
    p.add_argument("target", choices=list_targets())
    args = p.parse_args()
    print(json.dumps(get_target(args.target), indent=2))
