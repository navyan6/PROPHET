# PROPHET

**P**hylogenetic **R**obustness-**O**ptimized **P**eptide **H**ypercone-guided **E**volutionary **T**herapeutics

PROPHET designs antiviral peptides that are robust to viral escape mutations. Given a viral protein alignment and phylogenetic tree, it learns the evolutionary landscape of the protein using Direct Coupling Analysis (DCA), samples realistic escape variants via Gibbs sampling, then uses those variants to guide a flow-matching peptide diffusion model (MOG-DFM) to design peptides that bind broadly across the variant landscape.

---

## How it works

**Stage 1 — Evolutionary landscape learning**
1. Reads a multiple sequence alignment (MSA) and phylogenetic tree for a viral protein
2. Computes per-site evolutionary rates (λ_i) and substitution matrices (Q_i) via Fitch parsimony across 100 bootstrap trees
3. Fits a Direct Coupling Analysis (DCA) model via pseudolikelihood maximization
4. Gibbs-samples 500 escape variants from the DCA energy landscape
5. Optionally filters variants by ESM-2 pseudo-log-likelihood to remove biologically unrealistic sequences

**Stage 2 — Robust peptide design**
1. For each designed peptide, computes a CVaR robustness score — the mean PeptiVerse binding score over the worst-η fraction of escape variants
2. Uses MOG-DFM (flow matching) with CVaR guidance to generate peptides that bind both the wildtype and escape variants
3. Runs 5 comparison modes: `prophet`, `wt_only`, `random_variants`, `uniform_leaves`, `esm_only_variants`

---

## Registered targets

| Target | Protein | Length | Holdout |
|--------|---------|--------|---------|
| `hiv_protease` | HIV-1 protease | 99 aa | Clade holdout |
| `hcv_ns3` | HCV NS3 protease | 181 aa | Clade holdout |
| `zika_ns3` | Zika NS3 protease | 185 aa | Clade holdout |
| `wnv_ns3` | WNV NS3 protease | 185 aa | Clade holdout |
| `sars_mpro` | SARS-CoV-2 Mpro | ~306 aa | Clade holdout |
| `denv2_e` | DENV2 Envelope | 495 aa | Clade holdout |
| `rsv_f` | RSV Fusion protein | 574 aa | Clade holdout |

---

## Quick start (PARCC cluster)

```bash
# Run full pipeline (Stage 1 + Stage 2) for any registered target
./submit.sh hiv_protease

# Or submit directly
sbatch --export=TARGET=hcv_ns3 run_prophet.slurm
Results are written to results/{target}/stage1/ and results/{target}/comparison/.

Adding a new target
1. Prepare sequences

Download protein sequences (FASTA), deduplicate, and filter by length. Sequences must already be protein (amino acids).

2. Build train/test split


python scripts/make_clade_split.py \
    --fasta  your_protein.fasta \
    --target my_target \
    --min-len 150 --max-len 200 \
    --n-boot  100
This aligns sequences with MAFFT, builds a phylogenetic tree with FastTree, holds out a phylogenetically coherent clade (~20%) as the test set, and generates 100 bootstrap trees.

3. Register the target

Add an entry to configs/targets.py:


"my_target": {
    "alignment":  "data/my_target/alignments/train/my_target_train_aligned.fasta",
    "tree":       "data/my_target/trees/train/my_target_train_tree.nwk",
    "trees_file": "data/my_target/trees/train/my_target_bootstrap_trees.txt",
    "holdout":    "data/my_target/alignments/test/my_target_test_clade_holdout.fasta",
    "t_evo":      0.15,
    "protein":    True,
},
4. Submit


./submit.sh my_target
Repository structure

prophet/
  stage1.py          # Stage 1: DCA + Gibbs sampling
  stage2.py          # Stage 2: MOG-DFM peptide design
  run_comparison.py  # Table 2 method comparison runner
  eval/
    experiment1.py   # Variant quality evaluation

configs/
  targets.py         # Target registry

scripts/
  make_clade_split.py        # Universal train/test split script
  make_flavivirus_splits.py  # Flavivirus-specific split script
  score_holdout_robustness.py
  download_*.py              # NCBI sequence download scripts

MOG-DFM/             # Flow matching peptide design model
PeptiVerse/          # Peptide binding scorer
data/                # Alignments, trees, splits per target
results/             # Pipeline outputs per target
Dependencies

conda activate hadsbm
# Key packages: biopython, scikit-learn, joblib, numpy, torch, transformers
