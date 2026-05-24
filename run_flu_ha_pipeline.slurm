#!/bin/bash
# Full PROPHET pipeline for Influenza HA: translate → align → Stage 1 → Stage 2.
#
#SBATCH --job-name=prophet_flu_ha
#SBATCH --account=pranam-lab
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:B200:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/prophet_flu_ha_%j.out
#SBATCH --error=logs/prophet_flu_ha_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=navyanori6@gmail.com

REPO=/vast/projects/pranam/lab/$USER/hadsbm-hiv
PYTHON=/vast/home/n/nnori/.conda/envs/hadsbm/bin/python
export HF_HOME=/vast/projects/pranam/lab/$USER/.cache/huggingface

cd "$REPO"
mkdir -p logs

echo "=============================="
echo "Job: $SLURM_JOB_ID — Flu HA PROPHET pipeline"
echo "=============================="

# Step 1: Translate + split + build trees (CPU, ~2h for 100 bootstrap trees)
echo "[Step 1] Translating and building trees ..."
$PYTHON scripts/prep_virus_protein.py \
    --nt-fasta      alignments/flu_ha_aligned.fasta \
    --name          flu_ha \
    --out-dir       data/flu_ha \
    --n-bootstraps  100 \
    --test-frac     0.2 \
    --seed          42

# Step 2: Stage 1 — DCA + Gibbs sampling
echo "[Step 2] Stage 1 ..."
mkdir -p results/flu_ha_prophet_final

$PYTHON prophet/stage1.py \
    --tree          data/flu_ha/trees/train/flu_ha_train_tree.nwk \
    --trees-file    data/flu_ha/trees/train/flu_ha_bootstrap_trees.txt \
    --fasta         data/flu_ha/alignments/train/flu_ha_train_aligned.fasta \
    --protein \
    --out-dir       results/flu_ha_prophet_final \
    --prefix        flu_ha_prophet \
    --n-bootstraps  1 \
    --sample-variants 500 \
    --burn-in       200 \
    --t-evo         0.15 \
    --energy-mode   paper_dca \
    --ensemble-mode \
    --esm-device    cuda:0 \
    --esm-model     facebook/esm2_t33_650M_UR50D \
    --esm-filter-delta 20 \
    --seed          42

# Step 3: Stage 2 — peptide design
echo "[Step 3] Stage 2 ..."
VARIANTS=$(ls results/flu_ha_prophet_final/flu_ha_prophet_*esm_filtered*.fasta 2>/dev/null | head -1)
if [ -z "$VARIANTS" ]; then
    VARIANTS=$(ls results/flu_ha_prophet_final/flu_ha_prophet_*variants*.fasta 2>/dev/null | head -1)
fi

if [ -z "$VARIANTS" ]; then
    echo "ERROR: No variant FASTA found in results/flu_ha_prophet_final/"
    exit 1
fi

# WT = consensus of train alignment (majority rule)
WT_SEQ=$($PYTHON -c "
from Bio import SeqIO
from Bio.Align import MultipleSeqAlignment
recs = list(SeqIO.parse('data/flu_ha/alignments/train/flu_ha_train_aligned.fasta', 'fasta'))
aln = MultipleSeqAlignment(recs)
wt = ''
for i in range(aln.get_alignment_length()):
    col = aln[:, i]
    counts = {}
    for aa in col:
        if aa != '-': counts[aa] = counts.get(aa, 0) + 1
    wt += max(counts, key=counts.get) if counts else 'X'
print(wt)
")

STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p results/flu_ha_ablations

$PYTHON prophet/stage2.py \
    --variants-fasta  "$VARIANTS" \
    --wt-seq          "$WT_SEQ" \
    --device          cuda:0 \
    --dfm-device      cuda:0 \
    --n-designs       500 \
    --n-steps         200 \
    --peptide-length  10 \
    --beta            5.0 \
    --eta             0.1 \
    --tau-bind        7.5 \
    --peptiverse-normalization raw \
    --variant-limit   149 \
    --out-json        "results/flu_ha_ablations/prophet-${STAMP}.json" \
    --verbose-sampling

echo "Flu HA PROPHET pipeline complete."
echo "Results: results/flu_ha_ablations/prophet-${STAMP}.json"
