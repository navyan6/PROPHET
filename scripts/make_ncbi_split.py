#!/usr/bin/env python3
"""
Build a phylogenetic tree on NCBI HIV protease sequences, do a clade-based
train/holdout split, and write aligned FASTAs for both sets.
"""
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from Bio import Phylo, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

REPO = Path(__file__).resolve().parent.parent

FULL_ALIGNED  = REPO / "alignments/hiv_protease_aligned_full.fasta"
OUT_DIR       = REPO / "data/ncbi_split"
TREE_FASTA    = OUT_DIR / "ncbi_subset_aligned.fasta"
OUT_TREE      = OUT_DIR / "ncbi_subset_tree.nwk"
TRAIN_FASTA   = OUT_DIR / "ncbi_train_aligned.fasta"
HOLDOUT_FASTA = OUT_DIR / "ncbi_holdout_aligned.fasta"

SAMPLE_SIZE  = 1000   # total sequences to build tree on
MIN_HOLDOUT  = 150
MAX_HOLDOUT  = 250
SEED         = 42
WT = "PQVTLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"

random.seed(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load sequences (keep gapped alignment for tree building) ───────────────
print("Loading full alignment...")
all_records = []
seen_ungapped = set()
for r in SeqIO.parse(FULL_ALIGNED, "fasta"):
    s_ungapped = str(r.seq).replace("-", "")
    if len(s_ungapped) != 99 or "X" in s_ungapped or s_ungapped in seen_ungapped:
        continue
    seen_ungapped.add(s_ungapped)
    nid = r.id.split()[0].replace("|", "_").replace(".", "_")
    edits = sum(a != b for a, b in zip(s_ungapped, WT))
    if 3 <= edits <= 30:   # reasonable range, excludes clear outliers
        all_records.append((nid, str(r.seq), s_ungapped))  # (id, gapped, ungapped)

print(f"  {len(all_records)} unique valid sequences (edit 3-30 from WT)")

# ── 2. Sample and build tree ──────────────────────────────────────────────────
sampled = random.sample(all_records, min(SAMPLE_SIZE, len(all_records)))
print(f"\nBuilding tree on {len(sampled)} sequences...")

tree_records = [SeqRecord(Seq(gapped), id=nid, description="") for nid, gapped, _ in sampled]
SeqIO.write(tree_records, TREE_FASTA, "fasta")

result = subprocess.run(
    ["FastTree", "-quiet", "-lg", str(TREE_FASTA)],
    capture_output=True, text=True
)
if result.returncode != 0:
    print(f"FastTree error:\n{result.stderr}", file=sys.stderr)
    sys.exit(1)
OUT_TREE.write_text(result.stdout)
print(f"  Tree → {OUT_TREE}")

# ── 3. Find a clade for holdout ───────────────────────────────────────────────
print(f"\nFinding holdout clade ({MIN_HOLDOUT}–{MAX_HOLDOUT} leaves)...")
tree = Phylo.read(str(OUT_TREE), "newick")
tree.root_at_midpoint()

lookup = {nid: (gapped, ungapped) for nid, gapped, ungapped in sampled}

candidates = []
for clade in tree.find_clades(order="level"):
    leaves = [c for c in clade.get_terminals() if c.name in lookup]
    n = len(leaves)
    if not (MIN_HOLDOUT <= n <= MAX_HOLDOUT):
        continue
    seqs = [lookup[c.name][1] for c in leaves]
    edits = [sum(a != b for a, b in zip(s, WT)) for s in seqs]
    candidates.append((abs(np.mean(edits) - 9.18), n, leaves, seqs))

if not candidates:
    print("ERROR: no suitable holdout clade found. Adjust MIN/MAX_HOLDOUT.")
    sys.exit(1)

candidates.sort(key=lambda x: x[0])
_, best_n, holdout_leaves, holdout_seqs = candidates[0]
holdout_ids = {c.name for c in holdout_leaves}

edits = [sum(a != b for a, b in zip(s, WT)) for s in holdout_seqs]
print(f"  Holdout clade: {best_n} sequences")
print(f"  Edit from WT:  mean={np.mean(edits):.2f}, std={np.std(edits):.2f}, min={min(edits)}, max={max(edits)}")

# ── 4. Split into train and holdout ──────────────────────────────────────────
train_records   = [(nid, gapped, ungapped) for nid, gapped, ungapped in sampled if nid not in holdout_ids]
holdout_records = [(nid, lookup[c.name][0], lookup[c.name][1]) for c in holdout_leaves]

print(f"  Training: {len(train_records)} sequences")
print(f"  Holdout:  {len(holdout_records)} sequences")

# ── 5. Save aligned FASTAs ────────────────────────────────────────────────────
SeqIO.write(
    [SeqRecord(Seq(gapped), id=nid, description="") for nid, gapped, _ in train_records],
    TRAIN_FASTA, "fasta"
)
SeqIO.write(
    [SeqRecord(Seq(ungapped), id=nid, description="") for nid, _, ungapped in holdout_records],
    HOLDOUT_FASTA, "fasta"
)

print(f"\nSaved → {TRAIN_FASTA}")
print(f"Saved → {HOLDOUT_FASTA}")
print("\nNext steps:")
print(f"  1. python build_trees.py {TRAIN_FASTA} --skip-align --out-dir data/ncbi_split/trees --n-bootstraps 100")
print(f"  2. Run Stage 1 with --fasta {TRAIN_FASTA} --protein --ensemble-mode ...")
print(f"  3. Evaluate with --held-out-fasta {HOLDOUT_FASTA}")
