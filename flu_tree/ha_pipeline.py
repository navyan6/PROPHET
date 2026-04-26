import random
import subprocess
from Bio import SeqIO, Phylo
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT_FASTA   = "sequence.fasta"
HA_FASTA      = "ha_only.fasta"
SAMPLED_FASTA = "ha_sampled.fasta"
ALIGNED_FASTA = "ha_aligned.fasta"
OUTPUT_TREE   = "ha_tree.nwk"
OUTPUT_FIG    = "ha_tree.png"

HA_MIN = 1500
HA_MAX = 1800
SAMPLE_N = 1000

ha_records = []
total = 0
for record in SeqIO.parse(INPUT_FASTA, "fasta"):
    total += 1
    header = record.description.lower()
    if "hemagglutinin" in header or " ha " in header:
        if HA_MIN <= len(record.seq) <= HA_MAX:
            ha_records.append(record)

print(f"Total sequences parsed      : {total}")
print(f"HA sequences (1500–1800 bp) : {len(ha_records)}")

SeqIO.write(ha_records, HA_FASTA, "fasta")
print(f"Written to                  : {HA_FASTA}")

if len(ha_records) > SAMPLE_N:
    sampled = random.sample(ha_records, SAMPLE_N)
else:
    sampled = ha_records

SeqIO.write(sampled, SAMPLED_FASTA, "fasta")
print(f"Sampled                     : {len(sampled)} sequences → {SAMPLED_FASTA}")

print("\nRunning MAFFT alignment...")
result = subprocess.run(
    ["mafft", "--auto", "--quiet", "--thread", "-1", SAMPLED_FASTA],
    capture_output=True, text=True
)
if result.returncode != 0:
    raise RuntimeError(f"MAFFT failed: {result.stderr}")
with open(ALIGNED_FASTA, "w") as f:
    f.write(result.stdout)
print(f"Alignment complete → {ALIGNED_FASTA}")


print("\nRunning FastTree...")
result = subprocess.run(
    ["FastTree", "-nt", "-gtr", "-quiet", ALIGNED_FASTA],
    capture_output=True, text=True
)
if result.returncode != 0:
    raise RuntimeError(f"FastTree failed: {result.stderr}")
with open(OUTPUT_TREE, "w") as f:
    f.write(result.stdout)
print(f"Tree saved → {OUTPUT_TREE}")


print("\nPlotting tree...")
tree = Phylo.read(OUTPUT_TREE, "newick")
n_tips = len(tree.get_terminals())
fig, ax = plt.subplots(figsize=(18, max(14, n_tips * 0.15)))
Phylo.draw(tree, axes=ax, do_show=False)
ax.set_title("Influenza A Hemagglutinin (HA) Phylogenetic Tree", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=150)
print(f"Figure saved → {OUTPUT_FIG}")
