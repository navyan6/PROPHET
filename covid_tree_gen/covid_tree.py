from Bio import Phylo
import matplotlib.pyplot as plt
import subprocess
import os

FASTA_IN      = "covdata.fasta"
ALIGNED_FASTA = "covdata_aligned.fasta"
OUTPUT_TREE   = "covid_tree.nwk"
OUTPUT_FIG    = "covid_tree.png"


def align_sequences(fasta_in, fasta_out):
    result = subprocess.run(
        ["mafft", "--auto", "--quiet", "--thread", "-1", fasta_in],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"MAFFT failed: {result.stderr}")
    with open(fasta_out, "w") as f:
        f.write(result.stdout)
    print("MAFFT alignment complete.")


def build_tree(aligned_fasta, nwk_path):
    result = subprocess.run(
        ["FastTree", "-nt", "-gtr", "-quiet", aligned_fasta],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"FastTree failed: {result.stderr}")
    with open(nwk_path, "w") as f:
        f.write(result.stdout)
    print(f"Tree saved to {nwk_path}")


def plot_tree(nwk_path, fig_path):
    tree = Phylo.read(nwk_path, "newick")
    n_tips = len(tree.get_terminals())
    fig, ax = plt.subplots(figsize=(16, max(12, n_tips * 0.15)))
    Phylo.draw(tree, axes=ax, do_show=False)
    ax.set_title("hCoV-19 Phylogenetic Tree", fontsize=13)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    align_sequences(FASTA_IN, ALIGNED_FASTA)
    build_tree(ALIGNED_FASTA, OUTPUT_TREE)
    plot_tree(OUTPUT_TREE, OUTPUT_FIG)
