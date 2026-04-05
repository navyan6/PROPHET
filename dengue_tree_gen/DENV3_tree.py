import pandas as pd
from Bio import SeqIO, Phylo
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import matplotlib.pyplot as plt
import subprocess
import os
import tempfile

CSV_FILE = "cluster2and6.obs.csv"
OUTPUT_TREE = "DENV3_tree.nwk"
OUTPUT_FIG  = "DENV3_tree.png"

#remove non-human hosts and unknown virus types; keep Human + type 3 only
def load_and_filter(csv_file):
    df = pd.read_csv(csv_file, index_col=0)
    df = df[(df["Host"] == "Human") & (df["Virus Type"] == "3")]
    df = df.reset_index(drop=True)
    print(f"Sequences after filtering: {len(df)}")
    return df


def build_records(df):
    records = []
    for _, row in df.iterrows():
        accession = str(row["GenBank Accession"]).strip()
        year      = str(row["Collection Date"]).split(".")[0]
        country   = str(row["Country"]).strip()
        seq_id    = f"{accession}|{year}|{country}"
        records.append(SeqRecord(Seq(str(row["seq"])), id=seq_id, description=""))
    return records


def write_fasta(records, path):
    SeqIO.write(records, path, "fasta")


def align_sequences(fasta_in, fasta_out):
    #Align sequences with MAFFT
    result = subprocess.run(
        ["mafft", "--auto", "--quiet", fasta_in],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"MAFFT failed: {result.stderr}")
    with open(fasta_out, "w") as f:
        f.write(result.stdout)
    print("MAFFT Alignment complete.")



def build_tree_fasttree(aligned_fasta, nwk_path):
    result = subprocess.run(
        ["FastTree", "-wag", "-quiet", aligned_fasta],
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
    _, ax = plt.subplots(figsize=(14, max(8, n_tips * 0.18)))
    Phylo.draw(tree, axes=ax, do_show=False)
    ax.set_title("DENV3 Phylogenetic Tree (Human hosts, Cluster 2 & 6)", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    print(f"Figure saved to {fig_path}")
    plt.show()


def main():
    df = load_and_filter(CSV_FILE)
    records = build_records(df)

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_fasta     = os.path.join(tmpdir, "raw.fasta")
        aligned_fasta = os.path.join(tmpdir, "aligned.fasta")

        write_fasta(records, raw_fasta)
        align_sequences(raw_fasta, aligned_fasta)
        build_tree_fasttree(aligned_fasta, OUTPUT_TREE)

    plot_tree(OUTPUT_TREE, OUTPUT_FIG)


if __name__ == "__main__":
    main()
