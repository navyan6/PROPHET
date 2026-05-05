#!/usr/bin/env python3
"""
scripts/get_consensus.py
Compute the majority-rule consensus sequence from a multiple sequence alignment.

Works for both protein and (codon-translated) nucleotide alignments.
For nucleotide alignments pass --nucleotide; the script translates each sequence
before computing the consensus.

Usage
-----
  # Protein alignment
  python scripts/get_consensus.py flu_tree/ha_aligned.fasta

  # Nucleotide alignment (auto-translates)
  python scripts/get_consensus.py flu_tree/ha_aligned.fasta --nucleotide

  # Suppress gaps entirely (gaps never win a column)
  python scripts/get_consensus.py flu_tree/ha_aligned.fasta --no-gap-consensus

Output: one line, the consensus sequence, printed to stdout.
Optionally write to a file with --out.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from Bio import SeqIO
from Bio.Align import MultipleSeqAlignment


# ─────────────────────────── codon table ────────────────────────────────────
CODON_TABLE: dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def _translate(nt: str) -> str:
    """Translate a nucleotide string (gaps stripped) to protein."""
    nt_clean = nt.upper().replace("-", "").replace(".", "")
    aa = []
    for i in range(0, len(nt_clean) - 2, 3):
        codon = nt_clean[i : i + 3]
        aa.append(CODON_TABLE.get(codon, "X"))
    # drop stop codon if terminal
    if aa and aa[-1] == "*":
        aa.pop()
    return "".join(aa)


def _majority_consensus(
    seqs: list[str],
    allow_gap_win: bool = False,
) -> str:
    """Column-wise majority-rule consensus over a list of aligned strings."""
    if not seqs:
        return ""
    width = max(len(s) for s in seqs)
    consensus = []
    for i in range(width):
        col: dict[str, int] = {}
        for s in seqs:
            ch = s[i] if i < len(s) else "-"
            if ch == "-" and not allow_gap_win:
                continue
            col[ch] = col.get(ch, 0) + 1
        if not col:
            consensus.append("-")
        else:
            consensus.append(max(col, key=col.get))
    return "".join(consensus)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Majority-rule consensus from a FASTA alignment."
    )
    ap.add_argument("fasta", help="Path to aligned FASTA file")
    ap.add_argument(
        "--nucleotide",
        action="store_true",
        help="Input is a nucleotide alignment; translate each sequence first.",
    )
    ap.add_argument(
        "--no-gap-consensus",
        dest="no_gap",
        action="store_true",
        default=False,
        help="Never let a gap character win a column vote (default: gaps excluded).",
    )
    ap.add_argument("--out", default=None, help="Write consensus to this file.")
    args = ap.parse_args()

    records = list(SeqIO.parse(args.fasta, "fasta"))
    if not records:
        print(f"[error] No sequences found in {args.fasta}", file=sys.stderr)
        sys.exit(1)

    if args.nucleotide:
        seqs = [_translate(str(rec.seq)) for rec in records]
    else:
        seqs = [str(rec.seq).upper() for rec in records]

    consensus = _majority_consensus(seqs, allow_gap_win=not args.no_gap)
    # Strip leading/trailing gaps from final consensus
    consensus = consensus.replace("-", "")

    print(consensus)
    if args.out:
        Path(args.out).write_text(consensus + "\n")
        print(f"[saved] {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
