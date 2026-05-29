#!/usr/bin/env python3
"""
Download HCV NS3 protease-domain protein sequences from NCBI.

Strategy: search NCBI Nucleotide for complete HCV genome sequences, then
extract the NS3 protein translation from CDS feature annotations.

Targets:
  - HCV genotype 1a (taxid 31648)  → train + clade holdout
  - HCV genotype 1b (taxid 31649)  → optional cross-subtype holdout

The NS3 protease domain is the N-terminal 181 aa of NS3 (full NS3 ~631 aa).

Usage:
    python scripts/download_hcv_ns3.py --out-dir data/hcv_ns3
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


PROTEASE_END = 181      # NS3 protease domain length
MIN_FULL_NS3 = 580      # min length to accept an NS3 sequence before trimming

GENOME_RANGE = {
    "hcv_ns3": (9000, 10000),
}

# 0-indexed start of NS3 in the HCV polyprotein (genotype 1 reference, ~1026)
# Genotype 2/3/4 vary slightly; polyprotein fallback handles the offset
NS3_POLYPROTEIN_OFFSET = {
    "hcv_ns3": 1026,
}

TARGETS = {
    "hcv_ns3": {"taxid": "11103", "label": "Hepacivirus C (all HCV)", "n_max": 3000},
}

NS3_KEYWORDS = frozenset({
    "ns3", "nonstructural protein 3", "non-structural protein 3",
    "ns3 protease", "ns3/4a", "ns3 helicase", "ns3 protein",
})


def _matches_ns3(feat) -> bool:
    for key in ("product", "gene", "note"):
        val = " ".join(feat.qualifiers.get(key, [])).lower()
        if any(kw in val for kw in NS3_KEYWORDS):
            return True
    return False


def extract_ns3(rec, polyprotein_offset: int = 1026) -> str | None:
    # Priority 1: individual NS3 CDS
    for feat in rec.features:
        if feat.type != "CDS":
            continue
        if not _matches_ns3(feat):
            continue
        aa = feat.qualifiers.get("translation", [""])[0]
        if len(aa) >= MIN_FULL_NS3:
            return aa

    # Priority 2: mat_peptide
    for feat in rec.features:
        if feat.type != "mat_peptide":
            continue
        if not _matches_ns3(feat):
            continue
        try:
            nt = feat.extract(rec.seq)
            aa = str(nt.translate(to_stop=True))
            if len(aa) >= MIN_FULL_NS3:
                return aa
        except Exception:
            pass

    # Priority 3: polyprotein + fixed offset
    for feat in rec.features:
        if feat.type != "CDS":
            continue
        product = " ".join(feat.qualifiers.get("product", [])).lower()
        if "polyprotein" not in product:
            continue
        aa = feat.qualifiers.get("translation", [""])[0]
        if len(aa) < 2500:
            continue
        end = polyprotein_offset + 631
        if end > len(aa):
            continue
        ns3 = aa[polyprotein_offset:end]
        if len(ns3) >= MIN_FULL_NS3:
            return ns3

    return None


def fetch_ns3(key: str, taxid: str, label: str, n_max: int, email: str) -> list[SeqRecord]:
    Entrez.email = email
    lo, hi = GENOME_RANGE[key]
    offset = NS3_POLYPROTEIN_OFFSET[key]

    query = f"txid{taxid}[Organism:exp] AND {lo}:{hi}[Sequence Length]"
    print(f"  Searching NCBI Nucleotide for '{label}' ...")
    handle = Entrez.esearch(db="nucleotide", term=query, retmax=n_max, usehistory="y")
    result = Entrez.read(handle); handle.close()
    webenv    = result["WebEnv"]
    query_key = result["QueryKey"]
    total     = int(result["Count"])
    fetch_n   = min(n_max, total)
    print(f"    Found {total} records; fetching {fetch_n} ...")

    if fetch_n == 0:
        return []

    records, seen = [], set()
    batch = 20
    for start in range(0, fetch_n, batch):
        handle = Entrez.efetch(
            db="nucleotide", rettype="gb", retmode="text",
            retstart=start, retmax=batch,
            webenv=webenv, query_key=query_key,
        )
        try:
            for rec in SeqIO.parse(handle, "genbank"):
                aa = extract_ns3(rec, polyprotein_offset=offset)
                if aa and len(aa) >= MIN_FULL_NS3:
                    acc = rec.id.split(".")[0]
                    if acc not in seen:
                        seen.add(acc)
                        records.append(SeqRecord(Seq(aa), id=acc, description=""))
        except Exception as e:
            print(f"    WARNING: parse error at offset {start}: {e}")
        finally:
            handle.close()
        time.sleep(0.4)
        done = min(start + batch, fetch_n)
        if done % 100 == 0 or done == fetch_n:
            print(f"    Processed {done}/{fetch_n} genomes, {len(records)} NS3 found")

    return records


def trim_to_protease(records: list[SeqRecord], end: int = PROTEASE_END) -> list[SeqRecord]:
    out, seen = [], set()
    for rec in records:
        seq = str(rec.seq).replace("-", "").replace("X", "")
        domain = seq[:end]
        if len(domain) < end or "*" in domain:
            continue
        if domain in seen:
            continue
        seen.add(domain)
        out.append(SeqRecord(Seq(domain), id=rec.id, description=""))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir",      default="data/hcv_ns3", type=Path)
    ap.add_argument("--email",        default="navyanori6@gmail.com")
    ap.add_argument("--protease-end", type=int, default=PROTEASE_END)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for key, cfg in TARGETS.items():
        print(f"\n=== {cfg['label']} ({key}) ===")
        raw = fetch_ns3(key, cfg["taxid"], cfg["label"], cfg["n_max"], args.email)
        trimmed = trim_to_protease(raw, end=args.protease_end)
        print(f"  {len(raw)} NS3 extracted → {len(trimmed)} unique protease-domain seqs")
        out_path = args.out_dir / f"{key}_protease.fasta"
        SeqIO.write(trimmed, str(out_path), "fasta")
        print(f"  Wrote {out_path}")

    print("\nNext:")
    print("  python scripts/make_hcv_splits.py --in-dir data/hcv_ns3")
    print("  rsync -avz data/hcv_ns3/ \\")
    print("    nnori@login.betty.parcc.upenn.edu:/vast/projects/pranam/lab/nnori/hadsbm-hiv/data/hcv_ns3/")


if __name__ == "__main__":
    main()
