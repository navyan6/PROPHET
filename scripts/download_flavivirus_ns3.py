#!/usr/bin/env python3
"""
Download NS3 protease-domain protein sequences for flaviviruses from NCBI.

Strategy: search NCBI Nucleotide for complete genome sequences, then extract
the NS3 protein translation from CDS feature annotations. Falls back to
mat_peptide features (for polyprotein records), then to polyprotein + fixed
offset if needed.

Targets:
  - Dengue virus 3 (DENV3, taxid 11070)  → train
  - Dengue virus 1 (DENV1, taxid 11053)  → cross-serotype holdout
  - Zika virus     (taxid 64320)          → train + clade holdout
  - West Nile virus(taxid 11082)          → train + clade holdout

Usage:
    python scripts/download_flavivirus_ns3.py --out-dir data/flavivirus_ns3
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


PROTEASE_END = 185      # trim NS3 to protease domain (N-terminal 185 aa)
MIN_FULL_NS3 = 580      # require near-full NS3 before trimming (~619 aa)

# Genome length ranges (bp) per virus — used to filter nucleotide search
GENOME_RANGE = {
    "denv3": (10000, 11200),
    "denv1": (10000, 11200),
    "zika":  (10000, 11200),
    "wnv":   (10000, 11500),
}

# Known NS3 start position in flavivirus polyprotein (0-indexed, approximate).
# Used as last-resort fallback only.
NS3_POLYPROTEIN_OFFSET = {
    "denv3": 1474,
    "denv1": 1474,
    "zika":  1481,
    "wnv":   1512,
}

TARGETS = {
    "denv3": {"taxid": "11070", "label": "Dengue virus 3", "n_max": 500},
    "denv1": {"taxid": "11053", "label": "Dengue virus 1", "n_max": 300},
    "zika":  {"taxid": "64320", "label": "Zika virus",     "n_max": 1500},
    "wnv":   {"taxid": "11082", "label": "West Nile virus", "n_max": 2000},
}

NS3_KEYWORDS = frozenset({
    "ns3", "nonstructural protein 3", "non-structural protein 3",
    "ns3 protein", "ns3/hel", "ns3 helicase", "ns3 protease",
})


def _product_matches_ns3(feat) -> bool:
    for key in ("product", "gene", "note"):
        val = " ".join(feat.qualifiers.get(key, [])).lower()
        if any(kw in val for kw in NS3_KEYWORDS):
            return True
    return False


def extract_ns3(rec, polyprotein_offset: int = 1474) -> str | None:
    """Try to extract the NS3 amino acid sequence from a GenBank record."""
    # Priority 1: individual NS3 CDS feature
    for feat in rec.features:
        if feat.type != "CDS":
            continue
        if not _product_matches_ns3(feat):
            continue
        aa = feat.qualifiers.get("translation", [""])[0]
        if len(aa) >= MIN_FULL_NS3:
            return aa

    # Priority 2: mat_peptide (used in some RefSeq records)
    for feat in rec.features:
        if feat.type != "mat_peptide":
            continue
        if not _product_matches_ns3(feat):
            continue
        try:
            nt = feat.extract(rec.seq)
            aa = str(nt.translate(to_stop=True))
            if len(aa) >= MIN_FULL_NS3:
                return aa
        except Exception:
            pass

    # Priority 3: polyprotein CDS + fixed offset (last resort)
    for feat in rec.features:
        if feat.type != "CDS":
            continue
        product = " ".join(feat.qualifiers.get("product", [])).lower()
        if "polyprotein" not in product:
            continue
        aa = feat.qualifiers.get("translation", [""])[0]
        if len(aa) < 3000:
            continue
        end = polyprotein_offset + 619
        if end > len(aa):
            continue
        ns3 = aa[polyprotein_offset:end]
        if len(ns3) >= MIN_FULL_NS3:
            return ns3

    return None


def fetch_ns3_from_genomes(
    key: str,
    taxid: str,
    label: str,
    n_max: int,
    email: str,
) -> list[SeqRecord]:
    Entrez.email = email
    lo, hi = GENOME_RANGE[key]
    offset = NS3_POLYPROTEIN_OFFSET[key]

    # Search nucleotide for complete genome sequences in the expected size range
    query = (
        f"txid{taxid}[Organism:exp] "
        f"AND {lo}:{hi}[Sequence Length]"
    )
    print(f"  Searching NCBI Nucleotide for '{label}' genomes ...")
    handle = Entrez.esearch(db="nucleotide", term=query, retmax=n_max,
                            usehistory="y")
    result = Entrez.read(handle)
    handle.close()
    webenv    = result["WebEnv"]
    query_key = result["QueryKey"]
    total     = int(result["Count"])
    fetch_n   = min(n_max, total)
    print(f"    Found {total} records; fetching {fetch_n} ...")

    if fetch_n == 0:
        return []

    records = []
    batch = 20
    seen_acc: set[str] = set()
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
                    if acc not in seen_acc:
                        seen_acc.add(acc)
                        records.append(
                            SeqRecord(Seq(aa), id=acc, description="")
                        )
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
    ap.add_argument("--out-dir",       default="data/flavivirus_ns3", type=Path)
    ap.add_argument("--email",         default="navyanori6@gmail.com")
    ap.add_argument("--protease-end",  type=int, default=PROTEASE_END)
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    for key, cfg in TARGETS.items():
        print(f"\n=== {cfg['label']} ({key}) ===")
        raw = fetch_ns3_from_genomes(
            key, cfg["taxid"], cfg["label"], cfg["n_max"], args.email
        )
        trimmed = trim_to_protease(raw, end=args.protease_end)
        print(f"  {len(raw)} NS3 extracted → {len(trimmed)} unique protease-domain seqs")

        out_path = out / f"{key}_ns3_protease.fasta"
        SeqIO.write(trimmed, str(out_path), "fasta")
        print(f"  Wrote {out_path}")

    print("\nNext:")
    print("  rsync -avz data/flavivirus_ns3/ \\")
    print("    nnori@login.betty.parcc.upenn.edu:/vast/projects/pranam/lab/nnori/hadsbm-hiv/data/flavivirus_ns3/")
    print("  python scripts/make_flavivirus_splits.py --in-dir data/flavivirus_ns3")


if __name__ == "__main__":
    main()
