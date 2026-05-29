#!/usr/bin/env python3
"""
Download Influenza A H1N1 neuraminidase (NA, N1 subtype) protein sequences from NCBI.

Searches NCBI Nucleotide for influenza A NA segment 6 sequences, extracts
the neuraminidase protein translation, and deduplicates.

Usage:
    python scripts/download_flu_na.py --out-dir data/flu_na
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

NA_LEN = 469           # expected N1 NA protein length (aa)
MIN_NA_AA = 450        # minimum acceptable length

NA_KEYWORDS = frozenset({
    "neuraminidase", "na protein", "na ",
    "n1 neuraminidase", "neuraminidase (na)",
})

TARGETS = {
    "flu_na": {
        "taxid":   "11520",    # Influenza A virus
        "label":   "Influenza A NA (N1 subtype)",
        "n_max":   3000,
        # segment 6 CDS + flanking UTRs: 1413bp + ~200bp = 1350-1650
        "seg_len": (1350, 1650),
        # neuraminidase keyword in any field; CDS extraction filters to NA
        "query_extra": 'AND neuraminidase[All Fields]',
    },
}


def _matches_na(feat) -> bool:
    for key in ("product", "gene", "note"):
        val = " ".join(feat.qualifiers.get(key, [])).lower()
        if any(kw in val for kw in NA_KEYWORDS):
            return True
    return False


def extract_na(rec) -> str | None:
    for feat in rec.features:
        if feat.type != "CDS":
            continue
        if not _matches_na(feat):
            continue
        aa = feat.qualifiers.get("translation", [""])[0]
        if len(aa) >= MIN_NA_AA:
            return aa
    return None


def fetch_na(key: str, cfg: dict, email: str) -> list[SeqRecord]:
    Entrez.email = email
    lo, hi = cfg["seg_len"]
    query = (
        f"txid{cfg['taxid']}[Organism:exp] "
        f"AND {lo}:{hi}[Sequence Length] "
        f"{cfg.get('query_extra', '')}"
    )
    print(f"  Searching NCBI for '{cfg['label']}' ...")
    handle = Entrez.esearch(db="nucleotide", term=query,
                            retmax=cfg["n_max"], usehistory="y")
    result = Entrez.read(handle); handle.close()
    webenv    = result["WebEnv"]
    query_key = result["QueryKey"]
    total     = int(result["Count"])
    fetch_n   = min(cfg["n_max"], total)
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
                aa = extract_na(rec)
                if aa:
                    acc = rec.id.split(".")[0]
                    if acc not in seen:
                        seen.add(acc)
                        records.append(SeqRecord(Seq(aa), id=acc, description=""))
        except Exception as e:
            print(f"    WARNING at offset {start}: {e}")
        finally:
            handle.close()
        time.sleep(0.4)
        done = min(start + batch, fetch_n)
        if done % 200 == 0 or done == fetch_n:
            print(f"    Processed {done}/{fetch_n}, {len(records)} NA found")

    return records


def deduplicate(records: list[SeqRecord]) -> list[SeqRecord]:
    out, seen = [], set()
    for rec in records:
        seq = str(rec.seq).replace("-", "").replace("X", "")
        if len(seq) < MIN_NA_AA or "*" in seq:
            continue
        if seq in seen:
            continue
        seen.add(seq)
        out.append(SeqRecord(Seq(seq), id=rec.id, description=""))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/flu_na", type=Path)
    ap.add_argument("--email",   default="navyanori6@gmail.com")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for key, cfg in TARGETS.items():
        print(f"\n=== {cfg['label']} ===")
        raw = fetch_na(key, cfg, args.email)
        deduped = deduplicate(raw)
        print(f"  {len(raw)} NA extracted → {len(deduped)} unique sequences")
        out = args.out_dir / f"{key}_raw.fasta"
        SeqIO.write(deduped, str(out), "fasta")
        print(f"  Wrote {out}")

    print("\nNext:")
    print("  python scripts/make_flu_na_splits.py --in-dir data/flu_na")


if __name__ == "__main__":
    main()
