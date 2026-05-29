#!/usr/bin/env python3
"""
Download betacoronavirus main protease (Mpro / nsp5 / 3CLpro) protein sequences from NCBI.

Searches NCBI Protein for 3C-like proteinase sequences from betacoronaviruses
(taxid 694002), which includes SARS-CoV-2, SARS-CoV-1, MERS-CoV, and bat
coronaviruses — providing the phylogenetic diversity needed for DCA.

Usage:
    python scripts/download_sars_mpro.py --out-dir data/sars_mpro
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

MPRO_LEN = 306          # canonical SARS-CoV-2 Mpro length (aa)
MIN_MPRO_AA = 290
MAX_MPRO_AA = 320

MPRO_KEYWORDS = frozenset({
    "3c-like proteinase", "3cl protease", "3clpro", "main protease",
    "nsp5", "ns5", "3c-like", "3chymotrypsin-like",
})

TARGETS = {
    "sars_mpro": {
        "taxid":   "694002",   # Betacoronavirus (includes SARS-CoV-2, SARS-CoV-1, MERS, bat CoV)
        "label":   "Betacoronavirus Mpro (3CLpro / nsp5)",
        "n_max":   5000,
        "db":      "protein",
        "min_len": MIN_MPRO_AA,
        "max_len": MAX_MPRO_AA,
    },
}


def fetch_mpro(key: str, cfg: dict, email: str) -> list[SeqRecord]:
    Entrez.email = email
    lo, hi = cfg["min_len"], cfg["max_len"]
    query = (
        f"txid{cfg['taxid']}[Organism:exp] "
        f"AND {lo}:{hi}[Sequence Length] "
        f"AND (\"3C-like proteinase\"[Title] OR \"3CLpro\"[Title] "
        f"OR \"main protease\"[Title] OR \"nsp5\"[Title] "
        f"OR \"3C-like\"[Title])"
    )
    print(f"  Searching NCBI Protein for '{cfg['label']}' ...")
    handle = Entrez.esearch(db="protein", term=query,
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
    batch = 100
    for start in range(0, fetch_n, batch):
        handle = Entrez.efetch(
            db="protein", rettype="fasta", retmode="text",
            retstart=start, retmax=batch,
            webenv=webenv, query_key=query_key,
        )
        try:
            for rec in SeqIO.parse(handle, "fasta"):
                aa = str(rec.seq).replace("*", "")
                if len(aa) < lo or len(aa) > hi:
                    continue
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
        if done % 500 == 0 or done == fetch_n:
            print(f"    Processed {done}/{fetch_n}, {len(records)} Mpro found")

    return records


def deduplicate(records: list[SeqRecord], min_len: int) -> list[SeqRecord]:
    out, seen = [], set()
    for rec in records:
        seq = str(rec.seq).replace("-", "").replace("X", "")
        if len(seq) < min_len or "*" in seq:
            continue
        if seq in seen:
            continue
        seen.add(seq)
        out.append(SeqRecord(Seq(seq), id=rec.id, description=""))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/sars_mpro", type=Path)
    ap.add_argument("--email",   default="navyanori6@gmail.com")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for key, cfg in TARGETS.items():
        print(f"\n=== {cfg['label']} ===")
        raw = fetch_mpro(key, cfg, args.email)
        deduped = deduplicate(raw, cfg["min_len"])
        print(f"  {len(raw)} Mpro fetched → {len(deduped)} unique sequences")
        out = args.out_dir / f"{key}_raw.fasta"
        SeqIO.write(deduped, str(out), "fasta")
        print(f"  Wrote {out}")

    print("\nNext:")
    print("  python scripts/make_sars_mpro_splits.py --in-dir data/sars_mpro")


if __name__ == "__main__":
    main()
