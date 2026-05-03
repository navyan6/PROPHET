#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from Bio import SeqIO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prophet.stage2 import AffinityScorer, cvar_robust_score


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
AA = set("ACDEFGHIKLMNPQRSTVWY")


def _default_tau_bind(peptiverse_normalization: str) -> float:
    if peptiverse_normalization == "raw":
        return 8.0
    return 0.5


def _clean_peptide(seq: str) -> str:
    seq = seq.strip().upper().replace(" ", "").replace("-", "")
    return "".join(ch for ch in seq if ch in AA)


def _load_fasta(path: Path) -> list[str]:
    return [
        _clean_peptide(str(rec.seq))
        for rec in SeqIO.parse(str(path), "fasta")
        if str(rec.seq).strip()
    ]


def _seqs_from_csv(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    candidates = [
        "peptide",
        "Peptide Sequence",
        "sequence",
        "aa_seq",
        "amino_acid_sequence",
    ]
    out = []
    for row in rows:
        for col in candidates:
            val = row.get(col)
            if val:
                pep = _clean_peptide(val)
                if pep:
                    out.append(pep)
                    break
    return out


def _seq_from_pdb(path: Path) -> str:
    residues: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            resname = line[17:20].strip().upper()
            chain = line[21].strip()
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            residues.append(key + (AA3_TO_1.get(resname, "X"),))
    return _clean_peptide("".join(r[-1] for r in residues))


def _load_peptides(path: Path, input_format: str) -> list[str]:
    if input_format == "csv":
        return _seqs_from_csv(path)
    if input_format == "fasta":
        return _load_fasta(path)
    if input_format == "pdb":
        return [_seq_from_pdb(path)]
    if input_format == "pdb_dir":
        return [_seq_from_pdb(p) for p in sorted(path.glob("*.pdb"))]
    raise ValueError(f"Unsupported input format: {input_format}")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate generated peptide baselines with PeptiVerse")
    p.add_argument("--input", required=True, help="CSV, FASTA, PDB, or directory of PDB files")
    p.add_argument("--input-format", choices=["csv", "fasta", "pdb", "pdb_dir"], required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--wt-seq", required=True)
    p.add_argument("--test-fasta", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--tau-bind", type=float, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--peptiverse-normalization", choices=["minmax", "raw"], default="raw")
    p.add_argument("--peptiverse-min", type=float, default=7.0)
    p.add_argument("--peptiverse-max", type=float, default=9.0)
    p.add_argument("--dedupe", action="store_true")
    args = p.parse_args()

    peptides = [p for p in _load_peptides(Path(args.input), args.input_format) if p]
    if args.dedupe:
        peptides = list(dict.fromkeys(peptides))
    if not peptides:
        raise ValueError(f"No peptide sequences loaded from {args.input}")

    test_variants = _load_fasta(Path(args.test_fasta))
    if not test_variants:
        raise ValueError(f"No test variants loaded from {args.test_fasta}")

    tau_bind = (
        args.tau_bind
        if args.tau_bind is not None
        else _default_tau_bind(args.peptiverse_normalization)
    )
    scorer = AffinityScorer(
        mode="peptiverse",
        device=args.device,
        peptiverse_normalization=args.peptiverse_normalization,
        peptiverse_min=args.peptiverse_min,
        peptiverse_max=args.peptiverse_max,
    )
    rows = []
    for pep in peptides:
        wt_score = float(scorer(pep, args.wt_seq))
        per_variant = np.array([float(scorer(pep, v)) for v in test_variants], dtype=np.float64)
        rows.append(
            {
                "method": args.method,
                "peptide": pep,
                "wt_score": wt_score,
                "robust_score": cvar_robust_score(per_variant, args.eta),
                "mean_score": float(np.mean(per_variant)),
                "min_score": float(np.min(per_variant)),
                "retention_at_tau": float(np.mean(per_variant >= tau_bind)),
                "tau_bind": tau_bind,
                "omega": [0.0, 0.0],
                "per_variant": per_variant.tolist(),
            }
        )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved {len(rows)} evaluated peptides -> {out_path}")


if __name__ == "__main__":
    main()
