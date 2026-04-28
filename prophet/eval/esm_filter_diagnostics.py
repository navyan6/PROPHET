#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from Bio import SeqIO


def _count_fasta_records(path: Path) -> int:
    return sum(1 for _ in SeqIO.parse(str(path), "fasta"))


def main() -> None:
    p = argparse.ArgumentParser(description="ESM filter diagnostics from Stage 1 variant outputs")
    p.add_argument("--accepted-fasta", required=True, help="Final accepted variants FASTA")
    p.add_argument("--requested-samples", type=int, required=True, help="Requested --sample-variants count")
    p.add_argument("--burn-in", type=int, default=None, help="Burn-in used during sampling (optional)")
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    accepted_path = Path(args.accepted_fasta)
    accepted = _count_fasta_records(accepted_path)
    requested = int(args.requested_samples)
    rejection = max(0, requested - accepted)
    rejection_rate = float(rejection / requested) if requested > 0 else float("nan")

    out = {
        "accepted_fasta": str(accepted_path),
        "requested_samples": requested,
        "accepted_samples": accepted,
        "rejected_samples": rejection,
        "rejection_rate": rejection_rate,
    }
    if args.burn_in is not None:
        out["burn_in"] = int(args.burn_in)

    print(json.dumps(out, indent=2))
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Saved ESM diagnostics -> {out_path}")


if __name__ == "__main__":
    main()
