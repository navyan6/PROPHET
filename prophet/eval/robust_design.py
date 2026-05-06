from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
import sys

import numpy as np
from Bio import SeqIO


def evaluate_peptide_robustness(
    peptides: list[str],
    wt_target: str,
    escape_variants: list[str],
    aff_fn: Callable[[str, str], float],
    tau_bind: float = 0.5,
) -> dict:
    """
    Evaluate each peptide on WT and held-out escape variants.
    """
    rows = []
    for pep in peptides:
        wt_score = float(aff_fn(pep, wt_target))
        esc_scores = np.array([float(aff_fn(pep, v)) for v in escape_variants], dtype=np.float64)
        mean_escape = float(np.mean(esc_scores)) if esc_scores.size else float("nan")
        min_escape = float(np.min(esc_scores)) if esc_scores.size else float("nan")
        retention = float(np.mean(esc_scores >= tau_bind)) if esc_scores.size else float("nan")
        rows.append(
            {
                "peptide": pep,
                "wt_score": wt_score,
                "mean_escape": mean_escape,
                "min_escape": min_escape,
                "retention": retention,
            }
        )

    if not rows:
        return {"per_peptide": [], "aggregate": {}}

    return {
        "per_peptide": rows,
        "aggregate": {
            "mean_wt_score": float(np.mean([r["wt_score"] for r in rows])),
            "mean_mean_escape": float(np.mean([r["mean_escape"] for r in rows])),
            "mean_min_escape": float(np.mean([r["min_escape"] for r in rows])),
            "mean_retention": float(np.mean([r["retention"] for r in rows])),
        },
    }


def _default_tau_bind(affinity_mode: str, peptiverse_normalization: str) -> float:
    if affinity_mode == "peptiverse" and peptiverse_normalization == "raw":
        return 8.0
    return 0.5


def _load_peptides(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    peptides = []
    for row in rows:
        pep = row.get("peptide") or row.get("sequence")
        if pep:
            peptides.append(str(pep).strip().upper())
    return peptides


def _load_fasta(path: Path) -> list[str]:
    return [
        str(rec.seq).strip().upper().replace("-", "")
        for rec in SeqIO.parse(str(path), "fasta")
        if str(rec.seq).strip()
    ]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate Stage 2 designs on held-out escape variants"
    )
    p.add_argument("--designs-json", required=True)
    p.add_argument("--wt-seq", required=True)
    p.add_argument("--escape-fasta", required=True)
    p.add_argument("--out-json", default=None)
    p.add_argument("--tau-bind", type=float, default=None)
    p.add_argument("--affinity-mode", choices=["surrogate", "peptiverse"], default="peptiverse")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--peptiverse-normalization", choices=["minmax", "raw"], default="raw")
    p.add_argument("--peptiverse-min", type=float, default=7.0)
    p.add_argument("--peptiverse-max", type=float, default=9.0)
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from prophet.stage2 import AffinityScorer

    peptides = _load_peptides(Path(args.designs_json))
    escape_variants = _load_fasta(Path(args.escape_fasta))
    tau_bind = (
        args.tau_bind
        if args.tau_bind is not None
        else _default_tau_bind(args.affinity_mode, args.peptiverse_normalization)
    )
    scorer = AffinityScorer(
        device=args.device,
        peptiverse_normalization=args.peptiverse_normalization,
        peptiverse_min=args.peptiverse_min,
        peptiverse_max=args.peptiverse_max,
    )
    results = evaluate_peptide_robustness(
        peptides=peptides,
        wt_target=args.wt_seq.strip().upper().replace("-", ""),
        escape_variants=escape_variants,
        aff_fn=scorer,
        tau_bind=tau_bind,
    )
    results["inputs"] = {
        "designs_json": str(Path(args.designs_json)),
        "escape_fasta": str(Path(args.escape_fasta)),
        "n_peptides": len(peptides),
        "n_escape_variants": len(escape_variants),
        "affinity_mode": args.affinity_mode,
        "peptiverse_normalization": args.peptiverse_normalization,
        "peptiverse_min": args.peptiverse_min,
        "peptiverse_max": args.peptiverse_max,
        "tau_bind": tau_bind,
    }
    print(json.dumps(results, indent=2))
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved robustness metrics -> {out_path}")


if __name__ == "__main__":
    main()
