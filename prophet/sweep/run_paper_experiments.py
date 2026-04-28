#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("[run]", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Run paper-style PROPHET experiment bundle")
    p.add_argument("--repo-root", default=".", help="Repository root")
    p.add_argument("--tree", required=True)
    p.add_argument("--fasta", required=True)
    p.add_argument("--wt-seq", required=True)
    p.add_argument("--prefix", default="paper_run")
    p.add_argument("--out-dir", default="data/prophet")
    p.add_argument("--sample-variants", type=int, default=500)
    p.add_argument("--n-designs", type=int, default=250)
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--auto-calibrate-tevo", action="store_true")
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    prophet_dir = repo_root / "prophet"
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stage1 = prophet_dir / "stage1.py"
    stage2 = prophet_dir / "stage2.py"
    pareto = prophet_dir / "eval" / "pareto.py"
    eta_sens = prophet_dir / "eval" / "eta_sensitivity.py"
    esm_diag = prophet_dir / "eval" / "esm_filter_diagnostics.py"

    variants_fasta = out_dir / f"{args.prefix}_gibbs_variants.fasta"
    designs_json = out_dir / f"{args.prefix}_stage2_designs.json"
    pareto_json = out_dir / f"{args.prefix}_pareto.json"
    eta_json = out_dir / f"{args.prefix}_eta_sensitivity.json"
    esm_json = out_dir / f"{args.prefix}_esm_rejection.json"
    calib_json = out_dir / f"{args.prefix}_tevo_calibration.json"
    summary_json = out_dir / f"{args.prefix}_summary.json"

    stage1_cmd = [
        sys.executable,
        str(stage1),
        "--tree", args.tree,
        "--fasta", args.fasta,
        "--prefix", args.prefix,
        "--out-dir", str(out_dir),
        "--sample-variants", str(args.sample_variants),
        "--seed", str(args.seed),
    ]
    if args.auto_calibrate_tevo:
        stage1_cmd.extend(["--auto-calibrate-tevo", "--calibration-json", str(calib_json)])
    _run(stage1_cmd, cwd=repo_root)

    _run(
        [
            sys.executable,
            str(stage2),
            "--variants-fasta", str(variants_fasta),
            "--wt-seq", args.wt_seq,
            "--out-json", str(designs_json),
            "--n-designs", str(args.n_designs),
            "--n-steps", str(args.n_steps),
            "--eta", str(args.eta),
            "--seed", str(args.seed),
        ],
        cwd=repo_root,
    )

    _run(
        [
            sys.executable,
            str(pareto),
            "--designs-json", str(designs_json),
            "--out-json", str(pareto_json),
        ],
        cwd=repo_root,
    )
    _run(
        [
            sys.executable,
            str(eta_sens),
            "--designs-json", str(designs_json),
            "--out-json", str(eta_json),
        ],
        cwd=repo_root,
    )
    _run(
        [
            sys.executable,
            str(esm_diag),
            "--accepted-fasta", str(variants_fasta),
            "--requested-samples", str(args.sample_variants),
            "--out-json", str(esm_json),
        ],
        cwd=repo_root,
    )

    summary = {
        "prefix": args.prefix,
        "outputs": {
            "variants_fasta": str(variants_fasta),
            "designs_json": str(designs_json),
            "pareto_json": str(pareto_json),
            "eta_sensitivity_json": str(eta_json),
            "esm_rejection_json": str(esm_json),
            "tevo_calibration_json": str(calib_json) if args.auto_calibrate_tevo else None,
        },
        "params": vars(args),
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] summary -> {summary_json}")


if __name__ == "__main__":
    main()
