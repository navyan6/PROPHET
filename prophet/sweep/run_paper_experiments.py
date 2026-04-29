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
    p.add_argument("--trees-file", default=None, help="Optional file containing additional tree paths")
    p.add_argument("--tree-subsample-j", type=int, default=None,
                   help="Optional number of trees to subsample for Stage 1 averaging")
    p.add_argument("--fasta", required=True)
    p.add_argument("--wt-seq", required=True)
    p.add_argument("--prefix", default="paper_run")
    p.add_argument("--out-dir", default="data/prophet")
    p.add_argument("--sample-variants", type=int, default=500)
    p.add_argument("--n-designs", type=int, default=500)
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--peptide-length", type=int, default=10)
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--auto-calibrate-tevo", action="store_true")
    p.add_argument("--energy-mode", choices=["paper_dca", "dca_plus_qi"], default="dca_plus_qi",
                   help="Stage 1 Gibbs energy mode; paper bundle defaults to lambda + DCA + Qi")
    p.add_argument("--protein", action="store_true", help="Input FASTA is already protein aligned")
    p.add_argument("--dfm-ckpt", default=None, help="MOG-DFM peptide checkpoint")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dfm-device", default=None)
    p.add_argument("--affinity-mode", choices=["surrogate", "peptiverse"], default="peptiverse")
    p.add_argument("--peptiverse-normalization", choices=["minmax", "raw"], default="minmax")
    p.add_argument("--peptiverse-min", type=float, default=7.0)
    p.add_argument("--peptiverse-max", type=float, default=9.0)
    p.add_argument(
        "--design-modes",
        default="prophet,wt_only,random_variants,uniform_leaves",
        help="Comma-separated Stage 2 design modes to run",
    )
    p.add_argument("--guidance-variants-fasta", default=None)
    p.add_argument("--escape-fasta", default=None, help="Held-out escape variants for Table 2-style evaluation")
    p.add_argument("--tau-bind", type=float, default=0.5)
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
    robust_eval = prophet_dir / "eval" / "robust_design.py"

    variants_fasta = out_dir / f"{args.prefix}_gibbs_variants.fasta"
    designs_jsons: dict[str, Path] = {}
    pareto_jsons: dict[str, Path] = {}
    eta_jsons: dict[str, Path] = {}
    robust_jsons: dict[str, Path] = {}
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
        "--energy-mode", args.energy_mode,
    ]
    if args.trees_file:
        stage1_cmd.extend(["--trees-file", args.trees_file])
    if args.tree_subsample_j is not None:
        stage1_cmd.extend(["--tree-subsample-j", str(args.tree_subsample_j)])
    if args.protein:
        stage1_cmd.append("--protein")
    if args.auto_calibrate_tevo:
        stage1_cmd.extend(["--auto-calibrate-tevo", "--calibration-json", str(calib_json)])
    _run(stage1_cmd, cwd=repo_root)

    modes = [m.strip() for m in str(args.design_modes).split(",") if m.strip()]
    for mode in modes:
        designs_json = out_dir / f"{args.prefix}_{mode}_stage2_designs.json"
        pareto_json = out_dir / f"{args.prefix}_{mode}_pareto.json"
        eta_json = out_dir / f"{args.prefix}_{mode}_eta_sensitivity.json"
        robust_json = out_dir / f"{args.prefix}_{mode}_robust_design.json"
        designs_jsons[mode] = designs_json
        pareto_jsons[mode] = pareto_json
        eta_jsons[mode] = eta_json
        robust_jsons[mode] = robust_json

        stage2_cmd = [
            sys.executable,
            str(stage2),
            "--variants-fasta", str(variants_fasta),
            "--wt-seq", args.wt_seq,
            "--out-json", str(designs_json),
            "--n-designs", str(args.n_designs),
            "--n-steps", str(args.n_steps),
            "--peptide-length", str(args.peptide_length),
            "--eta", str(args.eta),
            "--seed", str(args.seed),
            "--design-mode", mode,
            "--affinity-mode", args.affinity_mode,
            "--device", args.device,
            "--peptiverse-normalization", args.peptiverse_normalization,
            "--peptiverse-min", str(args.peptiverse_min),
            "--peptiverse-max", str(args.peptiverse_max),
        ]
        if args.dfm_ckpt:
            stage2_cmd.extend(["--dfm-ckpt", args.dfm_ckpt])
        if args.dfm_device:
            stage2_cmd.extend(["--dfm-device", args.dfm_device])
        if args.guidance_variants_fasta:
            stage2_cmd.extend(["--guidance-variants-fasta", args.guidance_variants_fasta])
        _run(stage2_cmd, cwd=repo_root)

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
        if args.escape_fasta:
            robust_cmd = [
                sys.executable,
                str(robust_eval),
                "--designs-json", str(designs_json),
                "--wt-seq", args.wt_seq,
                "--escape-fasta", args.escape_fasta,
                "--out-json", str(robust_json),
                "--tau-bind", str(args.tau_bind),
                "--affinity-mode", args.affinity_mode,
                "--device", args.device,
                "--peptiverse-normalization", args.peptiverse_normalization,
                "--peptiverse-min", str(args.peptiverse_min),
                "--peptiverse-max", str(args.peptiverse_max),
            ]
            _run(robust_cmd, cwd=repo_root)
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
            "designs_json": {k: str(v) for k, v in designs_jsons.items()},
            "pareto_json": {k: str(v) for k, v in pareto_jsons.items()},
            "eta_sensitivity_json": {k: str(v) for k, v in eta_jsons.items()},
            "robust_design_json": {k: str(v) for k, v in robust_jsons.items()} if args.escape_fasta else None,
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
