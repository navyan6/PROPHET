#!/usr/bin/env python3
"""
PROPHET Stage 2 — Table 2 method comparison runner.

Runs 5 design modes in parallel across GPUs for a single target:
  1. prophet           — full PROPHET guidance
  2. wt_only           — WT-only guidance (baseline)
  3. random_variants   — randomly mutated variants as guidance
  4. uniform_leaves    — training alignment sequences as guidance
  5. esm_only_variants — ESM-only sampled variants as guidance

Usage:
  python prophet/run_comparison.py \
      --target hiv_protease \
      --out-dir results/hiv_protease/comparison \
      --gpus 0,1,2,3,4

  # Or directly with file paths (no configs/targets.py needed):
  python prophet/run_comparison.py \
      --variants-fasta results/hiv_protease/stage1/hiv_protease_gibbs_variants.fasta \
      --alignment data/pre_stage1_split/alignments/train/hiv_train_aligned.fasta \
      --out-dir results/hiv_protease/comparison \
      --gpus 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from Bio import SeqIO
from Bio.Align import MultipleSeqAlignment

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON    = Path(sys.executable)
STAGE2    = REPO_ROOT / "prophet" / "stage2.py"

CKPT_RELATIVE = Path("MOG-DFM/ckpt/peptide/cnn_epoch200_lr0.0001_embed512_hidden256_loss3.1051.ckpt")

TABLE2_MODES = [
    ("prophet",           ""),
    ("wt_only",           "--design-mode wt_only"),
    ("random_variants",   "--design-mode random_variants"),
    ("uniform_leaves",    "--design-mode uniform_leaves"),
    ("esm_only_variants", "--design-mode esm_only_variants"),
]


def _build_consensus(alignment_fasta: Path) -> str:
    records = list(SeqIO.parse(str(alignment_fasta), "fasta"))
    aln = MultipleSeqAlignment(records)
    consensus = ""
    for i in range(aln.get_alignment_length()):
        col = aln[:, i]
        counts: dict[str, int] = {}
        for aa in col:
            if aa not in "-X*.":
                counts[aa] = counts.get(aa, 0) + 1
        consensus += max(counts, key=counts.get) if counts else "X"
    return consensus


def _launch(
    name: str,
    variants_fasta: Path,
    alignment_fasta: Path,
    wt_seq: str,
    extra_args: str,
    gpu: int,
    out_dir: Path,
    ckpt: Path,
    n_designs: int,
    n_steps: int,
    peptide_length: int,
    beta: float,
    tau_bind: float,
    seed: int,
    guidance_var_limit: int,
    stamp: str,
) -> subprocess.Popen:
    out_json = out_dir / f"{name}.json"
    log_file = out_dir / f"{name}.log"

    # uniform_leaves needs the training alignment as guidance variants
    mode_args = extra_args
    if "uniform_leaves" in extra_args:
        mode_args += f" --guidance-variants-fasta {alignment_fasta}"

    cmd = [
        str(PYTHON), str(STAGE2),
        "--variants-fasta",          str(variants_fasta),
        "--wt-seq",                   wt_seq,
        "--out-json",                 str(out_json),
        "--n-designs",                str(n_designs),
        "--n-steps",                  str(n_steps),
        "--peptide-length",           str(peptide_length),
        "--beta",                     str(beta),
        "--dfm-ckpt",                 str(ckpt),
        "--device",                   f"cuda:{gpu}",
        "--dfm-device",               f"cuda:{gpu}",
        "--peptiverse-normalization", "raw",
        "--tau-bind",                 str(tau_bind),
        "--seed",                     str(seed),
        "--guidance-var-limit",       str(guidance_var_limit),
        "--verbose-sampling",
        *shlex.split(mode_args),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("HF_HOME",        "/vast/projects/pranam/lab/nnori/.cache/huggingface")
    env.setdefault("TORCH_HOME",     "/vast/projects/pranam/lab/nnori/.cache/torch")
    env.setdefault("XDG_CACHE_HOME", "/vast/projects/pranam/lab/nnori/.cache")

    print(f"  [{name}] GPU {gpu} → {out_json.name}")
    log_fh = open(log_file, "w", encoding="utf-8")
    return subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env,
                            stdout=log_fh, stderr=subprocess.STDOUT)


def main() -> None:
    p = argparse.ArgumentParser(description="PROPHET Table 2 comparison runner")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--target", default=None,
                     help="Target name from configs/targets.py (e.g. hiv_protease)")
    grp.add_argument("--variants-fasta", default=None,
                     help="Path to Gibbs variant FASTA (stage 1 output)")
    p.add_argument("--alignment", default=None,
                   help="Path to training alignment FASTA (required when --variants-fasta is used)")
    p.add_argument("--out-dir",   required=True, help="Output directory for JSON results")
    p.add_argument("--gpus",      default="0,1,2,3,4",
                   help="Comma-separated GPU indices (default: 0,1,2,3,4)")
    p.add_argument("--ckpt",      default=None,
                   help="Path to MOG-DFM checkpoint (default: REPO_ROOT/MOG-DFM/ckpt/...)")
    p.add_argument("--n-designs",       type=int,   default=500)
    p.add_argument("--n-steps",         type=int,   default=200)
    p.add_argument("--peptide-length",  type=int,   default=10)
    p.add_argument("--beta",            type=float, default=5.0)
    p.add_argument("--tau-bind",        type=float, default=8.0,
                   help="Binding score threshold for Ret. metric (default: 8.0)")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--guidance-var-limit", type=int, default=50,
                   help="Subsample guidance variants during DFM sampling (speeds up scoring)")
    args = p.parse_args()

    # Resolve paths
    if args.target is not None:
        sys.path.insert(0, str(REPO_ROOT))
        from configs.targets import get_target
        cfg = get_target(args.target)
        variants_fasta = _find_variants_fasta(args.target, cfg)
        alignment_fasta = REPO_ROOT / cfg["alignment"]
    else:
        if not args.variants_fasta or not args.alignment:
            p.error("--variants-fasta and --alignment are required when --target is not set")
        variants_fasta = Path(args.variants_fasta)
        alignment_fasta = Path(args.alignment)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.ckpt) if args.ckpt else (REPO_ROOT / CKPT_RELATIVE)
    if not ckpt.exists():
        sys.exit(f"ERROR: MOG-DFM checkpoint not found: {ckpt}")

    gpus = [int(g.strip()) for g in args.gpus.split(",")]

    if not variants_fasta.exists():
        sys.exit(f"ERROR: variants FASTA not found: {variants_fasta}")
    if not alignment_fasta.exists():
        sys.exit(f"ERROR: alignment FASTA not found: {alignment_fasta}")

    print(f"Building consensus WT from {alignment_fasta.name} ...")
    wt_seq = _build_consensus(alignment_fasta)
    print(f"  WT length={len(wt_seq)}: {wt_seq[:40]}{'...' if len(wt_seq) > 40 else ''}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nLaunching {len(TABLE2_MODES)} comparison modes on GPUs {gpus} ...")

    procs = []
    for i, (mode_name, mode_extra) in enumerate(TABLE2_MODES):
        gpu = gpus[i % len(gpus)]
        proc = _launch(
            name=mode_name,
            variants_fasta=variants_fasta,
            alignment_fasta=alignment_fasta,
            wt_seq=wt_seq,
            extra_args=mode_extra,
            gpu=gpu,
            out_dir=out_dir,
            ckpt=ckpt,
            n_designs=args.n_designs,
            n_steps=args.n_steps,
            peptide_length=args.peptide_length,
            beta=args.beta,
            tau_bind=args.tau_bind,
            seed=args.seed,
            guidance_var_limit=args.guidance_var_limit,
            stamp=stamp,
        )
        procs.append((mode_name, proc))

    print(f"\nAll {len(procs)} modes launched. Waiting for completion...")
    failed = []
    for name, proc in procs:
        rc = proc.wait()
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        print(f"  [{name}] {status}")
        if rc != 0:
            failed.append(name)

    if failed:
        print(f"\nWARNING: {len(failed)} mode(s) failed: {', '.join(failed)}")
        sys.exit(1)

    print(f"\nAll comparison modes complete. Results in: {out_dir}")
    for mode_name, _ in TABLE2_MODES:
        jf = out_dir / f"{mode_name}.json"
        print(f"  {jf.name}: {'exists' if jf.exists() else 'MISSING'}")


def _find_variants_fasta(target: str, cfg: dict) -> Path:
    """Find the best variants FASTA for a target: ESM-filtered preferred, else raw Gibbs."""
    prefix = cfg.get("out_prefix", target)
    stage1_dir = REPO_ROOT / "results" / target / "stage1"
    candidates = sorted(stage1_dir.glob(f"{prefix}_*esm_filtered*.fasta"))
    if candidates:
        return candidates[0]
    raw = stage1_dir / f"{prefix}_gibbs_variants.fasta"
    if raw.exists():
        return raw
    sys.exit(
        f"ERROR: No variants FASTA found for target '{target}' in {stage1_dir}.\n"
        f"  Run Stage 1 first: sbatch --export=TARGET={target} run_prophet.slurm"
    )


if __name__ == "__main__":
    main()
