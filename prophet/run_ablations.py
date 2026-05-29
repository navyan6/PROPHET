#!/usr/bin/env python3
"""
Automated ablation workflow for PROPHET Stage 2 on HIV-1 protease.

Covers all experiments in the paper:
  Table 2  — method comparison baselines (prophet, wt_only, uniform_leaves,
              random_variants, esm_only_variants)
  Table 4  — Stage-1 component ablations (note: -DCA / -lambda variants require
              Stage-1 re-runs; see NOTE below)
  Table 5  — CVaR eta sensitivity sweep
  Table 6  — Gibbs sample count M sensitivity sweep (variant-limit proxy)
  Table 7  — Tree count J sensitivity sweep (pre-computed variant FASTAs needed)

NOTE on Table 4 Stage-1 ablations (-DCA, -lambda, -ESM):
  These three conditions require re-running Stage 1 with the flags
  --ablate-zero-dca-couplings, --ablate-flatten-lambda, or without
  --esm-filter-delta, then running Stage 2 on the resulting variant FASTAs.
  Set STAGE1_ABLATION_VARIANT_DIR to a directory that already contains those
  FASTAs (produced by running stage1.py separately), or leave it None to skip.

Jobs are spread across GPUs using round-robin assignment.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from Bio import SeqIO
from Bio.Align import MultipleSeqAlignment

# ---------------------------------------------------------------------------
# CONFIG — PARCC paths (vast storage, account pranam-lab)
# ---------------------------------------------------------------------------
REPO_ROOT = Path("/vast/projects/pranam/lab/nnori/hadsbm-hiv")
PYTHON    = Path(sys.executable)   # uses whichever python launched this script
ALIGNMENT_FASTA = REPO_ROOT / "alignments" / "hiv_sequences_aligned.fasta"
VARIANTS_FASTA  = (
    REPO_ROOT / "results" / "hiv_prophet_final"
    / "hiv_prophet_t015_esm_filtered_d20.fasta"
)
CKPT    = (
    REPO_ROOT / "MOG-DFM" / "ckpt" / "peptide"
    / "cnn_epoch200_lr0.0001_embed512_hidden256_loss3.1051.ckpt"
)
STAGE2  = REPO_ROOT / "prophet" / "stage2.py"
OUTDIR  = REPO_ROOT / "results" / "ablations"

# Unfiltered Gibbs variants (500) — used for t4_no_esm and M > 149
UNFILTERED_FASTA = (
    REPO_ROOT / "results" / "hiv_prophet_final"
    / "hiv_prophet_t015_variants.fasta"
)
# Training alignment sequences — used as guidance for uniform_leaves ablation
TRAINING_LEAVES_FASTA = (
    REPO_ROOT / "data" / "pre_stage1_split" / "alignments" / "train"
    / "hiv_train_aligned.fasta"
)

# Stage-1 ablation variant FASTAs (Table 4).  Set to None to skip.
STAGE1_ABLATION_VARIANT_DIR: Path | None = (
    REPO_ROOT / "results" / "stage1_ablations"
)

N_DESIGNS      = 500
BETA           = 5.0
PEPTIVERSE_NORM = "raw"
TAU_BIND       = 8.0    # threshold for Ret. column in Tables 2/4/5/6/7
SEED           = 42
GPUS           = list(range(8))   # 8× B200 on DGX node
PEPTIDE_LENGTH = 10
N_STEPS        = 200

# Sensitivity sweep values (Tables 5, 6, 7)
ETA_VALUES     = [1.0, 0.5, 0.1]           # Table 5 — CVaR eta
M_VALUES       = [50, 100, 250, 500, 1000]  # Table 6 — variant subset M
J_SWEEP_DIRS   = {                          # Table 7 — tree count J
    # Map J value to the variant FASTA produced by a Stage-1 run with that J.
    # J=100 already done (main run). J=25/50 produced by run_stage1_j_sweep.slurm.
    # J=200 needs 200 bootstrap trees first (run build_trees with --n-bootstraps 200).
    25:  REPO_ROOT / "results" / "stage1_j_sweep" / "J_25"  / "hiv_train_gibbs_variants.fasta",
    50:  REPO_ROOT / "results" / "stage1_j_sweep" / "J_50"  / "hiv_train_gibbs_variants.fasta",
    100: REPO_ROOT / "results" / "hiv_prophet_final" / "hiv_prophet_t015_esm_filtered_d20.fasta",
    200: REPO_ROOT / "results" / "stage1_j_sweep" / "J_200" / "hiv_train_gibbs_variants.fasta",
}
# ---------------------------------------------------------------------------

OUTDIR.mkdir(parents=True, exist_ok=True)
RUN_STAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")
LAUNCH_LOG = OUTDIR / f"launch-{RUN_STAMP}.log"

# ---------------------------------------------------------------------------
# 1. Compute consensus WT sequence from the alignment (majority-rule)
# ---------------------------------------------------------------------------
records   = list(SeqIO.parse(str(ALIGNMENT_FASTA), "fasta"))
alignment = MultipleSeqAlignment(records)
consensus = ""
for i in range(alignment.get_alignment_length()):
    column   = alignment[:, i]
    aa_counts: dict[str, int] = {}
    for aa in column:
        if aa == "-":
            continue
        aa_counts[aa] = aa_counts.get(aa, 0) + 1
    consensus += max(aa_counts, key=aa_counts.get) if aa_counts else "N"

print(f"Consensus WT sequence (len={len(consensus)}):\n{consensus}\n")
with open(LAUNCH_LOG, "a", encoding="utf-8") as f:
    f.write(f"\n=== launch {datetime.now().isoformat(timespec='seconds')} ===\n")
    f.write(f"run_stamp={RUN_STAMP}\n")
    f.write(f"consensus_len={len(consensus)}\nconsensus={consensus}\n")

# ---------------------------------------------------------------------------
# 2. Build the experiment list
#    Each entry: (name, variants_fasta, extra_stage2_args_str)
# ---------------------------------------------------------------------------
experiments: list[tuple[str, Path, str]] = []

# --- Table 2: design-mode comparison ---
TABLE2_MODES = [
    ("prophet",            ""),
    ("wt_only",            "--design-mode wt_only"),
    ("uniform_leaves",     f"--design-mode uniform_leaves --guidance-variants-fasta {TRAINING_LEAVES_FASTA}"),
    ("random_variants",    "--design-mode random_variants"),
    ("esm_only_variants",  "--design-mode esm_only_variants"),
]
for mode_name, mode_args in TABLE2_MODES:
    experiments.append((f"t2_{mode_name}", VARIANTS_FASTA, mode_args))

# --- Table 4: Stage-1 component ablations ---
if STAGE1_ABLATION_VARIANT_DIR is not None:
    abl_dir = STAGE1_ABLATION_VARIANT_DIR
    stage1_ablations = [
        ("t4_no_dca",    abl_dir / "hiv_no_dca_gibbs_variants.fasta",    "--variant-limit 149"),
        ("t4_no_lambda", abl_dir / "hiv_no_lambda_gibbs_variants.fasta", "--variant-limit 149"),
        ("t4_no_esm",    UNFILTERED_FASTA,                                ""),
    ]
    for abl_name, abl_fasta, abl_args in stage1_ablations:
        if abl_fasta.exists():
            experiments.append((abl_name, abl_fasta, abl_args))
        else:
            print(f"[skip] {abl_name}: variant FASTA not found at {abl_fasta}")

# --- Table 5: CVaR eta sensitivity ---
for eta in ETA_VALUES:
    experiments.append((f"t5_eta_{eta}", VARIANTS_FASTA, f"--eta {eta}"))

# --- Table 6: variant subset M sensitivity ---
# ESM-filtered set has 149 variants; use unfiltered (500) for M > 149
for m in M_VALUES:
    fasta = VARIANTS_FASTA if m <= 149 else UNFILTERED_FASTA
    experiments.append((f"t6_M_{m}", fasta, f"--variant-limit {m}"))

# --- Table 7: tree count J sensitivity ---
for j, j_fasta in J_SWEEP_DIRS.items():
    if j_fasta.exists():
        experiments.append((f"t7_J_{j}", j_fasta, ""))
    else:
        print(f"[skip] t7_J_{j}: variant FASTA not found at {j_fasta}")

print(f"Total experiments to launch: {len(experiments)}\n")

# ---------------------------------------------------------------------------
# 3. Launch jobs — round-robin across GPUs
# ---------------------------------------------------------------------------
def _launch(name: str, variants_fasta: Path, extra_args: str, gpu: int) -> subprocess.Popen:
    out_json = OUTDIR / f"{name}-{RUN_STAMP}.json"
    log_file = OUTDIR / f"{name}-{RUN_STAMP}.log"
    cmd = [
        str(PYTHON),
        str(STAGE2),
        "--variants-fasta",          str(variants_fasta),
        "--wt-seq",                   consensus,
        "--out-json",                 str(out_json),
        "--n-designs",                str(N_DESIGNS),
        "--n-steps",                  str(N_STEPS),
        "--peptide-length",           str(PEPTIDE_LENGTH),
        "--beta",                     str(BETA),
        "--dfm-ckpt",                 str(CKPT),
        "--device",                   "cuda:0",
        "--dfm-device",               "cuda:0",
        "--peptiverse-normalization", PEPTIVERSE_NORM,
        "--tau-bind",                 str(TAU_BIND),
        "--seed",                     str(SEED),
        "--guidance-var-limit",       "50",
        "--verbose-sampling",
        *shlex.split(extra_args),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("HF_HOME",       "/vast/projects/pranam/lab/nnori/.cache/huggingface")
    env.setdefault("TORCH_HOME",    "/vast/projects/pranam/lab/nnori/.cache/torch")
    env.setdefault("XDG_CACHE_HOME","/vast/projects/pranam/lab/nnori/.cache")

    cmd_str = " ".join(shlex.quote(p) for p in cmd)
    print(f"Launching {name} on GPU {gpu}")
    log_fh = open(log_file, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    with open(LAUNCH_LOG, "a", encoding="utf-8") as lf:
        lf.write(
            "\t".join([
                f"name={name}",
                f"pid={proc.pid}",
                f"gpu={gpu}",
                f"out_json={out_json}",
                f"log_file={log_file}",
                f"cmd={cmd_str}",
            ]) + "\n"
        )
    return proc


procs = []
for idx, (name, vfasta, extra) in enumerate(experiments):
    gpu = GPUS[idx % len(GPUS)]
    procs.append(_launch(name, vfasta, extra, gpu))

print(f"\nAll {len(experiments)} jobs launched. Waiting for completion...")
for proc in procs:
    proc.wait()

print(f"\nAll {len(experiments)} jobs complete.")
print("Results in:", OUTDIR)
print("Launch manifest:", LAUNCH_LOG)
