#!/usr/bin/env python3
"""
Automated ablation workflow for PROPHET Stage 2 on HIV-1 protease.
- Computes consensus WT sequence from aligned train variants
- Runs ablation experiments (CVaR eta, Gibbs/leaves, etc.)
- Spreads jobs across available GPUs using nohup
- Writes logs and output JSONs to a results directory
"""
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from Bio import SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.Align import AlignInfo

# --- CONFIG ---
REPO_ROOT = Path("/scratch/pranamlab/kimberly/PROPHET")
PYTHON = REPO_ROOT / "venv" / "bin" / "python"
ALIGNMENT_FASTA = REPO_ROOT / "alignments" / "hiv_sequences_aligned.fasta"
VARIANTS_FASTA = REPO_ROOT / "results" / "all_trees_stage1_train_only" / "hiv_train_gibbs_variants.fasta"
CKPT = REPO_ROOT / "MOG-DFM" / "ckpt" / "peptide" / "cnn_epoch200_lr0.0001_embed512_hidden256_loss3.1051.ckpt"
STAGE2 = REPO_ROOT / "prophet" / "stage2.py"
OUTDIR = REPO_ROOT / "results" / "ablations"
N_DESIGNS = 30
BETA = 5.0
# PEPTIVERSE_MIN = 7.0
# PEPTIVERSE_MAX = 9.0
PEPTIVERSE_NORM = "raw"
SEED = 42
GPUS = [4, 5, 6, 7]  # Adjust as needed
PEPTIDE_LENGTH = 10
N_STEPS = 20

OUTDIR.mkdir(parents=True, exist_ok=True)
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LAUNCH_LOG = OUTDIR / f"launch-{RUN_STAMP}.log"

# --- 1. Compute consensus WT sequence from alignment ---
# --- 1. Compute consensus WT sequence from alignment (manual majority rule) ---
records = list(SeqIO.parse(str(ALIGNMENT_FASTA), "fasta"))
from Bio.Align import MultipleSeqAlignment
alignment = MultipleSeqAlignment(records)
consensus = ""
for i in range(alignment.get_alignment_length()):
    column = alignment[:, i]
    aa_counts = {}
    for aa in column:
        if aa == "-":
            continue
        aa_counts[aa] = aa_counts.get(aa, 0) + 1
    if aa_counts:
        consensus += max(aa_counts, key=aa_counts.get)
    else:
        consensus += "N"  # ambiguous
print(f"Consensus WT sequence (len={len(consensus)}):\n{consensus}\n")
with open(LAUNCH_LOG, "a", encoding="utf-8") as launch_log:
    launch_log.write(f"\n=== launch {datetime.now().isoformat(timespec='seconds')} ===\n")
    launch_log.write(f"run_stamp={RUN_STAMP}\n")
    launch_log.write(f"consensus_len={len(consensus)}\n")
    launch_log.write(f"consensus={consensus}\n")


# --- 2. Define ablation settings and launch jobs ---
ablations = [
    ("cvar_eta_1.0", "--eta 1.0"),
    ("cvar_eta_0.5", "--eta 0.5"),
    ("cvar_eta_0.1", "--eta 0.1"),
    ("gibbs_leaves", "--design-mode uniform_leaves"),
    # Add more ablations here as needed, e.g. ("no_dca", "--no-dca", 4)
]

if len(ablations) > len(GPUS):
    raise ValueError(f"Need at least {len(ablations)} GPUs, but GPUS={GPUS}")

for (name, extra_args), gpu in zip(ablations, GPUS):

    out_json = OUTDIR / f"{name}-{RUN_STAMP}.json"
    log_file = OUTDIR / f"{name}-{RUN_STAMP}.log"
    cmd = [
        str(PYTHON),
        str(STAGE2),
        "--variants-fasta", str(VARIANTS_FASTA),
        "--wt-seq", consensus,
        "--out-json", str(out_json),
        "--n-designs", str(N_DESIGNS),
        "--n-steps", str(N_STEPS),
        "--peptide-length", str(PEPTIDE_LENGTH),
        "--beta", str(BETA),
        "--dfm-ckpt", str(CKPT),
        # CUDA_VISIBLE_DEVICES exposes one physical GPU to this process, so
        # PyTorch must address it as cuda:0 inside the child process.
        "--device", "cuda:0",
        "--dfm-device", "cuda:0",
        "--peptiverse-normalization", PEPTIVERSE_NORM,
        "--seed", str(SEED),
        "--verbose-sampling",
        *shlex.split(extra_args),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("HF_HOME", "/scratch/pranamlab/kimberly/model_cache/hf")
    env.setdefault("TORCH_HOME", "/scratch/pranamlab/kimberly/model_cache/torch")
    env.setdefault("XDG_CACHE_HOME", "/scratch/pranamlab/kimberly/model_cache")

    cmd_str = " ".join(shlex.quote(part) for part in cmd)
    print(f"Launching ablation: {name} on GPU {gpu}")
    print("[run]", cmd_str, f"> {log_file} 2>&1")
    with open(log_file, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    with open(LAUNCH_LOG, "a", encoding="utf-8") as launch_log:
        launch_log.write(
            "\t".join(
                [
                    f"name={name}",
                    f"pid={proc.pid}",
                    f"gpu={gpu}",
                    "device=cuda:0",
                    f"out_json={out_json}",
                    f"log_file={log_file}",
                    f"cmd={cmd_str}",
                ]
            )
            + "\n"
        )

print("All ablation jobs launched. Monitor logs in:", OUTDIR)
print("Launch manifest:", LAUNCH_LOG)
