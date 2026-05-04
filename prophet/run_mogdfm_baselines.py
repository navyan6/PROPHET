#!/usr/bin/env python3
"""Launch standalone MOG-DFM baseline Stage 2 runs."""
from __future__ import annotations

import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path("/scratch/pranamlab/kimberly/PROPHET")
PYTHON = REPO_ROOT / "venv" / "bin" / "python"
STAGE2_BASELINES = REPO_ROOT / "prophet" / "stage2_mog_baselines.py"
OUTDIR = REPO_ROOT / "results" / "mogdfm_baselines"

VARIANTS_FASTA = REPO_ROOT / "results" / "all_trees_stage1_train_only" / "hiv_train_gibbs_variants.fasta"
WT_SEQ = "PQVTLWQKPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
DFM_CKPT = REPO_ROOT / "MOG-DFM" / "ckpt" / "peptide" / "cnn_epoch200_lr0.0001_embed512_hidden256_loss3.1051.ckpt"

N_DESIGNS = 500
N_STEPS = 200
PEPTIDE_LENGTH = 10
BETA = 5.0
SEED = 42
TAU_BIND = 8.0
GUIDANCE_VAR_LIMIT = 50
GPUS = [4, 5, 6, 7]
MOG_DFM_STEP_LOG_EVERY = "10"

MODES = [
    "wt_only",
    "uniform_leaves",
    "random_variants",
    "esm_only_variants",
]


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    launch_log = OUTDIR / "launch.log"
    with launch_log.open("a", encoding="utf-8") as f:
        f.write(f"\n=== launch {datetime.now().isoformat(timespec='seconds')} ===\n")

    for idx, mode in enumerate(MODES):
        gpu = GPUS[idx % len(GPUS)]
        out_json = OUTDIR / f"hiv_train_{mode}_stage2_mogdfm.json"
        log_file = OUTDIR / f"hiv_train_{mode}.log"
        guidance_fasta = OUTDIR / f"hiv_train_{mode}_guidance_variants.fasta"

        cmd = [
            str(PYTHON),
            str(STAGE2_BASELINES),
            "--variants-fasta", str(VARIANTS_FASTA),
            "--wt-seq", WT_SEQ,
            "--out-json", str(out_json),
            "--n-designs", str(N_DESIGNS),
            "--n-steps", str(N_STEPS),
            "--peptide-length", str(PEPTIDE_LENGTH),
            "--beta", str(BETA),
            "--seed", str(SEED),
            "--design-mode", mode,
            "--dfm-ckpt", str(DFM_CKPT),
            "--device", "cuda:0",
            "--dfm-device", "cuda:0",
            "--peptiverse-normalization", "raw",
            "--tau-bind", str(TAU_BIND),
            "--guidance-var-limit", str(GUIDANCE_VAR_LIMIT),
            "--guidance-out-fasta", str(guidance_fasta),
            "--verbose-sampling",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONUNBUFFERED"] = "1"
        env["MOG_DFM_STEP_LOG_EVERY"] = MOG_DFM_STEP_LOG_EVERY
        env.setdefault("HF_HOME", "/scratch/pranamlab/kimberly/model_cache/hf")
        env.setdefault("TORCH_HOME", "/scratch/pranamlab/kimberly/model_cache/torch")
        env.setdefault("XDG_CACHE_HOME", "/scratch/pranamlab/kimberly/model_cache")

        cmd_str = " ".join(shlex.quote(part) for part in cmd)
        print(f"Launching {mode} on physical GPU {gpu}")
        print("[run]", cmd_str, f"> {log_file} 2>&1")
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        with launch_log.open("a", encoding="utf-8") as f:
            f.write(
                "\t".join(
                    [
                        f"mode={mode}",
                        f"pid={proc.pid}",
                        f"gpu={gpu}",
                        "device=cuda:0",
                        f"out_json={out_json}",
                        f"guidance_fasta={guidance_fasta}",
                        f"log_file={log_file}",
                        f"cmd={cmd_str}",
                    ]
                )
                + "\n"
            )

    print("All MOG-DFM baseline jobs launched.")
    print("Launch manifest:", launch_log)


if __name__ == "__main__":
    main()
