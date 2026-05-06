#!/usr/bin/env python3
"""
scripts/eval_boltz.py
Re-score PROPHET / PepTune designed peptides using Boltz-2 binding affinity
prediction, which can approach FEP-level accuracy ~1000× faster.

Boltz-2 outputs two complementary metrics:
  affinity_probability_binary  — probability of being a binder (0–1), use for
                                  hit discovery / binder vs. decoy screening.
  affinity_pred_value          — predicted log10(IC50 μM), use for lead
                                  optimization / ranking among binders.

This script:
  1. Reads a PROPHET-format designs JSON (list of {peptide, wt_score, ...}).
  2. For each peptide, writes a Boltz YAML input (protein target + peptide).
  3. Runs `boltz predict` on each YAML.
  4. Parses the output predictions JSON.
  5. Augments the designs with boltz_binder_prob and boltz_affinity_pred,
     then writes a new JSON.

Usage
-----
  # Basic (one peptide at a time — safe, no OOM)
  python scripts/eval_boltz.py \\
      --designs-json results/ablations/t2_prophet.json \\
      --target-seq PQVTLWQRPLVTIKIGGQL... \\
      --out-json results/ablations/t2_prophet_boltz.json \\
      --boltz-bin boltz \\
      --device cuda:0

  # Evaluate only top-N by wt_score (fast sanity check)
  python scripts/eval_boltz.py \\
      --designs-json results/ablations/t2_prophet.json \\
      --target-seq PQVTLWQRPLVTIKIGGQL... \\
      --out-json results/ablations/t2_prophet_boltz_top20.json \\
      --top-n 20 \\
      --min-wt-score 7.5

Requirements
------------
  pip install boltz[cuda]   (or: cd boltz && pip install -e .[cuda])
  boltz predict --help      (verify CLI is available)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Boltz YAML template (Boltz-2 format, protein–protein affinity)
# ──────────────────────────────────────────────────────────────────────────────
_BOLTZ_YAML = """\
version: 1
sequences:
  - protein:
      id: A
      sequence: {target}
  - protein:
      id: B
      sequence: {peptide}
properties:
  - affinity:
      binder: B
"""


def _write_yaml(path: Path, target: str, peptide: str) -> None:
    path.write_text(_BOLTZ_YAML.format(target=target, peptide=peptide))


def _run_boltz(
    boltz_bin: str,
    yaml_path: Path,
    out_dir: Path,
    device: str = "cuda:0",
    cache_dir: str | None = None,
) -> dict:
    """Run boltz predict and return the parsed predictions dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        boltz_bin, "predict", str(yaml_path),
        "--out_dir", str(out_dir),
        "--devices", "1",
        "--accelerator", "gpu" if device.startswith("cuda") else "cpu",
        "--override",   # allow re-running in same out_dir
    ]
    if cache_dir:
        cmd.extend(["--cache", cache_dir])
    env = os.environ.copy()
    env.setdefault("HF_HOME", "/scratch/pranamlab/kimberly/model_cache/hf")
    env.setdefault("TRANSFORMERS_CACHE", env["HF_HOME"])
    env.setdefault("BOLTZ_CACHE", "/scratch/pranamlab/kimberly/boltz_cache")
    if device.startswith("cuda:"):
        gpu_idx = device.split(":")[1]
        env["CUDA_VISIBLE_DEVICES"] = gpu_idx

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    except FileNotFoundError:
        print(f"[boltz error] CLI not found: {boltz_bin}", file=sys.stderr)
        return {}

    if result.returncode != 0:
        print(f"[boltz command] {' '.join(cmd)}", file=sys.stderr)
        print(f"[boltz stderr] {result.stderr[-2000:]}", file=sys.stderr)
        return {}

    # Boltz writes predictions to out_dir/predictions/<name>/...
    # The affinity values are in the top-level predictions JSON
    pred_jsons = list(out_dir.rglob("*predictions*.json"))
    if not pred_jsons:
        # Try flat JSON from newer boltz versions
        pred_jsons = list(out_dir.rglob("*.json"))

    for pj in pred_jsons:
        try:
            with pj.open() as f:
                data = json.load(f)
            # Boltz-2 flat format: dict with affinity keys at top level
            if "affinity_pred_value" in data or "affinity_probability_binary" in data:
                return data
            # Boltz-2 nested: {predictions: [{affinity_pred_value: ...}]}
            if isinstance(data, list) and data:
                if "affinity_pred_value" in data[0]:
                    return data[0]
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, dict) and "affinity_pred_value" in v:
                        return v
                    if isinstance(v, list) and v and "affinity_pred_value" in v[0]:
                        return v[0]
        except Exception:
            continue

    manifest = next(out_dir.rglob("manifest.json"), None)
    if manifest is not None:
        try:
            print(f"[boltz manifest] {manifest.read_text()[:2000]}", file=sys.stderr)
        except Exception:
            pass
    files = [str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file()]
    print(
        "[boltz warning] No affinity prediction JSON found. "
        "Boltz-2 affinity requires the affinity binder to be a ligand; "
        "protein-peptide inputs may only produce structure/confidence outputs.",
        file=sys.stderr,
    )
    print(f"[boltz files] {files[:30]}", file=sys.stderr)
    if result.stdout.strip():
        print(f"[boltz stdout] {result.stdout[-2000:]}", file=sys.stderr)
    if result.stderr.strip():
        print(f"[boltz stderr] {result.stderr[-2000:]}", file=sys.stderr)
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate peptide designs with Boltz-2 binding affinity."
    )
    ap.add_argument("--designs-json", required=True,
                    help="PROPHET/PepTune output JSON (list of design dicts).")
    ap.add_argument("--target-seq", required=True,
                    help="Full target protein sequence.")
    ap.add_argument("--out-json", required=True,
                    help="Output JSON path (designs augmented with Boltz scores).")
    ap.add_argument("--boltz-bin", default="boltz",
                    help="Path to `boltz` CLI (default: boltz on PATH).")
    ap.add_argument("--boltz-cache", default=os.environ.get("BOLTZ_CACHE"),
                    help="Optional Boltz cache directory for checkpoints/data.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--top-n", type=int, default=None,
                    help="Only evaluate top-N designs by wt_score.")
    ap.add_argument("--min-wt-score", type=float, default=None,
                    help="Skip designs with wt_score below this threshold.")
    ap.add_argument("--keep-tmp", action="store_true",
                    help="Keep temporary YAML / boltz output directories.")
    args = ap.parse_args()

    target_seq = args.target_seq.strip().replace("-", "").upper()

    with open(args.designs_json) as f:
        designs: list[dict] = json.load(f)
    if not designs:
        print("[error] Empty designs JSON.", file=sys.stderr)
        sys.exit(1)

    # Filter / select designs to evaluate
    eval_designs = designs
    if args.min_wt_score is not None:
        eval_designs = [d for d in eval_designs
                        if (d.get("wt_score") or 0) >= args.min_wt_score]
        print(f"After min-wt-score filter: {len(eval_designs)} designs.",
              file=sys.stderr)
    if args.top_n is not None:
        eval_designs.sort(key=lambda d: d.get("wt_score") or 0, reverse=True)
        eval_designs = eval_designs[: args.top_n]
        print(f"After top-{args.top_n}: {len(eval_designs)} designs.",
              file=sys.stderr)

    print(f"Evaluating {len(eval_designs)} peptides with Boltz-2 ...",
          file=sys.stderr)

    # Build lookup: peptide → boltz results
    boltz_cache: dict[str, dict] = {}
    unique_peptides = list(dict.fromkeys(d["peptide"] for d in eval_designs))

    tmp_ctx = tempfile.TemporaryDirectory()
    tmp_dir = Path(tmp_ctx.name)

    try:
        for i, peptide in enumerate(unique_peptides):
            yaml_path = tmp_dir / f"input_{i:05d}.yaml"
            out_dir   = tmp_dir / f"out_{i:05d}"
            _write_yaml(yaml_path, target_seq, peptide)

            pred = _run_boltz(
                args.boltz_bin,
                yaml_path,
                out_dir,
                args.device,
                args.boltz_cache,
            )
            boltz_cache[peptide] = pred

            aff  = pred.get("affinity_pred_value")
            prob = pred.get("affinity_probability_binary")
            status = f"aff={aff:.3f}" if aff is not None else "no_aff"
            print(f"  [{i+1:4d}/{len(unique_peptides)}] {peptide[:20]:20s} "
                  f"prob={prob}  {status}", file=sys.stderr)

    finally:
        if not args.keep_tmp:
            tmp_ctx.cleanup()

    n_success = sum(1 for pred in boltz_cache.values() if pred)
    if unique_peptides and n_success == 0:
        print(
            "[error] Boltz produced no usable affinity predictions. "
            "For protein-peptide inputs this is expected: Boltz-2 affinity "
            "currently requires the binder chain to be a ligand.",
            file=sys.stderr,
        )
        sys.exit(2)
    if n_success < len(unique_peptides):
        print(
            f"[warning] Boltz produced affinity predictions for "
            f"{n_success}/{len(unique_peptides)} unique peptides.",
            file=sys.stderr,
        )

    # Augment all designs (not just eval subset) — mark unevaluated ones
    augmented = []
    for d in designs:
        entry = dict(d)
        pred  = boltz_cache.get(d["peptide"])
        if pred is not None:
            entry["boltz_binder_prob"]  = pred.get("affinity_probability_binary")
            entry["boltz_affinity_pred"] = pred.get("affinity_pred_value")
        else:
            entry["boltz_binder_prob"]   = None
            entry["boltz_affinity_pred"] = None
        augmented.append(entry)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(augmented, f, indent=2)
    print(f"\nSaved {len(augmented)} entries → {out_path}", file=sys.stderr)

    # Summary
    evaluated = [d for d in augmented if d["boltz_binder_prob"] is not None]
    if evaluated:
        import statistics
        probs = [d["boltz_binder_prob"] for d in evaluated
                 if d["boltz_binder_prob"] is not None]
        affs  = [d["boltz_affinity_pred"] for d in evaluated
                 if d["boltz_affinity_pred"] is not None]
        print("\n=== Boltz-2 summary ===", file=sys.stderr)
        if probs:
            print(f"  binder_prob   : mean={statistics.mean(probs):.3f}  "
                  f"median={statistics.median(probs):.3f}  "
                  f"n_binders(>0.5)={sum(1 for p in probs if p > 0.5)}/{len(probs)}",
                  file=sys.stderr)
        if affs:
            print(f"  affinity_pred : mean={statistics.mean(affs):.3f}  "
                  f"median={statistics.median(affs):.3f}  "
                  f"min={min(affs):.3f}",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
