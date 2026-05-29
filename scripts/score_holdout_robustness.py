#!/usr/bin/env python3
"""
Score Stage 2 design JSONs against the holdout clade using PeptiVerse.

For each ablation JSON, scores every peptide against all holdout sequences
and computes holdout CVaR_0.1 (robust), mean, min, and retention.
Writes per-file *_holdout.json and a combined holdout_summary.csv.

Skips files where *_holdout.json already exists (restartable).

Usage (PARCC, 1 GPU):
    python scripts/score_holdout_robustness.py \
        --ablations-dir results/ablations \
        --holdout-fasta data/pre_stage1_split/alignments/test/hiv_test_clade_holdout.fasta \
        --out-dir       results/holdout_scores \
        --device        cuda:0

    # Also score RFdiffusion baseline:
    python scripts/score_holdout_robustness.py \
        --ablations-dir results/ablations \
        --rfd-json      results/hiv_prophet_final/rfd_baseline_binding.json \
        --holdout-fasta data/pre_stage1_split/alignments/test/hiv_test_clade_holdout.fasta \
        --out-dir       results/holdout_scores \
        --device        cuda:0
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prophet.stage2 import AffinityScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_fasta(path: Path) -> list[str]:
    seqs, cur = [], []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur).replace("-", "").upper())
            cur = []
        else:
            cur.append(line.strip())
    if cur:
        seqs.append("".join(cur).replace("-", "").upper())
    return [s for s in seqs if s]


def cvar(scores: np.ndarray, eta: float = 0.1) -> float:
    k = max(1, int(len(scores) * eta))
    return float(np.sort(scores)[:k].mean())


def score_file(
    json_path: Path,
    holdout_seqs: list[str],
    scorer: AffinityScorer,
    tau: float,
    out_dir: Path,
    peptide_key: str = "peptide",
) -> dict:
    stem = json_path.stem
    out_path = out_dir / f"{stem}_holdout.json"

    if out_path.exists():
        print(f"  [skip] {stem}: holdout scores already exist")
        data = json.load(out_path.open())
        return aggregate_holdout(data, tau)

    def _drop_per_variant(pairs):
        return {k: v for k, v in pairs if k != "per_variant"}

    raw = json.load(json_path.open(), object_pairs_hook=_drop_per_variant)
    if isinstance(raw, dict) and "designs" in raw:
        designs = raw["designs"]
    elif isinstance(raw, list):
        designs = raw
    else:
        designs = []

    n = len(designs)
    print(f"  Scoring {stem}: {n} peptides × {len(holdout_seqs)} holdout seqs ...", flush=True)
    t0 = time.time()

    results = []
    for i, d in enumerate(designs):
        pep = d.get(peptide_key) or d.get("sequence", "")
        if not pep:
            continue
        scores = scorer.score_variants_batched(pep, holdout_seqs)
        results.append({
            "peptide":           pep,
            "wt_score":          d.get("wt_score"),
            "holdout_robust":    cvar(scores, eta=0.1),
            "holdout_mean":      float(scores.mean()),
            "holdout_min":       float(scores.min()),
            "holdout_retention": float((scores >= tau).mean()),
        })
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{n}  ({elapsed:.0f}s)", flush=True)

    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {out_path.name}  ({time.time()-t0:.0f}s)", flush=True)

    return aggregate_holdout(results, tau)


def aggregate_holdout(results: list[dict], tau: float) -> dict:
    if not results:
        return {}
    rb  = [r["holdout_robust"]    for r in results if r.get("holdout_robust")    is not None]
    mn  = [r["holdout_mean"]      for r in results if r.get("holdout_mean")      is not None]
    mi  = [r["holdout_min"]       for r in results if r.get("holdout_min")       is not None]
    ret = [r["holdout_retention"] for r in results if r.get("holdout_retention") is not None]
    wt  = [r["wt_score"]          for r in results if r.get("wt_score")          is not None]
    return {
        "n":              len(results),
        "holdout_robust": float(np.mean(rb))  if rb  else float("nan"),
        "holdout_mean":   float(np.mean(mn))  if mn  else float("nan"),
        "holdout_min":    float(np.mean(mi))  if mi  else float("nan"),
        "holdout_ret":    float(np.mean(ret)) if ret else float("nan"),
        "wt_score":       float(np.mean(wt))  if wt  else float("nan"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TAG_LABEL = {
    "t2_prophet":           "PROPHET",
    "t2_prophet_rescue":    "PROPHET",
    "t2_wt_only":           "MOG-DFM (WT only)",
    "t2_random_variants":   "MOG-DFM (Random variants)",
    "t2_uniform_leaves":    "MOG-DFM (Uniform leaves)",
    "t2_esm_only_variants": "MOG-DFM (ESM only)",
    "t4_no_dca":            "−DCA",
    "t4_no_lambda":         "−λ weighting",
    "t4_no_esm":            "−ESM filter",
    "t5_eta_0.1":           "CVaR η=0.1",
    "t5_eta_0.5":           "CVaR η=0.5",
    "t5_eta_1.0":           "CVaR η=1.0",
    "t6_M_50":              "M=50",
    "t6_M_100":             "M=100",
    "t6_M_250":             "M=250",
    "t6_M_500":             "M=500",
    "t6_M_1000":            "M=1000",
    "t7_J_25":              "J=25",
    "t7_J_50":              "J=50",
    "t7_J_100":             "J=100",
    "t7_J_200":             "J=200",
    "t5_tevo_t05":          "T_evo=0.5",
    "t5_tevo_t10":          "T_evo=1.0",
    "t5_tevo_t20":          "T_evo=2.0",
    "t5_tevo_t50":          "T_evo=5.0",
    # Conservation penalty sweep — pick best, that becomes full PROPHET
    "conserv_cw001":        "PROPHET (cw=0.01)",
    "conserv_cw005":        "PROPHET (cw=0.05)",
    "conserv_cw01":         "PROPHET (cw=0.1)",
    # −conservation ablation (conserv_weight=0, i.e. current t2_prophet runs)
    "t4_no_conserv":        "−Conservation",
    "t4_no_bootstrap":      "−Bootstrap ensemble",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablations-dir",  default="results/ablations")
    ap.add_argument("--holdout-fasta",  required=True)
    ap.add_argument("--out-dir",        default="results/holdout_scores")
    ap.add_argument("--rfd-json",       default=None,
                    help="Optional RFdiffusion baseline JSON to also score.")
    ap.add_argument("--device",         default="cuda:0")
    ap.add_argument("--tau",            type=float, default=8.0)
    ap.add_argument("--tags",           nargs="*", default=None,
                    help="Only process files whose tag starts with one of these prefixes.")
    ap.add_argument("--tag-prefix",    default=None,
                    help="Strip this prefix from file tags before TAG_LABEL lookup "
                         "(e.g. 'dengue_ns3_' so files match standard table keys).")
    args = ap.parse_args()

    abl_dir  = Path(args.ablations_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading holdout sequences from {args.holdout_fasta} ...")
    holdout_seqs = load_fasta(Path(args.holdout_fasta))
    print(f"  {len(holdout_seqs)} holdout sequences loaded.")

    print("Loading PeptiVerse scorer ...")
    scorer = AffinityScorer(device=args.device, peptiverse_normalization="raw")

    # Collect JSON files, pick most recent per tag
    jsons = sorted(abl_dir.glob("*.json"))
    best: dict[str, Path] = {}
    for p in jsons:
        raw_tag = p.stem.split("-")[0]
        # Strip optional virus prefix so flavivirus files match standard table keys
        tag = raw_tag[len(args.tag_prefix):] if args.tag_prefix and raw_tag.startswith(args.tag_prefix) else raw_tag
        if tag not in TAG_LABEL:
            continue
        if args.tags and not any(raw_tag.startswith(t) for t in args.tags):
            continue
        if tag not in best or p.stem > best[tag].stem:
            best[tag] = p

    rows = []

    # Score RFdiffusion baseline if provided
    if args.rfd_json:
        rfd_path = Path(args.rfd_json)
        if rfd_path.exists():
            print(f"\n[RFdiffusion]")
            agg = score_file(rfd_path, holdout_seqs, scorer, args.tau, out_dir)
            rows.append({"tag": "rfd", "label": "RFdiffusion", **agg})
        else:
            print(f"[skip] RFdiffusion: {rfd_path} not found")

    # Score ablation JSONs
    seen_prophet = False
    for tag in sorted(best):
        label = TAG_LABEL[tag]
        if "prophet" in tag.lower():
            if seen_prophet:
                continue
            seen_prophet = True
        print(f"\n[{tag}]")
        agg = score_file(best[tag], holdout_seqs, scorer, args.tau, out_dir)
        rows.append({"tag": tag, "label": label, **agg})

    # Write summary CSV
    if rows:
        summary_path = out_dir / "holdout_summary.csv"
        fields = list(rows[0].keys())
        with summary_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved summary: {summary_path}")

        # Print markdown table
        print("\n=== Holdout Robustness Summary ===")
        print(f"{'Method':<30} {'HoldRobust↑':>12} {'HoldMean↑':>10} {'HoldMin↑':>9} {'HoldRet↑':>9}")
        print("-" * 75)
        for r in rows:
            def f(v): return f"{v:.3f}" if isinstance(v, float) and v == v else "—"
            print(f"{r['label']:<30} {f(r.get('holdout_robust')):>12} "
                  f"{f(r.get('holdout_mean')):>10} {f(r.get('holdout_min')):>9} "
                  f"{f(r.get('holdout_ret')):>9}")
    else:
        print("No files processed.")


if __name__ == "__main__":
    main()
