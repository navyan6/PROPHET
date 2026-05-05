#!/usr/bin/env python3
"""
scripts/omega_pilot_sweep.py

Run a fast pilot sweep over omega (Pareto weight) values using only N_PILOT
peptides per omega point, plot wt_score vs robust_score for each omega to
find the best tradeoff, then optionally run the full 500-peptide job locked
to that best omega.

omega[0] = weight on WT binding score
omega[1] = weight on robustness score (CVaR across escape variants)
The two always sum to 1.

Usage
-----
  # Step 1 — pilot only (fast, ~5 peptides per omega, produces a plot)
  python scripts/omega_pilot_sweep.py \\
      --variants-fasta results/all_trees_stage1_train_only/hiv_train_gibbs_variants.fasta \\
      --wt-seq PQVTLWQRPLVTIKIGGQL... \\
      --dfm-ckpt MOG-DFM/ckpt/peptide/cnn_epoch200_lr0.0001_embed512_hidden256_loss3.1051.ckpt \\
      --out-plot results/omega_pilot.png \\
      --n-pilot 5 --n-omega 10 \\
      --device cuda:0

  # Step 2 — take printed best omega, then run full job:
  #   bash scripts/run_hiv_stage2_mode.sh prophet 0

  # OR do both in one command:
  python scripts/omega_pilot_sweep.py ... --run-full --n-designs 500 \\
      --full-out-json results/hiv_stage2/hiv_train_prophet_best_omega.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prophet.stage2 import (
    AffinityScorer,
    _load_variants_fasta,
    mog_dfm_guided_design,
)
from transformers import AutoTokenizer


def _load_model(dfm_ckpt: str, dfm_device: str):
    """Load MOG-DFM the same way stage2.py main() does."""
    import sys as _sys
    mogdfm_dir = str(ROOT / "MOG-DFM")
    if mogdfm_dir not in _sys.path:
        _sys.path.insert(0, mogdfm_dir)
    from models.peptide_classifiers import load_solver  # noqa: E402
    print(f"Loading DFM model from {dfm_ckpt} ...", file=sys.stderr)
    model = load_solver(dfm_ckpt, vocab_size=24, device=dfm_device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    print("DFM model loaded.", file=sys.stderr)
    return model, tokenizer


def _run_omega(
    omega: list[float],
    wt_seq: str,
    variants: list[str],
    aff_fn: AffinityScorer,
    dfm_model,
    dfm_tokenizer,
    n_pilot: int,
    peptide_length: int,
    n_steps: int,
    eta: float,
    dfm_device: str,
    seed: int,
) -> list[dict]:
    results = mog_dfm_guided_design(
        wt_seq=wt_seq,
        eval_variants=variants,
        aff_fn=aff_fn,
        peptide_length=peptide_length,
        n_designs=n_pilot,
        eta=eta,
        design_mode="prophet",
        dfm_model=dfm_model,
        dfm_tokenizer=dfm_tokenizer,
        dfm_device=dfm_device,
        fixed_omega=omega,
        n_steps=n_steps,
        seed=seed,
        verbose=False,
    )
    return [
        {
            "omega_w": omega[0],
            "wt_score": r.wt_score,
            "robust_score": r.robust_score,
            "peptide": r.peptide,
        }
        for r in results
    ]


def _best_omega(rows: list[dict]) -> float:
    """
    Pick the omega[0] value whose peptides have the highest
    product of mean_wt_score * mean_robust_score.
    This penalises extreme omegas that sacrifice one objective entirely.
    """
    omega_ws = sorted(set(r["omega_w"] for r in rows))
    best_w, best_score = 0.5, -1.0
    for w in omega_ws:
        subset = [r for r in rows if r["omega_w"] == w]
        mean_wt = float(np.mean([r["wt_score"] for r in subset]))
        mean_rb = float(np.mean([r["robust_score"] for r in subset]))
        combined = mean_wt * mean_rb
        if combined > best_score:
            best_score = combined
            best_w = w
    return best_w


def _plot(rows: list[dict], out_path: str, best_w: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    omega_ws = sorted(set(r["omega_w"] for r in rows))
    colors = cm.viridis(np.linspace(0, 1, len(omega_ws)))
    color_map = {w: c for w, c in zip(omega_ws, colors)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: scatter wt_score vs robust_score, coloured by omega
    ax = axes[0]
    for w in omega_ws:
        subset = [r for r in rows if r["omega_w"] == w]
        xs = [r["wt_score"] for r in subset]
        ys = [r["robust_score"] for r in subset]
        label = f"ω={w:.2f}" + (" ★" if abs(w - best_w) < 1e-6 else "")
        ax.scatter(xs, ys, color=color_map[w], label=label, alpha=0.8, s=50)
    ax.set_xlabel("WT binding score (PeptiVerse)")
    ax.set_ylabel("Robust score (CVaR)")
    ax.set_title("Pilot Pareto front — coloured by omega[0]")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)

    # Right: mean scores per omega
    ax2 = axes[1]
    mean_wts, mean_rbs, combined = [], [], []
    for w in omega_ws:
        subset = [r for r in rows if r["omega_w"] == w]
        mw = float(np.mean([r["wt_score"] for r in subset]))
        mr = float(np.mean([r["robust_score"] for r in subset]))
        mean_wts.append(mw)
        mean_rbs.append(mr)
        combined.append(mw * mr)

    ax2.plot(omega_ws, mean_wts, "o-",  label="mean WT score",      color="steelblue")
    ax2.plot(omega_ws, mean_rbs, "s-",  label="mean robust score",   color="darkorange")
    ax2.plot(omega_ws, combined, "^--", label="product (selection)", color="green", alpha=0.7)
    ax2.axvline(best_w, color="red", linestyle=":", linewidth=1.5,
                label=f"best ω={best_w:.2f}")
    ax2.set_xlabel("omega[0]  (weight on WT binding)")
    ax2.set_ylabel("Score")
    ax2.set_title("Mean scores vs omega weight")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved → {out_path}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pilot omega sweep for PROPHET Stage 2."
    )
    ap.add_argument("--variants-fasta",  required=True)
    ap.add_argument("--wt-seq",          required=True)
    ap.add_argument("--dfm-ckpt",        required=True)
    ap.add_argument("--out-plot",        default="results/omega_pilot.png")
    ap.add_argument("--out-json",        default=None,
                    help="Save pilot rows as JSON (optional).")
    ap.add_argument("--n-pilot",         type=int, default=5,
                    help="Peptides per omega point (default 5).")
    ap.add_argument("--n-omega",         type=int, default=10,
                    help="Number of omega grid points (default 10).")
    ap.add_argument("--eta",             type=float, default=0.1)
    ap.add_argument("--peptide-length",  type=int,   default=10)
    ap.add_argument("--n-steps",         type=int,   default=50,
                    help="DFM steps per pilot run (keep low, e.g. 50).")
    ap.add_argument("--device",          default="cuda:0")
    ap.add_argument("--dfm-device",      default="cuda:0")
    ap.add_argument("--seed",            type=int,   default=42)
    ap.add_argument("--peptiverse-normalization", choices=["raw", "minmax"],
                    default="raw")
    ap.add_argument("--run-full",        action="store_true",
                    help="After pilot, run full job with the best omega.")
    ap.add_argument("--n-designs",       type=int, default=500,
                    help="Designs for the full run (--run-full only).")
    ap.add_argument("--full-out-json",
                    default="results/hiv_stage2/hiv_train_prophet_best_omega.json",
                    help="Output JSON for full run (--run-full only).")
    args = ap.parse_args()

    wt_seq   = args.wt_seq.strip().replace("-", "").upper()
    variants = _load_variants_fasta(args.variants_fasta)
    print(f"Loaded {len(variants)} variants.", file=sys.stderr)

    aff_fn = AffinityScorer(
        device=args.device,
        peptiverse_normalization=args.peptiverse_normalization,
    )
    dfm_model, dfm_tokenizer = _load_model(args.dfm_ckpt, args.dfm_device)

    omega_values = np.linspace(0.0, 1.0, args.n_omega).tolist()
    all_rows: list[dict] = []

    print(f"\nPilot: {args.n_pilot} peptides × {args.n_omega} omega points\n",
          file=sys.stderr)
    for i, w in enumerate(omega_values):
        omega = [float(w), float(1.0 - w)]
        print(f"[{i+1}/{args.n_omega}] omega=({omega[0]:.2f}, {omega[1]:.2f})",
              file=sys.stderr)
        rows = _run_omega(
            omega=omega,
            wt_seq=wt_seq,
            variants=variants,
            aff_fn=aff_fn,
            dfm_model=dfm_model,
            dfm_tokenizer=dfm_tokenizer,
            n_pilot=args.n_pilot,
            peptide_length=args.peptide_length,
            n_steps=args.n_steps,
            eta=args.eta,
            dfm_device=args.dfm_device,
            seed=args.seed + i,
        )
        all_rows.extend(rows)
        mean_wt = float(np.mean([r["wt_score"] for r in rows]))
        mean_rb = float(np.mean([r["robust_score"] for r in rows]))
        print(f"    mean wt={mean_wt:.3f}  mean robust={mean_rb:.3f}",
              file=sys.stderr)

    best_w = _best_omega(all_rows)
    print(f"\n★  Best omega[0] = {best_w:.4f}  →  omega = [{best_w:.4f}, {1-best_w:.4f}]",
          flush=True)

    _plot(all_rows, args.out_plot, best_w)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(all_rows, f, indent=2)
        print(f"Pilot JSON saved → {args.out_json}", file=sys.stderr)

    if args.run_full:
        print(
            f"\n[full run] {args.n_designs} designs, fixed omega=[{best_w:.4f}, {1-best_w:.4f}]",
            file=sys.stderr,
        )
        full_results = mog_dfm_guided_design(
            wt_seq=wt_seq,
            eval_variants=variants,
            aff_fn=aff_fn,
            peptide_length=args.peptide_length,
            n_designs=args.n_designs,
            eta=args.eta,
            design_mode="prophet",
            dfm_model=dfm_model,
            dfm_tokenizer=dfm_tokenizer,
            dfm_device=args.dfm_device,
            fixed_omega=[best_w, 1.0 - best_w],
            n_steps=200,
            seed=args.seed,
            verbose=True,
        )
        out = [
            {
                "peptide":       r.peptide,
                "wt_score":      r.wt_score,
                "robust_score":  r.robust_score,
                "mean_score":    r.mean_score,
                "min_score":     r.min_score,
                "omega":         r.omega,
            }
            for r in full_results
        ]
        Path(args.full_out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.full_out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Full run saved → {args.full_out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
