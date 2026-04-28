#!/usr/bin/env python3
"""
Create visualization panels for PROPHET Stage-1 outputs.

Generates plots for each prefix:
  - lambda profile + top sites
  - mean Qi substitution heatmap
  - Qi heatmap at top-variable site
  - J coupling strength heatmap (Frobenius norm)
  - top coupled residue-pair bar chart
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

AA = "ACDEFGHIKLMNPQRSTVWY"


def plot_lambda(lambda_arr: np.ndarray, out_path: Path, prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(lambda_arr))
    ax.plot(x, lambda_arr, linewidth=1.5)
    ax.set_title(f"{prefix}: per-site mutation rates (lambda)")
    ax.set_xlabel("Alignment position")
    ax.set_ylabel("lambda")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_lambda_top(lambda_arr: np.ndarray, out_path: Path, prefix: str, k: int = 20) -> None:
    top = np.argsort(lambda_arr)[-k:][::-1]
    vals = lambda_arr[top]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(k), vals)
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels([str(i) for i in top], rotation=60, ha="right", fontsize=8)
    ax.set_title(f"{prefix}: top-{k} variable sites by lambda")
    ax.set_xlabel("Site index")
    ax.set_ylabel("lambda")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_qi_mean(qi: np.ndarray, out_path: Path, prefix: str) -> None:
    mean_q = qi.mean(axis=0)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mean_q, aspect="auto", cmap="viridis")
    ax.set_title(f"{prefix}: mean substitution matrix Qi (averaged across sites)")
    ax.set_xticks(np.arange(20))
    ax.set_yticks(np.arange(20))
    ax.set_xticklabels(list(AA), fontsize=8)
    ax.set_yticklabels(list(AA), fontsize=8)
    ax.set_xlabel("to amino acid")
    ax.set_ylabel("from amino acid")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_qi_at_site(qi: np.ndarray, site: int, out_path: Path, prefix: str) -> None:
    m = qi[site].copy()
    # Hide low-information "uniform prior-like" rows (all ~1/20),
    # which usually come from smoothing where no transitions were observed.
    uniform = np.full(20, 1.0 / 20.0, dtype=float)
    row_deviation = np.max(np.abs(m - uniform[None, :]), axis=1)
    low_info_rows = row_deviation < 1e-4
    m[low_info_rows, :] = np.nan

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.cm.magma.copy()
    cmap.set_bad(color="#d9d9d9")
    vmax = np.nanpercentile(m, 99)
    vmin = np.nanpercentile(m, 5)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, 1.0
    im = ax.imshow(m, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f"{prefix}: Qi at top-variable site {site} (low-info rows masked)")
    ax.set_xticks(np.arange(20))
    ax.set_yticks(np.arange(20))
    ax.set_xticklabels(list(AA), fontsize=8)
    ax.set_yticklabels(list(AA), fontsize=8)
    ax.set_xlabel("to amino acid")
    ax.set_ylabel("from amino acid")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def compute_j_frobenius(J: np.ndarray) -> np.ndarray:
    jf = np.sqrt((J ** 2).sum(axis=(2, 3)))
    np.fill_diagonal(jf, 0.0)
    return jf


def plot_j_heatmap(jf: np.ndarray, out_path: Path, prefix: str) -> None:
    # Robust clipping for sparse heavy-tailed coupling matrices.
    vmax = float(np.percentile(jf, 99.9))
    vmax = max(vmax, 1e-6)
    clipped = np.clip(jf, 0.0, vmax)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(clipped, aspect="auto", cmap="plasma", norm=LogNorm(vmin=1e-6, vmax=vmax))
    ax.set_title(f"{prefix}: coupling strength ||J[i,j]||_F (log, 99.9% clipped)")
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_j_top_pairs(jf: np.ndarray, out_path: Path, prefix: str, k: int = 20) -> None:
    upper = np.triu_indices_from(jf, k=1)
    vals = jf[upper]
    idx = np.argsort(vals)[-k:][::-1]
    top_vals = vals[idx]
    top_pairs = [(int(upper[0][i]), int(upper[1][i])) for i in idx]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(np.arange(k), top_vals)
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels([f"({i},{j})" for i, j in top_pairs], rotation=60, ha="right", fontsize=8)
    ax.set_title(f"{prefix}: top-{k} coupled site pairs")
    ax.set_ylabel("||J[i,j]||_F")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def visualize_prefix(prophet_dir: Path, out_dir: Path, prefix: str) -> None:
    lambda_path = prophet_dir / f"{prefix}_lambda.npy"
    qi_path = prophet_dir / f"{prefix}_Qi.npz"
    h_path = prophet_dir / f"{prefix}_h.npy"
    j_path = prophet_dir / f"{prefix}_J.npz"

    if not (lambda_path.exists() and qi_path.exists() and h_path.exists() and j_path.exists()):
        print(f"[skip] {prefix}: missing one or more required files")
        return

    out_prefix_dir = out_dir / prefix
    out_prefix_dir.mkdir(parents=True, exist_ok=True)

    lam = np.load(lambda_path)
    qi = np.load(qi_path)["Qi"]
    top_site = int(np.argmax(lam))

    plot_lambda(lam, out_prefix_dir / "lambda_profile.png", prefix)
    plot_lambda_top(lam, out_prefix_dir / "lambda_top_sites.png", prefix)
    plot_qi_mean(qi, out_prefix_dir / "Qi_mean_heatmap.png", prefix)
    plot_qi_at_site(qi, top_site, out_prefix_dir / "Qi_top_site_heatmap.png", prefix)

    J = np.load(j_path)["J"]
    jf = compute_j_frobenius(J)
    plot_j_heatmap(jf, out_prefix_dir / "J_coupling_heatmap.png", prefix)
    plot_j_top_pairs(jf, out_prefix_dir / "J_top_pairs.png", prefix)
    print(f"[ok] {prefix}: plots written to {out_prefix_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize PROPHET outputs")
    p.add_argument("--prophet-dir", type=Path, default=Path("data/prophet"))
    p.add_argument("--out-dir", type=Path, default=Path("data/prophet/plots"))
    p.add_argument(
        "--prefixes",
        nargs="*",
        default=["hiv_algo1", "flu_algo1", "dengue_algo1", "covid_algo1"],
        help="Output prefixes to visualize",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    prophet_dir = args.prophet_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for prefix in args.prefixes:
        try:
            visualize_prefix(prophet_dir, out_dir, prefix)
        except Exception as e:
            print(f"[error] {prefix}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

