#!/usr/bin/env python3
"""
Build Tables 2, 4, 5, 6, 7 from run_ablations.py / run_stage2_tevo.slurm outputs.

Usage (on PARCC after Stage 2 jobs finish):
    python scripts/make_paper_tables.py \
        --ablations-dir results/ablations \
        --out-dir       results/tables

Reads any JSON whose name matches t2_*, t4_*, t5_*, t6_*, t7_*, or t5_tevo_*.
Outputs one CSV + markdown per table, plus a combined summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_designs(path: Path) -> list[dict]:
    with path.open() as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "designs" in data:
        return data["designs"]
    return []


def aggregate(designs: list[dict], tau: float = 7.5) -> dict:
    if not designs:
        return {}
    wt   = [float(d["wt_score"])        for d in designs if "wt_score"        in d]
    mn   = [float(d["mean_score"])       for d in designs if "mean_score"       in d]
    mi   = [float(d["min_score"])        for d in designs if "min_score"        in d]
    ret  = [float(d["retention_score"])  for d in designs if "retention_score"  in d]
    rb   = [float(d["robust_score"])     for d in designs if "robust_score"     in d]

    # retention: fraction of designs where wt_score >= tau
    wt_ret = mean(1.0 if w >= tau else 0.0 for w in wt) if wt else float("nan")

    return {
        "n":            len(designs),
        "mean_wt":      mean(wt)  if wt  else float("nan"),
        "mean_mean":    mean(mn)  if mn  else float("nan"),
        "mean_min":     mean(mi)  if mi  else float("nan"),
        "mean_ret":     mean(ret) if ret else float("nan"),
        "mean_robust":  mean(rb)  if rb  else float("nan"),
        "wt_ret":       wt_ret,
    }


def fmt(x, decimals: int = 3) -> str:
    if x is None or x != x:  # nan check
        return "—"
    if isinstance(x, float):
        return f"{x:.{decimals}f}"
    return str(x)


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}")


def write_md(header: list[str], rows: list[list[str]], path: Path) -> None:
    col_w = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(header)]
    sep   = "|" + "|".join("-" * (w + 2) for w in col_w) + "|"
    hdr   = "|" + "|".join(f" {h:<{w}} " for h, w in zip(header, col_w)) + "|"
    lines = [hdr, sep]
    for r in rows:
        lines.append("|" + "|".join(f" {v:<{w}} " for v, w in zip(r, col_w)) + "|")
    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablations-dir", default="results/ablations")
    ap.add_argument("--out-dir",       default="results/tables")
    ap.add_argument("--tau",           type=float, default=7.5)
    args = ap.parse_args()

    abl_dir = Path(args.ablations_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsons = sorted(abl_dir.glob("*.json"))
    if not jsons:
        sys.exit(f"No JSONs found in {abl_dir}")

    # Group by prefix (t2_, t4_, t5_eta_, t5_tevo_, t6_, t7_)
    groups: dict[str, list[tuple[str, Path]]] = {}
    for p in jsons:
        name = p.stem          # e.g. t2_prophet-20260522_224257
        tag  = name.split("-")[0]   # e.g. t2_prophet
        groups.setdefault(tag, []).append((name, p))

    # Use the most recent file for each tag (in case of reruns)
    best: dict[str, Path] = {}
    for tag, files in groups.items():
        best[tag] = sorted(files, key=lambda x: x[0])[-1][1]

    def get(tag: str) -> tuple[str, list[dict]] | tuple[None, None]:
        p = best.get(tag)
        if p is None:
            return None, None
        d = load_designs(p)
        return p.stem, d

    tau = args.tau

    # ── Table 2: method comparison ──────────────────────────────────────────
    t2_methods = [
        ("PROPHET",           "t2_prophet"),
        ("PROPHET (rescue)",  "t2_prophet_rescue"),
        ("WT only",           "t2_wt_only"),
        ("Random variants",   "t2_random_variants"),
        ("Uniform leaves",    "t2_uniform_leaves"),
        ("ESM only",          "t2_esm_only_variants"),
    ]
    t2_rows_csv, t2_rows_md = [], []
    for label, tag in t2_methods:
        _, d = get(tag)
        if d is None:
            print(f"  [skip] {tag}: no JSON")
            continue
        a = aggregate(d, tau)
        t2_rows_csv.append({"method": label, **a})
        t2_rows_md.append([label, fmt(a["mean_wt"]), fmt(a["mean_mean"]),
                           fmt(a["mean_min"]), fmt(a["mean_ret"])])

    write_csv(t2_rows_csv, out_dir / "table2_method_comparison.csv")
    write_md(["Method", "WT↑", "Mean↑", "Min↑", "Ret.↑"], t2_rows_md,
             out_dir / "table2_method_comparison.md")

    # ── Table 4: Stage-1 ablations ──────────────────────────────────────────
    t4_methods = [
        ("Full PROPHET",     "t2_prophet"),
        ("Full PROPHET",     "t2_prophet_rescue"),
        ("−DCA",             "t4_no_dca"),
        ("−λ weighting",     "t4_no_lambda"),
        ("−ESM filter",      "t4_no_esm"),
        ("CVaR η=1.0",       "t5_eta_1.0"),
        ("CVaR η=0.5",       "t5_eta_0.5"),
        ("CVaR η=0.1",       "t5_eta_0.1"),
    ]
    seen_prophet = False
    t4_rows_csv, t4_rows_md = [], []
    for label, tag in t4_methods:
        if "prophet" in tag:
            if seen_prophet:
                continue
            seen_prophet = True
        _, d = get(tag)
        if d is None:
            print(f"  [skip] {tag}: no JSON")
            continue
        a = aggregate(d, tau)
        t4_rows_csv.append({"ablation": label, **a})
        t4_rows_md.append([label, fmt(a["mean_wt"]), fmt(a["mean_mean"]),
                           fmt(a["mean_min"]), fmt(a["mean_ret"])])

    write_csv(t4_rows_csv, out_dir / "table4_ablations.csv")
    write_md(["Ablation", "WT↑", "Mean↑", "Min↑", "Ret.↑"], t4_rows_md,
             out_dir / "table4_ablations.md")

    # ── Table 5: CVaR eta sensitivity ───────────────────────────────────────
    t5_eta_tags = [
        ("η=0.1", "t5_eta_0.1"),
        ("η=0.5", "t5_eta_0.5"),
        ("η=1.0", "t5_eta_1.0"),
    ]
    t5_rows_csv, t5_rows_md = [], []
    for label, tag in t5_eta_tags:
        _, d = get(tag)
        if d is None:
            print(f"  [skip] {tag}: no JSON")
            continue
        a = aggregate(d, tau)
        t5_rows_csv.append({"eta": label, **a})
        t5_rows_md.append([label, fmt(a["mean_wt"]), fmt(a["mean_mean"]),
                           fmt(a["mean_min"]), fmt(a["mean_ret"])])

    write_csv(t5_rows_csv, out_dir / "table5_eta_sensitivity.csv")
    write_md(["η", "WT↑", "Mean↑", "Min↑", "Ret.↑"], t5_rows_md,
             out_dir / "table5_eta_sensitivity.md")

    # ── Table 5 (T_evo): T_evo sensitivity ──────────────────────────────────
    tevo_tags = [
        ("T=0.5",  "t5_tevo_t05"),
        ("T=1.0",  "t5_tevo_t10"),
        ("T=2.0",  "t5_tevo_t20"),
        ("T=5.0",  "t5_tevo_t50"),
    ]
    tevo_rows_csv, tevo_rows_md = [], []
    for label, tag in tevo_tags:
        # match any file starting with this tag
        matches = [p for t, p in [(t, p) for t, p in
                   [(p.stem.split("-")[0], p) for p in jsons]
                   if t == tag or t.startswith(tag)]]
        if not matches:
            print(f"  [skip] {tag}: no JSON")
            continue
        d = load_designs(sorted(matches)[-1])
        a = aggregate(d, tau)
        tevo_rows_csv.append({"T_evo": label, **a})
        tevo_rows_md.append([label, fmt(a["mean_wt"]), fmt(a["mean_mean"]),
                             fmt(a["mean_min"]), fmt(a["mean_ret"])])

    write_csv(tevo_rows_csv, out_dir / "table5_tevo_sensitivity.csv")
    write_md(["T_evo", "WT↑", "Mean↑", "Min↑", "Ret.↑"], tevo_rows_md,
             out_dir / "table5_tevo_sensitivity.md")

    # ── Table 6: M sensitivity ───────────────────────────────────────────────
    t6_tags = [
        ("M=50",   "t6_M_50"),
        ("M=100",  "t6_M_100"),
        ("M=250",  "t6_M_250"),
        ("M=500",  "t6_M_500"),
        ("M=1000", "t6_M_1000"),
    ]
    t6_rows_csv, t6_rows_md = [], []
    for label, tag in t6_tags:
        _, d = get(tag)
        if d is None:
            print(f"  [skip] {tag}: no JSON")
            continue
        a = aggregate(d, tau)
        t6_rows_csv.append({"M": label, **a})
        t6_rows_md.append([label, fmt(a["mean_wt"]), fmt(a["mean_mean"]),
                           fmt(a["mean_min"]), fmt(a["mean_ret"])])

    write_csv(t6_rows_csv, out_dir / "table6_M_sensitivity.csv")
    write_md(["M", "WT↑", "Mean↑", "Min↑", "Ret.↑"], t6_rows_md,
             out_dir / "table6_M_sensitivity.md")

    # ── Table 7: J sensitivity ───────────────────────────────────────────────
    t7_tags = [
        ("J=25",  "t7_J_25"),
        ("J=50",  "t7_J_50"),
        ("J=100", "t7_J_100"),
        ("J=200", "t7_J_200"),
    ]
    t7_rows_csv, t7_rows_md = [], []
    for label, tag in t7_tags:
        _, d = get(tag)
        if d is None:
            print(f"  [skip] {tag}: no JSON")
            continue
        a = aggregate(d, tau)
        t7_rows_csv.append({"J": label, **a})
        t7_rows_md.append([label, fmt(a["mean_wt"]), fmt(a["mean_mean"]),
                           fmt(a["mean_min"]), fmt(a["mean_ret"])])

    write_csv(t7_rows_csv, out_dir / "table7_J_sensitivity.csv")
    write_md(["J", "WT↑", "Mean↑", "Min↑", "Ret.↑"], t7_rows_md,
             out_dir / "table7_J_sensitivity.md")

    print(f"\nDone. Tables in {out_dir}/")


if __name__ == "__main__":
    main()
