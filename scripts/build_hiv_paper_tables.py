#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median


DEFAULT_DESIGNS = {
    "prophet": ("hiv_train_prophet_stage2_peptiverse.json", "hiv_train_stage2_peptiverse_gpu1.json"),
    "wt_only": "hiv_train_wt_only_stage2_peptiverse.json",
    "random_variants": "hiv_train_random_variants_stage2_peptiverse.json",
    "uniform_leaves": "hiv_train_uniform_leaves_stage2_peptiverse.json",
    "peptune": "hiv_train_peptune_stage2_peptiverse.json",
    "peptune_unconditional": "hiv_train_peptune_unconditional_stage2_peptiverse.json",
    "rfdiffusion": "hiv_train_rfdiffusion_stage2_peptiverse.json",
}


def _load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_first_json(stage2_dir: Path, names: str | tuple[str, ...]) -> tuple[Path, object | None]:
    if isinstance(names, str):
        names = (names,)
    for name in names:
        path = stage2_dir / name
        data = _load_json(path)
        if data:
            return path, data
    return stage2_dir / names[0], None


def _fmt(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def _design_summary(rows: list[dict]) -> dict[str, float | int | str]:
    wt = [float(r["wt_score"]) for r in rows if "wt_score" in r]
    rb = [float(r["robust_score"]) for r in rows if "robust_score" in r]
    mn = [float(r["mean_score"]) for r in rows if "mean_score" in r]
    mi = [float(r["min_score"]) for r in rows if "min_score" in r]
    top = max(rows, key=lambda r: (float(r.get("robust_score", 0.0)), float(r.get("wt_score", 0.0))))
    return {
        "n_designs": len(rows),
        "best_peptide": top.get("peptide", ""),
        "best_wt_score": float(top.get("wt_score", 0.0)),
        "best_robust_score": float(top.get("robust_score", 0.0)),
        "mean_wt_score": mean(wt) if wt else float("nan"),
        "median_wt_score": median(wt) if wt else float("nan"),
        "mean_robust_score": mean(rb) if rb else float("nan"),
        "median_robust_score": median(rb) if rb else float("nan"),
        "mean_variant_score": mean(mn) if mn else float("nan"),
        "mean_min_variant_score": mean(mi) if mi else float("nan"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Build table-ready CSVs from HIV Stage 2 outputs")
    p.add_argument("--stage2-dir", default="results/hiv_stage2")
    p.add_argument("--out-dir", default="results/hiv_stage2/tables")
    args = p.parse_args()

    stage2_dir = Path(args.stage2_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict] = []
    eta_rows: list[dict] = []

    for mode, design_names in DEFAULT_DESIGNS.items():
        designs_path, designs = _load_first_json(stage2_dir, design_names)
        if not designs:
            continue

        row = {"method": mode, "designs_json": str(designs_path)}
        row.update(_design_summary(designs))

        pareto = _load_json(stage2_dir / f"hiv_train_{mode}_pareto.json")
        if pareto:
            row["n_pareto"] = pareto.get("n_pareto")
            row["hypervolume"] = pareto.get("hypervolume")

        robust = _load_json(stage2_dir / f"hiv_train_{mode}_robust_design.json")
        if robust:
            agg = robust.get("aggregate", {})
            inputs = robust.get("inputs", {})
            row["heldout_mean_wt_score"] = agg.get("mean_wt_score")
            row["heldout_mean_escape"] = agg.get("mean_mean_escape")
            row["heldout_mean_min_escape"] = agg.get("mean_min_escape")
            row["heldout_retention_tau"] = inputs.get("tau_bind")
            row["heldout_mean_retention_at_tau"] = agg.get("mean_retention")
        elif any("retention_at_tau" in r for r in designs):
            vals = [float(r["retention_at_tau"]) for r in designs if "retention_at_tau" in r]
            if vals:
                taus = {float(r["tau_bind"]) for r in designs if "tau_bind" in r}
                row["heldout_retention_tau"] = taus.pop() if len(taus) == 1 else None
                row["heldout_mean_retention_at_tau"] = mean(vals)

        eta = _load_json(stage2_dir / f"hiv_train_{mode}_eta_sensitivity.json")
        if eta:
            for eta_value, metrics in eta.get("eta_metrics", {}).items():
                eta_rows.append({"method": mode, "eta": eta_value, **metrics})

        rows_out.append(row)

    summary_csv = out_dir / "hiv_stage2_summary.csv"
    if rows_out:
        fieldnames = sorted({key for row in rows_out for key in row})
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_out)

    eta_csv = out_dir / "hiv_eta_sensitivity.csv"
    if eta_rows:
        fieldnames = sorted({key for row in eta_rows for key in row})
        with eta_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(eta_rows)

    markdown_path = out_dir / "hiv_stage2_summary.md"
    with markdown_path.open("w", encoding="utf-8") as f:
        f.write("| Method | N | Best peptide | Best robust | Mean robust | Hypervolume | Held-out retention |\n")
        f.write("|---|---:|---|---:|---:|---:|---:|\n")
        for row in rows_out:
            f.write(
                "| {method} | {n} | `{pep}` | {best_rb} | {mean_rb} | {hv} | {ret} |\n".format(
                    method=row.get("method", ""),
                    n=_fmt(row.get("n_designs")),
                    pep=row.get("best_peptide", ""),
                    best_rb=_fmt(row.get("best_robust_score")),
                    mean_rb=_fmt(row.get("mean_robust_score")),
                    hv=_fmt(row.get("hypervolume")),
                    ret=_fmt(row.get("heldout_mean_retention_at_tau")),
                )
            )

    print(f"Wrote {summary_csv}")
    print(f"Wrote {eta_csv}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
