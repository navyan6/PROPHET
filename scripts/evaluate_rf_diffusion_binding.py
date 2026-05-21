#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prophet.stage2 import AffinityScorer


AA = set("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_HIV_WT = (
    "PQVTLWQKPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEIC"
    "GHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
)


def clean_sequence(seq: str) -> str:
    seq = seq.strip().upper().replace(" ", "").replace("-", "")
    return "".join(ch for ch in seq if ch in AA)


def parse_metric(token: str) -> tuple[str, Any] | None:
    if ":" not in token:
        return None
    key, value = token.split(":", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return key, int(value)
        return key, float(value)
    except ValueError:
        return key, value


def parse_rfdiffusion_results(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            pair = parts[-1]
            if "/" not in pair:
                raise ValueError(
                    f"{path}:{line_no}: expected final target/peptide sequence pair"
                )
            target, peptide = pair.rsplit("/", 1)
            row: dict[str, Any] = {
                "source_file": str(path),
                "source_line": line_no,
                "target": clean_sequence(target),
                "peptide": clean_sequence(peptide),
            }
            for token in parts[:-1]:
                metric = parse_metric(token)
                if metric is not None:
                    row[metric[0]] = metric[1]
            if row["peptide"]:
                rows.append(row)
    return rows


def parse_peptide_list(path: Path, target_seq: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            peptide = clean_sequence(raw)
            if not peptide:
                continue
            rows.append(
                {
                    "source_file": str(path),
                    "source_line": line_no,
                    "target": clean_sequence(target_seq or DEFAULT_HIV_WT),
                    "peptide": peptide,
                }
            )
    return rows


def parse_scored_csv(path: Path, target_seq: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for line_no, row in enumerate(reader, start=2):
            peptide = clean_sequence(str(row.get("peptide", "")))
            if not peptide:
                continue
            target = clean_sequence(str(row.get("target", "") or target_seq or DEFAULT_HIV_WT))
            out: dict[str, Any] = {
                "source_file": row.get("source_file") or str(path),
                "source_line": row.get("source_line") or line_no,
                "target": target,
                "peptide": peptide,
            }
            for key, value in row.items():
                if key in out or value in (None, ""):
                    continue
                metric = parse_metric(f"{key}:{value}")
                if metric is not None:
                    out[metric[0]] = metric[1]
            rows.append(out)
    return rows


def parse_scored_json(path: Path, target_seq: str | None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("per_peptide", data.get("rows", []))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list or object with per_peptide/rows")

    rows: list[dict[str, Any]] = []
    for i, row in enumerate(data, start=1):
        if not isinstance(row, dict):
            continue
        peptide = clean_sequence(str(row.get("peptide") or row.get("sequence") or ""))
        if not peptide:
            continue
        target = clean_sequence(str(row.get("target", "") or target_seq or DEFAULT_HIV_WT))
        out = dict(row)
        out.update(
            {
                "source_file": row.get("source_file") or str(path),
                "source_line": row.get("source_line") or i,
                "target": target,
                "peptide": peptide,
            }
        )
        rows.append(out)
    return rows


def load_fasta(path: Path) -> list[dict[str, str]]:
    variants: list[dict[str, str]] = []
    name: str | None = None
    seq_parts: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    variants.append({"name": name, "sequence": clean_sequence("".join(seq_parts))})
                name = line[1:].split()[0] or f"variant_{len(variants) + 1}"
                seq_parts = []
            else:
                seq_parts.append(line)
    if name is not None:
        variants.append({"name": name, "sequence": clean_sequence("".join(seq_parts))})
    return [v for v in variants if v["sequence"]]


def load_rows(path: Path, input_format: str, target_seq: str | None) -> list[dict[str, Any]]:
    if input_format == "rfdiffusion":
        return parse_rfdiffusion_results(path)
    if input_format == "peptides":
        return parse_peptide_list(path, target_seq)
    if input_format == "csv":
        return parse_scored_csv(path, target_seq)
    if input_format == "json":
        return parse_scored_json(path, target_seq)
    raise ValueError(f"Unsupported input format: {input_format}")


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (row["target"], row["peptide"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    keys: list[str] = []
    preferred = [
        "rank",
        "binding_score",
        "wt_score",
        "mean_score",
        "cvar10_score",
        "min_score",
        "retention_at_tau",
        "tau_bind",
        "n_variants",
        "peptide",
        "target",
        "design",
        "n",
        "mpnn",
        "plddt",
        "i_ptm",
        "i_pae",
        "rmsd",
        "source_file",
        "source_line",
    ]
    for key in preferred:
        if any(key in row for row in rows):
            keys.append(key)
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            csv_row = {
                key: json.dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(csv_row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score RFdiffusion peptide outputs with PROPHET's affinity scorer."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input file. Repeat for multiple files.",
    )
    parser.add_argument(
        "--input-format",
        choices=["rfdiffusion", "peptides", "csv", "json"],
        default="rfdiffusion",
        help=(
            "Use rfdiffusion for lines ending in target/peptide, peptides for one "
            "peptide per line, or csv/json for previously scored outputs."
        ),
    )
    parser.add_argument(
        "--target-seq",
        default=None,
        help=(
            "Target sequence for peptide-list inputs. Defaults to the HIV WT sequence "
            "used by the stage-2 metrics scripts. RFdiffusion result files use their "
            "per-line target unless --override-target is set."
        ),
    )
    parser.add_argument(
        "--override-target",
        action="store_true",
        help="Use --target-seq for every row, including RFdiffusion target/peptide inputs.",
    )
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N loaded rows; useful for quick PeptiVerse tests.",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--escape-fasta",
        default=None,
        help="FASTA of escape variants. When set, compute mean/min/retention across variants.",
    )
    parser.add_argument(
        "--tau-bind",
        type=float,
        default=8.0,
        help="Retention threshold for raw PeptiVerse scores.",
    )
    parser.add_argument(
        "--include-per-variant",
        action="store_true",
        help="Store per-variant scores in the JSON/CSV output.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["binding_score", "wt_score", "mean_score", "cvar10_score", "min_score", "retention_at_tau"],
        default="mean_score",
        help="Column to rank by after scoring.",
    )
    parser.add_argument(
        "--peptiverse-normalization",
        choices=["raw", "minmax"],
        default="raw",
    )
    parser.add_argument("--peptiverse-min", type=float, default=7.0)
    parser.add_argument("--peptiverse-max", type=float, default=9.0)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for input_file in args.input:
        rows.extend(load_rows(Path(input_file), args.input_format, args.target_seq))

    if args.override_target:
        if not args.target_seq:
            raise ValueError("--override-target requires --target-seq")
        target = clean_sequence(args.target_seq)
        for row in rows:
            row["target"] = target

    if args.dedupe:
        rows = dedupe_rows(rows)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No peptide rows loaded.")

    scorer = AffinityScorer(
        device=args.device,
        peptiverse_normalization=args.peptiverse_normalization,
        peptiverse_min=args.peptiverse_min,
        peptiverse_max=args.peptiverse_max,
    )

    variants = load_fasta(Path(args.escape_fasta)) if args.escape_fasta else []
    if args.escape_fasta and not variants:
        raise ValueError(f"No variants loaded from {args.escape_fasta}")
    variant_seqs = [v["sequence"] for v in variants]

    for i, row in enumerate(rows, start=1):
        # Score against WT using batched path
        wt_scores = scorer.score_variants_batched(row["peptide"], [row["target"]])
        wt_score = float(wt_scores[0])
        row["binding_score"] = wt_score
        row["wt_score"] = wt_score

        if variant_seqs:
            import numpy as np
            var_arr = scorer.score_variants_batched(row["peptide"], variant_seqs)
            eta = 0.1
            k = max(1, int(len(var_arr) * eta))
            cvar = float(np.sort(var_arr)[:k].mean())
            row["mean_score"] = float(var_arr.mean())
            row["min_score"] = float(var_arr.min())
            row["cvar10_score"] = cvar
            row["retention_at_tau"] = float((var_arr >= args.tau_bind).mean())
            row["tau_bind"] = args.tau_bind
            row["n_variants"] = len(var_arr)
            if args.include_per_variant:
                row["per_variant"] = [
                    {"name": variant["name"], "score": float(score)}
                    for variant, score in zip(variants, var_arr)
                ]
            print(
                f"[{i:04d}/{len(rows):04d}] {row['peptide']} "
                f"wt={wt_score:.4f} mean={row['mean_score']:.4f} "
                f"cvar10={cvar:.4f} ret={row['retention_at_tau']:.3f}",
                flush=True,
            )
        else:
            print(
                f"[{i:04d}/{len(rows):04d}] {row['peptide']} "
                f"score={row['binding_score']:.4f}",
                flush=True,
            )

    rows.sort(key=lambda row: row.get(args.sort_by, float("-inf")), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    out_csv = Path(args.out_csv)
    write_csv(rows, out_csv)
    print(f"Saved CSV: {out_csv}")

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
