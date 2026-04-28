from __future__ import annotations

import csv
from pathlib import Path


def load_resistance_positions(csv_path: str | Path, pos_column: str = "position", one_indexed: bool = True) -> set[int]:
    """
    Load resistance positions from a CSV exported from Stanford/LANL curation.
    """
    out: set[int] = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if pos_column not in reader.fieldnames:
            raise ValueError(f"Column '{pos_column}' not found in {csv_path}")
        for row in reader:
            raw = (row.get(pos_column) or "").strip()
            if not raw:
                continue
            p = int(raw)
            out.add(p - 1 if one_indexed else p)
    return out

