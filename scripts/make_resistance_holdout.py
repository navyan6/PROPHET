#!/usr/bin/env python3
"""
Build a drug-resistant HIV protease holdout FASTA.

Generates:
  - All single major PI resistance mutants (~30 sequences)
  - Common double/triple resistance combinations (~20 sequences)

Writes to data/hiv_resistance_holdout.fasta

Usage:
    python scripts/make_resistance_holdout.py
"""
from __future__ import annotations
from pathlib import Path

WT = "PQVTLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"

# Major PI resistance mutations (position 1-based, WT aa, mutant aa list)
MAJOR = [
    (24,  "L", ["I"]),
    (30,  "D", ["N"]),
    (32,  "V", ["I"]),
    (46,  "M", ["I", "L"]),
    (47,  "I", ["A", "V"]),
    (48,  "G", ["M", "V"]),
    (50,  "I", ["L", "V"]),
    (54,  "I", ["A", "L", "M", "S", "T", "V"]),
    (76,  "L", ["A"]),
    (82,  "V", ["A", "F", "L", "S", "T"]),
    (84,  "I", ["A", "C", "V"]),
    (88,  "N", ["D", "S"]),
    (90,  "L", ["M"]),
]

# Clinically observed multi-drug resistant combinations
# Source: Stanford HIVDB common genotypes
COMBOS = [
    # nelfinavir resistance
    [("D30N", 30, "D", "N"), ("L90M", 90, "L", "M")],
    # saquinavir/lopinavir resistance
    [("G48V", 48, "G", "V"), ("L90M", 90, "L", "M")],
    # lopinavir/ritonavir resistance
    [("M46I", 46, "M", "I"), ("I54V", 54, "I", "V"), ("V82A", 82, "V", "A")],
    [("M46I", 46, "M", "I"), ("I54V", 54, "I", "V"), ("L90M", 90, "L", "M")],
    # darunavir resistance
    [("I50V", 50, "I", "V"), ("I84V", 84, "I", "V")],
    [("V32I", 32, "V", "I"), ("I47V", 47, "I", "V"), ("I54M", 54, "I", "M")],
    # tipranavir resistance
    [("L24I", 24, "L", "I"), ("M46L", 46, "M", "L"), ("I54V", 54, "I", "V"), ("V82T", 82, "V", "T")],
    # broad MDR
    [("M46I", 46, "M", "I"), ("I54V", 54, "I", "V"), ("V82A", 82, "V", "A"), ("I84V", 84, "I", "V")],
    [("L24I", 24, "L", "I"), ("M46I", 46, "M", "I"), ("I54V", 54, "I", "V"),
     ("V82A", 82, "V", "A"), ("L90M", 90, "L", "M")],
    # atazanavir resistance
    [("I50L", 50, "I", "L"), ("I84V", 84, "I", "V"), ("L90M", 90, "L", "M")],
    [("G48V", 48, "G", "V"), ("I84V", 84, "I", "V"), ("L90M", 90, "L", "M")],
]


def apply_mutations(seq: str, muts: list[tuple]) -> str:
    s = list(seq)
    for *_, pos, wt_aa, mut_aa in muts:
        assert s[pos - 1] == wt_aa, f"WT mismatch at {pos}: expected {wt_aa}, got {s[pos-1]}"
        s[pos - 1] = mut_aa
    return "".join(s)


def main():
    records: list[tuple[str, str]] = []

    # Single mutants
    for pos, wt_aa, mut_aas in MAJOR:
        assert WT[pos - 1] == wt_aa, f"WT check failed at {pos}"
        for mut in mut_aas:
            label = f"{wt_aa}{pos}{mut}"
            seq = list(WT)
            seq[pos - 1] = mut
            records.append((label, "".join(seq)))

    # Multi-mutant combinations
    for combo in COMBOS:
        label = "_".join(f"{wt}{pos}{mut}" for _, pos, wt, mut in combo)
        seq = WT
        for _, pos, wt_aa, mut_aa in combo:
            s = list(seq)
            s[pos - 1] = mut_aa
            seq = "".join(s)
        records.append((label, seq))

    # Deduplicate
    seen = set()
    unique = []
    for label, seq in records:
        if seq not in seen:
            seen.add(seq)
            unique.append((label, seq))

    out = Path("data/hiv_resistance_holdout.fasta")
    with out.open("w") as f:
        for label, seq in unique:
            f.write(f">{label}\n{seq}\n")

    print(f"Written {len(unique)} resistance variants to {out}")
    print("Single mutants: major PI resistance positions")
    print("Combos: clinically observed MDR genotypes")
    print()
    print("To score on PARCC:")
    print("  python scripts/score_holdout_robustness.py \\")
    print("    --ablations-dir results/ablations \\")
    print("    --holdout-fasta data/hiv_resistance_holdout.fasta \\")
    print("    --out-dir results/resistance_holdout_scores \\")
    print("    --device cuda:0 --tau 8.0")


if __name__ == "__main__":
    main()
