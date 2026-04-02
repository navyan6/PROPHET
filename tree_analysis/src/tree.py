"""
Turn UniProt-style JSON into a FASTA of HIV protease sequences.

  1. Load the wild-type sequence.
  2. Find where mature protease starts inside that string.
  3. For each VARIANT feature, if it is a single amino-acid change inside the
     protease window, build a new 99-letter sequence (WT protease with that
     one change) and write it as one FASTA record.

That matches how UniProt lists natural variants: one substitution at a time on
the reference sequence.
"""

from __future__ import annotations

import json
from pathlib import Path

# Length of mature HIV protease (amino acids).
PROTEASE_LEN = 99

PROTEASE_START_MARKERS = ("PQVTLWQR", "PQITLWQR")


def protease_start_index(full_polyprotein: str) -> int:
    for marker in PROTEASE_START_MARKERS:
        position = full_polyprotein.find(marker)
        if position != -1:
            return position
    return -1


def read_first_sequence_from_fasta(fasta_path: Path) -> str:
    """
    Read a simple one-sequence FASTA file and return the amino-acid string.

    Stops after the first record (first '>' to next '>' or end of file).
    """
    current_name: str | None = None
    sequence_chunks: list[str] = []

    with open(fasta_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line == "":
                continue
            if line.startswith(">"):
                if current_name is not None:
                    return "".join(sequence_chunks)
                current_name = line[1:].split()[0]
                sequence_chunks = []
            else:
                sequence_chunks.append(line)

    if current_name is None:
        raise ValueError(f"No FASTA records found in: {fasta_path}")
    return "".join(sequence_chunks)


def read_wildtype_polyprotein(
    variants_json: Path,
    wildtype_fasta: Path | None = None,
) -> str:
    """
      1. If wildtype_fasta is given and the file exists, use its first sequence.
      2. Otherwise read key "sequence" from variants_json (UniProt JSON).
    """
    if wildtype_fasta is not None and wildtype_fasta.is_file():
        return read_first_sequence_from_fasta(wildtype_fasta)

    with open(variants_json, encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        sequence = data.get("sequence", "")
    else:
        sequence = ""

    if sequence == "":
        raise ValueError(
            f"No 'sequence' in {variants_json}. Add it, or pass a wildtype FASTA file."
        )
    return sequence


def generate_variants_from_json(
    json_file: str | Path,
    output_file: str | Path,
    *,
    wildtype_fasta: Path | None = None,
    verbose: bool = False,
) -> None:
    """
    Read UniProt JSON, write one FASTA sequence per usable variant
    """
    json_path = Path(json_file)
    output_path = Path(output_file)

    with open(json_path, encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        features = data.get("features", [])
    elif isinstance(data, list):
        features = data
    else:
        features = []

    full_polyprotein = read_wildtype_polyprotein(json_path, wildtype_fasta)

    start = protease_start_index(full_polyprotein)
    if start < 0:
        raise ValueError(
            "Could not find protease start motif in the reference sequence. "
            f"Expected one of: {PROTEASE_START_MARKERS}"
        )
    if len(full_polyprotein) < start + PROTEASE_LEN:
        raise ValueError("Reference sequence is too short to contain full protease.")

    if verbose:
        print("Total features in JSON:", len(features))

    output_rows: list[tuple[str, str]] = []

    for feature_index, feature in enumerate(features):
        try:
            # Skip non-variant features when type is present.
            feature_type = feature.get("type")
            if feature_type is not None and feature_type != "VARIANT":
                continue

            position_zero_based = int(feature["begin"]) - 1
            wild_type_aa = feature["wildType"]
            alt_aa = feature["alternativeSequence"]

            if len(wild_type_aa) != 1 or len(alt_aa) != 1:
                continue

            if position_zero_based < start or position_zero_based >= start + PROTEASE_LEN:
                continue

            local_index = position_zero_based - start
            protease_letters = list(
                full_polyprotein[start : start + PROTEASE_LEN]
            )

            if protease_letters[local_index] != wild_type_aa:
                continue

            protease_letters[local_index] = alt_aa
            new_sequence = "".join(protease_letters)
            record_name = f"var_{feature_index}"
            output_rows.append((record_name, new_sequence))

        except (KeyError, TypeError, ValueError) as err:
            if verbose:
                print(f"Skipping feature {feature_index}: {err}")

    if verbose:
        print("Protease variants written:", len(output_rows))

    with open(output_path, "w", encoding="utf-8") as handle:
        for name, seq in output_rows:
            handle.write(f">{name}\n{seq}\n")

    if verbose:
        print("Wrote:", output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="UniProt JSON VARIANT features -> protease FASTA"
    )
    parser.add_argument("--json", type=Path, default=Path("hiv-variants.json"))
    parser.add_argument("--out", type=Path, default=Path("hiv_sequences.fasta"))
    parser.add_argument(
        "--wt-fasta",
        type=Path,
        default=None,
        help="Optional polyprotein WT (first record); default: JSON 'sequence'",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    cli_args = parser.parse_args()

    generate_variants_from_json(
        cli_args.json,
        cli_args.out,
        wildtype_fasta=cli_args.wt_fasta,
        verbose=cli_args.verbose,
    )
