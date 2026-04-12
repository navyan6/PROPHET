from Bio import SeqIO

INPUT_FILE  = "covdata.fasta"
OUTPUT_FILE = "spike.fasta"

# SARS-CoV-2 Spike gene coordinates (1-based, inclusive)
SPIKE_START = 21563
SPIKE_END   = 25384
MIN_LENGTH  = 25000

# Convert to 0-based Python slice indices
spike_slice = slice(SPIKE_START - 1, SPIKE_END)

total = 0
skipped = 0
extracted_records = []

for record in SeqIO.parse(INPUT_FILE, "fasta"):
    total += 1

    # Skip sequences that are too short to contain the full Spike region
    if len(record.seq) < MIN_LENGTH:
        skipped += 1
        continue

    # Extract the Spike region and preserve the original header
    spike_record = record[spike_slice]
    spike_record.id = record.id
    spike_record.description = record.description
    extracted_records.append(spike_record)

# Write all extracted Spike sequences to the output file
SeqIO.write(extracted_records, OUTPUT_FILE, "fasta")

print(f"Total sequences processed : {total}")
print(f"Successfully extracted    : {len(extracted_records)}")
print(f"Skipped (too short)       : {skipped}")
print(f"Output written to         : {OUTPUT_FILE}")
