#!/bin/bash
# Thin wrapper around run_prophet.slurm.
#
# Usage: ./submit.sh <target> [extra sbatch flags]
#
# Examples:
#   ./submit.sh hiv_protease
#   ./submit.sh flu_ha --time=18:00:00
#   ./submit.sh sars_mpro --partition=gpu
#
# Registered targets: hiv_protease, hcv_ns3, flu_ha, flu_na,
#                     sars_mpro, zika_ns3, wnv_ns3

set -e

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <target> [extra sbatch flags]"
    echo "Targets: hiv_protease hcv_ns3 flu_ha flu_na sars_mpro zika_ns3 wnv_ns3"
    exit 1
fi

TARGET="$1"
shift

sbatch --job-name="prophet_${TARGET}" --export="TARGET=${TARGET}" "$@" run_prophet.slurm
