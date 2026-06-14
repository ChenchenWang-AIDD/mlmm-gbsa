#!/usr/bin/env bash
# MM-GBSA / MM-PBSA wrapper script
# Usage: bash gbsa.sh [extra args for gbsa.py]
#   e.g. bash gbsa.sh --method gb        # run GBSA
#        bash gbsa.sh --method pb      # run PBSA
set -euo pipefail

eval "$(mamba shell hook --shell=bash)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/gbsa.py" \
    --workdir "${SCRIPT_DIR}" \
    --datadir "${SCRIPT_DIR}/md_output" \
    --ligand-resname UNK \
    --method gb \
    --igb 5 \
    --saltcon 0.15 \
    --istrng 0.15 \
    --startframe 1 \
    --endframe 100 \
    --interval 1 \
    --forcefields amber/ff14SB.xml amber/tip3p_standard.xml \
    --ligand-ff openff_unconstrained-2.0.0.offxml \
    "$@"
