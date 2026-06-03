#!/usr/bin/env bash
#
# Sweep LiteFNO `model.rank` on gray_scott_reaction_diffusion to produce a
# params vs vRMSE Pareto plot. Each rank gets its own metrics file and
# checkpoint subdir. Trains 50 epochs per rank (shorter than the 200-epoch
# baseline; enough for the tradeoff curve to take shape).
#
# Usage:
#   scripts/run_rank_sweep.sh [--gpu N] [--epochs E] [--ranks "8 16 32 64 128"]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

EPOCHS=50
RANKS="8 16 32 64 128"
DATASET="gray_scott_reaction_diffusion"
CONFIG="configs/experiments/litefno_${DATASET}.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) export CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --ranks) RANKS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

LOG_DIR="outputs/logs/${DATASET}_rank_sweep"
CKPT_ROOT="outputs/checkpoints/${DATASET}_rank_sweep"
mkdir -p "${LOG_DIR}" "${CKPT_ROOT}"

echo ">>> rank sweep on ${DATASET}, epochs=${EPOCHS}, ranks=${RANKS}"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && echo ">>> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

for RANK in ${RANKS}; do
  METRICS="${LOG_DIR}/r${RANK}.jsonl"
  CKPT="${CKPT_ROOT}/r${RANK}"
  if [[ -s "${METRICS}" ]] && grep -q '"test_vrmse"' "${METRICS}"; then
    echo ">>> rank=${RANK}: ${METRICS} already has test_vrmse, skipping"
    continue
  fi
  : > "${METRICS}"  # truncate so partial prior runs don't pollute the plot
  echo ">>> rank=${RANK}: training -> ${METRICS}"
  litefno train --config "${CONFIG}" \
    --set "model.rank=${RANK}" \
    --set "training.epochs=${EPOCHS}" \
    --set "logging.metrics_path=${METRICS}" \
    --set "training.checkpoint_dir=${CKPT}"
done

echo ">>> sweep complete. Plot with:"
echo "    python scripts/plot_rank_sweep.py"
