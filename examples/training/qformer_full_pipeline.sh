#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Q-Former full pipeline orchestrator.
#
# Runs:
#   STAGE 1: ``qformer_pretrain_custom.sh``       — pretrain on YOUR dataset
#   STAGE 2: ``qformer_finetune_benchmarks.sh``   — finetune + eval on
#                                                    LIBERO / MetaWorld / RoboTwin
#
# Both stages use the same ``OUTPUT_ROOT`` so Stage 2 can find Stage 1's
# checkpoints automatically. All env vars accepted by either stage script
# are forwarded transparently — see those scripts for the full list.
#
# ── Quick start (Runpod) ──────────────────────────────────────────────────────
#   HF_USER=myuser \
#   CUSTOM_DATASET_REPO_ID=myuser/my_103h_dataset \
#   CUSTOM_DATASET_ROOT=/workspace/datasets/my_103h_dataset \
#   bash examples/training/qformer_full_pipeline.sh
#
# ── Stage skips ───────────────────────────────────────────────────────────────
#   SKIP_STAGE1=true     # Stage 2 only (assumes Stage 1 checkpoints already exist
#                        # under ${OUTPUT_ROOT}/stage1/)
#   SKIP_STAGE2=true     # Stage 1 only
#
# ── Multi-GPU / multi-node ────────────────────────────────────────────────────
# Set ``ACCELERATE_LAUNCH_ARGS`` and it will be passed through to BOTH stages.
# IMPORTANT: when running across nodes, launch this orchestrator on EVERY
# node simultaneously (only ``--machine_rank`` differs). Each iteration's
# ``accelerate launch`` rendezvouses across nodes via ``--main_process_ip``.
#
#   # 2 nodes × 8 GPUs:
#   #  Node 0 (rank=0):
#   ACCELERATE_LAUNCH_ARGS="--multi_gpu --num_machines=2 --num_processes=16 \
#     --machine_rank=0 --main_process_ip=10.0.0.1 --main_process_port=29500 \
#     --mixed_precision=bf16" \
#     HF_USER=myuser CUSTOM_DATASET_REPO_ID=... \
#     bash examples/training/qformer_full_pipeline.sh
#
#   #  Node 1 (rank=1):  same command with --machine_rank=1.
#
# ── Examples ──────────────────────────────────────────────────────────────────
#   # Resume after a crashed Stage 2 — only run the benchmark phase
#   SKIP_STAGE1=true HF_USER=myuser \
#     bash examples/training/qformer_full_pipeline.sh
#
#   # Pretrain only, finetune later
#   SKIP_STAGE2=true HF_USER=myuser \
#     CUSTOM_DATASET_REPO_ID=myuser/my_103h_dataset \
#     bash examples/training/qformer_full_pipeline.sh
#
#   # Just LIBERO + MetaWorld at Stage 2 (skip RoboTwin which needs special install)
#   HF_USER=myuser \
#     CUSTOM_DATASET_REPO_ID=myuser/my_103h_dataset \
#     BENCHMARKS="libero metaworld" \
#     bash examples/training/qformer_full_pipeline.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

: "${HF_USER:?Set HF_USER to your Hugging Face Hub username}"

SKIP_STAGE1="${SKIP_STAGE1:-false}"
SKIP_STAGE2="${SKIP_STAGE2:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${SKIP_STAGE1}" != "true" ]]; then
  : "${CUSTOM_DATASET_REPO_ID:?STAGE 1 requires CUSTOM_DATASET_REPO_ID (or set SKIP_STAGE1=true)}"
  echo "================================================================"
  echo "  Pipeline ▶ STAGE 1: pretrain on custom data"
  echo "================================================================"
  bash "${SCRIPT_DIR}/qformer_pretrain_custom.sh"
else
  echo ">>> SKIP_STAGE1=true → skipping pretrain stage"
fi

if [[ "${SKIP_STAGE2}" != "true" ]]; then
  echo "================================================================"
  echo "  Pipeline ▶ STAGE 2: per-benchmark finetune + eval"
  echo "================================================================"
  bash "${SCRIPT_DIR}/qformer_finetune_benchmarks.sh"
else
  echo ">>> SKIP_STAGE2=true → skipping finetune+eval stage"
fi

echo "================================================================"
echo "  Pipeline complete: $(date)"
echo "================================================================"
