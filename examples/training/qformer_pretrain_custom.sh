#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1 — Q-Former pretrain on a custom teleop dataset.
#
# Continues training ``lerobot/smolvla_base`` on your own dataset (already on
# disk in a Runpod pod or a HF Hub repo), with the Q-Former enabled. Produces
# one checkpoint per (N, seed) at:
#
#   ${OUTPUT_ROOT}/stage1/qformer_n${N}_s${SEED}/checkpoints/last/pretrained_model
#
# These checkpoints are the starting point for STAGE 2
# (``qformer_finetune_benchmarks.sh``), which finetunes + evals on
# LIBERO / MetaWorld / RoboTwin.
#
# ── What gets logged to W&B ───────────────────────────────────────────────────
#   train/loss, train/lr, train/grad_norm, train/throughput   (every log_freq)
#   model checkpoints as W&B artifacts                        (every save_freq)
#
# Stage 1 has NO mid-training eval (your custom dataset has no sim env to roll
# out in). Use STAGE 2 for success-rate metrics.
#
# ── Quick start (Runpod) ──────────────────────────────────────────────────────
#   HF_USER=myuser \
#   CUSTOM_DATASET_REPO_ID=myuser/my_103h_dataset \
#   CUSTOM_DATASET_ROOT=/workspace/datasets/my_103h_dataset \
#   bash examples/training/qformer_pretrain_custom.sh
#
# ── Required env vars ─────────────────────────────────────────────────────────
#   HF_USER                   HF Hub username (used to namespace repo IDs)
#   CUSTOM_DATASET_REPO_ID    Repo id of your custom dataset (HF Hub style)
#
# ── Common overrides ──────────────────────────────────────────────────────────
#   N_VALUES                  Q-Former queries to sweep   (default: "8 16 32 64 128")
#   SEEDS                     Random seeds                (default: "0")
#   CUSTOM_DATASET_ROOT       Local dir of the dataset    (default: unset → HF Hub)
#   POLICY_PATH               Starting checkpoint         (default: lerobot/smolvla_base)
#   STEPS                     Pretrain steps              (default: 100000)
#   BATCH_SIZE                Pretrain batch size         (default: 64)
#   LR                        Optimizer LR               (default: 1e-4)
#   DECAY_LR                  Cosine decay target LR      (default: 2.5e-6)
#   QFORMER_NUM_LAYERS                                    (default: 2)
#   QFORMER_SA_EVERY_N        Self-attn cadence           (default: 1)
#   USE_PEFT                  true|false  apply LoRA      (default: false)
#   PEFT_R                    LoRA rank if USE_PEFT       (default: 64)
#   LORA_VLM                  Also LoRA SmolVLM text bb   (default: false)
#   OUTPUT_ROOT               Run output root             (default: ./outputs/qformer_pipeline)
#   WANDB_PROJECT             W&B project (Stage 1)       (default: smolvla-qformer-pretrain)
#   WANDB_ENABLE              true|false                  (default: true)
#   SAVE_FREQ                 Steps between checkpoints   (default: 25000)
#   DRY_RUN                   true|false                  (default: false)
#   SKIP_IF_EXISTS            true|false                  (default: true)
#
# ── Multi-GPU / multi-node ────────────────────────────────────────────────────
# Set ``ACCELERATE_LAUNCH_ARGS`` to non-empty to wrap each training run with
# ``accelerate launch``. NOTE: BATCH_SIZE in this script is per-process; the
# effective batch size becomes BATCH_SIZE × num_processes.
#
#   # Single node, 8 GPUs:
#   ACCELERATE_LAUNCH_ARGS="--multi_gpu --num_processes=8 --mixed_precision=bf16" \
#     HF_USER=myuser CUSTOM_DATASET_REPO_ID=... \
#     bash examples/training/qformer_pretrain_custom.sh
#
#   # 2 nodes × 8 GPUs (run on EACH node, only differing in --machine_rank):
#   #   Node 0:
#   ACCELERATE_LAUNCH_ARGS="--multi_gpu --num_machines=2 --num_processes=16 \
#     --machine_rank=0 --main_process_ip=10.0.0.1 --main_process_port=29500 \
#     --mixed_precision=bf16" \
#     HF_USER=myuser CUSTOM_DATASET_REPO_ID=... \
#     bash examples/training/qformer_pretrain_custom.sh
#   #   Node 1:
#   ACCELERATE_LAUNCH_ARGS="--multi_gpu --num_machines=2 --num_processes=16 \
#     --machine_rank=1 --main_process_ip=10.0.0.1 --main_process_port=29500 \
#     --mixed_precision=bf16" \
#     HF_USER=myuser CUSTOM_DATASET_REPO_ID=... \
#     bash examples/training/qformer_pretrain_custom.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

: "${HF_USER:?Set HF_USER to your Hugging Face Hub username}"
: "${CUSTOM_DATASET_REPO_ID:?Set CUSTOM_DATASET_REPO_ID to your dataset repo id (e.g. myuser/my_103h_dataset)}"

N_VALUES="${N_VALUES:-8 16 32 64 128}"
SEEDS="${SEEDS:-0}"
CUSTOM_DATASET_ROOT="${CUSTOM_DATASET_ROOT:-}"
POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_base}"
# Default raised vs. per-task pretraining: the merged dataset is much larger,
# so we want longer training to actually walk over it. SmolVLA's own base was
# trained for ~200k steps over a comparable multi-task pool.
STEPS="${STEPS:-200000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-4}"
DECAY_LR="${DECAY_LR:-2.5e-6}"
QFORMER_NUM_LAYERS="${QFORMER_NUM_LAYERS:-2}"
QFORMER_SA_EVERY_N="${QFORMER_SA_EVERY_N:-1}"
USE_PEFT="${USE_PEFT:-false}"
PEFT_R="${PEFT_R:-64}"
LORA_VLM="${LORA_VLM:-false}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/qformer_pipeline}"
WANDB_PROJECT="${WANDB_PROJECT:-smolvla-qformer-pretrain}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
SAVE_FREQ="${SAVE_FREQ:-25000}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-true}"
ACCELERATE_LAUNCH_ARGS="${ACCELERATE_LAUNCH_ARGS:-}"

# SmolVLA defaults push_to_hub=True. After 5 hours of training, a missing
# HF_TOKEN crashes the run *after* the checkpoint is already saved — and
# ``set -euo pipefail`` stops the rest of the sweep. Default to OFF; opt in
# explicitly with ``PUSH_TO_HUB=true`` (then also export HF_TOKEN).
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"

# ──────────────────────────────────────────────────────────────────────────────
# Frame timestamp tolerance.
#
# After ``aggregate_datasets()`` concatenates per-episode MP4s into multi-episode
# files, per-episode timestamps drift by a few microseconds from what's stored
# in the parquet (pure float32 precision). The default 100 µs tolerance is
# tighter than that drift, which raises ``FrameTimestampError``. Bumping to
# 1 ms (still ≪ frame interval at 30 Hz = 33 ms) is safe and silences it.
# ──────────────────────────────────────────────────────────────────────────────
TOLERANCE_S="${TOLERANCE_S:-0.001}"

# ──────────────────────────────────────────────────────────────────────────────
# Camera rename map.
#
# ``lerobot/smolvla_base`` was trained with cameras named ``camera1/2/3``.
# The MCAP converter writes ``observation.images.{left,right,top}`` to be
# human-readable. Without a rename, ``make_policy`` fails with a
# Feature mismatch error. Default below maps:
#   left  → camera1
#   right → camera2
#   top   → camera3
#
# Override with RENAME_MAP='{...}' if your dataset uses different keys, or set
# RENAME_MAP='{}' to disable.
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_RENAME_MAP='{"observation.images.left": "observation.images.camera1", "observation.images.right": "observation.images.camera2", "observation.images.top": "observation.images.camera3"}'
RENAME_MAP="${RENAME_MAP:-${DEFAULT_RENAME_MAP}}"

if [[ "${LORA_VLM}" == "true" ]]; then
  TRAIN_EXPERT_ONLY="false"
else
  TRAIN_EXPERT_ONLY="true"
fi

STAGE1_ROOT="${OUTPUT_ROOT}/stage1"
mkdir -p "${STAGE1_ROOT}"
SUMMARY_LOG="${STAGE1_ROOT}/stage1_summary.txt"
date | tee -a "${SUMMARY_LOG}" >/dev/null
echo "Stage 1 (SmolVLA-style merged-dataset base pretrain) started" | tee -a "${SUMMARY_LOG}"
echo "  HF_USER=${HF_USER}  N_VALUES='${N_VALUES}'  SEEDS='${SEEDS}'" | tee -a "${SUMMARY_LOG}"
echo "  Merged dataset: ${CUSTOM_DATASET_REPO_ID}  (root='${CUSTOM_DATASET_ROOT:-<hub>}')" | tee -a "${SUMMARY_LOG}"
echo "  Base policy:    ${POLICY_PATH}" | tee -a "${SUMMARY_LOG}"
echo "  Steps:          ${STEPS}  Batch (per-proc): ${BATCH_SIZE}  LR=${LR}" | tee -a "${SUMMARY_LOG}"
echo "  W&B project:    ${WANDB_PROJECT}" | tee -a "${SUMMARY_LOG}"

run() {
  echo "+ $*" | tee -a "${SUMMARY_LOG}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    return 0
  fi
  "$@"
}

# Wraps a lerobot-train invocation with ``accelerate launch`` when
# ``ACCELERATE_LAUNCH_ARGS`` is non-empty, otherwise calls ``lerobot-train``
# directly. Usage: ``run_lerobot_train --policy.path=... --dataset...``
run_lerobot_train() {
  if [[ -n "${ACCELERATE_LAUNCH_ARGS}" ]]; then
    # ``accelerate launch`` needs a Python script/module; we use ``-m`` to
    # invoke the same entry point as the ``lerobot-train`` console script.
    # shellcheck disable=SC2086
    run uv run accelerate launch ${ACCELERATE_LAUNCH_ARGS} \
      -m lerobot.scripts.lerobot_train "$@"
  else
    run uv run lerobot-train "$@"
  fi
}

pretrain_one() {
  local n="$1"
  local seed="$2"
  local run_name="qformer_n${n}_s${seed}"
  local output_dir="${STAGE1_ROOT}/${run_name}"
  local repo_id="${HF_USER}/smolvla_pretrain_${run_name}"

  if [[ "${SKIP_IF_EXISTS}" == "true" && -d "${output_dir}/checkpoints" ]]; then
    echo ">>> [skip-stage1] ${run_name}: checkpoint exists" | tee -a "${SUMMARY_LOG}"
    return 0
  fi

  # Auto-clean a partial output dir from a previous failed attempt, so
  # ``lerobot-train``'s "output dir already exists" check doesn't fire.
  if [[ -d "${output_dir}" && ! -d "${output_dir}/checkpoints" ]]; then
    echo ">>> [clean-partial] ${run_name}: removing partial ${output_dir}" | tee -a "${SUMMARY_LOG}"
    rm -rf "${output_dir}"
  fi

  echo ">>> [stage1-pretrain] ${run_name}" | tee -a "${SUMMARY_LOG}"
  local args=(
    --policy.path="${POLICY_PATH}"
    --policy.use_qformer=true
    --policy.qformer_num_queries="${n}"
    --policy.qformer_num_layers="${QFORMER_NUM_LAYERS}"
    --policy.qformer_self_attn_every_n_layers="${QFORMER_SA_EVERY_N}"
    --policy.lora_target_vlm_text_model="${LORA_VLM}"
    --policy.train_expert_only="${TRAIN_EXPERT_ONLY}"
    --policy.optimizer_lr="${LR}"
    --policy.scheduler_decay_lr="${DECAY_LR}"
    --policy.repo_id="${repo_id}"
    --policy.push_to_hub="${PUSH_TO_HUB}"
    --dataset.repo_id="${CUSTOM_DATASET_REPO_ID}"
    --steps="${STEPS}"
    --batch_size="${BATCH_SIZE}"
    --seed="${seed}"
    --save_freq="${SAVE_FREQ}"
    --output_dir="${output_dir}"
    --job_name="stage1_${run_name}"
    --wandb.enable="${WANDB_ENABLE}"
    --wandb.project="${WANDB_PROJECT}"
    --tolerance_s="${TOLERANCE_S}"
  )
  if [[ -n "${CUSTOM_DATASET_ROOT}" ]]; then
    args+=(--dataset.root="${CUSTOM_DATASET_ROOT}")
  fi
  if [[ -n "${RENAME_MAP}" && "${RENAME_MAP}" != "{}" ]]; then
    args+=(--rename_map="${RENAME_MAP}")
  fi
  if [[ "${USE_PEFT}" == "true" ]]; then
    args+=(--peft.method_type=LORA --peft.r="${PEFT_R}")
  fi

  run_lerobot_train "${args[@]}"
}

for N in ${N_VALUES}; do
  for SEED in ${SEEDS}; do
    pretrain_one "${N}" "${SEED}"
  done
done

echo "Stage 1 finished: $(date)" | tee -a "${SUMMARY_LOG}"
echo "Outputs under:   ${STAGE1_ROOT}"
echo "Summary log:     ${SUMMARY_LOG}"
echo "Next: bash examples/training/qformer_finetune_benchmarks.sh"
