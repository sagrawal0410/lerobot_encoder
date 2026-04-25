#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 — Per-benchmark finetune + success-rate eval.
#
# For each benchmark in BENCHMARKS, for each (N, seed):
#   1. Load the STAGE 1 checkpoint produced by ``qformer_pretrain_custom.sh``
#      (or fall back to ``lerobot/smolvla_base`` if Stage 1 was skipped).
#   2. Finetune on the benchmark's training dataset.
#   3. Mid-training eval logs ``eval/pc_success`` per suite to W&B.
#   4. Run a deeper final standalone eval, push results to W&B via
#      ``log_eval_to_wandb.py``.
#
# Each benchmark gets its own W&B project so dashboards don't get tangled.
#
# ── Supported benchmarks ──────────────────────────────────────────────────────
#   libero       → HuggingFaceVLA/libero, suites: spatial/object/goal/long
#   metaworld    → lerobot/metaworld_mt50, suites: easy/medium/hard/very_hard
#   robotwin     → lerobot/robotwin_unified, comma-separated task list
#
# ── Quick start ───────────────────────────────────────────────────────────────
#   HF_USER=myuser bash examples/training/qformer_finetune_benchmarks.sh
#
# ── Common overrides ──────────────────────────────────────────────────────────
#   HF_USER                   (required) HF Hub username
#   N_VALUES                  Q-Former queries to sweep   (default: "8 16 32 64 128")
#   SEEDS                     Random seeds                (default: "0")
#   BENCHMARKS                space-sep list              (default: "libero metaworld robotwin")
#   STAGE1_OUTPUT_ROOT        Where Stage 1 wrote checkpoints
#                                                         (default: ./outputs/qformer_pipeline/stage1)
#   FALLBACK_POLICY_PATH      Used if a Stage 1 ckpt is missing
#                                                         (default: lerobot/smolvla_base)
#   STEPS                     Finetune steps              (default: 50000)
#   BATCH_SIZE                                            (default: 64)
#   LR                                                    (default: 1e-4)
#   DECAY_LR                                              (default: 2.5e-6)
#   EVAL_FREQ                 Mid-train eval cadence      (default: 10000)
#   EVAL_N_EPISODES_TRAIN     Episodes per task per eval  (default: 3)
#   EVAL_N_EPISODES_FINAL     Episodes for final eval     (default: 10)
#   SAVE_FREQ                                             (default: 25000)
#   USE_PEFT                  true|false                  (default: false)
#   PEFT_R                                                (default: 64)
#   LORA_VLM                                              (default: false)
#   OUTPUT_ROOT                                           (default: ./outputs/qformer_pipeline)
#   WANDB_PROJECT_PREFIX      W&B project prefix          (default: smolvla-qformer)
#                             → projects:  ${PREFIX}-${benchmark}
#   WANDB_ENABLE                                          (default: true)
#   RUN_FINAL_EVAL            true|false                  (default: true)
#   DRY_RUN                   true|false                  (default: false)
#   SKIP_IF_EXISTS                                        (default: true)
#
#   # Per-benchmark eval-suite overrides (rarely needed):
#   LIBERO_EVAL_TASKS         (default: libero_spatial,libero_object,libero_goal,libero_10)
#   METAWORLD_EVAL_TASKS      (default: medium)
#   ROBOTWIN_EVAL_TASKS       (default: beat_block_hammer,handover_block,stack_blocks_two)
#
#   # Per-benchmark training-dataset overrides (rarely needed):
#   LIBERO_DATASET            (default: HuggingFaceVLA/libero)
#   METAWORLD_DATASET         (default: lerobot/metaworld_mt50)
#   ROBOTWIN_DATASET          (default: lerobot/robotwin_unified)
#
# ── Multi-GPU / multi-node ────────────────────────────────────────────────────
# Set ``ACCELERATE_LAUNCH_ARGS`` to wrap the ``lerobot-train`` calls with
# ``accelerate launch`` (see ``qformer_pretrain_custom.sh`` header for examples).
# ``lerobot-eval`` (the standalone final-eval) is always single-process — it
# rolls out one env per worker and parallelises via VectorEnv, so distributing
# it across nodes provides no speedup.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

: "${HF_USER:?Set HF_USER to your Hugging Face Hub username}"

N_VALUES="${N_VALUES:-8 16 32 64 128}"
SEEDS="${SEEDS:-0}"
BENCHMARKS="${BENCHMARKS:-libero metaworld robotwin}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/qformer_pipeline}"
STAGE1_OUTPUT_ROOT="${STAGE1_OUTPUT_ROOT:-${OUTPUT_ROOT}/stage1}"
FALLBACK_POLICY_PATH="${FALLBACK_POLICY_PATH:-lerobot/smolvla_base}"
STEPS="${STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-4}"
DECAY_LR="${DECAY_LR:-2.5e-6}"
EVAL_FREQ="${EVAL_FREQ:-10000}"
EVAL_N_EPISODES_TRAIN="${EVAL_N_EPISODES_TRAIN:-3}"
EVAL_N_EPISODES_FINAL="${EVAL_N_EPISODES_FINAL:-10}"
SAVE_FREQ="${SAVE_FREQ:-25000}"
USE_PEFT="${USE_PEFT:-false}"
PEFT_R="${PEFT_R:-64}"
LORA_VLM="${LORA_VLM:-false}"
WANDB_PROJECT_PREFIX="${WANDB_PROJECT_PREFIX:-smolvla-qformer}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
RUN_FINAL_EVAL="${RUN_FINAL_EVAL:-true}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-true}"
ACCELERATE_LAUNCH_ARGS="${ACCELERATE_LAUNCH_ARGS:-}"

# ──────────────────────────────────────────────────────────────────────────────
# Per-benchmark step / batch / eval overrides.
#
# By default each benchmark inherits ``STEPS``, ``BATCH_SIZE``, ``EVAL_FREQ``,
# ``EVAL_N_EPISODES_TRAIN`` from the top-level vars above. Override per
# benchmark with ``LIBERO_STEPS``, ``METAWORLD_BATCH_SIZE``, etc. to match a
# specific published recipe. Examples (matching LeRobot docs as of 2026):
#
#   # SmolVLA general user-finetune recipe (smolvla.mdx)
#   STEPS=20000 BATCH_SIZE=64
#
#   # LIBERO doc example (libero.mdx, 1-GPU illustration)
#   LIBERO_STEPS=100000 LIBERO_BATCH_SIZE=4 LIBERO_EVAL_FREQ=1000
#
#   # MetaWorld doc example (metaworld.mdx, 1-GPU illustration)
#   METAWORLD_STEPS=100000 METAWORLD_BATCH_SIZE=4 METAWORLD_EVAL_FREQ=1000
# ──────────────────────────────────────────────────────────────────────────────
LIBERO_STEPS="${LIBERO_STEPS:-${STEPS}}"
METAWORLD_STEPS="${METAWORLD_STEPS:-${STEPS}}"
ROBOTWIN_STEPS="${ROBOTWIN_STEPS:-${STEPS}}"
LIBERO_BATCH_SIZE="${LIBERO_BATCH_SIZE:-${BATCH_SIZE}}"
METAWORLD_BATCH_SIZE="${METAWORLD_BATCH_SIZE:-${BATCH_SIZE}}"
ROBOTWIN_BATCH_SIZE="${ROBOTWIN_BATCH_SIZE:-${BATCH_SIZE}}"
LIBERO_EVAL_FREQ="${LIBERO_EVAL_FREQ:-${EVAL_FREQ}}"
METAWORLD_EVAL_FREQ="${METAWORLD_EVAL_FREQ:-${EVAL_FREQ}}"
ROBOTWIN_EVAL_FREQ="${ROBOTWIN_EVAL_FREQ:-${EVAL_FREQ}}"
LIBERO_EVAL_N_EPISODES_TRAIN="${LIBERO_EVAL_N_EPISODES_TRAIN:-${EVAL_N_EPISODES_TRAIN}}"
METAWORLD_EVAL_N_EPISODES_TRAIN="${METAWORLD_EVAL_N_EPISODES_TRAIN:-${EVAL_N_EPISODES_TRAIN}}"
ROBOTWIN_EVAL_N_EPISODES_TRAIN="${ROBOTWIN_EVAL_N_EPISODES_TRAIN:-${EVAL_N_EPISODES_TRAIN}}"

# Frame-timestamp tolerance — same reasoning as Stage 1 (aggregator drift).
TOLERANCE_S="${TOLERANCE_S:-0.001}"

# Optional camera rename map. Stage-1 → Stage-2 normally needs no rename
# because both LIBERO/MetaWorld/RoboTwin and the Stage-1 saved policy use
# ``observation.images.camera{1,2,3}``. Override per-benchmark via
# ``LIBERO_RENAME_MAP``, ``METAWORLD_RENAME_MAP``, ``ROBOTWIN_RENAME_MAP`` if
# the benchmark dataset uses different camera keys.
RENAME_MAP="${RENAME_MAP:-}"
LIBERO_RENAME_MAP="${LIBERO_RENAME_MAP:-${RENAME_MAP}}"
METAWORLD_RENAME_MAP="${METAWORLD_RENAME_MAP:-${RENAME_MAP}}"
ROBOTWIN_RENAME_MAP="${ROBOTWIN_RENAME_MAP:-${RENAME_MAP}}"

# SmolVLA defaults push_to_hub=True, which crashes the post-training step with
# 401 Unauthorized when no HF write token is set. Default OFF; opt in with
# ``PUSH_TO_HUB=true`` (then also export HF_TOKEN).
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"

LIBERO_DATASET="${LIBERO_DATASET:-HuggingFaceVLA/libero}"
METAWORLD_DATASET="${METAWORLD_DATASET:-lerobot/metaworld_mt50}"
ROBOTWIN_DATASET="${ROBOTWIN_DATASET:-lerobot/robotwin_unified}"

LIBERO_EVAL_TASKS="${LIBERO_EVAL_TASKS:-libero_spatial,libero_object,libero_goal,libero_10}"
METAWORLD_EVAL_TASKS="${METAWORLD_EVAL_TASKS:-medium}"
ROBOTWIN_EVAL_TASKS="${ROBOTWIN_EVAL_TASKS:-beat_block_hammer,handover_block,stack_blocks_two}"

if [[ "${LORA_VLM}" == "true" ]]; then
  TRAIN_EXPERT_ONLY="false"
else
  TRAIN_EXPERT_ONLY="true"
fi

STAGE2_ROOT="${OUTPUT_ROOT}/stage2"
mkdir -p "${STAGE2_ROOT}"
SUMMARY_LOG="${STAGE2_ROOT}/stage2_summary.txt"
date | tee -a "${SUMMARY_LOG}" >/dev/null
echo "Stage 2 (per-benchmark finetune + eval) started" | tee -a "${SUMMARY_LOG}"
echo "  HF_USER=${HF_USER}  N_VALUES='${N_VALUES}'  SEEDS='${SEEDS}'" | tee -a "${SUMMARY_LOG}"
echo "  Benchmarks:        ${BENCHMARKS}" | tee -a "${SUMMARY_LOG}"
echo "  Stage 1 root:      ${STAGE1_OUTPUT_ROOT}" | tee -a "${SUMMARY_LOG}"
echo "  W&B prefix:        ${WANDB_PROJECT_PREFIX}-<benchmark>" | tee -a "${SUMMARY_LOG}"

run() {
  echo "+ $*" | tee -a "${SUMMARY_LOG}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    return 0
  fi
  "$@"
}

# Wrap lerobot-train with ``accelerate launch`` when multi-GPU/multi-node is
# requested via ``ACCELERATE_LAUNCH_ARGS``. lerobot-eval is always called
# directly (single-process) since its parallelism is via VectorEnv workers,
# not DDP.
run_lerobot_train() {
  if [[ -n "${ACCELERATE_LAUNCH_ARGS}" ]]; then
    # shellcheck disable=SC2086
    run uv run accelerate launch ${ACCELERATE_LAUNCH_ARGS} \
      -m lerobot.scripts.lerobot_train "$@"
  else
    run uv run lerobot-train "$@"
  fi
}

resolve_starting_checkpoint() {
  local n="$1"
  local seed="$2"
  local stage1_ckpt="${STAGE1_OUTPUT_ROOT}/qformer_n${n}_s${seed}/checkpoints/last/pretrained_model"
  if [[ -d "${stage1_ckpt}" ]]; then
    echo "${stage1_ckpt}"
  else
    echo "${FALLBACK_POLICY_PATH}"
  fi
}

resolve_benchmark_config() {
  local bench="$1"
  case "${bench}" in
    libero)
      BENCH_DATASET="${LIBERO_DATASET}"
      BENCH_EVAL_TASKS="${LIBERO_EVAL_TASKS}"
      BENCH_RENAME_MAP="${LIBERO_RENAME_MAP}"
      BENCH_STEPS="${LIBERO_STEPS}"
      BENCH_BATCH_SIZE="${LIBERO_BATCH_SIZE}"
      BENCH_EVAL_FREQ="${LIBERO_EVAL_FREQ}"
      BENCH_EVAL_N_EPISODES_TRAIN="${LIBERO_EVAL_N_EPISODES_TRAIN}"
      ;;
    metaworld)
      BENCH_DATASET="${METAWORLD_DATASET}"
      BENCH_EVAL_TASKS="${METAWORLD_EVAL_TASKS}"
      BENCH_RENAME_MAP="${METAWORLD_RENAME_MAP}"
      BENCH_STEPS="${METAWORLD_STEPS}"
      BENCH_BATCH_SIZE="${METAWORLD_BATCH_SIZE}"
      BENCH_EVAL_FREQ="${METAWORLD_EVAL_FREQ}"
      BENCH_EVAL_N_EPISODES_TRAIN="${METAWORLD_EVAL_N_EPISODES_TRAIN}"
      ;;
    robotwin)
      BENCH_DATASET="${ROBOTWIN_DATASET}"
      BENCH_EVAL_TASKS="${ROBOTWIN_EVAL_TASKS}"
      BENCH_RENAME_MAP="${ROBOTWIN_RENAME_MAP}"
      BENCH_STEPS="${ROBOTWIN_STEPS}"
      BENCH_BATCH_SIZE="${ROBOTWIN_BATCH_SIZE}"
      BENCH_EVAL_FREQ="${ROBOTWIN_EVAL_FREQ}"
      BENCH_EVAL_N_EPISODES_TRAIN="${ROBOTWIN_EVAL_N_EPISODES_TRAIN}"
      ;;
    *)
      echo "ERROR: unknown benchmark '${bench}'" | tee -a "${SUMMARY_LOG}"
      return 1
      ;;
  esac
}

finetune_one() {
  local bench="$1"
  local n="$2"
  local seed="$3"
  resolve_benchmark_config "${bench}"
  local run_name="qformer_${bench}_n${n}_s${seed}"
  local output_dir="${STAGE2_ROOT}/${bench}/qformer_n${n}_s${seed}"
  local repo_id="${HF_USER}/smolvla_${run_name}"
  local starting_ckpt
  starting_ckpt="$(resolve_starting_checkpoint "${n}" "${seed}")"
  local wandb_project="${WANDB_PROJECT_PREFIX}-${bench}"

  if [[ "${SKIP_IF_EXISTS}" == "true" && -d "${output_dir}/checkpoints" ]]; then
    echo ">>> [skip-stage2] ${run_name}: checkpoint exists" | tee -a "${SUMMARY_LOG}"
    return 0
  fi

  # Auto-clean partial output dirs (existing dir with no checkpoints) so a
  # previous crashed attempt doesn't trip lerobot-train's existence check.
  if [[ -d "${output_dir}" && ! -d "${output_dir}/checkpoints" ]]; then
    echo ">>> [clean-partial] ${run_name}: removing partial ${output_dir}" | tee -a "${SUMMARY_LOG}"
    rm -rf "${output_dir}"
  fi

  echo ">>> [stage2-finetune] ${run_name}  (start=${starting_ckpt})" | tee -a "${SUMMARY_LOG}"
  local args=(
    --policy.path="${starting_ckpt}"
    --policy.use_qformer=true
    --policy.qformer_num_queries="${n}"
    --policy.lora_target_vlm_text_model="${LORA_VLM}"
    --policy.train_expert_only="${TRAIN_EXPERT_ONLY}"
    --policy.optimizer_lr="${LR}"
    --policy.scheduler_decay_lr="${DECAY_LR}"
    --policy.repo_id="${repo_id}"
    --policy.push_to_hub="${PUSH_TO_HUB}"
    --dataset.repo_id="${BENCH_DATASET}"
    --env.type="${bench}"
    --env.task="${BENCH_EVAL_TASKS}"
    --steps="${BENCH_STEPS}"
    --batch_size="${BENCH_BATCH_SIZE}"
    --seed="${seed}"
    --eval_freq="${BENCH_EVAL_FREQ}"
    --eval.n_episodes="${BENCH_EVAL_N_EPISODES_TRAIN}"
    --eval.batch_size=1
    --save_freq="${SAVE_FREQ}"
    --output_dir="${output_dir}"
    --job_name="${run_name}"
    --wandb.enable="${WANDB_ENABLE}"
    --tolerance_s="${TOLERANCE_S}"
    --wandb.project="${wandb_project}"
  )
  if [[ -n "${BENCH_RENAME_MAP}" && "${BENCH_RENAME_MAP}" != "{}" ]]; then
    args+=(--rename_map="${BENCH_RENAME_MAP}")
  fi
  if [[ "${USE_PEFT}" == "true" ]]; then
    args+=(--peft.method_type=LORA --peft.r="${PEFT_R}")
  fi

  run_lerobot_train "${args[@]}"
}

final_eval_one() {
  local bench="$1"
  local n="$2"
  local seed="$3"
  resolve_benchmark_config "${bench}"
  local run_name="qformer_${bench}_n${n}_s${seed}"
  local output_dir="${STAGE2_ROOT}/${bench}/qformer_n${n}_s${seed}"
  local eval_dir="${output_dir}/eval_final"
  local checkpoint_dir="${output_dir}/checkpoints/last/pretrained_model"
  local wandb_project="${WANDB_PROJECT_PREFIX}-${bench}"

  if [[ ! -d "${output_dir}/checkpoints" ]]; then
    echo ">>> [skip-final-eval] ${run_name}: no checkpoint dir" | tee -a "${SUMMARY_LOG}"
    return 0
  fi

  echo ">>> [stage2-final-eval] ${run_name} (${EVAL_N_EPISODES_FINAL} episodes/task)" | tee -a "${SUMMARY_LOG}"
  mkdir -p "${eval_dir}"
  run uv run lerobot-eval \
    --policy.path="${checkpoint_dir}" \
    --env.type="${bench}" \
    --env.task="${BENCH_EVAL_TASKS}" \
    --eval.batch_size=1 \
    --eval.n_episodes="${EVAL_N_EPISODES_FINAL}" \
    --output_dir="${eval_dir}"

  echo ">>> [wandb-log] ${run_name}_final_eval" | tee -a "${SUMMARY_LOG}"
  run uv run python examples/training/log_eval_to_wandb.py \
    --eval_dir="${eval_dir}" \
    --wandb_project="${wandb_project}" \
    --wandb_run_name="${run_name}_final_eval" \
    --wandb_enable="${WANDB_ENABLE}" \
    --tags "N=${n}" "seed=${seed}" "stage=final" "benchmark=${bench}"
}

for BENCHMARK in ${BENCHMARKS}; do
  echo "=== Benchmark: ${BENCHMARK} ===" | tee -a "${SUMMARY_LOG}"
  for N in ${N_VALUES}; do
    for SEED in ${SEEDS}; do
      finetune_one "${BENCHMARK}" "${N}" "${SEED}"
      if [[ "${RUN_FINAL_EVAL}" == "true" ]]; then
        final_eval_one "${BENCHMARK}" "${N}" "${SEED}"
      fi
    done
  done
done

echo "Stage 2 finished: $(date)" | tee -a "${SUMMARY_LOG}"
echo "Outputs under:   ${STAGE2_ROOT}"
echo "Summary log:     ${SUMMARY_LOG}"
