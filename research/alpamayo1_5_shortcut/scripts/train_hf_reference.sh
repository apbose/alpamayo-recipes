#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
readonly RECIPE_DIR="$REPO_ROOT/recipes/alpamayo1_5_sft"
readonly ENV_DIR="${ALPAMAYO_ENV:-$RECIPE_DIR/a1_5_sft}"
readonly TORCHRUN_BIN="${TORCHRUN_BIN:-$ENV_DIR/bin/torchrun}"
readonly MANIFEST_DIR="${MANIFEST_DIR:-$PROJECT_DIR/manifests/hf_stream_300gb}"
readonly TRAIN_MANIFEST="${TRAIN_MANIFEST:-$MANIFEST_DIR/train.json}"
readonly VAL_MANIFEST="${VAL_MANIFEST:-$MANIFEST_DIR/val.json}"
readonly HF_CACHE_DIR="${HF_CACHE_DIR:-/home/scratch.abose_sw/alpamayo-assets/hf_on_demand_smoke/cache}"
readonly NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
readonly DATALOADER_WORKERS="${DATALOADER_WORKERS:-1}"
readonly MAX_STEPS="${MAX_STEPS:-}"
readonly CONFIG_NAME="${CONFIG_NAME:-sft_stage2_trajectory_shortcut_hf_reference}"
readonly SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
readonly RUN_NAME="${RUN_NAME:-alpamayo15_hf300gb_reference_1epoch}"
readonly RUN_ROOT="${RUN_ROOT:-/home/scratch.abose_sw/alpamayo-assets/runs/$RUN_NAME}"
readonly OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/trainer_output}"
readonly STATUS_FILE="$RUN_ROOT/STATUS"

: "${BASE_CHECKPOINT:?Set BASE_CHECKPOINT to the Alpamayo-1.5 A1-format wrapper}"

for required in "$TORCHRUN_BIN" "$BASE_CHECKPOINT" "$TRAIN_MANIFEST" \
                "$VAL_MANIFEST" "$MANIFEST_DIR/summary.json"; do
  if [[ ! -e "$required" ]]; then
    printf 'Required path not found: %s\n' "$required" >&2
    exit 2
  fi
done
if [[ -e "$RUN_ROOT" ]]; then
  printf 'Refusing to overwrite existing run: %s\n' "$RUN_ROOT" >&2
  exit 2
fi
if [[ ! "$NPROC_PER_NODE" =~ ^[1-8]$ ]]; then
  printf 'NPROC_PER_NODE must be between 1 and 8.\n' >&2
  exit 2
fi
if [[ ! "$DATALOADER_WORKERS" =~ ^[0-9]+$ ]]; then
  printf 'DATALOADER_WORKERS must be a non-negative integer.\n' >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR" "/tmp/alpamayo15-$RUN_NAME-triton"
printf 'status=running\nstarted_utc=%s\nrun_name=%s\nnproc_per_node=%s\nmax_steps=%s\nsource_coverage_gb=301.7651036\ntraining_rows=5295\ndataloader_workers_per_rank=%s\nloss_estimator=reference_partition\nbootstrap_every=8\nroute_conditioning=false\n' \
  "$(date -u +%FT%TZ)" "$RUN_NAME" "$NPROC_PER_NODE" "${MAX_STEPS:-epoch}" "$DATALOADER_WORKERS" \
  > "$STATUS_FILE"

finished=0
record_exit() {
  local exit_code=$?
  if (( finished == 0 )); then
    printf 'status=failed\nfinished_utc=%s\nexit_code=%s\n' \
      "$(date -u +%FT%TZ)" "$exit_code" > "$STATUS_FILE"
  fi
}
trap record_exit EXIT INT TERM

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HOME="${HF_HOME:-/home/abose_sw/.cache/huggingface}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-/home/abose_sw/.cache/huggingface/token}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export DS_IGNORE_CUDA_DETECTION=1
export TRITON_CACHE_DIR="/tmp/alpamayo15-$RUN_NAME-triton"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

trainer_overrides=(
  "trainer.num_train_epochs=1"
  "trainer.per_device_train_batch_size=1"
  "trainer.gradient_accumulation_steps=1"
  "trainer.dataloader_num_workers=$DATALOADER_WORKERS"
  "trainer.logging_steps=1"
  "+trainer.save_strategy=$SAVE_STRATEGY"
  "trainer.save_total_limit=1"
  "+trainer.eval_strategy=no"
  "trainer.report_to=none"
)
if (( DATALOADER_WORKERS > 0 )); then
  trainer_overrides+=(
    "trainer.dataloader_persistent_workers=true"
    "trainer.dataloader_prefetch_factor=2"
  )
else
  trainer_overrides+=(
    "trainer.dataloader_persistent_workers=false"
    "trainer.dataloader_prefetch_factor=null"
  )
fi
if [[ -n "$MAX_STEPS" ]]; then
  trainer_overrides+=("+trainer.max_steps=$MAX_STEPS")
fi

cd "$RECIPE_DIR"
set +e
"$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC_PER_NODE" \
  -m alpamayo1_5_sft.train_hf \
  --config-path pkg://alpamayo1_5_sft/configs \
  --config-name "$CONFIG_NAME" \
  "model.pretrained_model_name_or_path=$BASE_CHECKPOINT" \
  +model.attn_implementation=eager \
  "data.train_dataset.hf_cache_dir=$HF_CACHE_DIR" \
  "data.train_dataset.annotations_path=$TRAIN_MANIFEST" \
  "data.val_dataset.hf_cache_dir=$HF_CACHE_DIR" \
  "data.val_dataset.annotations_path=$VAL_MANIFEST" \
  "paths.output_dir=$OUTPUT_DIR" \
  "${trainer_overrides[@]}" \
  > "$RUN_ROOT/train.log" 2>&1
train_status=$?
set -e
if [[ "$train_status" -ne 0 ]]; then
  exit "$train_status"
fi

checkpoint="none"
latest_checkpoint="$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -V | tail -n 1)"
if [[ -n "$latest_checkpoint" ]]; then
  checkpoint="$latest_checkpoint"
fi
printf 'status=complete\nfinished_utc=%s\nexit_code=0\ncheckpoint=%s\n' \
  "$(date -u +%FT%TZ)" "$checkpoint" > "$STATUS_FILE"
finished=1
