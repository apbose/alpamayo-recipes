#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
readonly RECIPE_DIR="$REPO_ROOT/recipes/alpamayo1_5_sft"
readonly ENV_DIR="${ALPAMAYO_ENV:-$RECIPE_DIR/a1_5_sft}"
readonly TORCHRUN_BIN="${TORCHRUN_BIN:-$ENV_DIR/bin/torchrun}"
readonly TRAIN_MANIFEST="${TRAIN_MANIFEST:-$PROJECT_DIR/manifests/route_less_19chunks/train.json}"
readonly VAL_MANIFEST="${VAL_MANIFEST:-$PROJECT_DIR/manifests/route_less_19chunks/val.json}"
readonly TRAIN_CHUNKS='[420,727,728,1519,1657,2368,2372,2447,2868]'
readonly VAL_CHUNKS='[214,224,276,968,982,1984,2599]'
readonly MAX_STEPS="${MAX_STEPS:-10692}"

: "${DATASET_DIR:?Set DATASET_DIR to the local PhysicalAI-AV root}"
: "${BASE_CHECKPOINT:?Set BASE_CHECKPOINT to the Alpamayo-1.5 A1-format wrapper}"
readonly OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/train_route_less}"

for required in "$TORCHRUN_BIN" "$TRAIN_MANIFEST" "$VAL_MANIFEST"; do
  if [[ ! -e "$required" ]]; then
    printf 'Required path not found: %s\n' "$required" >&2
    exit 2
  fi
done
if [[ ! -d "$DATASET_DIR" || ! -d "$BASE_CHECKPOINT" ]]; then
  printf 'DATASET_DIR and BASE_CHECKPOINT must both be existing directories.\n' >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  printf 'Refusing to overwrite existing output: %s\n' "$OUTPUT_DIR" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export DS_IGNORE_CUDA_DETECTION=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/alpamayo15-shortcut-triton}"

cd "$RECIPE_DIR"
"$TORCHRUN_BIN" --standalone --nproc_per_node=1 \
  -m alpamayo1_5_sft.train_hf \
  --config-path pkg://alpamayo1_5_sft/configs \
  --config-name sft_stage2_trajectory_shortcut \
  "model.pretrained_model_name_or_path=$BASE_CHECKPOINT" \
  +model.attn_implementation=eager \
  "data.train_dataset.local_dir=$DATASET_DIR" \
  "data.train_dataset.annotations_path=$TRAIN_MANIFEST" \
  "data.train_dataset.chunk_ids=$TRAIN_CHUNKS" \
  "data.val_dataset.local_dir=$DATASET_DIR" \
  "data.val_dataset.annotations_path=$VAL_MANIFEST" \
  "data.val_dataset.chunk_ids=$VAL_CHUNKS" \
  "paths.output_dir=$OUTPUT_DIR" \
  "+trainer.max_steps=$MAX_STEPS" \
  trainer.num_train_epochs=1 \
  trainer.per_device_train_batch_size=1 \
  trainer.per_device_eval_batch_size=1 \
  trainer.dataloader_num_workers=2 \
  trainer.logging_steps=1 \
  +trainer.save_strategy=steps \
  "trainer.save_steps=$MAX_STEPS" \
  trainer.save_total_limit=1 \
  +trainer.eval_strategy=no \
  trainer.report_to=none
