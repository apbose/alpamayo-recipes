#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
readonly RECIPE_DIR="$REPO_ROOT/recipes/alpamayo1_5_sft"
readonly ENV_DIR="${ALPAMAYO_ENV:-$RECIPE_DIR/a1_5_sft}"
readonly TORCHRUN_BIN="${TORCHRUN_BIN:-$ENV_DIR/bin/torchrun}"
readonly BENCHMARK="$SCRIPT_DIR/benchmark_inference_steps.py"
readonly MANIFEST_DIR="${MANIFEST_DIR:-$PROJECT_DIR/manifests/route_less_19chunks}"

: "${DATASET_DIR:?Set DATASET_DIR to the local PhysicalAI-AV root}"
: "${CHECKPOINT:?Set CHECKPOINT to a released wrapper or trained checkpoint}"
readonly OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/benchmark}"

for required in "$TORCHRUN_BIN" "$BENCHMARK" "$MANIFEST_DIR/val.json"; do
  if [[ ! -e "$required" ]]; then
    printf 'Required path not found: %s\n' "$required" >&2
    exit 2
  fi
done
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
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/alpamayo15-shortcut-benchmark-triton}"

cd "$RECIPE_DIR"
"$TORCHRUN_BIN" --standalone --nproc_per_node=1 "$BENCHMARK" \
  --checkpoint "$CHECKPOINT" \
  --config-name sft_stage2_trajectory_shortcut \
  --dataset "$DATASET_DIR" \
  --manifest-dir "$MANIFEST_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --attention-backend eager \
  --steps 10 8 4 2 \
  --num-traj-samples 6 \
  --seed 42 \
  --warmup-samples 1
