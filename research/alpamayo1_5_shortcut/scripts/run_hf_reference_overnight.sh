#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
readonly WORKSPACE="$(cd "$REPO_ROOT/../.." && pwd)"
readonly RECIPE_DIR="$REPO_ROOT/recipes/alpamayo1_5_sft"
readonly PYTHON="$RECIPE_DIR/a1_5_sft/bin/python"
readonly TORCHRUN="$RECIPE_DIR/a1_5_sft/bin/torchrun"
readonly TRAIN_LAUNCHER="$SCRIPT_DIR/train_hf_reference.sh"
readonly VALIDATOR="$SCRIPT_DIR/validate_reference_checkpoint.py"
readonly BENCHMARK="$SCRIPT_DIR/benchmark_inference_steps.py"
readonly GATE="$SCRIPT_DIR/evaluate_two_step_gate.py"
readonly MANIFEST_DIR="$PROJECT_DIR/manifests/hf_stream_300gb"
readonly EVAL_MANIFEST_DIR="$PROJECT_DIR/manifests/route_less_19chunks"
readonly LOCAL_DATASET="/home/scratch.abose_sw/alpamayo-assets/physical_ai_av"
readonly BASE_CHECKPOINT_PATH="/home/scratch.abose_sw/alpamayo-assets/checkpoints/Alpamayo-1.5-10B-A1-format"
readonly RUN_NAME="${RUN_NAME:-alpamayo15_hf300gb_reference_5295clips_1epoch_20260913}"
readonly TRAIN_ROOT="/home/scratch.abose_sw/alpamayo-assets/runs/$RUN_NAME"
readonly PIPELINE_DIR="$WORKSPACE/results/${RUN_NAME}_pipeline"
readonly CANDIDATE_DIR="$WORKSPACE/results/${RUN_NAME}_eval_10_4_2_1_32clips"
readonly BASELINE="$WORKSPACE/results/alpamayo1_5_route_less_released_eval_10_8_4_2_32clips/benchmark_results.json"
readonly SMOKE_STATUS="/home/scratch.abose_sw/alpamayo-assets/runs/alpamayo15_hf300gb_reference_8gpu_smoke_1step/STATUS"
readonly EXPECTED_STEPS=662
readonly TRAINING_ROWS=5295

for required in "$PYTHON" "$TORCHRUN" "$TRAIN_LAUNCHER" "$VALIDATOR" \
                "$BENCHMARK" "$GATE" "$BASE_CHECKPOINT_PATH" \
                "$MANIFEST_DIR/summary.json" \
                "$EVAL_MANIFEST_DIR/summary.json" "$BASELINE" "$SMOKE_STATUS"; do
  if [[ ! -e "$required" ]]; then
    printf 'Required path not found: %s\n' "$required" >&2
    exit 2
  fi
done
if ! grep -q '^status=complete$' "$SMOKE_STATUS"; then
  printf 'Eight-GPU HF/reference smoke did not pass; refusing full launch.\n' >&2
  exit 2
fi
for target in "$TRAIN_ROOT" "$PIPELINE_DIR" "$CANDIDATE_DIR"; do
  if [[ -e "$target" ]]; then
    printf 'Refusing to overwrite experiment path: %s\n' "$target" >&2
    exit 2
  fi
done

mkdir -p "$PIPELINE_DIR" "$CANDIDATE_DIR"
phase=training
finished=0
record_exit() {
  local exit_code=$?
  if (( finished == 0 )); then
    printf 'status=failed\nfinished_utc=%s\nphase=%s\nexit_code=%s\n' \
      "$(date -u +%FT%TZ)" "$phase" "$exit_code" > "$PIPELINE_DIR/STATUS"
  fi
}
trap record_exit EXIT INT TERM
printf 'status=running\nstarted_utc=%s\nphase=%s\nsource_coverage_gb=301.7651036\ntraining_clips=5295\nworld_size=8\nexpected_optimizer_steps=662\nloss_estimator=reference_partition\nbranch_allocation=1_shortcut_7_flow\nroute_conditioning=false\n' \
  "$(date -u +%FT%TZ)" "$phase" > "$PIPELINE_DIR/STATUS"

env BASE_CHECKPOINT="$BASE_CHECKPOINT_PATH" RUN_NAME="$RUN_NAME" \
  "$TRAIN_LAUNCHER" > "$PIPELINE_DIR/training_launcher.log" 2>&1
checkpoint="$(awk -F= '$1 == "checkpoint" {print $2}' "$TRAIN_ROOT/STATUS")"
if [[ -z "$checkpoint" || "$checkpoint" == "none" || ! -d "$checkpoint" ]]; then
  printf 'Training completed without a checkpoint.\n' >&2
  exit 3
fi

phase=checkpoint_validation
printf 'status=running\nupdated_utc=%s\nphase=%s\ncheckpoint=%s\n' \
  "$(date -u +%FT%TZ)" "$phase" "$checkpoint" > "$PIPELINE_DIR/STATUS"
"$PYTHON" "$VALIDATOR" \
  --checkpoint "$checkpoint" \
  --train-log "$TRAIN_ROOT/train.log" \
  --output "$PIPELINE_DIR/checkpoint_validation.json" \
  --expected-steps "$EXPECTED_STEPS" \
  --world-size 8 \
  --training-rows "$TRAINING_ROWS" \
  > "$PIPELINE_DIR/checkpoint_validation.log" 2>&1

phase=matched_evaluation
printf 'status=running\nupdated_utc=%s\nphase=%s\ncheckpoint=%s\n' \
  "$(date -u +%FT%TZ)" "$phase" "$checkpoint" > "$PIPELINE_DIR/STATUS"
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/abose_sw/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export DS_IGNORE_CUDA_DETECTION=1
export TRITON_CACHE_DIR="/tmp/alpamayo15-$RUN_NAME-eval-triton"
mkdir -p "$TRITON_CACHE_DIR"
cd "$RECIPE_DIR"
"$TORCHRUN" --standalone --nproc_per_node=1 "$BENCHMARK" \
  --checkpoint "$checkpoint" \
  --config-name sft_stage2_trajectory_shortcut \
  --dataset "$LOCAL_DATASET" \
  --manifest-dir "$EVAL_MANIFEST_DIR" \
  --output-dir "$CANDIDATE_DIR" \
  --attention-backend eager \
  --steps 10 4 2 1 \
  --num-traj-samples 6 \
  --seed 42 \
  --warmup-samples 1 \
  > "$PIPELINE_DIR/evaluation.log" 2>&1

phase=two_step_gate
printf 'status=running\nupdated_utc=%s\nphase=%s\ncheckpoint=%s\n' \
  "$(date -u +%FT%TZ)" "$phase" "$checkpoint" > "$PIPELINE_DIR/STATUS"
"$PYTHON" "$GATE" \
  --baseline "$BASELINE" \
  --candidate "$CANDIDATE_DIR/benchmark_results.json" \
  --output "$CANDIDATE_DIR/two_step_gate.json" \
  > "$PIPELINE_DIR/two_step_gate.log" 2>&1

printf 'status=complete\nfinished_utc=%s\nphase=complete\nexit_code=0\ncheckpoint=%s\nbenchmark=%s\ngate=%s\n' \
  "$(date -u +%FT%TZ)" "$checkpoint" \
  "$CANDIDATE_DIR/benchmark_results.json" "$CANDIDATE_DIR/two_step_gate.json" \
  > "$PIPELINE_DIR/STATUS"
finished=1
