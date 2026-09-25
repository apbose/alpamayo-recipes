#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
readonly WORKSPACE_ROOT="$(cd "$REPO_ROOT/../.." && pwd)"
readonly RECIPE_DIR="$REPO_ROOT/recipes/alpamayo1_5_sft"
readonly PYTHON_BIN="$RECIPE_DIR/a1_5_sft/bin/python"
readonly TORCHRUN_BIN="$RECIPE_DIR/a1_5_sft/bin/torchrun"
readonly BENCHMARK_SCRIPT="$SCRIPT_DIR/benchmark_inference_steps.py"
readonly GATE_SCRIPT="$SCRIPT_DIR/evaluate_two_step_gate.py"
readonly SUMMARIZER_SCRIPT="$SCRIPT_DIR/summarize_reference_followups.py"
readonly MANIFEST_DIR="$PROJECT_DIR/manifests/route_less_19chunks"
readonly DATASET_ROOT=/home/scratch.abose_sw/alpamayo-assets/physical_ai_av
readonly REFERENCE_CHECKPOINT_PATH=/home/scratch.abose_sw/alpamayo-assets/runs/alpamayo15_hf300gb_reference_5295clips_1epoch_20260913_r2/trainer_output/checkpoint-662
readonly RELEASED_CHECKPOINT_PATH=/home/scratch.abose_sw/alpamayo-assets/checkpoints/Alpamayo-1.5-10B-A1-format
readonly V1_CHECKPOINT_PATH=/home/scratch.abose_sw/alpamayo-assets/runs/alpamayo1_5_route_less_full_10692rows_1epoch/trainer_output/checkpoint-10692
readonly SUITE_RUN_NAME="${SUITE_RUN_NAME:-alpamayo15_hf300gb_reference_followups_20260913}"
readonly SUITE_DIR="$WORKSPACE_ROOT/results/$SUITE_RUN_NAME"
readonly LOG_DIR="$SUITE_DIR/logs"

for required in "$PYTHON_BIN" "$TORCHRUN_BIN" "$BENCHMARK_SCRIPT" \
                "$GATE_SCRIPT" "$SUMMARIZER_SCRIPT" "$MANIFEST_DIR/summary.json" \
                "$MANIFEST_DIR/val.json" "$MANIFEST_DIR/test.json" "$DATASET_ROOT" \
                "$REFERENCE_CHECKPOINT_PATH" "$RELEASED_CHECKPOINT_PATH" \
                "$V1_CHECKPOINT_PATH"; do
  if [[ ! -e "$required" ]]; then
    printf 'Required path not found: %s\n' "$required" >&2
    exit 2
  fi
done
if [[ -e "$SUITE_DIR" ]]; then
  printf 'Refusing to overwrite experiment suite: %s\n' "$SUITE_DIR" >&2
  exit 2
fi
mkdir -p "$LOG_DIR"

phase=initializing
finished=0
record_exit() {
  local exit_code=$?
  if (( finished == 0 )); then
    printf 'status=failed\nfinished_utc=%s\nphase=%s\nexit_code=%s\n' \
      "$(date -u +%FT%TZ)" "$phase" "$exit_code" > "$SUITE_DIR/STATUS"
  fi
}
trap record_exit EXIT INT TERM

write_running_status() {
  printf 'status=running\nupdated_utc=%s\nphase=%s\ncompleted_experiments=%s\ntotal_experiments=8\nexecution=sequential_single_gpu\ngpu=0\n' \
    "$(date -u +%FT%TZ)" "$phase" "$completed_experiments" > "$SUITE_DIR/STATUS"
}

export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}"
export HF_HOME=/home/abose_sw/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export DS_IGNORE_CUDA_DETECTION=1
export TRITON_CACHE_DIR="/tmp/alpamayo15-$SUITE_RUN_NAME-triton"
mkdir -p "$TRITON_CACHE_DIR"

completed_experiments=0
printf 'status=running\nstarted_utc=%s\nphase=%s\ncompleted_experiments=0\ntotal_experiments=8\nexecution=sequential_single_gpu\ngpu=0\n' \
  "$(date -u +%FT%TZ)" "$phase" > "$SUITE_DIR/STATUS"

run_benchmark() {
  local label=$1
  local checkpoint=$2
  local seed=$3
  local eval_split=$4
  shift 4
  local output_dir="$SUITE_DIR/$label"
  phase="$label"
  write_running_status
  if [[ -e "$output_dir" ]]; then
    printf 'Refusing existing benchmark output: %s\n' "$output_dir" >&2
    exit 2
  fi
  cd "$RECIPE_DIR"
  "$TORCHRUN_BIN" --standalone --nproc_per_node=1 "$BENCHMARK_SCRIPT" \
    --checkpoint "$checkpoint" \
    --config-name sft_stage2_trajectory_shortcut \
    --dataset "$DATASET_ROOT" \
    --manifest-dir "$MANIFEST_DIR" \
    --output-dir "$output_dir" \
    --eval-split "$eval_split" \
    --attention-backend eager \
    --steps 10 8 4 2 1 \
    --num-traj-samples 6 \
    --seed "$seed" \
    --warmup-samples 1 \
    "$@" > "$LOG_DIR/$label.log" 2>&1
  completed_experiments=$((completed_experiments + 1))
  write_running_status
}

# Primary matched comparison. Run these first so a valid gate is available early.
run_benchmark candidate_seed42 "$REFERENCE_CHECKPOINT_PATH" 42 val
run_benchmark released_seed42 "$RELEASED_CHECKPOINT_PATH" 42 val
phase=validation_two_step_gate
write_running_status
"$PYTHON_BIN" "$GATE_SCRIPT" \
  --baseline "$SUITE_DIR/released_seed42/benchmark_results.json" \
  --candidate "$SUITE_DIR/candidate_seed42/benchmark_results.json" \
  --output "$SUITE_DIR/candidate_seed42/two_step_gate_vs_released.json" \
  > "$LOG_DIR/validation_two_step_gate.log" 2>&1

# Stochastic repeatability on the exact same fixed validation clips.
run_benchmark candidate_seed43 "$REFERENCE_CHECKPOINT_PATH" 43 val
run_benchmark candidate_seed44 "$REFERENCE_CHECKPOINT_PATH" 44 val

# Isolate whether the learned d-conditioning branch matters at inference.
run_benchmark candidate_zero_adapter_seed42 "$REFERENCE_CHECKPOINT_PATH" 42 val \
  --zero-step-size-adapter

# Descriptive comparison with the original per-sample weighted shortcut pilot.
run_benchmark v1_seed42 "$V1_CHECKPOINT_PATH" 42 val

# Independent fixed test clips: candidate and released checkpoint, matched protocol.
run_benchmark candidate_test_seed42 "$REFERENCE_CHECKPOINT_PATH" 42 test
run_benchmark released_test_seed42 "$RELEASED_CHECKPOINT_PATH" 42 test
phase=test_two_step_gate
write_running_status
"$PYTHON_BIN" "$GATE_SCRIPT" \
  --baseline "$SUITE_DIR/released_test_seed42/benchmark_results.json" \
  --candidate "$SUITE_DIR/candidate_test_seed42/benchmark_results.json" \
  --output "$SUITE_DIR/candidate_test_seed42/two_step_gate_vs_released.json" \
  > "$LOG_DIR/test_two_step_gate.log" 2>&1

phase=summarizing
write_running_status
"$PYTHON_BIN" "$SUMMARIZER_SCRIPT" --suite-dir "$SUITE_DIR" \
  > "$LOG_DIR/summarizer.log" 2>&1

printf 'status=complete\nfinished_utc=%s\nphase=complete\ncompleted_experiments=8\ntotal_experiments=8\nreport=%s\nsummary=%s\n' \
  "$(date -u +%FT%TZ)" "$SUITE_DIR/REPORT.md" "$SUITE_DIR/summary.json" \
  > "$SUITE_DIR/STATUS"
finished=1
