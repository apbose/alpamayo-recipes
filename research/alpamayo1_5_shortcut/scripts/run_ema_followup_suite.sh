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
readonly VALIDATOR_SCRIPT="$SCRIPT_DIR/validate_ema_checkpoint.py"
readonly REFERENCE_VALIDATOR_SCRIPT="$SCRIPT_DIR/validate_reference_checkpoint.py"
readonly GATE_SCRIPT="$SCRIPT_DIR/evaluate_two_step_gate.py"
readonly BOOTSTRAP_SCRIPT="$SCRIPT_DIR/bootstrap_paired_step_regression.py"
readonly SUMMARIZER_SCRIPT="$SCRIPT_DIR/summarize_ema_experiment.py"
readonly TRAIN_RUN="${TRAIN_RUN:-/home/scratch.abose_sw/alpamayo-assets/runs/alpamayo15_hf300gb_reference_ema_5295clips_1epoch_20260915_r1}"
readonly EMA_CHECKPOINT="${EMA_CHECKPOINT:-$TRAIN_RUN/trainer_output/checkpoint-662}"
readonly TRAIN_LOG="${TRAIN_LOG:-$TRAIN_RUN/train.log}"
readonly CONTROLS_DIR="${CONTROLS_DIR:-$WORKSPACE_ROOT/results/alpamayo15_hf300gb_flow_control_followups_20260914_r1}"
readonly RELEASED_RESULT="$CONTROLS_DIR/released_128/benchmark_results.json"
readonly MANIFEST_DIR="$PROJECT_DIR/manifests/route_less_19chunks_128eval"
readonly DATASET_ROOT="${DATASET_ROOT:-/home/scratch.abose_sw/alpamayo-assets/physical_ai_av}"
readonly RUN_NAME="${RUN_NAME:-alpamayo15_hf300gb_reference_ema_followups_20260915_r1}"
readonly SUITE_DIR="$WORKSPACE_ROOT/results/$RUN_NAME"
readonly LOG_DIR="$SUITE_DIR/logs"
readonly COMPACT_DIR="$PROJECT_DIR/results/$RUN_NAME"

for required in "$PYTHON_BIN" "$TORCHRUN_BIN" "$BENCHMARK_SCRIPT" \
                "$VALIDATOR_SCRIPT" "$REFERENCE_VALIDATOR_SCRIPT" \
                "$GATE_SCRIPT" "$BOOTSTRAP_SCRIPT" \
                "$SUMMARIZER_SCRIPT" "$EMA_CHECKPOINT" "$TRAIN_LOG" \
                "$CONTROLS_DIR/shortcut_128/benchmark_results.json" \
                "$CONTROLS_DIR/flow_control_128/benchmark_results.json" \
                "$RELEASED_RESULT" "$MANIFEST_DIR/summary.json" \
                "$MANIFEST_DIR/val.json" "$DATASET_ROOT"; do
  if [[ ! -e "$required" ]]; then
    printf 'Required path not found: %s\n' "$required" >&2
    exit 2
  fi
done
if [[ -e "$SUITE_DIR" || -e "$COMPACT_DIR" ]]; then
  printf 'Refusing existing EMA result directory: %s or %s\n' \
    "$SUITE_DIR" "$COMPACT_DIR" >&2
  exit 2
fi
mkdir -p "$LOG_DIR"

phase=initializing
finished=0
completed_evaluations=0
record_exit() {
  local exit_code=$?
  if (( finished == 0 )); then
    printf 'status=failed\nfinished_utc=%s\nphase=%s\nexit_code=%s\n' \
      "$(date -u +%FT%TZ)" "$phase" "$exit_code" > "$SUITE_DIR/STATUS"
  fi
}
trap record_exit EXIT INT TERM

write_running_status() {
  printf 'status=running\nupdated_utc=%s\nphase=%s\ncompleted_evaluations=%s\ntotal_evaluations=2\nexecution=sequential_single_gpu\ngpu=0\n' \
    "$(date -u +%FT%TZ)" "$phase" "$completed_evaluations" > "$SUITE_DIR/STATUS"
}

export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}"
export HF_HOME=/home/abose_sw/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export DS_IGNORE_CUDA_DETECTION=1
export TRITON_CACHE_DIR="/tmp/alpamayo15-$RUN_NAME-triton"
mkdir -p "$TRITON_CACHE_DIR"

printf 'status=running\nstarted_utc=%s\nphase=%s\ncompleted_evaluations=0\ntotal_evaluations=2\nexecution=sequential_single_gpu\ngpu=0\n' \
  "$(date -u +%FT%TZ)" "$phase" > "$SUITE_DIR/STATUS"

phase=checkpoint_validation
write_running_status
"$PYTHON_BIN" "$VALIDATOR_SCRIPT" \
  --checkpoint "$EMA_CHECKPOINT" \
  --train-log "$TRAIN_LOG" \
  --output "$SUITE_DIR/ema_checkpoint_validation.json" \
  --expected-steps 662 > "$LOG_DIR/checkpoint_validation.log" 2>&1
"$PYTHON_BIN" "$REFERENCE_VALIDATOR_SCRIPT" \
  --checkpoint "$EMA_CHECKPOINT" \
  --train-log "$TRAIN_LOG" \
  --output "$SUITE_DIR/reference_checkpoint_validation.json" \
  --expected-steps 662 \
  --world-size 8 \
  --training-rows 5295 > "$LOG_DIR/reference_checkpoint_validation.log" 2>&1

run_benchmark() {
  local label=$1
  local inference_weights=$2
  shift 2
  phase="$label"
  write_running_status
  cd "$RECIPE_DIR"
  "$TORCHRUN_BIN" --standalone --nproc_per_node=1 "$BENCHMARK_SCRIPT" \
    --checkpoint "$EMA_CHECKPOINT" \
    --config-name sft_stage2_trajectory_shortcut_reference_ema \
    --shortcut-inference-weights "$inference_weights" \
    --expected-ema-updates 662 \
    --dataset "$DATASET_ROOT" \
    --manifest-dir "$MANIFEST_DIR" \
    --output-dir "$SUITE_DIR/$label" \
    --eval-split val \
    --attention-backend eager \
    --num-traj-samples 6 \
    --seed 42 \
    --warmup-samples 1 \
    --steps "$@" > "$LOG_DIR/$label.log" 2>&1
  completed_evaluations=$((completed_evaluations + 1))
  write_running_status
}

run_benchmark ema_weights_128 ema 10 4 2 1
run_benchmark student_weights_128 student 10 2

phase=two_step_gate
write_running_status
"$PYTHON_BIN" "$GATE_SCRIPT" \
  --baseline "$RELEASED_RESULT" \
  --candidate "$SUITE_DIR/ema_weights_128/benchmark_results.json" \
  --output "$SUITE_DIR/two_step_gate_vs_released.json" \
  > "$LOG_DIR/two_step_gate.log" 2>&1

phase=paired_bootstrap
write_running_status
"$PYTHON_BIN" "$BOOTSTRAP_SCRIPT" \
  --benchmark "$SUITE_DIR/ema_weights_128/benchmark_results.json" \
  --output "$SUITE_DIR/bootstrap_ema_10_vs_2.json" \
  > "$LOG_DIR/bootstrap_ema.log" 2>&1
"$PYTHON_BIN" "$BOOTSTRAP_SCRIPT" \
  --benchmark "$SUITE_DIR/student_weights_128/benchmark_results.json" \
  --output "$SUITE_DIR/bootstrap_student_10_vs_2.json" \
  > "$LOG_DIR/bootstrap_student.log" 2>&1

phase=summarizing
write_running_status
"$PYTHON_BIN" "$SUMMARIZER_SCRIPT" \
  --suite-dir "$SUITE_DIR" \
  --controls-dir "$CONTROLS_DIR" > "$LOG_DIR/summarizer.log" 2>&1

phase=publishing_compact_results
write_running_status
mkdir -p "$COMPACT_DIR/ema_weights_128" "$COMPACT_DIR/student_weights_128"
cp "$SUITE_DIR/REPORT.md" "$SUITE_DIR/summary.json" \
   "$SUITE_DIR/ema_checkpoint_validation.json" \
   "$SUITE_DIR/reference_checkpoint_validation.json" \
   "$SUITE_DIR/two_step_gate_vs_released.json" \
   "$SUITE_DIR/two_step_gate_vs_released.md" \
   "$SUITE_DIR/bootstrap_ema_10_vs_2.json" \
   "$SUITE_DIR/bootstrap_ema_10_vs_2.md" \
   "$SUITE_DIR/bootstrap_student_10_vs_2.json" \
   "$SUITE_DIR/bootstrap_student_10_vs_2.md" "$COMPACT_DIR/"
cp "$SUITE_DIR/ema_weights_128/benchmark_results.json" \
   "$COMPACT_DIR/ema_weights_128/"
cp "$SUITE_DIR/student_weights_128/benchmark_results.json" \
   "$COMPACT_DIR/student_weights_128/"

printf 'status=complete\nfinished_utc=%s\nphase=complete\ncompleted_evaluations=2\ntotal_evaluations=2\ncheckpoint=%s\nreport=%s\nsummary=%s\n' \
  "$(date -u +%FT%TZ)" "$EMA_CHECKPOINT" "$SUITE_DIR/REPORT.md" \
  "$SUITE_DIR/summary.json" > "$SUITE_DIR/STATUS"
cp "$SUITE_DIR/STATUS" "$COMPACT_DIR/STATUS"
finished=1
