#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
readonly WORKSPACE_ROOT="$(cd "$REPO_ROOT/../.." && pwd)"
readonly RECIPE_DIR="$REPO_ROOT/recipes/alpamayo1_5_sft"
readonly PYTHON_BIN="$RECIPE_DIR/a1_5_sft/bin/python"
readonly TORCHRUN_BIN="$RECIPE_DIR/a1_5_sft/bin/torchrun"
readonly TRAIN_SCRIPT="$SCRIPT_DIR/train_hf_flow_control.sh"
readonly VALIDATOR_SCRIPT="$SCRIPT_DIR/validate_flow_control_checkpoint.py"
readonly BENCHMARK_SCRIPT="$SCRIPT_DIR/benchmark_inference_steps.py"
readonly GATE_SCRIPT="$SCRIPT_DIR/evaluate_two_step_gate.py"
readonly BOOTSTRAP_SCRIPT="$SCRIPT_DIR/bootstrap_paired_step_regression.py"
readonly SUMMARIZER_SCRIPT="$SCRIPT_DIR/summarize_flow_control_experiments.py"
readonly STREAM_MANIFEST_DIR="$PROJECT_DIR/manifests/hf_stream_300gb"
readonly MANIFEST_32_DIR="$PROJECT_DIR/manifests/route_less_19chunks"
readonly MANIFEST_128_DIR="$PROJECT_DIR/manifests/route_less_19chunks_128eval"
readonly DATASET_ROOT=/home/scratch.abose_sw/alpamayo-assets/physical_ai_av
readonly RELEASED_CHECKPOINT_PATH=/home/scratch.abose_sw/alpamayo-assets/checkpoints/Alpamayo-1.5-10B-A1-format
readonly SHORTCUT_CHECKPOINT_PATH=/home/scratch.abose_sw/alpamayo-assets/runs/alpamayo15_hf300gb_reference_5295clips_1epoch_20260913_r2/trainer_output/checkpoint-662
readonly PRIOR_SUITE_DIR=/home/abose_sw/alphamayo/results/alpamayo15_hf300gb_reference_followups_20260913_r1
readonly FLOW_RUN_NAME="${FLOW_RUN_NAME:-alpamayo15_hf300gb_flow_control_5295clips_1epoch_20260914_r1}"
readonly FLOW_RUN_ROOT=/home/scratch.abose_sw/alpamayo-assets/runs/$FLOW_RUN_NAME
readonly SUITE_RUN_NAME="${SUITE_RUN_NAME:-alpamayo15_hf300gb_flow_control_followups_20260914_r1}"
readonly SUITE_DIR="$WORKSPACE_ROOT/results/$SUITE_RUN_NAME"
readonly LOG_DIR="$SUITE_DIR/logs"

for required in "$PYTHON_BIN" "$TORCHRUN_BIN" "$TRAIN_SCRIPT" \
                "$VALIDATOR_SCRIPT" "$BENCHMARK_SCRIPT" "$GATE_SCRIPT" \
                "$BOOTSTRAP_SCRIPT" "$SUMMARIZER_SCRIPT" "$DATASET_ROOT" \
                "$RELEASED_CHECKPOINT_PATH" "$SHORTCUT_CHECKPOINT_PATH" \
                "$STREAM_MANIFEST_DIR/train.json" "$STREAM_MANIFEST_DIR/val.json" \
                "$MANIFEST_32_DIR/val.json" "$MANIFEST_128_DIR/val.json" \
                "$PRIOR_SUITE_DIR/released_seed42/benchmark_results.json"; do
  if [[ ! -e "$required" ]]; then
    printf 'Required path not found: %s\n' "$required" >&2
    exit 2
  fi
done
for target in "$FLOW_RUN_ROOT" "$SUITE_DIR"; do
  if [[ -e "$target" ]]; then
    printf 'Refusing to overwrite experiment path: %s\n' "$target" >&2
    exit 2
  fi
done
mkdir -p "$LOG_DIR"

phase=initializing
completed_evaluations=0
finished=0
record_exit() {
  local exit_code=$?
  if (( finished == 0 )); then
    printf 'status=failed\nfinished_utc=%s\nphase=%s\nexit_code=%s\ncompleted_evaluations=%s\n' \
      "$(date -u +%FT%TZ)" "$phase" "$exit_code" "$completed_evaluations" \
      > "$SUITE_DIR/STATUS"
  fi
}
trap record_exit EXIT INT TERM

write_running_status() {
  printf 'status=running\nupdated_utc=%s\nphase=%s\ncompleted_evaluations=%s\ntotal_evaluations=7\n' \
    "$(date -u +%FT%TZ)" "$phase" "$completed_evaluations" \
    > "$SUITE_DIR/STATUS"
}

phase=flow_control_training
write_running_status
env BASE_CHECKPOINT="$RELEASED_CHECKPOINT_PATH" \
  FLOW_RUN_NAME="$FLOW_RUN_NAME" \
  NPROC_PER_NODE=8 \
  DATALOADER_WORKERS=1 \
  SAVE_STRATEGY=epoch \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  "$TRAIN_SCRIPT" > "$LOG_DIR/flow_training_launcher.log" 2>&1

flow_checkpoint="$(awk -F= '$1 == "checkpoint" {print $2}' "$FLOW_RUN_ROOT/STATUS")"
if [[ -z "$flow_checkpoint" || "$flow_checkpoint" == "none" || ! -d "$flow_checkpoint" ]]; then
  printf 'Flow-only training completed without a checkpoint.\n' >&2
  exit 3
fi

phase=flow_checkpoint_validation
write_running_status
"$PYTHON_BIN" "$VALIDATOR_SCRIPT" \
  --checkpoint "$flow_checkpoint" \
  --train-log "$FLOW_RUN_ROOT/train.log" \
  --output "$SUITE_DIR/flow_checkpoint_validation.json" \
  --expected-steps 662 \
  --world-size 8 \
  --training-rows 5295 \
  > "$LOG_DIR/flow_checkpoint_validation.log" 2>&1

export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}"
export HF_HOME=/home/abose_sw/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export DS_IGNORE_CUDA_DETECTION=1
export TRITON_CACHE_DIR="/tmp/alpamayo15-$SUITE_RUN_NAME-eval-triton"
mkdir -p "$TRITON_CACHE_DIR"

run_benchmark() {
  local label=$1
  local checkpoint=$2
  local manifest_dir=$3
  shift 3
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
    --manifest-dir "$manifest_dir" \
    --output-dir "$output_dir" \
    --eval-split val \
    --attention-backend eager \
    --num-traj-samples 6 \
    --seed 42 \
    --warmup-samples 1 \
    "$@" > "$LOG_DIR/$label.log" 2>&1
  completed_evaluations=$((completed_evaluations + 1))
  write_running_status
}

# Same 32-clip protocol as the completed shortcut/released suite.
run_benchmark flow_control_32 "$flow_checkpoint" "$MANIFEST_32_DIR" \
  --steps 10 8 4 2 1
"$PYTHON_BIN" "$GATE_SCRIPT" \
  --baseline "$PRIOR_SUITE_DIR/released_seed42/benchmark_results.json" \
  --candidate "$SUITE_DIR/flow_control_32/benchmark_results.json" \
  --output "$SUITE_DIR/flow_control_32/two_step_gate_vs_released.json" \
  > "$LOG_DIR/flow_control_32_gate.log" 2>&1

# Larger matched evidence on 128 deterministic held-out validation clips.
run_benchmark shortcut_128 "$SHORTCUT_CHECKPOINT_PATH" "$MANIFEST_128_DIR" \
  --steps 10 4 2
run_benchmark released_128 "$RELEASED_CHECKPOINT_PATH" "$MANIFEST_128_DIR" \
  --steps 10 4 2
run_benchmark flow_control_128 "$flow_checkpoint" "$MANIFEST_128_DIR" \
  --steps 10 4 2
"$PYTHON_BIN" "$GATE_SCRIPT" \
  --baseline "$SUITE_DIR/released_128/benchmark_results.json" \
  --candidate "$SUITE_DIR/shortcut_128/benchmark_results.json" \
  --output "$SUITE_DIR/shortcut_128/two_step_gate_vs_released.json" \
  > "$LOG_DIR/shortcut_128_gate.log" 2>&1
"$PYTHON_BIN" "$GATE_SCRIPT" \
  --baseline "$SUITE_DIR/released_128/benchmark_results.json" \
  --candidate "$SUITE_DIR/flow_control_128/benchmark_results.json" \
  --output "$SUITE_DIR/flow_control_128/two_step_gate_vs_released.json" \
  > "$LOG_DIR/flow_control_128_gate.log" 2>&1

# Inference-only scale intervention on the learned d adapter.
run_benchmark adapter_scale2 "$SHORTCUT_CHECKPOINT_PATH" "$MANIFEST_32_DIR" \
  --steps 10 2 1 --step-size-adapter-scale 2
run_benchmark adapter_scale4 "$SHORTCUT_CHECKPOINT_PATH" "$MANIFEST_32_DIR" \
  --steps 10 2 1 --step-size-adapter-scale 4
run_benchmark adapter_scale8 "$SHORTCUT_CHECKPOINT_PATH" "$MANIFEST_32_DIR" \
  --steps 10 2 1 --step-size-adapter-scale 8

phase=paired_bootstrap
write_running_status
"$PYTHON_BIN" "$BOOTSTRAP_SCRIPT" \
  --benchmark "$SUITE_DIR/flow_control_32/benchmark_results.json" \
  --output "$SUITE_DIR/bootstrap_flow_32_10_vs_2.json" \
  > "$LOG_DIR/bootstrap_flow_32.log" 2>&1
"$PYTHON_BIN" "$BOOTSTRAP_SCRIPT" \
  --benchmark "$SUITE_DIR/shortcut_128/benchmark_results.json" \
  --output "$SUITE_DIR/bootstrap_shortcut_128_10_vs_2.json" \
  > "$LOG_DIR/bootstrap_shortcut_128.log" 2>&1
"$PYTHON_BIN" "$BOOTSTRAP_SCRIPT" \
  --benchmark "$SUITE_DIR/flow_control_128/benchmark_results.json" \
  --output "$SUITE_DIR/bootstrap_flow_128_10_vs_2.json" \
  > "$LOG_DIR/bootstrap_flow_128.log" 2>&1

phase=summarizing
write_running_status
"$PYTHON_BIN" "$SUMMARIZER_SCRIPT" \
  --suite-dir "$SUITE_DIR" \
  --prior-suite-dir "$PRIOR_SUITE_DIR" \
  > "$LOG_DIR/summarizer.log" 2>&1

printf 'status=complete\nfinished_utc=%s\nphase=complete\ncompleted_evaluations=7\ntotal_evaluations=7\nflow_checkpoint=%s\nreport=%s\nsummary=%s\n' \
  "$(date -u +%FT%TZ)" "$flow_checkpoint" "$SUITE_DIR/REPORT.md" \
  "$SUITE_DIR/summary.json" > "$SUITE_DIR/STATUS"
finished=1
