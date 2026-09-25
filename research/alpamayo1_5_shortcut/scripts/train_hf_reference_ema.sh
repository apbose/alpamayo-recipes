#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG_NAME="sft_stage2_trajectory_shortcut_hf_reference_ema"
export RUN_NAME="${RUN_NAME:-alpamayo15_hf300gb_reference_ema_1epoch}"

exec "$SCRIPT_DIR/train_hf_reference.sh"
