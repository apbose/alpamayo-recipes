# Alpamayo 1.5 Stage-2 Shortcut Pilot

This directory is the handoff for an **unofficial research experiment** that
adds solver-step conditioning and Shortcut Models self-consistency to NVIDIA's
Alpamayo 1.5 Stage-2 Action Expert. The VLM is frozen during training.

The implementation is complete enough to reproduce the local pilot, but the
result did **not** pass the predeclared two-step quality-and-safety gate. It is
not an official NVIDIA benchmark and it is not evidence of safe autonomous
driving.

## Current outcome

- Pilot result base: `NVlabs/alpamayo-recipes` commit `6172113`; the sharing branch is rebased onto upstream commit `670b551`.
- Model: Alpamayo 1.5 10B, with an A1-format configuration wrapper over the
  unchanged released weights.
- Data: approximately 100 GB across 19 PhysicalAI-AV chunks.
- Training: 10,692 route-less windows from 891 official-training clips, one
  epoch, batch size one, NVIDIA B300.
- Evaluation: 32 clip-disjoint official-validation clips, one timestamp per
  clip, six stochastic candidates, seed 42, eager attention.
- Efficiency: two Action-Expert calls were 5.16x faster than ten calls.
- Gate: closed. Relative to the trained ten-step path, two-step minADE was
  14.07% worse and corner distance was 13.73% worse; safety was not evaluated.

See [docs/RESULTS.md](docs/RESULTS.md) for the full matched table.

## What this fork adds

| Area | File |
|---|---|
| Step-size adapter and shortcut target | `recipes/alpamayo1_5_sft/models/shortcut_modules.py` |
| Trainable model wrapper and losses | `recipes/alpamayo1_5_sft/models/shortcut_alpamayo_r1.py` |
| Shortcut model configuration | `recipes/alpamayo1_5_sft/configs/models/ar1_5_shortcut.yaml` |
| Route-less Stage-2 experiment | `recipes/alpamayo1_5_sft/configs/sft_stage2_trajectory_shortcut.yaml` |
| Navigation smoke experiment | `recipes/alpamayo1_5_sft/configs/sft_stage2_nav_shortcut.yaml` |
| Route-less PhysicalAI loader | `src/alpamayo/data/pai_trajectory.py` |
| Unit and configuration tests | `recipes/alpamayo1_5_sft/tests/test_shortcut_*.py` and `test_pai_trajectory.py` |
| Fixed manifests and compact evidence | this directory |

## Repository map

```text
research/alpamayo1_5_shortcut/
├── README.md
├── docs/
│   ├── DATA_AND_SPLITS.md
│   ├── METHOD.md
│   ├── ON_DEMAND_DATA.md
│   └── RESULTS.md
├── manifests/
│   ├── nav_smoke/
│   └── route_less_19chunks/
├── results/
│   ├── dataset_audit_summary.json
│   └── experiment_summary.json
└── scripts/
    ├── audit_local_physical_ai_trajectories.py
    ├── benchmark_inference_steps.py
    ├── benchmark_route_less.sh
    ├── evaluate_two_step_gate.py
    ├── make_route_less_pilot_manifests.py
    ├── run_tests.sh
    ├── train_route_less.sh
    ├── validate_local_physical_ai_trajectory_audit.py
    ├── validate_shortcut_checkpoint.py
    └── summarize_shortcut_training.py
```

## Environment

Follow `recipes/alpamayo1_5_sft/README.md` to create the recipe environment.
The wrappers below default to:

```text
recipes/alpamayo1_5_sft/a1_5_sft/bin/{python,torchrun}
```

Override that location with `ALPAMAYO_ENV=/path/to/environment`.

The released checkpoint must first be wrapped in the older A1 configuration
namespace expected by Stage 2:

```bash
python scripts/convert_checkpoint.py to-a1 \
  --input /path/to/Alpamayo-1.5-10B \
  --output /path/to/Alpamayo-1.5-10B-A1-format
```

This changes configuration/class names and symlinks the same weight shards; it
does not retrain or numerically convert the model.

## Reproduce

Run the CPU-oriented unit/configuration tests:

```bash
research/alpamayo1_5_shortcut/scripts/run_tests.sh
```

Launch the route-less one-epoch training run:

```bash
DATASET_DIR=/path/to/physical_ai_av \
BASE_CHECKPOINT=/path/to/Alpamayo-1.5-10B-A1-format \
OUTPUT_DIR=/scratch/runs/alpamayo1_5_shortcut \
CUDA_VISIBLE_DEVICES=0 \
research/alpamayo1_5_shortcut/scripts/train_route_less.sh
```

Benchmark either the released wrapper or a trained checkpoint:

```bash
DATASET_DIR=/path/to/physical_ai_av \
CHECKPOINT=/path/to/checkpoint \
OUTPUT_DIR=/scratch/results/shortcut_benchmark \
CUDA_VISIBLE_DEVICES=0 \
research/alpamayo1_5_shortcut/scripts/benchmark_route_less.sh
```

The checked-in scripts never contain tokens. Authenticate with Hugging Face
using its normal local credential store, and keep datasets/checkpoints outside
this Git repository.

## Read next

1. [Data and fixed splits](docs/DATA_AND_SPLITS.md)
2. [Method and loss](docs/METHOD.md)
3. [Matched results and gate](docs/RESULTS.md)
4. [What “on-demand” Hugging Face loading does and does not establish](docs/ON_DEMAND_DATA.md)
