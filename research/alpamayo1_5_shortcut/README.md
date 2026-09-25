# Alpamayo 1.5 Stage-2 Shortcut Experiments

This directory is the handoff for an **unofficial research experiment** that
adds solver-step conditioning and Shortcut Models self-consistency to NVIDIA's
Alpamayo 1.5 Stage-2 Action Expert. The VLM is frozen during training.

The implementation is complete enough to reproduce the local pilot, but the
result did **not** pass the predeclared two-step quality-and-safety gate. It is
not an official NVIDIA benchmark and it is not evidence of safe autonomous
driving.

## Start here: HF streaming and paper-style EMA

The same branch now includes the HF on-demand dataset backend, partitioned
flow/shortcut training, FP32 EMA teachers, the targeted 10-to-5 experiment, and
the reference-aligned full-hierarchy A6 implementation plus A7 control code.

- [Portable setup and commands](docs/STREAMING_AND_PAPER_QUICKSTART.md)
- [Completed 128-clip results, including the missing-cell evaluations](docs/RESULTS_128CLIP_2026-09-25.md)
- [Detailed experiment ledger and limitations](docs/ABLATION_LEDGER_2026-09-20.md)
- [Pinned reference fixture and license](third_party/README.md)

A6 training/evaluation are complete; A7's implementation is present but its
training has **not** run. This is a reference-target port into Alpamayo, not an
identical reproduction of the image-model paper. No two-step safety success
has been established. All training manifests are route-less and contain only
sample-selection metadata, not camera/motion payloads.

## Original local pilot (historical)

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

The subsequent follow-up adds official Hugging Face on-demand loading and a
reference-style 1-in-8 loss allocation. Its deterministic manifest covers
5,295 new training clips across 301.765 GB of source shards; the unattended
one-epoch run is documented in [docs/HF_STREAMING_300GB.md](docs/HF_STREAMING_300GB.md).
Training completed all 662 updates and its checkpoint passed the structural,
finite-value, sampler-balance, and adapter-update validator. A controlled
eight-run follow-up suite completed final quality, seed sensitivity, an adapter
ablation, and a second fixed test split. Two-step Action-Expert inference was
5.14x faster, but minADE regressed 25.47% on validation and 30.75% on test
relative to the same checkpoint at ten steps. The two-step quality gate remains
closed. See the compact report under
`results/alpamayo15_hf300gb_reference_followups_20260913_r1/`.

An EMA-teacher follow-up completed the same 5,295-clip, 662-update schedule.
The EMA implementation, FP32 state, checkpoint restore, and inference selection
all passed validation, but it did not improve two-step shortcut learning. The
EMA-trained student regressed 32.72% in minADE from ten to two calls, while EMA
weights regressed 19.38% because their ten-step baseline still lagged after the
short schedule. The formal quality gate failed. See
[docs/EMA_TEACHER_EXPERIMENT_2026-09-15.md](docs/EMA_TEACHER_EXPERIMENT_2026-09-15.md)
and the compact report under
`results/alpamayo15_hf300gb_reference_ema_followups_20260915_r1/`.

## What this fork adds

| Area | File |
|---|---|
| Step-size adapter and shortcut target | `recipes/alpamayo1_5_sft/models/shortcut_modules.py` |
| Trainable model wrapper and losses | `recipes/alpamayo1_5_sft/models/shortcut_alpamayo_r1.py` |
| Shortcut model configuration | `recipes/alpamayo1_5_sft/configs/models/ar1_5_shortcut.yaml` |
| Route-less Stage-2 experiment | `recipes/alpamayo1_5_sft/configs/sft_stage2_trajectory_shortcut.yaml` |
| Navigation smoke experiment | `recipes/alpamayo1_5_sft/configs/sft_stage2_nav_shortcut.yaml` |
| Route-less PhysicalAI loader | `src/alpamayo/data/pai_trajectory.py` |
| Optional official HF streaming interface | `src/alpamayo/data/pai_utils.py` and `src/alpamayo/data/pai.py` |
| Reference 1-in-8 model configuration | `recipes/alpamayo1_5_sft/configs/models/ar1_5_shortcut_reference.yaml` |
| FP32 EMA-teacher model configuration | `recipes/alpamayo1_5_sft/configs/models/ar1_5_shortcut_reference_ema.yaml` |
| 300 GB streaming/reference experiment | `recipes/alpamayo1_5_sft/configs/sft_stage2_trajectory_shortcut_hf_reference.yaml` |
| 300 GB EMA-teacher experiment | `recipes/alpamayo1_5_sft/configs/sft_stage2_trajectory_shortcut_hf_reference_ema.yaml` |
| Paper/reference target layout and formulas | `recipes/alpamayo1_5_sft/models/paper_shortcut_targets.py` |
| Paper-style EMA expert and A7 target switch | `recipes/alpamayo1_5_sft/models/paper_shortcut_alpamayo.py` |
| 64-target DDP/accumulation training driver | `recipes/alpamayo1_5_sft/train_paper_ema.py` |
| Full-hierarchy model configuration | `recipes/alpamayo1_5_sft/configs/models/ar1_5_shortcut_paper_ema.yaml` |
| Resumable missing-cell evaluation runner | `research/alpamayo1_5_shortcut/scripts/fill_missing_evaluations.py` |
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
│   ├── HF_STREAMING_300GB.md
│   ├── FLOW_CONTROL_AND_128CLIP_EXPERIMENTS.md
│   ├── EMA_TEACHER_EXPERIMENT_2026-09-15.md
│   └── RESULTS.md
├── manifests/
│   ├── nav_smoke/
│   ├── route_less_19chunks/
│   └── hf_stream_300gb/
├── results/
│   ├── dataset_audit_summary.json
│   └── experiment_summary.json
└── scripts/
    ├── audit_local_physical_ai_trajectories.py
    ├── benchmark_inference_steps.py
    ├── benchmark_route_less.sh
    ├── evaluate_two_step_gate.py
    ├── make_route_less_pilot_manifests.py
    ├── make_hf_streaming_manifests.py
    ├── benchmark_hf_streaming.py
    ├── bootstrap_paired_step_regression.py
    ├── train_hf_reference.sh
    ├── train_hf_flow_control.sh
    ├── train_hf_reference_ema.sh
    ├── run_hf_reference_overnight.sh
    ├── run_ema_followup_suite.sh
    ├── run_reference_followup_suite.sh
    ├── run_flow_control_experiments.sh
    ├── summarize_flow_control_experiments.py
    ├── summarize_reference_followups.py
    ├── validate_flow_control_checkpoint.py
    ├── validate_reference_checkpoint.py
    ├── validate_ema_checkpoint.py
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
5. [Measured 300 GB streaming/reference experiment](docs/HF_STREAMING_300GB.md)
6. [Matched flow-only control and 128-clip evaluation](docs/FLOW_CONTROL_AND_128CLIP_EXPERIMENTS.md)
7. [EMA teacher implementation and matched experiment](docs/EMA_TEACHER_EXPERIMENT_2026-09-15.md)
8. [Streaming and paper-style training quickstart](docs/STREAMING_AND_PAPER_QUICKSTART.md)
9. [Updated 128-clip ablation table](docs/RESULTS_128CLIP_2026-09-25.md)
8. [Targeted 10-to-5 EMA experiment](docs/TEN_TO_FIVE_EMA_EXPERIMENT_2026-09-19.md)
9. [Complete ablation ledger and full-hierarchy paper-target EMA port](docs/ABLATION_LEDGER_2026-09-20.md)

The paper-target EMA port has a separate entry point:
`research/alpamayo1_5_shortcut/scripts/run_paper_ema_suite.py --run-dir /scratch/NEW_RUN_DIRECTORY`.
Use the recipe environment's Python. It refuses an existing run directory and
gates the full run on tests, a saved training smoke checkpoint, and real EMA
reload/inference. Training needs HF access; evaluations use local data offline.
See the ledger before launching: this is a multi-factor alignment experiment,
not a claim of an identical DiT/image-paper reproduction.
