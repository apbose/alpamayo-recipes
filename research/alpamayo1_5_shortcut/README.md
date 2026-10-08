# Alpamayo 1.5 Stage-2 Shortcut Experiments

## Current results: corrected native gold evaluation (2026-10-08)

Use [native_eval/](native_eval/) for current released-checkpoint/native inference
comparisons. The older Stage-2/eager benchmark remains a historical matched
research protocol; it is **not** NVIDIA's native inference reproduction.

Completed: R0 (released 10B), A6 and A8 at 10/5 steps, all 644 gold clips,
six candidates per clip, zero skipped clips. A6/A8 use EMA weights at update 249.

| Checkpoint | 10-step minADE m | 5-step minADE m | 10-to-5 degradation |
|---|---:|---:|---:|
| Released 10B R0 | 0.8282 | 0.8732 | +5.43% |
| A6 | 0.8386 | 0.8805 | +5.00% |
| A8 (full-set diagnostic) | 0.8374 | 0.8792 | +4.99% |

A8's available training pool contains 147 gold clips; 112 were actually used.
The shared **497-clip** cohort excludes both A6/A8 fine-tuning pools. Its
10-to-5 degradation is 3.99% for R0, 3.84% for A6 and 3.68% for A8.
Released-model pretraining overlap is unknown. Five steps roughly halves
Action Expert time, not full-model time; the released models get that benefit
too. A6/A8 have slightly worse absolute minADE than R0 at both solver counts.
Small mean differences alone do not establish significance.

- [Aggregate scores and caveats](results/native_gold644_20261008/summary.json)
- [Quality and measured latency for both cohorts](results/native_gold644_20261008/comparison.csv)
- [Exact public clip-selection manifest](manifests/gold644_native/gold644.json)
- [Fine-tuning overlap audit and shared cohort IDs](manifests/gold644_native/overlap_audit.json)

### What changed in evaluation

The corrected input labels every camera/frame, starts the assistant at
`<|cot_start|>`, and lets the VLM generate reasoning before
`<|traj_future_start|>`. The Action Expert can then attend to scene **and**
generated-reasoning K/V. Ground-truth future poses are reserved for scoring.

The old 10B gold run used a trajectory-only prefix, disabled camera/frame
labels, and used different precision/attention/seed settings. The corrected
released 10B 10-step minADE is **0.8282 m**, versus **1.4646 m** under that old
protocol. This combined correction is not an isolated camera-label or
reasoning ablation. A6/A8 were trained with trajectory-only conditioning, so
native reasoning-conditioned evaluation also introduces a conditioning shift.

Native settings: FP16, VLM FlashAttention 2, expert SDPA, seed 42 per clip,
four cameras x four frames, 16 history poses, 64 future poses, six stochastic
reasoning/trajectory candidates. Full-model and expert timers exclude data
loading/decoding; one warm-up is excluded. Saved analytical adapter buffers
remain FP32. 10/5-step adapter inputs (d=0.1/0.2) were not explicit points of
the dyadic training hierarchy. No collision/off-road safety pass is claimed.

### Portable native-evaluation commands

1. Use the Stage-2 Python environment (Torch 2.8 / Transformers 4.57.1 in the
   recorded run), with FlashAttention and the official `physical_ai_av`
   package available. Authenticate to Hugging Face locally; never place a
   token in the plan or Git.
2. Obtain NVIDIA's public [Alpamayo 1.5 source](https://github.com/NVlabs/alpamayo1.5)
   at commit `7a8f1c781a826f09be53e1e211f26e947ec18019`. The runner requires a
   clean checkout. This does not switch the experiment to Alpamayo 2.
3. Copy `native_eval/plan.example.json` to `native_eval/plan.local.json`
   (ignored by Git), and replace asset paths. Use the original native released
   10B checkpoint as `checkpoint`, not its A1-format wrapper; point
   `action_checkpoint` at A6/A8's completed checkpoint-249. Keep its
   `COMPLETE.json`, config, and parent `protocol.json`.

Run from the recipes repository root:

```bash
python research/alpamayo1_5_shortcut/native_eval/run_native_gold_suite.py \
  --plan research/alpamayo1_5_shortcut/native_eval/plan.local.json \
  --native-source /path/to/alpamayo1.5 \
  --output /scratch/native-gold-evaluation \
  --gpu 0 --validate-only
```

Then replace `--validate-only` with `--smoke-only` for real-sample checks.
For the full evaluation, omit both flags. Use the same plan/output/source to
resume; the runner refuses changed provenance. The plan's cohort audit is
verified against each trained checkpoint's manifest hash, so it must not be
reused for an unrelated training population.

The runner snapshots our evaluation/helper code, checks an idle GPU, runs
tests, then executes trained checkpoints before released references. It saves
each clip durably; a bounded stage failure preserves progress and does not
block other checkpoints. Use `tmux` or `nohup` for unattended execution.
Monitor `status.json`, each stage's `status.json`, and `summary.json`.
Do not interpret `incomplete` as a completed benchmark.

A prior HF CDN 404 surfaced as `BadZipFile` because Python's ZIP parser
wrapped the HTTP error. A fresh read and CRC checks succeeded. The retry
helper now inspects exception causes/contexts, refreshes file access and
retries the **same** pinned sample. Actual corruption/auth errors are not
silently treated as success; no clips are replaced or skipped.

## Historical experiments and original implementation

The sections below describe earlier protocols and should not be read as
current native-inference baseline reproductions.


New prepared experiment: [A8, fresh streaming data per A6 phase](docs/A8_FRESH_DATA_A6_2026-09-25.md)
keeps A6's 249-update budget and checks zero overlap with all existing heldouts.
Its 249-update training completed September 25. Matched final-EMA evaluation
completed at 10/5/4 steps (minADE 1.2355/1.2475/1.2789 m). The interrupted
two-step evaluation was resumed separately September 28; see the A8 document
for the live report and source-preserving resume artifacts.

This directory is the handoff for an **unofficial research experiment** that
adds solver-step conditioning and Shortcut Models self-consistency to NVIDIA's
Alpamayo 1.5 Stage-2 Action Expert. The VLM is frozen during training.

The implementation is complete enough to reproduce the local pilot, but the
result did **not** pass the predeclared two-step quality-and-safety gate. It is
not an official NVIDIA benchmark and it is not evidence of safe autonomous
driving.

## Start here: HF streaming and paper-style EMA

- [New R0/A6 public gold-644 evaluation protocol and leakage audit](docs/GOLD_644_EVALUATION_2026-09-29.md)

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
