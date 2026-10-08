# R0/A6 evaluation on the public gold-644 set

> **Protocol update:** This document records the historical Stage-2/eager protocol, not the corrected native inference reproduction. Current native scores and commands are in [the project README](../README.md#current-results-corrected-native-gold-evaluation-2026-10-08). The 644-clip population is unchanged; input labels, reasoning context, precision, attention and seed policy were corrected together.

This is an inference-only comparison requested September 29, 2026. It uses
the existing released checkpoint (R0) and A6 checkpoint-249 EMA weights. No
training, quantization, checkpoint conversion, or GPU power changes are involved.

## Data and leakage audit

NVIDIA's checked-in
`recipes/alpamayo1_5_quant/1005_7cam_gold_eval_metadb_public.parquet` contains
644 rows and 644 unique clip IDs. Preserve its row order and use one timestamp,
`t0_relative=5_100_000`, following that recipe's `eval.py` default. Do not infer
timestamps from the parquet's `event_t0s`. This is not the old 128-clip set;
the two sets have zero clip overlap.

The underlying camera and ego-motion features are read on demand from
`nvidia/PhysicalAI-Autonomous-Vehicles` at pinned revision
`33f9bf447ed3bcb7d545ce13f4226f824214fafb`. No full-dataset download is requested.
The runner checks all clip IDs and required feature availability, then validates
two real streamed samples before GPU smoke tests.

Hash-verified manifests and saved source-selection plans establish:

| Training run | Gold clips in manifest | Gold clips actually used |
|---|---:|---:|
| A6 | 0 | 0 |
| A8 | 147 | 112 |

A8 is **not** included. Evaluating A8 on all 644 would not be a fully held-out
fine-tuning evaluation. Future training should exclude the entire gold manifest
if this set is to remain held out. NVIDIA's proprietary pretraining overlap is
not independently audited here.

## Matched inference

- R0 on GPU 0; A6 on GPU 1, on the same host.
- Both models: 10, 5, 4, 2 solver steps, in that order; six candidates per clip.
- Same images, history, route-less prompt, generated VLM rollout, metric code.
- Eager attention and BF16 autocast; seed 42 reset before each solver count.
- One warm-up sample per solver count, excluded from metric/timing averages.
- A6 uses verified EMA weights after 249 updates. R0's wrapper adapter is zeroed.
- All 644 rows must complete for a solver count to enter the comparison table.
  Failed data access/inference is an error, not an instruction to skip a clip.
- CUDA-synchronized expert-sampler and full-model-call wall times; neither
  includes data fetching/decoding or metric computation. Six candidates are
  already included in one call, not six separate model calls.

This is a matched comparison using the public gold manifest, **not an exact
reproduction of NVIDIA's quantization evaluator**, which uses FP16 and reseeds
per clip. Keep our protocol and those published scores distinct.

## Runner and artifacts

Runner: `scripts/evaluate_gold_r0_a6.py` under this research directory. It reuses
`scripts/benchmark_inference_steps.py`, now with opt-in HF access. Local-access
defaults and the model call remain unchanged. Both two-clip smoke tests at
10/2 steps must pass before either full sweep begins.

```bash
recipes/alpamayo1_5_sft/a1_5_sft/bin/python \
  research/alpamayo1_5_shortcut/scripts/evaluate_gold_r0_a6.py \
  --output-dir /path/to/workspace/results/alpamayo15_gold644_r0_a6_20260929_r1 \
  --doc /path/to/workspace/docs/ALPAMAYO15_GOLD644_R0_A6_2026-09-29.md \
  --gpus 0 1
```

Launch from the repository root using the existing model/token cache environment:
`HF_HOME=/path/to/huggingface-cache` and
`HF_HUB_CACHE=/path/to/huggingface-cache/hub`. Streaming requires network
access; it is not an offline-data evaluation. Detached processes survive SSH
disconnects, but not shutdown/reboot of the remote machine.

The runner refuses existing output/report paths and has no overall wall-clock
timeout. It records commands, configuration/source hashes, train/gold audit,
hardware, logs, live `status.json`, per-sample progress, and an auto-updated
report. A failure is recorded and stops the suite; this runner does not silently
resume or overwrite a prior run. The benchmark's per-clip JSONL files preserve
partial progress for inspection; partial clips are not silently aggregated as
full results. A future restart needs an explicitly separate output directory.

Upon successful completion it writes `comparison.csv` and 100,000-resample
paired clip-bootstrap intervals for A6 minus R0 at each step count. Confidence
intervals cover clip sampling, not training-seed variation or vehicle safety.

The live report and `status.json`, not this protocol note, establish whether the
evaluation is running or complete. Pending results are never reported as zero.

## Launch verification

The detached runner started on `umb-b300-dp-160` at 07:56 UTC on September 29
(PID 119181). Before GPU smoke inference, CUDA tensor checks passed on GPUs 0/1,
all 644 clips passed the pinned-index/required-feature checks, and two streamed
samples passed the existing shape/finite-value validator. Their initial raw
load times were 22.06 s and 14.20 s; these are data-access checks, not model
inference latency. Forty-two CPU/regression tests passed (5 new runner/config
tests, 21 dataset/paper tests, 9 A8 report/resume tests, 7 missing-cell tests).

Run directory:
`/path/to/workspace/results/alpamayo15_gold644_r0_a6_20260929_r1`.
Live report:
`/path/to/workspace/docs/ALPAMAYO15_GOLD644_R0_A6_2026-09-29.md`.

## Recovery after HTTP 499 (September 29, 23:30 UTC)

The original run stopped at 22:14 UTC during a camera ZIP range read. R0 raised
HTTP 499 while loading chunk 3126; the parent then terminated A6. Both complete
10/5-step sweeps remain unchanged in `alpamayo15_gold644_r0_a6_20260929_r1`.
Four-step progress was partial; two steps had not started.

At the user's request, the same runner was extended with `--resume-from` and
opt-in `--hf-stream-max-attempts 6`. The latter changes only the HF filesystem's
process-local backoff policy (408/429/499/500/502/503/504, waits 2/4/8/16/30 s).
Installed library files, model weights, losses and metric code are not changed.
Authentication failures are not retried; exhausted retries still fail the run.

- Recovery directory: `/path/to/workspace/results/alpamayo15_gold644_r0_a6_20260929_r2`.
- Live recovery report: `/path/to/workspace/docs/ALPAMAYO15_GOLD644_R0_A6_RECOVERY_2026-09-29.md`.
- Detached parent PID at launch: 1047394; GPUs 0 and 1; no overall phase deadline.
- Eleven gold/retry CPU tests passed, including 499 recovery, bounded exhaustion,
  no auth retry, no result overwrites and no cross-run latency speedup.
- Original 10/5 artifacts passed manifest/checkpoint/per-clip/aggregate checks
  and were imported read-only. Missing counts are 4 and 2 for both checkpoints.
- Four raw streamed windows passed validation: indices 0, 1, 444 and 445,
  including the vicinity of the previous failure. Two-clip GPU smoke gates must
  pass before full evaluation starts; consult the live status, not this note.
- Incomplete counts restart from clip zero with seed 42. Partial RNG state is
  not guessed and partial old four-step predictions are not spliced into new ones.
- Completed-count merges retain per-step source paths/hashes. Network retries
  stay outside model-call timing. No historical/new-run speedup is inferred.
- On completion, the runner writes both A6-minus-R0 paired bootstrap intervals
  and within-checkpoint 10-to-5/4/2 degradation intervals.

The request was to restart on-demand data access and evaluation, not download
the entire dataset. No full archive mirror download or new training was launched.

Revised slide 4 includes all eight R0/A6 × 10/5/4/2 table rows. Unfinished rows
are editable `Pending` cells, never zeros. The old deck is preserved; revised
deck: `docs/slides/ALPAMAYO15_EXPERIMENT_UPDATE_2026-09-29_v2.pptx` in the workspace.
