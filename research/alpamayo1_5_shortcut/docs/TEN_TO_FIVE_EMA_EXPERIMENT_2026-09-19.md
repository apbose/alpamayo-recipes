# Targeted Alpamayo 1.5 10-to-5 EMA Shortcut Experiment

Date: 2026-09-19

## Purpose

The previous shortcut experiments trained the dyadic hierarchy

\[
d \in \{0.125, 0.25, 0.5, 1.0\},
\]

which directly covers 8, 4, 2, and 1 solver calls, but not the released
10-step interval \(d=0.1\). This experiment is intentionally narrower: teach a
five-step student directly from the released model's ten-step scale.

This is an additional experiment. It does not change or overwrite the v1,
online-teacher, flow-control, or earlier EMA experiment paths.

## Training objective

For a ground-truth action trajectory \(x_1\), Gaussian noise \(x_0\), and
flow time \(t\),

\[
x_t = t x_1 + (1-t)x_0,
\qquad
v^* = x_1-x_0.
\]

The flow branch anchors the student at the ten-step interval:

\[
\mathcal L_{\mathrm{flow}}
=
\left\|
f_\theta(x_t,t,d=0.1,C)-v^*
\right\|_2^2.
\]

For a shortcut sample, the EMA teacher uses two half-steps of size \(0.1\):

\[
v_1=f_\phi(x_t,t,0.1,C),
\]

\[
x_{\mathrm{mid}}=x_t+0.1v_1,
\]

\[
v_2=f_\phi(x_{\mathrm{mid}},t+0.1,0.1,C),
\]

and supplies the stopped-gradient target

\[
v_{\mathrm{target}}
=
\operatorname{stopgrad}
\left(
\frac{v_1+v_2}{2}
\right).
\]

The student learns the same displacement with one \(D=0.2\) call:

\[
\mathcal L_{\mathrm{shortcut}}
=
\left\|
f_\theta(x_t,t,D=0.2,C)-v_{\mathrm{target}}
\right\|_2^2.
\]

Here \(C\) is the detached frozen-VLM scene KV cache. The EMA teacher is

\[
\phi_{k+1}=0.999\phi_k+0.001\theta_{k+1}.
\]

## Batch allocation

The global batch is eight samples: one sample on each of eight GPUs.

- Six ranks run empirical flow matching.
- Two ranks run shortcut self-consistency.
- DDP averaging therefore realizes 75% flow and 25% shortcut.
- `shortcut_bootstrap_every=4` assigns shortcut examples to two of every
  eight global positions.
- `shortcut_loss_weight=0.25` records and validates the intended allocation;
  branches are physically partitioned, not both evaluated on every sample.

## What changed

- `shortcut_require_dyadic_steps` is a new opt-in validation switch.
  Existing configs keep it `true`.
- The 10-to-5 config sets it to `false` only for \(D=0.2\), which still
  partitions \([0,1]\) exactly into five intervals.
- Flow step: \(d=0.1\).
- Shortcut step: \(D=0.2\), one level only.
- Teacher: float32 EMA, decay 0.999.
- Inference weights: EMA.
- Batch ratio: six flow plus two shortcut.
- AdamW weight decay: 0.1.
- VLM remains frozen; only Action Expert/projections and the step adapter train.

## Data and scale

- Fixed route-less PhysicalAI-AV training manifest: 5,295 clip-disjoint rows.
- Source archive coverage represented by the manifest: 301.765 GB.
- Manifest SHA-256:
  `21e94e04f441bceebb8c31159b660ff012669b7e2cbf2fdbb45243da00d1da5e`.
- Samples are fetched through the existing HF on-demand range-read path.
- Global batch: 8.
- Updates per full manifest pass: \(\lceil 5295/8\rceil=662\).
- Overnight target: 1,986 updates, or three passes.
- Expected duration from the earlier EMA run: approximately 11–12 hours.

## Verification

Focused tests:

```text
37 passed
```

The one-update eight-GPU smoke run completed with:

```text
shortcut branch D mean: 0.200000
shortcut loss on rank 0: 0.037910
DDP-averaged training loss: 0.2087
gradient norm: 0.4863
train runtime: 124.37 s
```

Smoke path:

```text
/home/scratch.abose_sw/alpamayo-assets/runs/alpamayo15_hf300gb_10to5_reference_ema_smoke_1step_20260919_r1
```

## Overnight run

Run name:

```text
alpamayo15_hf300gb_10to5_reference_ema_5295clips_3epochs_20260919_r2
```

Run root:

```text
/home/scratch.abose_sw/alpamayo-assets/runs/alpamayo15_hf300gb_10to5_reference_ema_5295clips_3epochs_20260919_r2
```

Check status and recent metrics with:

```bash
RUN_ROOT=/home/scratch.abose_sw/alpamayo-assets/runs/alpamayo15_hf300gb_10to5_reference_ema_5295clips_3epochs_20260919_r2
cat "$RUN_ROOT/STATUS"
tail -50 "$RUN_ROOT/train.log"
```

Checkpoints are scheduled at updates 662, 1,324, and 1,986. With
`save_total_limit=1`, only the newest checkpoint is retained.

## Relevant files

- Model config:
  `recipes/alpamayo1_5_sft/configs/models/ar1_5_shortcut_10to5_reference_ema.yaml`
- Experiment config:
  `recipes/alpamayo1_5_sft/configs/sft_stage2_trajectory_shortcut_hf_10to5_reference_ema.yaml`
- Launcher:
  `research/alpamayo1_5_shortcut/scripts/train_hf_10to5_reference_ema.sh`
- Step-grid support:
  `recipes/alpamayo1_5_sft/models/shortcut_modules.py`
- Model wiring and loss:
  `recipes/alpamayo1_5_sft/models/shortcut_alpamayo_r1.py`
- Tests:
  `recipes/alpamayo1_5_sft/tests/test_shortcut_modules.py` and
  `recipes/alpamayo1_5_sft/tests/test_shortcut_model_config.py`

## Evaluation after training

Use the same held-out manifest, seeds, stochastic candidate count, attention
backend, hardware, and metric code for both solver counts.

1. Evaluate the trained EMA weights at 10 steps.
2. Evaluate the same weights at 5 steps.
3. Compare ADE, minADE, corner distance, Action Expert latency, and end-to-end
   latency.
4. Treat trained-10 versus trained-5 as the isolated step-reduction comparison.
5. Do not claim a safety gate until collision, off-road, and kinematic checks
   are also available.

## 2026-09-20: training completed; detached offline evaluation launched

The three-epoch training run finished on September 19 at 18:48:32 UTC,
approximately 10 hours 23 minutes after launch. The retained checkpoint is
`trainer_output/checkpoint-1986`. Training completion is not a trajectory
quality result; the following evaluations measure that separately.

Before launching evaluation we verified:

- Exactly 1,986 optimizer updates and three epochs were saved.
- EMA tensors, the EMA update counter, shard index, and training logs passed
  `validate_ema_checkpoint.py` with `--expected-steps 1986`.
- The unchanged training manifest contains 5,295 unique clips. The evaluation
  manifest contains 128 unique clips, with **zero overlap** with those training
  clips.
- All 12 tests in `test_shortcut_model_config.py` passed, including equality
  of the training and local-evaluation model configurations.

The new local evaluation recipe is
`recipes/alpamayo1_5_sft/configs/sft_stage2_trajectory_shortcut_10to5_reference_ema.yaml`.
It preserves the trained flow interval `0.1`, shortcut interval `[0.2]`, and
6-flow/2-shortcut allocation. The benchmark additionally checks the loaded
shortcut settings against the checkpoint config so an older evaluation recipe
cannot silently substitute the earlier dyadic hierarchy.

The runner was launched at **17:26:43 UTC on September 20**, detached into its
own operating-system session, with no terminal input required:

```text
research/alpamayo1_5_shortcut/scripts/run_10to5_eval_suite.py
```

It uses local checkpoints and the restored local PhysicalAI-AV archives, with
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Missing assets cause a recorded
failure rather than a Hugging Face download. Closing SSH or Codex does not
stop the detached job; the remote machine must remain powered on.

Startup verification confirmed that the runner is reparented to PID 1 in its
own session, the benchmark process has both offline flags enabled, the loaded
EMA counter is 1,986, and all model attention backends resolve to eager. The
initial 10-step warm-up completed and the 128-clip evaluation loop started.

### Automatic queue

The three benchmarks run sequentially on GPU 0 to avoid concurrent benchmark
GPU contention. Each evaluates **10 and 5 solver steps** on the same 128 clips,
with six trajectory candidates, seed 42 reset per step count, one warm-up
sample per step count, and eager attention:

1. `ema_weights_128`: trained checkpoint, EMA inference weights.
2. `student_weights_128`: the same checkpoint, student inference weights.
3. `released_128`: released Alpamayo 1.5 checkpoint, zero-initialized adapter.

After each benchmark, the runner computes paired clip-bootstrap confidence
intervals for 5 versus 10 steps (100,000 resamples). After all three, it writes
`summary.json` and `REPORT.md`. Estimated total duration is roughly 2–3 hours;
initialization and local disk throughput can change this.

### Status and results

```text
/home/abose_sw/alphamayo/results/alpamayo15_10to5_eval_128clips_20260920_r1/
  STATUS
  status.json
  preflight.json
  ema_checkpoint_validation.json
  commands.jsonl
  logs/
  ema_weights_128/benchmark_results.json
  student_weights_128/benchmark_results.json
  released_128/benchmark_results.json
  bootstrap_*_10_vs_5.json
  bootstrap_*_10_vs_5.md
  summary.json                  # written after all comparisons complete
  REPORT.md                     # written after all comparisons complete
```

To check progress:

```bash
cat /home/abose_sw/alphamayo/results/alpamayo15_10to5_eval_128clips_20260920_r1/STATUS
tail -c 3000 /home/abose_sw/alphamayo/results/alpamayo15_10to5_eval_128clips_20260920_r1/logs/ema_weights_128.log
```

`status=running` reports the active phase; `status=complete` means the final
report exists; `status=failed` includes the failing phase and exception. The
runner stops on failure instead of silently skipping a comparison.

### Interpretation limits

- Trained 5 versus trained 10 measures step reduction within one checkpoint.
- EMA versus student measures inference-weight choice within that checkpoint.
- Trained versus released also includes three epochs of fine-tuning; this
  suite does **not** include a matched three-epoch flow-only training control.
- No collision/off-road safety gate is claimed.
- Model-call timing excludes dataset loading and decoding. Reciprocal latency
  in Hz is not an end-to-end closed-loop driving rate.
