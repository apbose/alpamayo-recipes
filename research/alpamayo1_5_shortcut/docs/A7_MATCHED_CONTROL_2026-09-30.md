# A7: matched target-supervision control, queued after gold644 evaluation

> Historical orchestration notes as of 2026-09-30, not a live job-status report. See the research README for the current published experiment status. Paths below are illustrative.

## Status at launch

Queued **2026-09-30 07:18:52 UTC** on `umb-b300-dp-160`.
Detached controller PID: **1560279**, parent PID 1, independent session, no terminal.
Verified state: `waiting_for_gold_evaluation`. **Training has not started.**

The September 23 A7 queue timed out during an A6 evaluation, before any A7
training. This is the first renewed A7 training attempt, not a continuation
from an existing A7 checkpoint.

Live status (supersedes this launch snapshot):

```text
/path/to/workspace/results/alpamayo15_a7_control_after_gold_20260930_r1/status.json
```

## Question this experiment answers

Does A6's two-half-step EMA teacher target help, compared with a ground-truth
velocity target at exactly the same noisy inputs, times, and step sizes?

| Setting | A6 | A7 |
|---|---|---|
| Initial checkpoint | Released R0 | Same released R0 |
| Training manifest | 5,295 clips | Same manifest and order |
| Global target batch | 64 | 64 |
| Optimizer updates | 249 | 249 |
| First 16 target slots | EMA two-half-step target | Direct flow velocity target |
| Remaining 48 slots | Direct flow velocity target | Direct flow velocity target |
| Source reuse, noise, t, d | Paper-target layout | Same assignments |
| d adapter / hierarchy | Full M=128 hierarchy | Unchanged |
| EMA decay / inference weights | 0.999 / EMA | 0.999 / EMA |
| VLM | Frozen | Frozen |

A7 is **not** adapter-free ordinary flow matching. It retains the d adapter,
all hierarchy inputs, and EMA updates; it removes teacher supervision only.
The teacher is not queried to form A7's targets, but EMA weights are still
maintained for matched inference. This differs from the older A3 control.

The source-reuse layout means 64 target pairs are not 64 distinct clips.
The exact repeated A6 plan exposes all 5,295 manifest rows but applies training
targets to 5,211 unique clips across the 249 updates. No new data pool is selected.

## Execution order and safeguards

```text
Current R0/A6 gold644 evaluation succeeds
                |
Verify both models have complete 10/5/4/2 results
                |
Wait until no GPU compute processes remain
                |
CPU tests + identical A6/A7 batch-plan/config audit
                |
2-update A7 smoke on 8 GPUs; save and reload
                |
1 validation clip at 10 and 2 steps
                |
Fresh R0 -> full A7 training: 249 updates on 8 GPUs
                |
A7 evaluation: 128 clips x [10, 5, 4, 2] steps
                |
A6 matched evaluation: same 128 clips and solver counts
                |
Paired bootstrap intervals + comparison report
```

- A failed/dead dependency stops the queue; it never authorizes early training.
- Current evaluation processes and results are not modified or interrupted.
- Full training starts from R0, **not** from the two-update smoke checkpoint.
- Training and inference phases have no overall wall-clock timeout in this
  opt-in queue mode. Individual HTTP requests/retries remain bounded.
- HF training reads get opt-in, process-local HTTP retries in the parent and
  spawned loader workers: six attempts for transient 408/429/499/5xx responses.
  No samples are skipped and retries do not draw model randomness.
- Existing losses, optimizer settings, sample generation and checkpoint formats
  are unchanged. No driver, GPU power, or attention-backend changes are made.
- No files or checkpoints are deleted. Save only smoke `checkpoint-2` and final
  `checkpoint-249`; do not save a second intermediate full-run checkpoint.
- Measured budget: **120.88 GiB**, including two checkpoint saves and a 20-GiB
  reserve; scratch had approximately 168 GiB free when checked. The queue checks
  storage again after the dependency completes.

## Evaluation and interpretation

This follow-up uses the existing **128-clip** held-out manifest, six stochastic
candidates, seed 42 reset per solver count, one excluded warm-up, eager attention,
and EMA inference. The new A6/A7 sweeps run sequentially on GPU 0 on the same host.
The ongoing **644-clip gold evaluation is separate**; its rows are not mixed into
this comparison. A7 is not automatically evaluated on gold644 in this queue.

Report both:

1. A7 versus A6 at the same step count: target-supervision ablation.
2. Each checkpoint's 5/4/2-step result versus its own 10-step result: step-reduction
   degradation, with paired clip-bootstrap intervals.

The report includes minADE, ADE, corner distance, expert latency, full-call latency,
and minADE percentage degradation relative to the checkpoint's own 10-step result.
Inference latency excludes dataset loading/decoding. Bootstrap uses 100,000 paired
clip resamples. No single-seed result or open-loop metric establishes vehicle safety.

The comparison is budget/architecture/data-plan matched, not bitwise cross-host
training reproducibility. The requested HF metadata revision is recorded, but the
official reader's uncached payload URL revision behavior remains an audit item;
this follow-up does not claim newly guaranteed payload pinning.

## Checks completed before launch

- Exact equality with A6's saved `raw_batch_plan.json`: passed.
- Shared optimizer, precision, d/source assignment, seed, EMA and checkpoint
  protocol fields: passed.
- Explicit overlap audit against existing held-outs plus gold644: **zero overlap**
  with 1,630 unique held-out clips in the combined exclusion set.
- CPU-only preflight artifacts:
  `/path/to/workspace/results/alpamayo15_a7_control_preflight_20260930_r1`.
- 32 training/source-layout/retry tests, 8 queue tests and 11 gold-recovery tests:
  **51 passed** after fixes. The spawned-worker retry test caught and verified a
  fix for working-directory-dependent imports. No CUDA training ran in these tests.
- Python compilation and `git diff --check`: passed.
- Detached process and waiting status verified; no A7 training process started.

## Timing estimate

After the current evaluation finishes: approximately **12–18 hours** for smoke,
training and both 128-clip evaluation sweeps. A6's recorded training took about
11.5 hours; A7 avoids teacher forward calls but still runs frozen-VLM conditioning,
data streaming and student updates. Throughput and network conditions can vary.
The current evaluation's remaining time is additional to this estimate.

## Files and live outputs

Existing runner extended, rather than adding a second launch script:

- `research/alpamayo1_5_shortcut/scripts/resume_paper_ema_with_control.py`
- `recipes/alpamayo1_5_sft/train_paper_ema.py`: opt-in worker HTTP retries only;
  the empirical-supervision mode already existed.
- `recipes/alpamayo1_5_sft/tests/test_training_stream_retry.py`
- `research/alpamayo1_5_shortcut/scripts/test_resume_paper_ema_with_control.py`

Controller output:

```text
/path/to/workspace/results/alpamayo15_a7_control_after_gold_20260930_r1/
    status.json
    commands.jsonl                  # populated after dependency releases
    logs/
    CONTROL_SMOKE_PASSED.json        # only after real smoke/reload succeeds
    a7_empirical_ema_eval/
    a6_matched_ema_eval/
    cross_model_pairs.json
    A7_vs_A6_*.json
    REPORT.md                       # written after evaluations complete
```

Training checkpoints and losses will be written under:

```text
/path/to/alpamayo-assets/runs/alpamayo15_paper_empirical_ema_5295clips_b64_20260930_r1/
```

Launch log:

```text
/path/to/workspace/results/alpamayo15_a7_control_after_gold_20260930_r1_launch.log
```

Launch arguments (already launched; **do not run a duplicate**):

```bash
recipes/alpamayo1_5_sft/a1_5_sft/bin/python \
  research/alpamayo1_5_shortcut/scripts/resume_paper_ema_with_control.py \
  --control-only \
  --after-evaluation /path/to/workspace/results/alpamayo15_gold644_r0_a6_20260929_r2 \
  --output-dir /path/to/workspace/results/alpamayo15_a7_control_after_gold_20260930_r1 \
  --control-run /path/to/alpamayo-assets/runs/alpamayo15_paper_empirical_ema_5295clips_b64_20260930_r1 \
  --hf-stream-max-attempts 6
```

The controller was started with a new session, ignored SIGHUP, disconnected stdin
and file-backed output. Laptop/SSH/Codex-session closure does not stop it; the remote
machine must remain running. No commit or push was performed.
