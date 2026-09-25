# Alpamayo 1.5 shortcut experiments: ablation ledger

Updated: 2026-09-25. Lower displacement error is better. All driving experiments
below freeze the VLM and train the Action Expert/projections. They use route-less
inputs: no fabricated navigation or human/generated CoC supervision.

The missing-cell suite is complete. See the portable
[completed 128-clip results](RESULTS_128CLIP_2026-09-25.md) for the current table.
A7 is implemented but untrained; references to queues below are historical.

## September 24: missing inference cells (historical execution record)

An evaluation-only detached suite was launched at 07:32 UTC on September 24.
It does not retrain or overwrite any checkpoint. It evaluates A2/A3/A4 at five
steps, A5 at four/two steps, and A6 at two steps, using the same 128 validation
windows, six candidates, seed 42 and eager attention as the historical table.
A4/A5/A6 use EMA weights; A2/A3 use student weights. A7 remains untrained.

The **automatically updated expanded table** is
[/home/abose_sw/alphamayo/docs/ALPAMAYO15_MISSING_EVAL_2026-09-24.md](/home/abose_sw/alphamayo/docs/ALPAMAYO15_MISSING_EVAL_2026-09-24.md).
The historical tables below are retained as provenance and are not a live dashboard.

- Suite: `/home/abose_sw/alphamayo/results/alpamayo15_missing_eval_128clips_20260924_r1`.
- Runner: `research/alpamayo1_5_shortcut/scripts/fill_missing_evaluations.py`.
- One process per checkpoint on physical GPUs 0/1/2/5/3 respectively.
- No benchmark wall-clock timeout; `--resume` skips completed solver settings.
- Checkpoint shard presence/config hashes, validation identity and zero training
  overlap passed preflight. Seven CPU-only runner/report regression tests passed.
- Original result JSON files are read-only. `REPORT.md`, `comparison.csv` and
  `summary.json` in the suite directory update after each full solver-count evaluation.
- New timings are explicitly separated from historical timings: enforced power
  limits differ between GPUs, and concurrent work can affect latency.
- No new one-step evaluation or collision/off-road safety claim is made.

The previous September 23 runner failed its eight-hour timeout at 21:36 UTC.
Its surviving A6 evaluation worker finished at 22:17 UTC, reproducing the saved
quality results. A7 did not start. This new missing-cell suite is independent
of that failed training queue and does not implicitly restart A7.

## Experiment inventory

| ID | Experiment | Training data | Updates / global target batch | Loss allocation | Teacher / evaluated weights | Step-size training |
|---|---|---|---|---|---|---|
| R0 | Released checkpoint | No training in this workspace | — | — | Released | Native flow model |
| A1 | Initial local pilot | 891 clips, 10,692 windows | 10,692 / 1 | Both losses per sample; 0.875 flow + 0.125 shortcut | Online / student | Flow 1/8; shortcut 1/4, 1/2, 1 |
| A2 | HF online partition | 5,295 clips, one window each | 662 / 8 | 7 flow-only + 1 shortcut-only | Online / student | Same M=8 ladder |
| A3 | HF ordinary flow control | Same 5,295 clips | 662 / 8 | 8 flow | No shortcut teacher / student | Native flow; no d adapter |
| A4 | Matched M=8 EMA follow-up | Same 5,295 clips | 662 / 8 | 7 flow-only + 1 shortcut-only | EMA 0.999 / EMA and student | Same M=8 ladder |
| A5 | Targeted 10-to-5 EMA | Same 5,295 clips; 3 passes | 1,986 / 8 | 6 flow-only + 2 shortcut-only | EMA 0.999 / EMA and student | Flow 0.1; shortcut 0.2 only |
| A6 | Paper-target EMA port (new) | Same 5,295-clip manifest; planned 3 raw-data passes | 249 / 64 | Original 16 bootstrap + 48 flow target layout, including source reuse | EMA 0.999 / EMA | Flow 1/128; all seven larger dyadic levels |
| A7 | Empirical-target control (queued) | Same manifest and exact A6 raw batch plan | 249 / 64 | 64 empirical velocity targets; A6 source/noise/time/d assignment retained | EMA 0.999 maintained / EMA | Same full M=128 hierarchy and adapter |

A6 training completed at 2026-09-21 11:36:12 UTC (249 updates), and all five
EMA solver-count evaluations completed on 128 clips. The released-checkpoint
comparison was interrupted by SIGHUP; the original `STATUS` is stale. The
September 23 detached queue below resumes the missing work and schedules A7.
The physical per-GPU microbatch remains one; eight GPUs and eight accumulation
rounds form a 64-target global batch. A6 is not retrained.

### What does "target batch" mean?

One target pair contains an entire noisy 64x2 action tensor, its context and
its velocity target. It is not one waypoint. In the reference implementation,
some underlying scenes occur in both flow and shortcut target pairs. Thus 64
target pairs do not imply 64 distinct clips receiving gradients.

## Results: keep evaluation populations separate

### Initial 32-clip pilot (not comparable directly with the 128-clip table)

| Model | 10-step minADE | 4-step minADE | 2-step minADE |
|---|---:|---:|---:|
| R0 released | 1.1766 | 1.2679 | 1.5032 |
| A1 local pilot | 0.7818 | 0.8128 | 0.8917 |

Source: [original experiment summary](../results/experiment_summary.json).
The 2-step A1 minADE regression against its own 10-step result was +14.07%.
Its faster expert did not establish acceptable two-step quality or safety.

### Fixed 128-clip validation, six candidates, seed 42, eager attention

All values are minADE in meters; missing values mean not evaluated here.

| Experiment / inference weights | 10 steps | 5 steps | 4 steps | 2 steps |
|---|---:|---:|---:|---:|
| R0 released | 1.3346 | 1.3648 | 1.3977 | 1.5832 |
| A2 online shortcut / student | 1.0957 | — | 1.1795 | 1.4214 |
| A3 flow-only / student | 1.0804 | — | 1.1850 | 1.4072 |
| A4 EMA-trained / EMA | 1.2072 | — | 1.2566 | 1.4411 |
| A4 same checkpoint / student | 1.0990 | — | — | 1.4586 |
| A5 targeted 10-to-5 / EMA | 1.0775 | 1.0897 | — | — |
| A5 same checkpoint / student | 1.0828 | 1.0907 | — | — |
| A6 paper-target / EMA | 1.2278 | 1.2409 | 1.2746 | Not queued |
| A7 empirical-target / EMA | Pending | Pending | Not queued | Not queued |

Sources:

- [A2/A3 flow-control report](../results/alpamayo15_hf300gb_flow_control_followups_20260914_r1/REPORT.md)
- [A4 EMA report](../results/alpamayo15_hf300gb_reference_ema_followups_20260915_r1/REPORT.md)
- A5: `/home/abose_sw/alphamayo/results/alpamayo15_10to5_eval_128clips_20260920_r1/REPORT.md`
- A6: `/home/scratch.abose_sw/alpamayo-assets/runs/alpamayo15_paper_ema_5295clips_b64_20260920_r1/ema_eval_128/benchmark_results.json`

### A6 completed EMA measurements (original host)

| Steps | minADE m | ADE m | Corner m | Expert ms | Model-call ms |
|---|---:|---:|---:|---:|---:|
| 128 | 1.2383 | 2.7615 | 1.1987 | 5369.47 | 15825.29 |
| 10 | 1.2278 | 2.5627 | 1.2099 | 416.36 | 10834.24 |
| 8 | 1.2204 | 2.5004 | 1.2075 | 332.58 | 10757.50 |
| 5 | 1.2409 | 2.2851 | 1.2388 | 206.67 | 10625.06 |
| 4 | 1.2746 | 2.2150 | 1.2774 | 164.79 | 10569.47 |

These are completed measurements, not placeholders. The resumed queue repeats
128/10/5 EMA inference on the new host to match A7's host/driver; it preserves
the original measurements and does not pool timing samples across machines.

### A5 quality and latency

| Weights | Steps | minADE | ADE | Corner | Expert ms | Model-call ms |
|---|---:|---:|---:|---:|---:|---:|
| A5 EMA | 10 | 1.0775 | 2.1294 | 1.1015 | 416.45 | 10818.72 |
| A5 EMA | 5 | 1.0897 | 2.0975 | 1.1109 | 206.79 | 10607.66 |
| A5 student | 10 | 1.0828 | 2.0612 | 1.1129 | 398.50 | 10800.96 |
| A5 student | 5 | 1.0907 | 2.1205 | 1.1096 | 197.83 | 10605.99 |

EMA 10-to-5: minADE +1.13%, ADE -1.50%, corner +0.85%; expert 2.01x
faster, but model-call latency only 1.95% lower. Paired clip-bootstrap minADE
95% interval: [-4.15%, +6.32%]. The released model also has only +2.26%
minADE regression at five steps. A5 therefore does not isolate a causal gain
from shortcut supervision; no matched three-epoch flow-only control exists.

## Which comparisons support which claims?

- A2 versus A3 controls data, update budget and the previous training schedule,
  but also changes the d-adapter and objective; it is an overall shortcut-versus-
  ordinary-flow comparison, not a loss-only ablation.
- A4 student versus A2 student is the earlier teacher-type comparison.
- A4 EMA versus A4 student isolates the inference weight choice.
- A5 EMA versus A5 student likewise isolates inference weight choice.
- A5 five versus A5 ten isolates solver-count change within a checkpoint.
- A5 versus A4 changes training duration, step hierarchy and loss allocation.
- A6 versus earlier runs changes batch size, hierarchy, source reuse, time/noise
  construction, d encoding, optimizer schedule/decay policy and precision.
  **A6 is a reference-method alignment experiment, not a one-factor ablation.**
- A7 versus A6 changes the target used in the 16 bootstrap-layout slots from
  two-half-step EMA predictions to empirical velocity. Source indices, noise,
  times, d values, architecture, optimizer, precision, seed, update count and
  EMA policy are held fixed. This is a target-supervision ablation, not an
  adapter-free ordinary-flow baseline. Different training hosts/numerical
  nondeterminism and single-seed uncertainty remain limitations.
- No experiment here establishes collision, off-road, traffic-rule or closed-loop
  safety. minADE selects the best of six candidates, not a deployed policy choice.

## A6: what is matched to the paper/reference implementation?

Paper: [One Step Diffusion via Shortcut Models, Appendix B](https://arxiv.org/html/2410.12557v3#A2).
Code pinned to commit `601004348667094e1b71f30942199759412d4432`:
[targets_shortcut.py](https://github.com/kvfrans/shortcut-models/blob/601004348667094e1b71f30942199759412d4432/targets_shortcut.py).
Local reference file SHA-256:
`adb2ac83febc1de012a7cdb713b9117d53616473ccd38f1104fd2475d8b1abc3`.

| Detail | A6 implementation |
|---|---|
| Flow/bootstrap fraction | Paper-reported 75% / 25%; public code default is instead 87.5% / 12.5% |
| Global target batch | 64, the paper's CelebA batch size |
| Base resolution | M=128 |
| Shortcut allocation | Exact `repeat` then remainder-fill rule, not cycling one level per update |
| Source mapping | Bootstrap inputs 0..15, flow inputs 0..47; original overlap preserved |
| Flow time | Uniform discrete ticks 0..127 divided by 128 |
| Shortcut time | Uniform grid ticks 0..(1/d)-1 multiplied by d |
| Interpolation | Original endpoint-noise factor 1e-5 retained |
| Target | Two EMA half-steps; midpoint and averaged velocity clipped to [-4,4] |
| Loss | Mean MSE over all 64 target pairs; no extra bootstrap multiplier |
| EMA | FP32, decay 0.999, after optimizer step, also selected for evaluation |
| Step input | `dt_base=-log2(d)`, 256 sinusoidal features, max period 10000 |
| Optimizer | AdamW lr=1e-4, betas=(0.9,0.999), eps=1e-8, decay 0.1 on all trainable parameters |
| Schedule / clipping | Constant LR, no warmup, no gradient clipping |
| CFG branch | Disabled; no image class conditioning exists in this trajectory task |

The exact bootstrap counts per 64-target update are:

| d | 1/64 | 1/32 | 1/16 | 1/8 | 1/4 | 1/2 | 1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Target pairs | 2 | 2 | 2 | 2 | 2 | 2 | 4 |

The remaining 48 use flow supervision with d=1/128.

### Equations

For clean actions x, Gaussian noise epsilon, and a grid-aligned time t:

$$
x_t = [1-(1-10^{-5})t]\epsilon + tx,
\qquad v_{\mathrm{flow}}=x-(1-10^{-5})\epsilon.
$$

For bootstrap step d:

$$
v_1=f_\phi(x_t,t,d/2,C),\quad
x_{\mathrm{mid}}=\operatorname{clip}(x_t+(d/2)v_1,-4,4),
$$

$$
v_2=f_\phi(x_{\mathrm{mid}},t+d/2,d/2,C),\quad
v_{\mathrm{target}}=\operatorname{stopgrad}\left[\operatorname{clip}\left((v_1+v_2)/2,-4,4\right)\right].
$$

$$
L=\frac{1}{64}\left[\sum_{i=1}^{16}L_{\mathrm{shortcut},i}
+\sum_{j=1}^{48}L_{\mathrm{flow},j}\right],\qquad
\phi_{k+1}=0.999\phi_k+0.001\theta_{k+1}.
$$

### Explicit differences from an identical paper reproduction

1. Alpamayo Action Expert and frozen VLM context replace the paper's DiT image
   network and image class labels. Actions use Alpamayo's unicycle conversion
   and normalization, not image VAE latents.
2. The d embedding is a residual addition to action tokens, not DiT adaLN.
   Its output layer starts at zero to preserve the released checkpoint;
   the first projection uses normal(0,0.02). Hidden width follows Alpamayo.
3. We fine-tune a released driving model rather than train an image model
   from scratch. Trainable weights and EMA are FP32, computation uses BF16;
   the frozen VLM retains its checkpoint precision.
4. Torch/CUDA replaces JAX/TPU. Injected-random-input parity tests verify the
   numerical target formulas; RNG streams are not claimed bitwise identical.
5. A shuffled finite clip manifest replaces TFDS. The driver records the raw
   batch plan and actual distinct training clips contributing targets.
6. Eight DDP ranks and eight accumulation rounds replace a physical batch of
   64. Parameters and EMA stay fixed throughout each accumulated update.
7. The initial budget is 249 updates, not the paper's 400k/800k. It draws 15,936
   raw rows and builds 15,936 target pairs (~3 raw-manifest passes). Because of
   original source reuse, only 11,952 selected source occurrences contribute
   before counting repeated flow/bootstrap targets. This is **not** three
   complete gradient-bearing passes over every clip. The recorded full-run
   plan selects **5,211 distinct training clips** from the 5,295-clip manifest;
   84 are not selected into target pairs by this finite shuffle/reuse plan.
8. EMA initialization weight after 249 updates is 0.999^249 ≈ 77.9%. This short
   pilot may under-adapt EMA; the decay is intentionally not retuned and the
   result cannot establish a fully converged paper-method comparison.

## A6 execution and artifacts

Run root:

```text
/home/scratch.abose_sw/alpamayo-assets/runs/alpamayo15_paper_ema_5295clips_b64_20260920_r1
```

Pipeline launch: 2026-09-20 23:45 UTC, detached OS session.

Verified smoke results (2026-09-20):

- 50 parity/regression tests passed, including direct execution of the pinned
  upstream target function with shared injected random inputs.
- Two real eight-GPU optimizer updates completed, each with exactly 48 flow
  and 16 shortcut targets, and EMA counts 1 then 2.
- Mean target-pair loss: 0.630821 then 0.430933; gradient norms 0.587647 then
  0.279855. These are execution checks, not validation-quality claims.
- Training computation for both smoke updates took 369.25 seconds, excluding
  initialization/saving. This suggests about 13 hours for 249 updates, plus
  several hours for the queued inference comparisons; data throughput varies.
- Full smoke checkpoint saved at 23:58 UTC. Reload verification confirmed the
  complete dyadic grid, FP32 EMA weights, eager attention and EMA counter 2.
  Actual 10- and 8-step reload inference both passed with finite trajectories.
- `SMOKE_PASSED.json` was written at 2026-09-21 00:00:36 UTC. The pipeline
  immediately launched the full eight-GPU, 249-update training command.
  Initial status at handoff was `initializing`; the full run's first optimizer
  update was not claimed as complete at that point.

```text
50 parity/regression tests
       ↓
2-update eight-GPU smoke, full checkpoint saved
       ↓
reload EMA and run real 10/8-step inference on one validation clip
       ↓  only if every check passes
249-update training, checkpoints at 83 / 166 / 249
       ↓
128-clip EMA evaluation: 128, 10, 8, 5, 4 steps
       ↓
released checkpoint: 128 and 10 steps
       ↓
paired clip-bootstrap intervals and REPORT.md
```

Power-of-two counts are on the training hierarchy. Ten and five are off-grid
interpolation diagnostics; do not confuse their status with A5's targeted
10-to-5 training. No one-step evaluation is queued.

Training uses the existing HF on-demand source and needs network access for
uncached data. Evaluation uses local archives with HF offline flags. Every
phase runs independently of the user's SSH/Codex session. Failures stop the
queue and are recorded rather than silently changing the experiment.

Artifacts: `STATUS`, `status.json`, `commands.jsonl`, `source_hashes.json`,
`logs/`, `smoke/`, `smoke_reload_eval/`, `SMOKE_PASSED.json`,
`training/protocol.json`, `training/raw_batch_plan.json`,
`training/metrics.jsonl`, `training/checkpoint-*`, `ema_eval_128/`,
`released_eval_128/`, `bootstrap_*.json`, `summary.json`, `REPORT.md`.

Files to inspect:

- `recipes/alpamayo1_5_sft/models/paper_shortcut_targets.py`: allocation and target formulas.
- `recipes/alpamayo1_5_sft/models/paper_shortcut_alpamayo.py`: frozen VLM/Action Expert integration.
- `recipes/alpamayo1_5_sft/train_paper_ema.py`: batch replay, accumulation, optimizer/EMA and saving.
- `recipes/alpamayo1_5_sft/tests/test_paper_shortcut.py`: original-source numerical parity and regression tests.
- `research/alpamayo1_5_shortcut/scripts/run_paper_ema_suite.py`: unattended queue.

## September 23: resumed comparisons and overnight A7 queue

Detached launch: **2026-09-23 07:22 UTC**, host `umb-b300-dp-128`, driver
`595.91.07`. Verified actual CUDA BF16 matrix multiplication and existing HF
authentication/access to the pinned dataset revision. All 53 tests passed,
including equality of the legacy target values before/after the opt-in change
and a test that A7 never calls the teacher to construct training targets.

Queue output root:

```text
/home/abose_sw/alphamayo/results/alpamayo15_a6_resume_a7_empirical_20260923_r1
```

A7 checkpoints/training logs:

```text
/home/scratch.abose_sw/alpamayo-assets/runs/alpamayo15_paper_empirical_ema_5295clips_b64_20260923_r1
```

The queue executes serially, with no training/evaluation GPU contention:

1. Validate saved A6 checkpoint/metrics/manifests and run 53 tests.
2. Compute the five previously missing A6 paired-clip bootstrap reports.
3. Resume released-model 128/10-step evaluation on 128 clips, GPU 0.
4. Repeat A6 EMA 128/10/5 evaluation on this host, GPU 0, then write
   `A6_COMPLETED_REPORT.md` and `A6_COMPLETE.json`.
5. Run a two-update A7 smoke on eight GPUs; require 64 empirical targets,
   zero teacher targets and exactly one EMA update per optimizer update.
6. Save/reload the A7 smoke checkpoint and test real 10/5-step inference.
7. Only if the gate passes, train fresh from the released checkpoint for
   **249 updates**, with the identical A6 seed-10 raw batch plan. Save at
   updates 125 and 249; retain the smoke checkpoint. No old files are deleted.
8. Evaluate A7 EMA at 128/10/5 steps on 128 clips, then generate within-model
   and A7-vs-A6 paired-bootstrap intervals, `summary.json` and `REPORT.md`.

Estimated total: roughly 15–18 hours, including 10–12 hours of training;
throughput and shared-storage latency can change this. All evaluation uses
local data and HF offline flags. Training uses HF on-demand reads and therefore
needs network access. Neither phase needs SSH or Codex to stay connected; the
remote machine must stay powered on. A failure stops the queue and is recorded
in `STATUS`/`status.json`. The new runner preserves the interrupted run's logs.

Important A7 semantics: d and t are **not** changed to the fixed flow anchor
for the 16 larger-step slots. Their original inputs are retained; only their
training targets become `x - (1 - 1e-5) * noise`, without teacher clipping.
Thus A7 tests the value of bootstrap supervision under matched conditioning.
EMA is still updated and evaluated, although it generates no A7 training
targets. This is deliberately different from A3's vanilla adapter-free flow
control. The `bootstrap_pairs`/`flow_pairs` fields in `protocol.json` describe
the original input layout; `teacher_target_pairs`/`empirical_target_pairs`
record the actual supervision, and runtime metrics must report 0/64.

Code/config entry points:

- `scripts/resume_paper_ema_with_control.py` (under this research directory).
- `recipes/alpamayo1_5_sft/train_paper_ema.py --supervision empirical_velocity`.
- `recipes/alpamayo1_5_sft/configs/models/ar1_5_paper_empirical_ema.yaml`.
- `recipes/alpamayo1_5_sft/configs/sft_stage2_trajectory_paper_empirical_ema.yaml`.
- `recipes/alpamayo1_5_sft/tests/test_paper_empirical_ablation.py`.

## Remaining ablations (not automatically launched)

1. Additional seeds for the A6/A7 target-supervision comparison after the
   queued A7 experiment completes; no additional seed is launched now.
2. Teacher online versus EMA while evaluating student weights in both cases.
3. EMA versus student inference within the same full-hierarchy checkpoint.
4. Paper 25% versus repository-default 12.5% bootstrap allocation.
5. Longer fixed-method training or an explicitly labeled EMA-decay ablation.
6. Physical safety/kinematic checks and eventually closed-loop evaluation.

Old invalid EMA reload runs and failed launches are not evidence. Retain their
logs for provenance, but exclude them from result tables. The valid A4 report
above uses the corrected persistent EMA counter and restored EMA weights.
