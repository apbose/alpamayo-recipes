# EMA teacher experiment

## Question

The matched 5,295-clip experiment found that online-teacher shortcut training
was essentially tied with ordinary flow-matching fine-tuning. This follow-up
tests one concrete missing part of the Shortcut Models method: an exponential
moving-average (EMA) network supplies the two stopped-gradient bootstrap
predictions, and EMA weights are used for evaluation.

This is a controlled ablation, not yet a literal reproduction of every image
paper hyperparameter. The completed online-teacher experiment used one
shortcut sample and seven flow samples per global batch of eight, with the
dyadic hierarchy `0.125 -> 0.25 -> 0.5 -> 1.0`. This run preserves those
choices and changes only the teacher/evaluation weights. The Shortcut Models
paper used a 25% bootstrap fraction, `M=128`, EMA targets/evaluation, and EMA
decay `0.999`; changing the fraction and hierarchy belongs in a later,
separately labelled experiment.

## Update rule and data flow

The trainable Action Expert is the student with parameters `theta`. A frozen
copy, `theta_ema`, contains the action input projection (including the `d`
adapter), transformer expert, and action output projection. The VLM is already
frozen and is not duplicated.

```text
Frozen VLM -> reusable scene KV cache
                         |
                         +-------------------------------+
                         |                               |
                         v                               v
              student s_theta(x_t,t,2d)       EMA teacher s_ema(x_t,t,d)
                         |                               |
                         |                    take first half step
                         |                               |
                         |                    EMA teacher at midpoint
                         |                               |
                         |                    stop-gradient average target
                         |                               |
                         +--------- MSE loss <-----------+
                                      |
                               optimizer updates theta
                                      |
                                      v
              theta_ema <- 0.999 theta_ema + 0.001 theta
```

The EMA update occurs once after every successful optimizer step, on every
DDP rank. Flow examples still use the empirical flow-matching target. The
reference partition remains one shortcut example plus seven flow examples per
global update.

## Implementation

| Concern | File |
|---|---|
| Full-precision EMA tensor update | `recipes/alpamayo1_5_sft/models/shortcut_modules.py` |
| EMA modules, target dispatch, checkpoint restore, and inference swap | `recipes/alpamayo1_5_sft/models/shortcut_alpamayo_r1.py` |
| Post-optimizer callback | `recipes/alpamayo1_5_sft/trainer.py` |
| Automatic callback registration | `recipes/alpamayo1_5_sft/train_hf.py` |
| Matched EMA model config | `recipes/alpamayo1_5_sft/configs/models/ar1_5_shortcut_reference_ema.yaml` |
| HF on-demand training config | `recipes/alpamayo1_5_sft/configs/sft_stage2_trajectory_shortcut_hf_reference_ema.yaml` |
| Local fixed-manifest evaluation config | `recipes/alpamayo1_5_sft/configs/sft_stage2_trajectory_shortcut_reference_ema.yaml` |
| Training launcher | `research/alpamayo1_5_shortcut/scripts/train_hf_reference_ema.sh` |
| Artifact validator | `research/alpamayo1_5_shortcut/scripts/validate_ema_checkpoint.py` |

EMA parameters are kept in FP32 even though the trainable student is BF16.
The mixed dtype matters at decay `0.999`: repeatedly rounding the moving
average to BF16 can suppress small updates. Hugging Face applies a global
checkpoint dtype during normal loading, so EMA tensors are explicitly recast
after initialization and directly restored from safetensors when resuming or
evaluating. The evaluator checks the actual runtime dtype and aborts on a
silent downgrade.

## Validation completed before the full run

- Focused unit/configuration suite: 43 tests passed, including saved-config
  recovery and restoration with non-persistent rotary buffers.
- Eight-B300 forward/backward/optimizer smoke: passed with finite shortcut
  loss `0.045377` and finite gradient norm `0.51171875`.
- Peak training allocation: approximately 50.1 GiB per 275-GiB B300.
- Saved one-step checkpoint: seven model shards plus optimizer, scheduler,
  trainer, and eight RNG states.
- EMA artifact validator: passed every check; 417 student tensors have EMA
  partners, the stored update counter is one, probe tensors are finite FP32,
  and student/EMA probes differ after training.
- Reload/inference smoke: the saved checkpoint completed one real validation
  clip at both ten and two solver calls with eager attention.

The first saved smoke (`...20260915_r1`) exposed a BF16 EMA cast and must not
be used. The corrected proof artifact is:

```text
/home/scratch.abose_sw/alpamayo-assets/runs/
  alpamayo15_hf300gb_reference_ema_checkpoint_smoke_1step_20260915_r2/
```

Its validation report is `ema_checkpoint_validation.json` inside that run.

## Full matched run

The full run started at `2026-09-15T09:16:54Z`:

```text
/home/scratch.abose_sw/alpamayo-assets/runs/
  alpamayo15_hf300gb_reference_ema_5295clips_1epoch_20260915_r1/
```

Fixed properties:

| Property | Value |
|---|---:|
| Training clips | 5,295 |
| Source-shard coverage | 301.765 GB |
| GPUs | 8 B300 |
| Per-device batch | 1 |
| Global batch | 8 |
| Planned optimizer updates | 662 |
| Shortcut allocation | 1 in 8 |
| Flow allocation | 7 in 8 |
| EMA decay | 0.999 |
| EMA storage/updates | FP32 |
| VLM | Frozen |
| Navigation text | Not used |

The run completed successfully at `2026-09-15T13:06:50Z`. All 662 optimizer
updates took 3 hours 43 minutes 53 seconds. The final reported training loss
was `0.480799`; all logged losses were finite. One transient loss/gradient
outlier recovered immediately, and gradient clipping remained configured at
`1.0`.

The strict validators passed:

- all seven model shards and trainer-state artifacts were present;
- the checkpoint contained 662 EMA updates and 417 matched student/EMA tensor
  pairs;
- every stored EMA tensor was FP32, finite, and paired with its student tensor;
- student and EMA probe parameters were non-identical after training;
- the distributed epoch contained 662 shortcut examples and 4,634 flow
  examples, including one distributed-padding row;
- shortcut levels were balanced as `d=0.25: 221`, `d=0.5: 221`, and
  `d=1.0: 220`; and
- mean logged shortcut loss was `0.331042`.

## Matched 128-clip evaluation

Every row below uses the same clip-disjoint validation manifest, six
stochastic trajectory candidates, seed 42 reset for each solver count, and
eager attention. Lower minADE is better.

| Solver calls | EMA weights | Same checkpoint, student weights | Online-teacher shortcut | Flow-only | Released |
|---:|---:|---:|---:|---:|---:|
| 10 | 1.2072 | 1.0990 | 1.0957 | **1.0804** | 1.3346 |
| 4 | 1.2566 | -- | **1.1795** | 1.1850 | 1.3977 |
| 2 | 1.4411 | 1.4586 | 1.4214 | **1.4072** | 1.5832 |
| 1 | 1.9123 | -- | -- | -- | -- |

The EMA-weight Action Expert took 416.79 ms at ten calls, 165.01 ms at four,
81.12 ms at two, and 38.99 ms at one. Thus ten-to-two inference was 5.14x
faster in the Action Expert.

The decisive within-checkpoint comparisons are:

| Weights evaluated | 10-step minADE | 2-step minADE | Regression |
|---|---:|---:|---:|
| EMA | 1.2072 | 1.4411 | +19.38% |
| Student | 1.0990 | 1.4586 | +32.72% |
| Earlier online-teacher student | 1.0957 | 1.4214 | +29.73% |
| Flow-only control | 1.0804 | 1.4072 | +30.25% |

For EMA weights, the paired-bootstrap 95% confidence interval for the
ten-to-two minADE change was **+11.54% to +28.24%**; the probability that the
regression exceeded the predeclared 10% threshold was `0.99109`. For the
EMA-trained student, the corresponding interval was **+21.72% to +45.93%**.

The formal gate therefore failed. Efficiency passed, but minADE and corner
distance exceeded the maximum 10% quality regression and collision/off-road
safety remains unevaluated.

## Interpretation

The EMA-teacher change did not produce a two-step advantage in this matched,
short run. The EMA-trained student is close to the old online-teacher model at
ten steps, but is 2.62% worse at two steps. It is also worse than the flow-only
control at two steps.

The EMA weights themselves show a smaller relative ten-to-two regression, but
that does not demonstrate better shortcut learning: their two-step minADE is
only 1.20% better than the same checkpoint's student weights, while their
ten-step minADE is 9.85% worse. With decay `0.999` and only 662 updates, the
EMA retains approximately `0.999^662 = 51.6%` of its initial released-model
weight. The result is consistent with a substantially lagging average.

Consequently:

- EMA storage, target generation, checkpointing, restoration, and inference
  selection are mechanically validated;
- EMA does not pass the two-step quality gate under the current training
  budget;
- the experiment should not be described as the paper's full training setup,
  because it retains the one-in-eight shortcut allocation and `M=8` dyadic
  hierarchy rather than the paper's 25% allocation and `M=128`; and
- a meaningful next EMA experiment requires a much longer schedule and/or a
  decay warmup chosen for the available update count, followed by the same
  matched controls and safety-capable evaluation.

The generated report is
`results/alpamayo15_hf300gb_reference_ema_followups_20260915_r1/REPORT.md` in
the workspace. Compact shareable artifacts are under
`research/alpamayo1_5_shortcut/results/alpamayo15_hf300gb_reference_ema_followups_20260915_r1/`.
