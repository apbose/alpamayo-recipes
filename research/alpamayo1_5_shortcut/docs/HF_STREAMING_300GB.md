# 300 GB Hugging Face streaming experiment

## Scope

This is a separate follow-up to the completed 19-chunk local v1 pilot. It does
not alter or overwrite that checkpoint path.

- Dataset: `nvidia/PhysicalAI-Autonomous-Vehicles`
- Immutable dataset revision: `33f9bf447ed3bcb7d545ce13f4226f824214fafb`
- Access: official `physical_ai_av.PhysicalAIAVDatasetInterface` with
  `maybe_stream=True`
- Conditioning: route-less; the feature catalog contains no `nav_text`
- Training split: 5,295 unique official-training clips, one `t0=7.0 s` window
  per clip
- Source shards: 55 chunks totaling exactly 301,765,103,600 decimal bytes
  (301.765 GB, 281.041 GiB)
- Evaluation: the same fixed 32 official-validation clips used by the v1 pilot
- Overlap: zero train/validation/test clip overlap

All 19 locally cached pilot chunks were excluded when choosing the new chunks.
The selected chunks are deterministic under seed 42. Exact file paths and byte
sizes are recorded in `manifests/hf_stream_300gb/shard_inventory.json`.

## What a packed archive contains

A ZIP is only the container. For every selected chunk, Stage 2 uses five ZIP
archives containing the actual per-clip driving payloads:

```text
chunk N
├── egomotion.chunk_N.zip
│   └── <clip_id>.egomotion.parquet
├── camera_cross_left_120fov.chunk_N.zip
│   └── <clip_id>.mp4 + frame timestamps
├── camera_front_wide_120fov.chunk_N.zip
├── camera_cross_right_120fov.chunk_N.zip
└── camera_front_tele_30fov.chunk_N.zip
```

The remote reader seeks to a selected clip member through Hugging Face rather
than first downloading the complete chunk archive. It then decodes four image
frames per camera and interpolates 16 history plus 64 future ego poses.
Streaming changes transport, not labels: no navigation command is added.

## Verified behavior

One previously uncached manifest row decoded successfully in 23.660 seconds:

```text
image_frames      (4, 4, 3, 1080, 1920) uint8
camera_indices    (4,)                    int64
ego_history_xyz   (1, 16, 3)             float32
ego_history_rot   (1, 16, 3, 3)          float32
ego_future_xyz    (1, 64, 3)             float32
ego_future_rot    (1, 64, 3, 3)          float32
```

An eight-sample fixed benchmark produced:

| Loader workers | Wall time | Throughput |
|---:|---:|---:|
| 0 | 157.303 s | 0.0509 samples/s |
| 2 | 60.622 s | 0.1320 samples/s |
| 4 | 34.108 s | 0.2345 samples/s |
| 8 | 18.472 s | 0.4331 samples/s |

This is a small systems measurement, not a statistically stable service-level
benchmark. It demonstrates useful concurrency and confirms that pure serial
streaming would starve the GPU.

A real eight-B300, one-optimizer-step smoke also passed. Its global batch had
one shortcut example and seven flow examples. Rank 0 reported shortcut loss
`0.045322` at `d=0.25`; the DDP-averaged loss was `0.230469` and the gradient
norm was finite (`0.511719`). The first cold step took about two minutes. A three-step smoke without
prefetch took 164.846 seconds; one loader worker per rank reduced the otherwise
identical smoke to 110.203 seconds by preparing later samples during the first
GPU pass. The losses were identical in both runs.

## Reference-style loss allocation

The completed v1 pilot computed both objectives for every physical batch-one
sample:

```text
L_v1 = 0.875 L_flow + 0.125 L_shortcut
```

The new optional `reference_partition` estimator follows the original Shortcut
Models allocation over a global batch of eight:

```text
global batch of 8
├── 1 sample: shortcut-only (two stopped-gradient half-step teachers + student)
└── 7 samples: flow-only

DDP mean = (1/8) L_shortcut + (7/8) mean(L_flow)
```

The branch and shortcut-level cursors are checkpoint-persistent. The teacher is
still the current online expert with stopped gradients; an EMA teacher is a
separate, untested ablation and this mode should not be described as fully
reference-identical.

That paragraph records the original run. The separately configured EMA
follow-up is documented in
[EMA_TEACHER_EXPERIMENT_2026-09-15.md](EMA_TEACHER_EXPERIMENT_2026-09-15.md);
it does not alter the original checkpoint or its reported results.

## Reproduce

Generate the measured manifest:

```bash
HF_TOKEN_PATH=/path/to/huggingface/token \
recipes/alpamayo1_5_sft/a1_5_sft/bin/python \
  research/alpamayo1_5_shortcut/scripts/make_hf_streaming_manifests.py \
  --output-dir research/alpamayo1_5_shortcut/manifests/hf_stream_300gb \
  --target-source-gb 300 \
  --cache-dir /scratch/hf-cache \
  --validation-manifest research/alpamayo1_5_shortcut/manifests/route_less_19chunks/val.json \
  --test-manifest research/alpamayo1_5_shortcut/manifests/route_less_19chunks/test.json
```

Run the loader benchmark with
`research/alpamayo1_5_shortcut/scripts/benchmark_hf_streaming.py`. Run training
with `research/alpamayo1_5_shortcut/scripts/train_hf_reference.sh`; the wrapper
requires `BASE_CHECKPOINT` and never embeds an authentication token.


## Completed full training

The detached run
`alpamayo15_hf300gb_reference_5295clips_1epoch_20260913_r2` completed on eight
B300 GPUs. It used local batch one per rank and one prefetch worker per rank.

- Optimizer updates: 662/662 (one complete manifest epoch)
- Distributed samples: 5,296, including one deterministic padding sample
- Runtime: 11,941.230 seconds (3 h 19 min)
- Reported mean training loss: 0.440085
- Rank-zero shortcut objectives: 662
- Shortcut-level counts: 221 at `d=0.25`, 221 at `d=0.5`, 220 at `d=1.0`
- Mean rank-zero shortcut loss: 0.064701
- Expected global allocation: 662 shortcut examples and 4,634 flow examples

The saved `checkpoint-662` passed every checkpoint-validator check. Its four
adapter tensors are finite and nonzero, all five model shards and eight RNG
states are present, and the saved branch/level cursors agree with the logged
allocation.

The first automatic evaluation attempt stopped before reading an evaluation
sample because its zero-worker DataLoader inherited non-null prefetch and
persistent-worker settings. The benchmark now explicitly clears both settings.
A real checkpoint smoke subsequently loaded the model and completed a 10-step
prediction, proving this was evaluation plumbing rather than a model or
checkpoint failure.

## Controlled follow-up evaluations

The detached suite
`alpamayo15_hf300gb_reference_followups_20260913_r1` runs eight benchmarks
sequentially on one B300 so latency is not contaminated by concurrent jobs:

1. Reference-partition checkpoint, validation split, seed 42
2. Released checkpoint, validation split, seed 42
3. Reference-partition checkpoint, validation split, seed 43
4. Reference-partition checkpoint, validation split, seed 44
5. Reference checkpoint with its step-size adapter zeroed after loading
6. Earlier v1 per-sample-weighted checkpoint (descriptive comparison only)
7. Reference-partition checkpoint on the independent fixed test split
8. Released checkpoint on the same independent fixed test split

Every run measures 10, 8, 4, 2, and 1 solver calls with six trajectory
candidates, eager attention, and one latency warm-up. The suite then runs
matched two-step gates on validation and test and generates `REPORT.md`,
`summary.json`, and `comparison.csv`. Compact outputs are published under
`research/alpamayo1_5_shortcut/results/alpamayo15_hf300gb_reference_followups_20260913_r1/`
after all eight runs and both gates succeed.

## Follow-up results

The suite completed on 2026-09-13. On the fixed 32-clip validation split, the
reference-partition checkpoint improved minADE relative to the released model
at every solver count: 36.52% at ten steps, 37.65% at two steps, and 27.23% at
one step. This includes the benefit of Action-Expert fine-tuning and does not by
itself demonstrate successful shortcut conditioning.

The matched within-checkpoint comparison is the shortcut decision:

| Split | 10-step minADE | 2-step minADE | Regression | Expert speedup |
|---|---:|---:|---:|---:|
| Validation | 0.7469 | 0.9372 | 25.47% | 5.14x |
| Independent test | 1.2348 | 1.6145 | 30.75% | 5.15x |

Both quality gates failed because the predeclared limit was 10% regression.
Validation corner distance regressed 26.64%; test corner distance regressed
27.05%. The efficiency checks passed. Safety remains unevaluated because the
public open-loop recipe supplies no collision, off-road, or traffic-rule
evaluator, so the overall gate remains closed independently of quality.

Across seeds 42, 43, and 44, two-step minADE was `0.9922 +/- 0.0544`, while
ten-step minADE was `0.7798 +/- 0.0555`; the step-reduction regression is thus
not specific to one random seed. Zeroing the trained step-size adapter after
checkpoint loading changed minADE by only -0.14% to +0.13% across solver
counts. Under this protocol, the adapter has no material measured effect; most
of the trained-versus-released gain therefore cannot be attributed to explicit
step-size conditioning.

The independent test split also limits overinterpretation: candidate minADE
improved over the released checkpoint by 19.42% at ten steps and 6.99% at two
steps, but one-step was 0.21% worse. The next model iteration should target a
stronger conditioning signal and/or teacher construction before any additional
one-step training claim.

Full compact evidence is in
`research/alpamayo1_5_shortcut/results/alpamayo15_hf300gb_reference_followups_20260913_r1/`.
