# HF streaming and paper-style shortcut training

Run commands from the repository root using the environment in
[`recipes/alpamayo1_5_sft/README.md`](../../../recipes/alpamayo1_5_sft/README.md).
The commands below assume that environment is active. No datasets, model weights,
tokens, or optimizer states are distributed in this fork. Authenticate with HF
using its normal credential store and obtain the required dataset/model access.
Never put a token in a YAML file or commit it.

## Code map

| Component | Source |
|---|---|
| Local/HF access switch | `src/alpamayo/data/pai.py` |
| Official HF interface compatibility layer | `src/alpamayo/data/pai_utils.py` |
| Explicit route-less `(clip_id, t0_relative)` windows | `src/alpamayo/data/pai_trajectory.py` |
| Step-size adapter, branch sampler and EMA update | `recipes/alpamayo1_5_sft/models/shortcut_modules.py` |
| Existing online/EMA shortcut model | `recipes/alpamayo1_5_sft/models/shortcut_alpamayo_r1.py` |
| Reference layout/interpolation/two-half-step targets | `recipes/alpamayo1_5_sft/models/paper_shortcut_targets.py` |
| A6 model and A7 supervision switch | `recipes/alpamayo1_5_sft/models/paper_shortcut_alpamayo.py` |
| DDP + gradient accumulation + post-update EMA | `recipes/alpamayo1_5_sft/train_paper_ema.py` |

The runtime remains **Alpamayo 1.5**, not Alpamayo 2. The dataset package implements
remote range reads; our adapter does not reimplement streaming. `hf_stream`
does not add navigation labels or load the separate reasoning annotations.

## 1. Check the implementation without downloading data

```bash
ALPAMAYO_ENV=/path/to/recipe/environment \
  bash research/alpamayo1_5_shortcut/scripts/run_tests.sh
```

This includes streaming/local compatibility mocks, zero adapter and gradient
checks, EMA state/callback tests, Hydra configurations, exact reference target
parity, A7's no-teacher-target switch, and missing-cell report tests. The small
MIT-licensed reference fixture is vendored and hash-checked; a separate JAX
installation or reference checkout is not necessary.

## 2. Validate HF on-demand sample loading

The committed `manifests/hf_stream_300gb/` contains 5,295 training clip windows
and the original 32-clip validation/test selections. The separate
`manifests/route_less_19chunks_128eval/` contains the later 128-clip evaluation
selection. Training is disjoint from both. The training driver pins the exact
5,295-row manifest hash to preserve the recorded experiment.

```bash
python research/alpamayo1_5_shortcut/scripts/benchmark_hf_streaming.py \
  --manifest research/alpamayo1_5_shortcut/manifests/hf_stream_300gb/train.json \
  --summary research/alpamayo1_5_shortcut/manifests/hf_stream_300gb/summary.json \
  --cache-dir /path/to/hf-cache \
  --output /path/to/streaming_smoke.json \
  --samples 8 --workers 0 2
```

Use an existing parent directory for `--output`. The validator checks four
cameras × four frames, sixteen history poses, sixty-four future poses and
finite floating-point tensors. It transfers actual data, not just metadata.

The immutable dataset revision is `33f9bf447ed3bcb7d545ce13f4226f824214fafb`.
The 301.765 GB number is **source-archive coverage**, not measured bytes downloaded.
Range access can still transfer substantial data and does not guarantee that
decoded samples persist between epochs.

## 3. Prepare the released checkpoint

```bash
python scripts/convert_checkpoint.py to-a1 \
  --input /path/to/Alpamayo-1.5-10B \
  --output /path/to/Alpamayo-1.5-10B-A1-format
```

This wraps configuration/class names and links unchanged weight shards.
Keep the input directory available while the symlinks are used.

## 4. A6: reference-aligned EMA training

First run a two-update smoke with a new output directory:

```bash
torchrun --standalone --nproc_per_node=8 \
  recipes/alpamayo1_5_sft/train_paper_ema.py \
  --checkpoint /path/to/Alpamayo-1.5-10B-A1-format \
  --manifest research/alpamayo1_5_shortcut/manifests/hf_stream_300gb/train.json \
  --hf-cache /path/to/hf-cache \
  --output-dir /path/to/runs/a6-smoke \
  --updates 2 --save-every 2 --workers 1 \
  --bootstrap-every 4 --supervision ema_bootstrap
```

For the recorded full budget, start **fresh from the released checkpoint** with
a different output directory, `--updates 249 --save-every 83`. This driver does
not implement full optimizer/RNG resume; do not treat `--checkpoint` as such.
The other world sizes must divide 64 and are subject to available GPU memory.

One update averages 64 target pairs: 16 EMA bootstrap targets and 48 direct flow
targets. On eight GPUs, local microbatch one and eight accumulation rounds
produce that mean before one optimizer step and one EMA update. Some underlying
scenes are reused between branches, matching the reference source selection.
The base resolution is 128; larger steps are 1/64 through 1. EMA decay is 0.999.

This ports the target algorithm, not the entire image-model experiment:
Alpamayo's frozen VLM/action expert, residual d injection, PyTorch/DDP and short
fine-tuning budget differ from DiT/JAX/from-scratch training. The paper-reported
bootstrap fraction is 1/4; the public repository default is 1/8. Both target
allocations are tested. See [the ledger](ABLATION_LEDGER_2026-09-20.md).

## 5. A7: matched empirical-target control

Use the same command, released checkpoint, seed and update budget, but a new
output directory and `--supervision empirical_velocity`. Only the 16 teacher
targets are replaced by empirical flow velocities. Inputs, source reuse, noise,
time, d values, architecture and EMA policy remain matched. All 64 targets then
use flow supervision. EMA is maintained for inference, not queried for targets.

**A7 code has passed tests; A7 training has not run in the recorded experiments.**

## 6. Evaluate a saved checkpoint

The matched benchmark below uses local validation archives; HF training does not
make this particular benchmark remote. It requires one GPU and never updates weights.

```bash
torchrun --standalone --nproc_per_node=1 \
  research/alpamayo1_5_shortcut/scripts/benchmark_inference_steps.py \
  --checkpoint /path/to/runs/a6/checkpoint-249 \
  --config-name sft_stage2_trajectory_shortcut_paper_ema \
  --dataset /path/to/local-physical-ai-av \
  --manifest-dir research/alpamayo1_5_shortcut/manifests/route_less_19chunks_128eval \
  --output-dir /path/to/results/a6-eval \
  --steps 10 5 4 2 --num-traj-samples 6 --seed 42 \
  --attention-backend eager --warmup-samples 1 \
  --shortcut-inference-weights ema --expected-ema-updates 249 \
  --verify-checkpoint-shortcut-config
```

For A7 use `sft_stage2_trajectory_paper_empirical_ema`. Missing-cell requests such
as `--steps 5` are supported; without a same-run ten-step reference, timing
speedups are left blank. Model-call latency excludes input loading/decoding.

The dated `run_*_suite.py` and `fill_missing_evaluations.py` helpers preserve
local experiment paths/provenance and are **not portable defaults**. Use the
explicit-path commands above on a teammate's machine. Evaluation-runner resume
is separate from training resume. Historical runners with fixed timeouts are
retained for provenance; the missing-cell runner has no benchmark wall-clock timeout.

## Results and limitations

See [the completed 128-clip table](RESULTS_128CLIP_2026-09-25.md). A5 has the best
measured minADE here but trained longer. A6 has not demonstrated an advantage
over earlier methods. No collision/off-road/closed-loop safety pass is claimed.
