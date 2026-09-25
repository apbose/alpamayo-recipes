# Flow-only control and larger evaluation

## Why this experiment is necessary

The 300 GB reference-partition checkpoint improved over the released model,
but an inference ablation found almost no change after zeroing its learned
step-size adapter. That leaves two competing explanations:

1. shortcut self-consistency improved the Action Expert even though the adapter
   itself is weak; or
2. ordinary Stage-2 flow-matching fine-tuning produced nearly all the gain.

The completed v1 checkpoint cannot answer this cleanly because it used a
different training population and update count. The new control changes only
the training objective.

## Matched flow-only control

Both checkpoints start from the same released Alpamayo 1.5 A1-format wrapper.

| Property | Reference shortcut | Flow-only control |
|---|---|---|
| Streamed manifest | Same 5,295 clips | Same 5,295 clips |
| Source archive coverage | 301.765 GB | 301.765 GB |
| Global batch | 8 | 8 |
| Optimizer updates | 662 | 662 |
| Seed and LR schedule | Same | Same |
| VLM | Frozen | Frozen |
| Trainable base Action Expert | Yes | Yes |
| Explicit step-size adapter | Yes | No |
| Shortcut bootstrap examples | 1 in 8 | None |
| Flow-matching examples | 7 in 8 | 8 in 8 |

The control uses NVIDIA's `TrainableAlpamayoR1` flow-matching forward path, not
the shortcut subclass. A real eight-B300 smoke passed before launch with loss
`0.2772` and gradient norm `0.5508`, both finite.

## Larger fixed evaluation

The original benchmark uses 32 validation clips. The local 19-chunk audit
contains 688 valid official-validation clips and 298 valid official-test clips,
so a deterministic manifest was generated with 128 validation and 128 test
clips at `t0=7.0 s`.

- Selection seed: 42
- Validation manifest SHA-256:
  `d7dc21959b746dacdfee2f559550b329783d4664db88d3cacb5590323fec9e62`
- Test manifest SHA-256:
  `b0a8f1dbcff437f4cff60bc6f1a3c722ce2065cd57e9bbe0a04bea66cde0add5`
- Train/validation/test clip overlap: zero
- Navigation conditioning: none

The released, reference-shortcut, and flow-only checkpoints are evaluated on
the same 128 validation clips at 10, 4, and 2 solver calls. Six candidates,
seed 42, eager attention, and one warm-up are fixed across checkpoints.

## Adapter-strength sweep

The existing scale-zero and scale-one results are extended with scales 2, 4,
and 8. Only the final projection of the trained adapter is multiplied after
loading; checkpoint files are never modified. Each scale is evaluated at 10,
2, and 1 solver calls on the fixed 32 validation clips.

This intervention answers whether the learned adapter direction is useful but
underweighted. It is not a substitute for retraining.

## Current bootstrap evidence

A 100,000-iteration paired clip bootstrap was applied to the completed
reference-checkpoint 10-versus-2 results:

| Split | Metric | Change | Paired 95% interval | P(regression > 10%) |
|---|---|---:|---:|---:|
| 32 validation clips | minADE | +25.47% | +7.67% to +47.98% | 0.952 |
| 32 validation clips | corner distance | +26.64% | +9.71% to +48.16% | 0.972 |
| 32 test clips | minADE | +30.75% | +13.15% to +55.89% | 0.992 |
| 32 test clips | corner distance | +27.05% | +12.71% to +46.39% | 0.992 |

The bootstrap resamples this fixed open-loop clip set. It is not a safety
confidence interval. The 128-clip experiment will provide a narrower and more
representative displacement estimate within the locally available chunks.

## Unattended pipeline

The detached pipeline is named
`alpamayo15_hf300gb_flow_control_followups_20260914_r1`. It performs:

1. full 662-step flow-only training on eight B300 GPUs;
2. structural and finite-value checkpoint validation;
3. flow-only evaluation on the original 32 clips at 10/8/4/2/1 steps;
4. released, shortcut, and flow-only evaluation on 128 clips at 10/4/2 steps;
5. adapter scales 2/4/8 at 10/2/1 steps;
6. paired bootstrap analyses and two-step gates; and
7. automatic JSON, CSV, and Markdown summarization.

Latency experiments run sequentially on one B300 after training to avoid GPU,
CPU, and archive-I/O contention. Compact outputs will be published under
`research/alpamayo1_5_shortcut/results/alpamayo15_hf300gb_flow_control_followups_20260914_r1/`
only after every phase succeeds. Until that directory exists, the pipeline is
still running and no flow-control conclusion should be quoted.

## Interpretation rule

- If shortcut clearly beats flow-only under the matched protocol, the shortcut
  objective added value beyond ordinary fine-tuning.
- If they are effectively equal, the existing gain should be attributed mainly
  to Stage-2 flow fine-tuning, not shortcut conditioning.
- If an adapter scale greater than one improves two-step quality, retraining
  with a stronger or normalized conditioning path is justified.
- Regardless of displacement results, collision/off-road safety remains
  unavailable in this public open-loop recipe.
