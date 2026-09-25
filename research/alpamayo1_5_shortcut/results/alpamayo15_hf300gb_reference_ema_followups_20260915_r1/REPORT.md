# Alpamayo 1.5 EMA-teacher shortcut experiment

- EMA checkpoint validation passed: **True**
- Sampler/accounting validation passed: **True**
- Optimizer/EMA updates: 662
- Fixed evaluation clips: 128
- Overall two-step gate passed: **False**

## minADE on the same 128 clips

| Solver calls | EMA weights | Same checkpoint, student weights | Online-teacher shortcut | Flow-only | Released |
|---:|---:|---:|---:|---:|---:|
| 10 | 1.2072 | 1.0990 | 1.0957 | 1.0804 | 1.3346 |
| 4 | 1.2566 | -- | 1.1795 | 1.1850 | 1.3977 |
| 2 | 1.4411 | 1.4586 | 1.4214 | 1.4072 | 1.5832 |
| 1 | 1.9123 | -- | -- | -- | -- |

## Two-step comparisons

| Comparison | minADE change | ADE change | Corner-distance change |
|---|---:|---:|---:|
| EMA 2 vs same EMA checkpoint 10 | +19.38% | -16.25% | +23.33% |
| EMA 2 vs online-teacher 2 | +1.39% | +5.10% | +1.98% |
| EMA-trained student 2 vs online-teacher 2 | +2.62% | +10.71% | +1.09% |
| EMA 2 vs flow-only 2 | +2.41% | +7.49% | +1.96% |
| EMA weights 2 vs student weights 2 | -1.20% | -5.06% | +0.88% |

## Runtime and uncertainty

- EMA runtime state: decay `0.999`, updates `662`, dtypes `['torch.float32']`, inference weights `ema`.
- EMA 10-to-2 Action-Expert speedup: 5.14x.
- Paired EMA 10-to-2 minADE change: +19.38% (95% CI +11.54% to +28.24%).

## Interpretation boundary

- Quality smoke gate passed: False.
- Efficiency gate passed: True.
- Safety evidence passed: False (`not_evaluated`).
- Lower displacement error is better. EMA weights versus the old online-teacher checkpoint combines training-target and evaluation-weight effects.
- The EMA-trained student versus the online-teacher checkpoint isolates the teacher-target change; both rows evaluate student weights.
- EMA versus student weights within the new checkpoint isolates the evaluation-weight swap.
- These are open-loop displacement metrics, not collision, off-road, or traffic-rule safety evidence.
