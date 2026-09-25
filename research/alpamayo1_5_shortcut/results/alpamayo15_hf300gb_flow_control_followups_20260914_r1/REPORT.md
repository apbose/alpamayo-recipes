# Alpamayo 1.5 matched flow-control follow-ups

- Flow-only checkpoint validation passed: **True**
- Flow-only optimizer steps: 662
- Mean logged flow loss: 0.502857

## Same-data 32-clip comparison

| Steps | Shortcut minADE | Flow-only minADE | Flow change | Shortcut regression vs 10 | Flow regression vs 10 |
|---|---|---|---|---|---|
| 10 | 0.7469 | 0.7283 | -2.49% | +0.00% | +0.00% |
| 8 | 0.7571 | 0.7356 | -2.84% | +1.37% | +1.00% |
| 4 | 0.7912 | 0.7684 | -2.88% | +5.93% | +5.51% |
| 2 | 0.9372 | 0.9627 | +2.72% | +25.47% | +32.18% |
| 1 | 1.4339 | 1.4480 | +0.99% | +91.97% | +98.81% |

## Larger 128-clip comparison

| Steps | Released minADE | Shortcut minADE | Flow-only minADE | Shortcut vs flow |
|---|---|---|---|---|
| 10 | 1.3346 | 1.0957 | 1.0804 | -1.40% |
| 4 | 1.3977 | 1.1795 | 1.1850 | +0.46% |
| 2 | 1.5832 | 1.4214 | 1.4072 | -1.00% |

## Adapter-strength sweep

| Scale | 10-step minADE | 2-step minADE | 2-step regression | 1-step minADE |
|---|---|---|---|---|
| 0 | 0.7475 | 0.9381 | +25.50% | 1.4358 |
| 1 | 0.7469 | 0.9372 | +25.47% | 1.4339 |
| 2 | 0.7467 | 0.9389 | +25.74% | 1.4309 |
| 4 | 0.7452 | 0.9489 | +27.33% | 1.4286 |
| 8 | 0.7451 | 0.9686 | +30.00% | 1.4196 |

## Paired 128-clip uncertainty

- Shortcut 10-to-2 minADE change: +29.73% (95% CI +20.15% to +41.04%).
- Flow-only 10-to-2 minADE change: +30.25% (95% CI +20.30% to +41.86%).

## Limits

- These are open-loop displacement results, not collision/off-road safety.
- Adapter scaling changes inference only; it does not optimize a new checkpoint.
- Latency runs were sequential on one B300 to avoid contention.
