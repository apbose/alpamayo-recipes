# Alpamayo 1.5 reference-shortcut follow-up results

All latency runs were executed sequentially on one NVIDIA B300. Each row uses six stochastic trajectory candidates and eager attention.

## Primary 32-clip validation comparison

| Steps | Released minADE | Candidate minADE | Gain | Candidate ADE | Candidate corner |
|---|---|---|---|---|---|
| 10 | 1.1766 | 0.7469 | +36.52% | 1.3857 | 0.7589 |
| 8 | 1.1844 | 0.7571 | +36.08% | 1.3632 | 0.7724 |
| 4 | 1.2679 | 0.7912 | +37.60% | 1.3330 | 0.8088 |
| 2 | 1.5032 | 0.9372 | +37.65% | 1.4247 | 0.9610 |
| 1 | 1.9705 | 1.4339 | +27.23% | 1.6177 | 1.4988 |

## Step reduction within the reference-partition checkpoint

| Steps | minADE | Change vs 10 | Expert speedup | End-to-end speedup |
|---|---|---|---|---|
| 10 | 0.7469 | +0.00% | 1.00x | 1.00x |
| 8 | 0.7571 | +1.37% | 1.25x | 1.01x |
| 4 | 0.7912 | +5.93% | 2.53x | 1.03x |
| 2 | 0.9372 | +25.47% | 5.14x | 1.02x |
| 1 | 1.4339 | +91.97% | 10.70x | 1.03x |

## Seed repeatability (42, 43, 44)

| Steps | minADE mean | minADE std | ADE mean | Corner mean |
|---|---|---|---|---|
| 10 | 0.7798 | 0.0555 | 1.6422 | 0.8038 |
| 8 | 0.7743 | 0.0334 | 1.5958 | 0.7994 |
| 4 | 0.8118 | 0.0362 | 1.5235 | 0.8499 |
| 2 | 0.9922 | 0.0544 | 1.5193 | 1.0255 |
| 1 | 1.4488 | 0.0130 | 1.6372 | 1.5101 |

## Step-size adapter ablation

The checkpoint is unchanged on disk. The ablation zeros only the adapter's final projection after loading.

| Steps | Trained minADE | Zero-adapter minADE | Zero-adapter change |
|---|---|---|---|
| 10 | 0.7469 | 0.7475 | +0.07% |
| 8 | 0.7571 | 0.7561 | -0.14% |
| 4 | 0.7912 | 0.7903 | -0.12% |
| 2 | 0.9372 | 0.9381 | +0.10% |
| 1 | 1.4339 | 1.4358 | +0.13% |

## Independent 32-clip test comparison

| Steps | Released minADE | Candidate minADE | Gain |
|---|---|---|---|
| 10 | 1.5324 | 1.2348 | +19.42% |
| 8 | 1.5534 | 1.2463 | +19.77% |
| 4 | 1.6262 | 1.3819 | +15.02% |
| 2 | 1.7359 | 1.6145 | +6.99% |
| 1 | 2.0269 | 2.0311 | -0.21% |

## Gate status

- Validation quality passed: **False**
- Validation efficiency passed: **True**
- Independent-test quality passed: **False**
- Independent-test efficiency passed: **True**
- Safety remains unavailable in the public open-loop recipe; therefore the overall deployment gate remains closed regardless of displacement quality.

## Interpretation limits

- This is open-loop trajectory displacement evaluation, not closed-loop AV safety.
- The v1-versus-reference comparison is descriptive because training data volume and update counts differ.
- End-to-end latency is VLM-dominated; Action-Expert latency is the relevant solver-step scaling measurement.
