# Paired solver-step bootstrap

- Reference step count: 10
- Candidate step count: 2
- Paired clips: 32
- Bootstrap iterations: 100,000

| Metric | Reference | Candidate | Relative change | 95% CI | P(change > 10%) |
|---|---:|---:|---:|---:|---:|
| min_ade | 0.7469 | 0.9372 | +25.47% | [+7.67%, +47.98%] | 0.952 |
| ade | 1.3857 | 1.4247 | +2.82% | [-8.90%, +19.42%] | 0.171 |
| corner_distance | 0.7589 | 0.9610 | +26.64% | [+9.71%, +48.16%] | 0.972 |

The interval resamples paired clips. It quantifies this fixed open-loop sample only; it is not a vehicle-safety confidence interval.
