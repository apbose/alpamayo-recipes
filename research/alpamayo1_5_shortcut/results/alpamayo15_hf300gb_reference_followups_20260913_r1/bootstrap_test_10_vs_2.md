# Paired solver-step bootstrap

- Reference step count: 10
- Candidate step count: 2
- Paired clips: 32
- Bootstrap iterations: 100,000

| Metric | Reference | Candidate | Relative change | 95% CI | P(change > 10%) |
|---|---:|---:|---:|---:|---:|
| min_ade | 1.2348 | 1.6145 | +30.75% | [+13.15%, +55.89%] | 0.992 |
| ade | 2.6198 | 2.2514 | -14.06% | [-26.46%, -1.11%] | 0.000 |
| corner_distance | 1.3442 | 1.7078 | +27.05% | [+12.71%, +46.39%] | 0.992 |

The interval resamples paired clips. It quantifies this fixed open-loop sample only; it is not a vehicle-safety confidence interval.
