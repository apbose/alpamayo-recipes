# Paired solver-step bootstrap

- Reference step count: 10
- Candidate step count: 2
- Paired clips: 128
- Bootstrap iterations: 100,000

| Metric | Reference | Candidate | Relative change | 95% CI | P(change > 10%) |
|---|---:|---:|---:|---:|---:|
| min_ade | 1.2072 | 1.4411 | +19.38% | [+11.54%, +28.24%] | 0.991 |
| ade | 2.4351 | 2.0395 | -16.25% | [-24.11%, -8.85%] | 0.000 |
| corner_distance | 1.1936 | 1.4722 | +23.33% | [+15.84%, +32.01%] | 1.000 |

The interval resamples paired clips. It quantifies this fixed open-loop sample only; it is not a vehicle-safety confidence interval.
