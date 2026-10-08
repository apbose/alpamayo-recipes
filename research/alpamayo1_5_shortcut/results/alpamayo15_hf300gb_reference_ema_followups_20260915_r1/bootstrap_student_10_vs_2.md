# Paired solver-step bootstrap

- Reference step count: 10
- Candidate step count: 2
- Paired clips: 128
- Bootstrap iterations: 100,000

| Metric | Reference | Candidate | Relative change | 95% CI | P(change > 10%) |
|---|---:|---:|---:|---:|---:|
| min_ade | 1.0990 | 1.4586 | +32.72% | [+21.72%, +45.93%] | 1.000 |
| ade | 2.1732 | 2.1483 | -1.15% | [-8.42%, +7.24%] | 0.005 |
| corner_distance | 1.1114 | 1.4594 | +31.31% | [+21.15%, +43.27%] | 1.000 |

The interval resamples paired clips. It quantifies this fixed open-loop sample only; it is not a vehicle-safety confidence interval.
