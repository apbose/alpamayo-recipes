# Results

## Evaluation protocol

Released and trained checkpoints were evaluated with the same route-less
loader, manifest, metric code, and hardware:

- 32 unique official-validation clips, one `t0 = 7 s` window per clip;
- manifest SHA-256 `0d70d2422e7dcc10800f8d16ba642af49d8113d627882436a7a46b3f0fae0d31`;
- six stochastic trajectory candidates per scene;
- random seed reset to 42 before every solver-count evaluation;
- one unmeasured warm-up sample per solver count;
- eager attention throughout;
- one NVIDIA B300 GPU.

Lower is better for all three quality metrics.

## Released Alpamayo 1.5 checkpoint

| Solver calls | minADE (m) | ADE (m) | Corner distance (m) | Expert latency (ms) | End-to-end (ms) |
|---:|---:|---:|---:|---:|---:|
| 10 | 1.1766 | 2.1159 | 1.1567 | 402.75 | 10,711.42 |
| 8 | 1.1844 | 2.0803 | 1.1738 | 321.61 | 10,625.59 |
| 4 | 1.2679 | 1.9758 | 1.2626 | 159.23 | 10,443.43 |
| 2 | 1.5032 | 1.9614 | 1.5286 | 78.14 | 10,381.72 |

This is one released checkpoint forced through different Euler solver budgets,
not four separately trained models.

## Trained shortcut checkpoint

| Solver calls | minADE (m) | ADE (m) | Corner distance (m) | Expert latency (ms) | End-to-end (ms) |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.7818 | 1.4117 | 0.8117 | 404.02 | 10,711.37 |
| 8 | 0.7975 | 1.4008 | 0.8208 | 322.18 | 10,636.62 |
| 4 | 0.8128 | 1.3655 | 0.8408 | 159.56 | 10,449.01 |
| 2 | 0.8917 | 1.4229 | 0.9231 | 78.26 | 10,402.06 |

At the same two-step budget, the trained pilot was better than the released
checkpoint by 40.68% minADE, 27.46% ADE, and 39.61% corner distance. This
confirms that training changed the model usefully on this small open-loop
subset; it does not establish generalization or safety.

## Predeclared two-step gate

The candidate two-step path was compared primarily with its own trained
high-step path. The allowed quality regression was 10%, the required
Action-Expert speedup was 4x, and a matching safety report was mandatory.

| Check | Two-step change versus trained 10-step | Result |
|---|---:|---|
| minADE | +14.07% | Fail |
| ADE | +0.79% | Pass |
| Corner distance | +13.73% | Fail |
| Action-Expert latency | 5.16x faster | Pass |
| Collision/off-road/traffic safety | Not evaluated | Unavailable/fail |

Decision: **do not pursue one-step yet**. This means the one-step benchmark was
withheld under the mentor's gate; it does not mean one-step code is impossible.

## Training evidence

- 10,692 optimizer updates completed.
- All recorded losses and gradients were finite.
- Flow loss, first 100 versus last 100 updates: `0.7662 -> 0.4257`.
- Shortcut loss: `0.05335 -> 0.02147`.
- Weighted objective: `0.6771 -> 0.3752`.
- Each shortcut interval was sampled exactly 3,564 times.
- The saved checkpoint contained five model shards, 1,172 indexed tensors,
  optimizer/scheduler/RNG state, and nonzero learned adapter tensors.

## Metric interpretation

For candidate `k`, ADE averages XY displacement error over the 64 future
points. minADE selects the lowest-ADE complete candidate among the six, then
averages over scenes. It is oracle-like because the future ground truth is
used to choose the candidate. In NVIDIA's current metric implementation, the
reported plain ADE selects candidate index zero because dummy log-probabilities
are tied; it is not the mean over six candidates. Corner distance is also an
open-loop geometric diagnostic, not a collision or off-road safety metric.

## Scope limitations

The experiment covers 19 chunks rather than the full corpus, uses only 32
validation clips, has no confidence intervals, is route-less, and has no
closed-loop safety evaluator. The 5.16x number applies only to the iterative
Action Expert. End-to-end latency improved by approximately 2.9% because the
frozen VLM and other work dominate the measured call.
