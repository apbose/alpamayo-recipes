# Alpamayo 1.5 two-step gate

- Overall gate passed: **False**
- Decision: **do_not_pursue_one_step**
- Quality smoke checks passed: False
- Action-Expert latency check passed: True
- Safety evidence passed: False

Evidence scope: 128-clip-open-loop-pilot. This is not statistically sufficient evidence for vehicle deployment.

## Quality

- min_ade: two-step=1.441144; passed=False
- ade: two-step=2.039533; passed=True
- corner_distance: two-step=1.472157; passed=False

## Safety

- Status: not_evaluated
- Reason: NVIDIA's public Stage-2 recipe reports displacement distance only; it has no collision, off-road, or traffic-rule evaluator. Finite outputs are necessary but are not safety evidence.
