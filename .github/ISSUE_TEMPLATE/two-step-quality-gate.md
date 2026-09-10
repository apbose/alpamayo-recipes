---
name: Two-step quality gate follow-up
about: Track work required before evaluating one-step shortcut inference
title: "[Shortcut] Improve the two-step quality gate"
labels: enhancement
assignees: ''
---

## Goal

Improve two-step Alpamayo 1.5 shortcut quality before evaluating one-step.

## Current gate

- minADE regression versus trained 10-step: +14.07% (fail)
- ADE regression: +0.79% (pass)
- corner-distance regression: +13.73% (fail)
- Action-Expert speedup: 5.16x (pass)
- safety evidence: unavailable (fail)

## Proposed decisions

- [ ] Confirm whether the primary reference is trained 10-step, trained 8-step,
      or released 10-step.
- [ ] Confirm discrete dyadic conditioning versus continuous arbitrary `d`.
- [ ] Decide whether to add direct `d=0.1` flow supervision.
- [ ] Decide whether to omit `D=1.0` until the two-step gate passes.
- [ ] Choose online versus EMA teacher.
- [ ] Choose a reference-faithful 7:1 flow/shortcut batch estimator.

## Evaluation requirements

- [ ] Evaluate all 688 available validation clips, or document a power analysis.
- [ ] Run multiple seeds and report confidence intervals.
- [ ] Preserve the fixed clip-disjoint manifest and protocol fingerprints.
- [ ] Add collision/off-road/traffic-rule or closed-loop safety evaluation.
- [ ] Do not report one-step as a success until this gate passes.
