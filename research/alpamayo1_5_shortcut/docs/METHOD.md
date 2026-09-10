# Method

## Stage-2 baseline

The released Alpamayo 1.5 Stage-2 path freezes the VLM and trains the Action
Expert. The VLM still runs to produce scene-context KV cache; “frozen” means
its parameters receive no optimizer updates, not that it is skipped.

```text
4-camera history + 16 ego-history poses + prompt
                         |
                   frozen VLM
                         |
                  scene KV cache
                         |
Gaussian actions x_t [64,2] + flow time t
                         |
                  Action Expert
                         |
             velocity field [64,2]
                         |
          repeated Euler solver updates
                         |
       acceleration/curvature [64,2]
                         |
              unicycle integration
                         |
             future XY trajectory
```

The 64 future ground-truth poses are differentiated/inverted into 64 clean
`(acceleration, curvature)` targets for training. They do not come from the
VLM.

## Solver-step adapter

The original action projection builds one 2,048-dimensional embedding for each
of the 64 action tokens using acceleration, curvature, and flow time `t`. We
add a residual branch for solver interval `d`:

```text
d
|
Fourier encoding (20 features)
|
Linear 20 -> 256 -> SiLU -> Linear 256 -> 2,048
|
add the same d embedding to all 64 action-token embeddings
```

Formally:

```text
e_i = ActionProjection(x_t[i], t) + StepAdapter(d),  i = 1..64
```

The final adapter layer is zero-initialized. A released checkpoint therefore
initially produces the same output as the unmodified projection. Training
updates the Action Expert, action projections, and adapter; the VLM remains
frozen. The adapter has 531,712 parameters.

Solver `d` is not the physical 0.1-second spacing between trajectory points.
It is the interval used to advance the entire noisy 64x2 action tensor through
the noise-to-data flow:

| `d` | Solver calls over `[0,1]` |
|---:|---:|
| 0.125 | 8 |
| 0.25 | 4 |
| 0.5 | 2 |
| 1.0 | 1 |

Ten-step inference uses `d = 0.1`, which was accepted by the Fourier adapter
but was not explicitly present in the dyadic training hierarchy.

## Training objective

Let clean actions be `x`, Gaussian noise be `eps`, and flow time be `t`:

```text
z_t = t*x + (1-t)*eps
flow target = x - eps
```

The flow anchor uses `d = 0.125`:

```text
L_flow = MSE(s(z_t, t, 0.125), x - eps)
```

For one sampled shortcut interval `D` from `{0.25, 0.5, 1.0}`, the online
stopped-gradient teacher makes two half steps:

```text
v1       = stopgrad(s(z_t,             t,       D/2))
z_mid    = clip(z_t + (D/2)*v1)
v2       = stopgrad(s(z_mid,           t+D/2,   D/2))
v_target = clip((v1 + v2)/2)
L_short  = MSE(s(z_t, t, D), v_target)
```

The pilot objective was:

```text
L = 0.875 * L_flow + 0.125 * L_short
```

Because the physical batch size was one, every sample ran both branches. This
matches the intended scalar weighting but is not the reference Shortcut Models
batch estimator, which normally assigns approximately seven eighths of a batch
to flow-only examples and one eighth to shortcut-only examples. The teacher
was the current online expert in evaluation/no-gradient mode, not an EMA copy.

## Training schedule

- One route-less `D` per update, balanced cycle over `0.25, 0.5, 1.0`.
- 10,692 updates: exactly 3,564 uses of each `D`.
- Per-device batch size one; no gradient accumulation.
- One epoch over all official-training windows in the local 19 chunks.
- VLM KV cache computed once per sample and detached/reused by the expert
  branches.

## Important limitations of this implementation

- Discrete dyadic step conditioning was trained; arbitrary continuous `d` was
  not established.
- `d = 0.1` for ten steps is outside the trained dyadic ladder.
- The current online teacher can move during training; no EMA-teacher ablation
  was run.
- `D = 1.0` was included during training even though one-step evaluation was
  withheld after the two-step gate failed.
- The main experiment omitted navigation conditioning because genuine route
  text was unavailable at scale.
