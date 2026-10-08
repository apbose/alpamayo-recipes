# A8: A6 with fresh streaming data in each phase

> **Protocol update:** Historical training/128-clip evaluation record. Later auditing found 147 gold-644 clips in A8's available pool and 112 in actual training. The old 'zero overlap' claim below refers only to the heldouts known when this schedule was built, not the later gold set. See [current native gold results and the shared 497-clip cohort](../README.md#current-results-corrected-native-gold-evaluation-2026-10-08).

Status: smoke and full training completed. All 249 updates finished at
2026-09-25 15:36 UTC (about 11 h 38 min training); checkpoints 83/166/249 saved.
Final EMA evaluation at 10/5/4/2 steps launched at 20:43 UTC (PID 568067).
The 10/5/4-step results completed; the worker stopped on SIGHUP on September 26
at 04:11 UTC before completing two steps. The old evaluation status is stale.
On September 28, a 22:40 UTC resume failed before inference because this host's
default HF cache lacked the Cosmos processor/config. The existing cache under
`/path/to/huggingface-cache` contained all required files. Only the missing
two-step evaluation was relaunched at 22:44 UTC with explicit `HF_HOME` and
`HF_HUB_CACHE`, on `umb-b300-dp-160`, GPU 0 (runner PID 19324, benchmark PID 19426).

Run artifacts: `/path/to/alpamayo-assets/runs/alpamayo15_a8_fresh_data_3x5295_b64_20260925_r1`.
Its `status.json` is the live authority; this document records the launch.

Evaluation artifacts: `/path/to/workspace/results/alpamayo15_a8_eval_128clips_20260925_r1`.
These original raw results remain unchanged. Current resume artifacts are
`/path/to/workspace/results/alpamayo15_a8_eval_2step_20260928_r2`.
The failed `r1` attempt is preserved; no checkpoint or dataset download was needed.
The resumed `status.json`, `REPORT.md`, `a8_vs_a6.csv` and
`merged_benchmark_results.json` retain the completed 10/5/4-step measurements and
add two-step results only when all 128 clips finish. The same report updates
`/path/to/workspace/docs/ALPAMAYO15_A8_EVAL_2026-09-25.md` automatically.
The evaluation-only runner is `scripts/evaluate_fresh_data_a8.py` in this project:
one GPU, sequential solver counts, unchanged benchmark, A6 quality comparisons,
paired clip-bootstrap and no wall-clock timeout. Historical A6 latency is not
used to claim a cross-run speedup.

Resume uses the same benchmark source, checkpoint, EMA count, manifest and
inference settings. Nine CPU report/resume tests passed; both saved A8 and A6
artifacts passed protocol validation. A direct one-rank NCCL process replaces
the elastic supervisor; runner and worker are detached and ignore SIGHUP.
There is no wall-clock timeout, but reboot or scheduler termination can stop it.
The current enforced GPU limit is 1100 W; no power setting was changed.
No two-step-versus-historical-ten-step latency speedup will be computed because
the host/power conditions differ. Per-step source paths/hashes are retained.

Partial minADE (metres), before resumed two-step evaluation completes:

| Steps | A6 | A8 | A8 versus A6 | A8 versus own 10 steps |
|---:|---:|---:|---:|---:|
| 10 | 1.2278 | 1.2355 | +0.63% | 0.00% |
| 5 | 1.2409 | 1.2475 | +0.54% | +0.97% |
| 4 | 1.2746 | 1.2789 | +0.34% | +3.51% |
| 2 | 1.4704 | Pending | — | — |

This is a launch-time snapshot; consult the automatically updated report above
for the final two-step cell and paired clip-bootstrap confidence intervals.

Verification: 77 pytest tests plus 7 evaluation-runner tests passed (84 total).
The default 249-update A6 batch plan was compared directly with the saved plan
from the completed A6 run and is identical. A8 phase one matches its first 83
batches exactly. No new commit/push was made as part of this experiment setup.

## Question

Does wider data coverage help the full-hierarchy A6 shortcut model more than
repeated exposure to one small training set, at the same optimizer-update budget?

A5 ran three passes of its 5,295-clip set, but used a different loss hierarchy
and batch size. **A6 is the primary matched comparator for A8**, not A5.

```text
Released Alpamayo 1.5 checkpoint
            |
            v
Phase 1: original A6 set      5,295 clips   updates   1–83
            |  keep weights, optimizer and EMA
            v
Phase 2: new streaming set   5,295 clips   updates  84–166
            |  keep weights, optimizer and EMA
            v
Phase 3: new streaming set   5,295 clips   updates 167–249
            |
            v
Evaluate on the SAME fixed 128 validation clips
```

These are three distinct-data phases, not three conventional epochs over one
dataset. The old A6 runner/defaults and completed checkpoints remain available.

## Fixed versus changed

Unchanged: released initialization, frozen VLM, A6 Action Expert and d adapter,
128-base-step dyadic hierarchy, 16 EMA bootstrap + 48 flow targets per global
batch of 64, eight-GPU/local-microbatch-one execution, 249 updates, LR 1e-4,
AdamW/weight decay 0.1, FP32 trainable/EMA parameters, BF16 compute, EMA 0.999
updated once per optimizer step, eager attention, route-less t0=7 s inputs.

Changed: phases two and three use newly selected official training clips.
Phase one is byte-for-byte the old A6 manifest and its first 83 raw batches match
A6. Phases two/three use deterministic seeds 11/12 for data shuffling; noise and
time seeds continue to use the global optimizer-update number. Each phase pads
its last raw batch from **itself**: 83 × 64 = 5,312 raw draws for 5,295 rows,
so 17 draws repeat. Unlike A6's continuous repeated stream, no raw batch crosses
phase boundaries. This small boundary/shuffle difference is documented, not
hidden as bitwise parity across all updates.

## Data identity and leakage protection

Bundle: `research/alpamayo1_5_shortcut/manifests/a8_fresh_data_3x5295/`.

| Phase | Available unique clips | Chunks | Source ZIP coverage | Updates | Unique clips receiving targets |
|---|---:|---:|---:|---:|---:|
| 1: original A6 | 5,295 | 55 | 301.765 GB | 83 | 3,976 |
| 2: fresh | 5,295 | 55 | 304.023 GB | 83 | 3,974 |
| 3: fresh | 5,295 | 55 | 305.329 GB | 83 | 3,975 |
| Total | 15,885 | 165 | 911.117 GB | 249 | 11,925 |

Coverage is the sum of full source archives touched, **not network bytes
downloaded**. The runtime reads selected clip members via HF range requests.

The reference A6 layout builds 64 targets from only 48 distinct source positions
per raw batch: positions 0–15 are reused in both branches and 48–63 are unused.
Consequently, do not claim all 15,885 manifest clips receive gradient supervision.
There are 15,936 target pairs, of which 3,984 are bootstrap and 11,952 are flow.
Distinct trained clips total 11,925, versus 5,211 in the recorded repeated A6 run.

Checks completed:

- Official pinned HF revision: `33f9bf447ed3bcb7d545ce13f4226f824214fafb`.
- Every selected clip is valid and in the official **train** split.
- Required camera, ego-motion, calibration and vehicle-dimension feature presence.
- Zero clip-ID overlap across the three phases; their chunks are also disjoint.
- Zero overlap against **986 unique held-out clips** collected from every existing
  validation/test manifest: navigation smoke, 32/128-clip evaluation and all
  timestamps/one-per-clip local holdouts. The 986 count is their union, not the
  number evaluated in the benchmark, which remains 128 validation clips.
- The builder excludes the original 19 local chunks from new selection.
- Training rechecks live holdouts and the bundle's recorded holdout IDs, verifies
  phase hashes and combined row order, then snapshots the actual training rows.
- Clip-ID matching rejects leakage even if t0 differs. No failure silently skips
  a row or substitutes another clip.

Only six real payload windows were decoded as a smoke test, **not all 15,885**.
With one process / zero workers, two windows took 44.30 s, 44.41 s and 36.79 s
for phases 1/2/3. These include decoding and are not reliable full-run throughput
estimates; cache state, range reads and eight-GPU concurrency will differ.

## Files

| Purpose | File |
|---|---|
| Official metadata selection | `research/alpamayo1_5_shortcut/scripts/make_fresh_epoch_manifests.py` |
| Hash, leakage and phase/batch checks | `recipes/alpamayo1_5_sft/models/paper_data_plan.py` |
| Optional fresh-data mode | `recipes/alpamayo1_5_sft/train_paper_ema.py` (`--fresh-data-schedule`) |
| Safe preparation/training runner | `research/alpamayo1_5_shortcut/scripts/run_fresh_data_a6.py` |
| Regression tests | `recipes/alpamayo1_5_sft/tests/test_paper_fresh_data.py` |

`schedule.json` records source revision, phase hashes, heldout IDs and metadata
audit. `phase_1.json` through `phase_3.json` are the exact selections; `train.json`
is their ordered concatenation. Worker prefetch does not switch any mutable
dataset: each planned target explicitly indexes the combined immutable rows.

Local preflight evidence is under
`/path/to/workspace/results/alpamayo15_a8_fresh_data_20260925/`:
`plan/protocol.json`, `plan/raw_batch_plan.json`, and the three
`phase_N_streaming_smoke.json` files. Planned coverage is not a completed-training claim.

## Run

From the repository root with the recipe environment active and HF authentication
available, this **prepares/checks only** (use a new output directory):

```bash
python research/alpamayo1_5_shortcut/scripts/run_fresh_data_a6.py \
  --run-dir /path/to/runs/a8-preflight \
  --checkpoint /path/to/Alpamayo-1.5-10B-A1-format \
  --hf-cache /path/to/hf-cache
```

To train, use another new directory and explicitly add `--train`. The runner
executes tests → CPU plan → two real samples per phase → two-update GPU smoke →
fresh 249-update training. Both smoke and full training initialize independently
from the released checkpoint, **not the trained A6 checkpoint**. No automatic
optimizer/RNG resume is implemented; failures stop the pipeline.

```bash
python research/alpamayo1_5_shortcut/scripts/run_fresh_data_a6.py \
  --run-dir /path/to/runs/a8-training \
  --checkpoint /path/to/Alpamayo-1.5-10B-A1-format \
  --hf-cache /path/to/hf-cache \
  --nproc-per-node 8 --train
```

Output: logs, commands, status and `training/checkpoint-83`, `checkpoint-166`,
`checkpoint-249`. The runner has no benchmark/training wall-clock deadline;
launch it with the site's normal detached job mechanism if leaving the session.
Reserve sufficient checkpoint space before launching.

After training, reuse the [matched benchmark command](STREAMING_AND_PAPER_QUICKSTART.md#6-evaluate-a-saved-checkpoint)
with the new checkpoint, unchanged A6 model config, EMA update count 249, steps
10/5/4/2 and the same 128-clip validation manifest. This runner **does not launch
evaluation automatically**. Report absolute quality and within-checkpoint
regression; do not claim safety without collision/off-road checks.

The primary test is A8 versus A6 at matched updates. A5 remains a useful reference,
but cannot isolate fresh-data benefit because its targets, hierarchy, batch size
and number of optimizer updates differ. EMA decay is deliberately not tuned here.
