## Goal

Determine whether Hugging Face on-demand PhysicalAI-AV access can feed
Alpamayo 1.5 Stage-2 training without unacceptable GPU starvation or data
mismatch.

## Context

Alpamayo 2 smoke inference uses `load_physical_aiavdataset(...,
maybe_stream=True)`. The current Alpamayo 1.5 shortcut recipe uses
`PhysicalAIAVDatasetLocalInterface`; streaming is not currently wired into its
training dataset. See
`research/alpamayo1_5_shortcut/docs/ON_DEMAND_DATA.md`.

## Scope

- [ ] Add an explicit HF-backed route-less loader/config; keep local mode intact.
- [ ] Pin the PhysicalAI-AV repository revision.
- [ ] Use a fixed, clip-disjoint 128-sample manifest.
- [ ] Compare every decoded field against local loading for a shared subset.
- [ ] Benchmark cold and warm cache with 1, 2, and 4 workers.
- [ ] Record load latency, GPU idle fraction, bytes transferred, cache growth,
      retries, and failures.
- [ ] Test interruption/resume and clear error reporting for expired auth.

## Acceptance criteria

- [ ] Tensor shapes, timestamps, camera order, and trajectories match local mode.
- [ ] Zero missing/corrupt samples in the fixed benchmark.
- [ ] Warm-cache input throughput does not materially starve the training GPU.
- [ ] Credentials and downloaded data remain outside Git.
- [ ] Results and recommendation are committed as a compact JSON/Markdown report.

## Non-goals

This issue does not migrate the experiment to Alpamayo 2 and does not claim
that a one-sample inference loader is already a production training pipeline.
