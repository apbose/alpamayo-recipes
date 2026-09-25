# On-demand PhysicalAI-AV loading

## Short answer

The screenshot is correct about **Alpamayo 2 smoke inference**: NVIDIA's public
loader can open missing PhysicalAI-AV feature archives through Hugging Face and
fetch the data needed for one `(clip_id, t0_us)` sample. It does not, by itself,
demonstrate that Alpamayo 1.5 Stage-2 training streams efficiently or is
supported without changes.

## What NVIDIA's code establishes

The official Alpamayo 2 function
[`load_physical_aiavdataset`](https://github.com/NVlabs/alpamayo2/blob/main/src/alpamayo2_super/load_physical_aiavdataset.py)
defaults `maybe_stream=True`. It requests ego-motion, camera, and calibration
features for one clip and timestamp. The official smoke path calls that loader
for one validation sample.

The underlying
[`PhysicalAIAVDatasetInterface`](https://github.com/NVlabs/physical_ai_av/blob/main/src/physical_ai_av/dataset.py)
passes `maybe_stream` to `open_file`. Its
[`HfRepoInterface.open_file`](https://github.com/NVlabs/physical_ai_av/blob/main/src/physical_ai_av/utils/hf_interface.py)
uses a local cached file when present; otherwise it opens the remote file with
Hugging Face's filesystem interface. For ZIP features, the reader then opens
the chunk archive and reads the member belonging to the requested clip.

```text
(clip_id, t0_us)
       |
map clip -> packed chunk files
       |
local cache exists? -- yes --> local read
       |
       no
       v
Hugging Face remote file / range-seek
       |
read selected clip members and decode frames
```

This is on-demand file access, not the Hugging Face `datasets` library's
`IterableDataset` training abstraction.

## Original local-only baseline

NVIDIA's Alpamayo 1.5 Stage-2 recipe uses
`alpamayo.data.pai_utils.PhysicalAIAVDatasetLocalInterface`. Our
`PAITrajectoryDataset` was built on that same local interface. Therefore the
original shortcut pilot required the selected camera/calibration/ego-motion
chunk files on disk. Local access remains the default; this branch now also
supports `access_mode=hf_stream` with `maybe_stream=True`.

The Alpamayo 2 inference loader also defaults to seven cameras, whereas this
Alpamayo 1.5 Stage-2 recipe uses four. It is a useful reference implementation,
not a drop-in training dataset for this experiment.

## Can we train on demand?

Yes: `PAITrajectoryDataset` now accepts the explicit HF-backed interface switch,
and the 5,295-clip training experiments completed using it. Randomized,
multi-worker training repeatedly seeks inside large packed archives; network
latency, retries, rate limits, worker contention, and cache behavior can leave
the GPU idle. Multiple windows from the same clip/chunk make persistent local
caching especially valuable.

Recommended modes:

| Mode | Good for | Main trade-off |
|---|---|---|
| Pure remote access | one-sample smoke or rare clips | minimum disk, least predictable throughput |
| On-demand persistent cache | expanding experiments | downloads only touched assets and reuses them |
| Pre-download selected chunks | long/repeated training | more disk, most reproducible throughput |

For the existing 891-clip/10,692-window run, keeping the 19 selected chunks
local is sensible because every chunk is reused many times. For scaling beyond
those chunks, an on-demand persistent cache is worth implementing and
benchmarking before downloading the complete corpus.

## Measured follow-up

The optional HF-backed interface has now passed exact local/remote parity on a
shared sample and raw shape/finite-value validation on a previously uncached
sample. An explicit 301.765 GB source-shard manifest covers 5,295 new training
clips; see [HF_STREAMING_300GB.md](HF_STREAMING_300GB.md). The full training
path remains a bounded experiment because streaming throughput, not disk
capacity, is the principal risk.

## Remaining throughput study

The interface switch and small-sample parity/throughput checks are implemented.
For a larger systems study, benchmark at least 128 fixed samples with:

- cold cache and warm cache;
- 1, 2, and 4 data-loader workers;
- sample-loading time and GPU idle fraction;
- bytes transferred and persistent cache growth;
- retry/failure count;
- equality of decoded tensor shapes and trajectory labels versus local data;
- a pinned PhysicalAI-AV revision for reproducibility.

Adopt it for training only if the warm-cache path is correct and input loading
does not materially starve the GPU. The GitHub issue template at
`.github/ISSUE_TEMPLATE/on-demand-physicalai-data.md` captures
these acceptance criteria.
