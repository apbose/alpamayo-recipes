# Data and fixed splits

## Dataset scope

The PhysicalAI-AV metadata index contains 306,152 clips. This pilot did not
use the full corpus. It used every valid clip found in 19 locally downloaded
chunks, approximately 100 GB for the four required cameras, camera calibration,
and ego-motion labels.

```text
Full metadata index: 306,152 clips
                 |
          select 19 chunks
                 |
        1,877 valid clips
                 |
       audit 12 t0 values/clip
                 |
       22,524 valid windows
```

A clip is a 20-second sensor recording. A training window selects one `t0`
inside a clip and loads:

- four cameras;
- four frames per camera at 10 Hz;
- 16 ego-history poses ending at `t0`;
- 64 future poses after `t0`, covering 6.4 seconds at 10 Hz.

The audit used `t0 = 2, 3, ..., 13` seconds, so every clip contributed 12
windows while preserving the required history and future margins. All 22,524
windows passed shape, finiteness, timestamp, rotation, and decode checks.

## Why split by clip

Nearby frames from one 20-second clip share the same road, traffic, lighting,
and actors. Splitting individual frames would leak almost-identical scenes
between training and validation. We retained PhysicalAI-AV's official
train/validation/test clip assignment and kept every window from a clip in the
same split.

| Official split in the 19 chunks | Unique clips | All 12 windows |
|---|---:|---:|
| Train | 891 | 10,692 |
| Validation | 688 | 8,256 |
| Test | 298 | 3,576 |
| Total | 1,877 | 22,524 |

For the matched pilot evaluation, the validation manifest contains 32 unique
clips and exactly one timestamp (`t0 = 7 s`) per clip. Selection was
deterministic, seed 42, and stratified to cover every available validation
chunk.

## Navigation versus route-less data

NVIDIA publishes `nav_demo_samples.json` with 20 timestamp-specific navigation
rows over 19 unique clips. Applying the official clip split produces:

| Split | Navigation rows | Unique clips |
|---|---:|---:|
| Train | 10 | 9 |
| Validation | 7 | 7 |
| Test | 3 | 3 |

Those rows were used for navigation smoke tests. They are too small for the
10,692-row pilot: only 10 official-training rows have genuine `nav_text`.
Navigation commands cannot safely be copied to other timestamps because the
maneuver and distance change with time. The full pilot therefore used the
explicit `PAITrajectoryDataset` route-less loader and did not fabricate route
text.

## Checked-in manifests

| Manifest | Rows | SHA-256 |
|---|---:|---|
| `route_less_19chunks/train.json` | 10,692 | `5035dc3e0d525f1cfeea530826fc023f0d9021117580b6a34950d4a7fed651b9` |
| `route_less_19chunks/val.json` | 32 | `0d70d2422e7dcc10800f8d16ba642af49d8113d627882436a7a46b3f0fae0d31` |
| `route_less_19chunks/test.json` | 32 | `1b45ff0ec46f0fb82703473fde67013add160e5156db7ce11bcb019faa1faf82` |
| `nav_smoke/nav_train.json` | 10 | `614da858fb9ab547bc14714aca0edc12f8ebfb3e345d2c40a62fbcca67471203` |
| `nav_smoke/nav_val.json` | 7 | `721a119f35947b509985108cfeb060e481f9cc09eb20892ac19c9e8355f928ef` |
| `nav_smoke/nav_test.json` | 3 | `4a0c77f323f4903177efd364dd6d623a2e32df32201e16984bd06b90f86a157b` |

The manifests contain identifiers and timestamps only. Camera archives,
checkpoints, and Hugging Face credentials are deliberately excluded from Git.
