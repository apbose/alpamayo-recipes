# SPDX-License-Identifier: Apache-2.0
"""Shared native-evaluation data access using NVIDIA\'s public PhysicalAI API.

No full-chunk download, no invented route/reasoning text, no skipped samples.
The golden manifest has mixed official splits: held-out status is guaranteed
by explicit clip exclusion from this experiment, not by relabeling its clips.
"""
from __future__ import annotations

from functools import wraps
from contextlib import contextmanager
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading
import huggingface_hub
from physical_ai_av import PhysicalAIAVDatasetInterface
import hashlib
import json
import math
from pathlib import Path

import torch

HF_REVISION = '33f9bf447ed3bcb7d545ce13f4226f824214fafb'
REQUIRED_FEATURES = ['egomotion', 'camera_cross_left_120fov', 'camera_front_wide_120fov',
                     'camera_cross_right_120fov', 'camera_front_tele_30fov']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure_retries(attempts=6):
    """Same process-local range-read retry policy used by the 10B gold runner."""
    from huggingface_hub import hf_file_system
    original = getattr(hf_file_system.http_backoff, '_stream_original', hf_file_system.http_backoff)

    @wraps(original)
    def retrying(method, url, **kwargs):
        kwargs.update(max_retries=attempts-1, base_wait_time=2, max_wait_time=30,
                      retry_on_status_codes=(408, 429, 499, 500, 502, 503, 504))
        return original(method, url, **kwargs)

    retrying._stream_original = original
    hf_file_system.http_backoff = retrying


class PinnedPhysicalAIAVDatasetInterface(PhysicalAIAVDatasetInterface):
    @contextmanager
    def open_file(self, filename, mode='rb', maybe_stream=False):
        # The installed official helper's remote path omits revision. Add @sha
        # here so remote bytes, not merely the metadata, use the recorded pin.
        cached=huggingface_hub.try_to_load_from_cache(
            filename=filename,cache_dir=self.cache_dir,**self.repo_snapshot_info)
        if isinstance(cached,str):
            with open(cached,mode) as handle:
                yield handle
        elif maybe_stream:
            with self.fs.open(f'datasets/{self.repo_id}@{self.revision}/{filename}',mode) as handle:
                yield handle
        else:
            raise FileNotFoundError(filename)


def official_interface(cache, revision=HF_REVISION):
    if revision != HF_REVISION:
        raise ValueError('This experiment requires the pinned A8 dataset revision')
    from physical_ai_av import PhysicalAIAVDatasetInterface
    configure_retries()
    return PinnedPhysicalAIAVDatasetInterface(revision=revision, cache_dir=cache)


def validate_rows(rows, avdi, split=None):
    if not rows or len({(r['clip_id'], r['t0_relative']) for r in rows}) != len(rows):
        raise ValueError('Empty manifest or duplicate sample windows')
    ids = list({r['clip_id'] for r in rows})
    selected = avdi.clip_index.loc[ids]
    if not selected['clip_is_valid'].all():
        raise ValueError('Invalid clips in official index')
    if split is not None and not selected['split'].eq(split).all():
        raise ValueError(f'Manifest contains non-{split} official clips')
    if not avdi.feature_presence.loc[ids, REQUIRED_FEATURES].all().all():
        raise ValueError('Required camera or ego-motion feature missing')
    for row in rows:
        if 'nav_text' in row or not 1_600_000 < row['t0_relative'] <= 13_600_000:
            raise ValueError('Invalid route-less row or timestamp')
        if 'split' in row and row['split'] != str(avdi.clip_index.at[row['clip_id'], 'split']):
            raise ValueError('Manifest split does not match official split')


def validate_sample(sample):
    assert sample['image_frames'].shape[:3] == (4, 4, 3)
    assert sample['ego_history_xyz'].shape[-2:] == (16, 3)
    assert sample['ego_future_xyz'].shape[-2:] == (64, 3)
    for value in sample.values():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            if not torch.isfinite(value).all():
                raise FloatingPointError('Nonfinite decoded sample')
    return sample


class OrderedPrefetch:
    """Bounded CPU/IO prefetch. Futures are consumed in exact sampler order.

    Each IO thread owns its dataset interface. No GPU work or RNG draws happen
    here. Exceptions stop the run; no failed clips are replaced or skipped.
    """
    def __init__(self, indices, factory, workers=4, capacity=8):
        if workers<1 or capacity<workers:
            raise ValueError('Prefetch capacity must be at least the positive worker count')
        self.indices=iter(indices)
        self.factory=factory
        self.local=threading.local()
        self.pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='hf-sample')
        self.pending=deque()
        for _ in range(capacity):
            if not self._submit():
                break

    def _load(self,index):
        if not hasattr(self.local,'dataset'):
            self.local.dataset=self.factory()
        return self.local.dataset[index]

    def _submit(self):
        try:
            index=next(self.indices)
        except StopIteration:
            return False
        self.pending.append((index,self.pool.submit(self._load,index)))
        return True

    def get(self,index):
        expected,future=self.pending.popleft()
        if index!=expected:
            raise ValueError(f'Prefetch order mismatch: {index} != {expected}')
        sample=future.result()
        self._submit()
        return sample

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True)
