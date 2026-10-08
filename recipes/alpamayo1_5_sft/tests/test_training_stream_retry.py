"""Opt-in transport settings must reach spawned workers, without CUDA/network."""
from functools import partial
import torch
from torch.utils.data import DataLoader, Dataset
from alpamayo1_5_sft.train_paper_ema import configure_training_stream_retries


class RetryProbe(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        from huggingface_hub import hf_file_system
        return hasattr(hf_file_system.http_backoff, '_gold_retry_original')


def test_default_leaves_library_unchanged():
    from huggingface_hub import hf_file_system
    original = hf_file_system.http_backoff
    assert configure_training_stream_retries(max_attempts=1) is None
    assert hf_file_system.http_backoff is original


def test_spawned_worker_receives_retry_policy():
    loader = DataLoader(RetryProbe(), batch_size=1, num_workers=1,
                        multiprocessing_context='spawn',
                        worker_init_fn=partial(configure_training_stream_retries, max_attempts=6))
    assert torch.equal(next(iter(loader)), torch.tensor([True]))
