"""Opt-in, process-local retries for the installed HF filesystem's HTTP reads.

No installed library file is edited. The original HF backoff implementation
still performs every request, including auth, Range headers and timeout handling.
Only its retry policy is changed; no sample is skipped and no RNG is consumed.
"""
from __future__ import annotations

from functools import wraps

RETRY_STATUS_CODES = (408, 429, 499, 500, 502, 503, 504)


def configure_hf_stream_retries(max_attempts: int = 6) -> dict:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    from huggingface_hub import hf_file_system
    policy = dict(max_attempts=max_attempts, retry_status_codes=list(RETRY_STATUS_CODES),
                  base_wait_seconds=2, max_wait_seconds=30, scope="hf_filesystem_http")
    # Idempotent even when called twice in a smoke/benchmark process.
    current = hf_file_system.http_backoff
    original = getattr(current, "_gold_retry_original", current)

    @wraps(original)
    def retrying_backoff(method, url, **kwargs):
        kwargs.update(max_retries=max_attempts - 1, base_wait_time=2,
                      max_wait_time=30, retry_on_status_codes=RETRY_STATUS_CODES)
        return original(method, url, **kwargs)

    retrying_backoff._gold_retry_original = original
    hf_file_system.http_backoff = retrying_backoff
    return policy
