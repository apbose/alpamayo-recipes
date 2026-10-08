# SPDX-License-Identifier: Apache-2.0
"""Bounded retries for remote sample reads, including ZIP-wrapped HTTP errors.

Never infer archive corruption from an outer BadZipFile alone. Conversely,
never classify a bare corrupt ZIP or a missing repository file as a transient
network failure. Reopening re-resolves the same pinned HF path, not new data.
"""
import json
import time
from urllib.parse import urlsplit

from requests.exceptions import RequestException


def exception_chain(error):
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.extend([current.__cause__, current.__context__])


def is_retryable_data_error(error):
    for current in exception_chain(error):
        if isinstance(current, (TimeoutError, ConnectionError)):
            return True
        if not isinstance(current, RequestException):
            continue
        response = getattr(current, 'response', None)
        if response is None:
            return True
        status = response.status_code
        host = urlsplit(response.url or '').hostname or ''
        if status in (408, 429, 499, 500, 502, 503, 504):
            return True
        # Observed failure was on a signed CDN URL, not a missing HF repo/path.
        if status == 404 and (host.endswith('.hf.co') or host.endswith('.huggingface.co')):
            return True
    return False


def safe_error_details(error):
    """Log types/statuses only: signed query strings and credentials stay out."""
    result = []
    for current in exception_chain(error):
        response = getattr(current, 'response', None)
        result.append(dict(type=type(current).__name__,
            http_status=getattr(response, 'status_code', None),
            host=urlsplit(getattr(response, 'url', '') or '').hostname))
    return result


def read_with_retries(load, refresh, *, label, attempts=4, sleep=time.sleep):
    if attempts < 1:
        raise ValueError('attempts must be positive')
    for attempt in range(1, attempts + 1):
        try:
            return load()
        except Exception as error:
            if not is_retryable_data_error(error) or attempt == attempts:
                raise
            print('DATA_RETRY', json.dumps(dict(sample=label, attempt=attempt,
                next_attempt=attempt + 1, error_chain=safe_error_details(error))), flush=True)
            refresh()
            sleep(min(2 ** attempt, 10))
