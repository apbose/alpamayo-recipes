# SPDX-License-Identifier: Apache-2.0
"""Recovery regression tests: network causes, bounded retries, queue order."""
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import BadZipFile

from requests import Response
from requests.exceptions import HTTPError

from native_eval_io import exception_chain, is_retryable_data_error, read_with_retries, safe_error_details


def http_error(status=404, url='https://us.aws.cdn.hf.co/archive?secret=not-for-logs'):
    response=Response(); response.status_code=status; response.url=url
    return HTTPError('request failed', response=response)


class NetworkRecoveryTest(unittest.TestCase):
    def test_zip_wrapped_cdn_404_is_retryable(self):
        error=BadZipFile('File is not a zip file')
        error.__context__=http_error()
        self.assertTrue(is_retryable_data_error(error))
        self.assertNotIn('secret',str(safe_error_details(error)))

    def test_cause_chain_and_cycle(self):
        inner=http_error(503); outer=BadZipFile('outer')
        outer.__cause__=inner; inner.__context__=outer
        self.assertEqual(len(list(exception_chain(outer))),2)
        self.assertTrue(is_retryable_data_error(outer))

    def test_bare_zip_corruption_not_network(self):
        self.assertFalse(is_retryable_data_error(BadZipFile('corrupt archive')))

    def test_missing_repo_and_auth_errors_not_retried(self):
        for status in (401,403,404):
            self.assertFalse(is_retryable_data_error(http_error(status,'https://huggingface.co/datasets/missing')))

    def test_fresh_handle_retry_returns_same_sample(self):
        calls=[]; refreshed=[]; sample=object()
        def load():
            calls.append(1)
            if len(calls)==1:
                error=BadZipFile('wrapped'); error.__context__=http_error()
                raise error
            return sample
        self.assertIs(read_with_retries(load,lambda:refreshed.append(1),label='clip',sleep=lambda _:None),sample)
        self.assertEqual((len(calls),len(refreshed)),(2,1))

    def test_retry_is_bounded_no_silent_skip(self):
        calls=[]; refreshed=[]
        def load():
            calls.append(1); raise http_error(503)
        with self.assertRaises(HTTPError):
            read_with_retries(load,lambda:refreshed.append(1),label='clip',attempts=3,sleep=lambda _:None)
        self.assertEqual((len(calls),len(refreshed)),(3,2))

    def test_corrupt_zip_is_not_swallowed(self):
        def load(): raise BadZipFile('real corruption')
        with self.assertRaises(BadZipFile):
            read_with_retries(load,lambda:self.fail('should not refresh'),label='clip',sleep=lambda _:None)

    def test_public_model_kinds_only(self):
        from run_native_gold_suite import KINDS, TRAINED
        self.assertEqual(KINDS, {'R0', 'A6', 'A8'})
        self.assertEqual(TRAINED, {'A6', 'A8'})

    def test_requested_stages_first_and_independent(self):
        from run_native_gold_suite import build_jobs, execute_jobs
        entries=[dict(name='R0',kind='R0'),dict(name='A6',kind='A6'),
                 dict(name='A8',kind='A8')]
        jobs=build_jobs(entries)
        self.assertEqual([(e['name'],s) for e,s in jobs],
            [('A6',10),('A6',5),('A8',10),('A8',5),('R0',10),('R0',5)])
        calls=[]
        def run(entry,steps):
            calls.append((entry['name'],steps))
            return entry['name']!='A6'
        outcomes=execute_jobs(jobs,run,lambda:None)
        self.assertEqual(len(calls),6)
        self.assertFalse(outcomes['A6_10step'])
        self.assertTrue(outcomes['A8_5step'])


if __name__=='__main__':
    unittest.main()
