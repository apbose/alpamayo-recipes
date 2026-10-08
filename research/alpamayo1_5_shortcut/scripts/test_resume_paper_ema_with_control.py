"""CPU-only tests for the deferred A7 queue; never start a GPU job."""
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from resume_paper_ema_with_control import (
    checkpoint_storage_budget, dependency_ready, validate_control_plan,
)


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def dependency(self, status, host=None):
        (self.root / 'status.json').write_text(json.dumps(dict(
            status=status, host=host or socket.gethostname(), pid=12345)))

    @patch('resume_paper_ema_with_control.os.kill')
    def test_running_dependency_waits(self, kill):
        self.dependency('evaluating')
        self.assertFalse(dependency_ready(self.root))
        kill.assert_called_once_with(12345, 0)

    @patch('resume_paper_ema_with_control.os.kill', side_effect=ProcessLookupError)
    def test_dead_dependency_does_not_launch(self, _kill):
        self.dependency('evaluating')
        with self.assertRaises(ProcessLookupError):
            dependency_ready(self.root)

    def test_failed_dependency_does_not_launch(self):
        self.dependency('failed')
        with self.assertRaisesRegex(RuntimeError, 'did not succeed'):
            dependency_ready(self.root)

    def test_different_host_rejected(self):
        self.dependency('evaluating', host='different-host')
        with self.assertRaisesRegex(ValueError, 'different host'):
            dependency_ready(self.root)

    @patch('evaluate_gold_r0_a6.validate_result')
    def test_complete_requires_both_validated_results(self, validate):
        self.dependency('complete')
        for name in ('R0', 'A6'):
            (self.root / f'{name}_merged_benchmark_results.json').write_text(json.dumps(
                dict(results={str(step): {} for step in (10, 5, 4, 2)})))
        self.assertTrue(dependency_ready(self.root))
        self.assertEqual(validate.call_count, 2)
        validate.side_effect = ValueError('incomplete results')
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            dependency_ready(self.root)

    def test_complete_label_cannot_hide_missing_sweeps(self):
        self.dependency('complete')
        (self.root / 'R0_merged_benchmark_results.json').write_text('{"results": {"10": {}}}')
        with self.assertRaisesRegex(ValueError, 'incomplete solver counts'):
            dependency_ready(self.root)

    def test_budget_counts_two_checkpoints_and_reserve(self):
        (self.root / 'weights').write_bytes(b'0123456789')
        (self.root / 'optimizer').write_bytes(b'01234')
        self.assertEqual(checkpoint_storage_budget(self.root), 30 + 20 * 1024**3)

    def test_different_plan_rejected(self):
        original, control = self.root / 'a6', self.root / 'a7'
        original.mkdir()
        control.mkdir()
        (original / 'raw_batch_plan.json').write_text('[[1,2]]')
        (control / 'raw_batch_plan.json').write_text('[[2,1]]')
        with self.assertRaisesRegex(ValueError, 'batch plans differ'):
            validate_control_plan(original, control)


if __name__ == '__main__':
    unittest.main()
