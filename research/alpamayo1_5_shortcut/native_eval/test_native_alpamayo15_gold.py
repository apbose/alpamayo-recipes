# SPDX-License-Identifier: Apache-2.0
"""CPU regression checks for the corrected native golden evaluation wrapper.

Run with the evaluator's PYTHONPATH/environment:
    python -m unittest discover -s research/alpamayo1_5_shortcut/native_eval -p test_native_alpamayo15_gold.py -v
These are not replacements for the mandatory native GPU smoke test.
"""
import copy
import unittest

import numpy as np
import torch
from alpamayo1_5 import helper
from evaluate_native_alpamayo15_gold import audit_prompt, metric


class NativeProtocolTest(unittest.TestCase):
    def messages(self, labels=True):
        return helper.create_message(torch.zeros(16, 3, 2, 2),
            camera_indices=torch.tensor([0, 1, 2, 6]) if labels else None)

    def test_native_reasoning_prefix_and_all_labels(self):
        report = audit_prompt(self.messages())
        self.assertEqual(report['assistant_prefix'], '<|cot_start|>')
        self.assertEqual(report['frame_labels'], 16)
        self.assertFalse(report['future_target_in_model_inputs'])

    def test_reject_old_trajectory_only_prefix(self):
        messages = copy.deepcopy(self.messages())
        messages[-1]['content'][0]['text'] = '<|traj_future_start|>'
        with self.assertRaises(AssertionError):
            audit_prompt(messages)

    def test_reject_missing_camera_frame_labels(self):
        with self.assertRaises(AssertionError):
            audit_prompt(self.messages(labels=False))

    def test_reject_a_missing_frame_label(self):
        messages = self.messages()
        content = messages[1]['content']
        content.remove(next(item for item in content if item.get('text') == 'frame 0 '))
        with self.assertRaises(AssertionError):
            audit_prompt(messages)

    def test_full_horizon_min_over_six_candidates(self):
        pred = np.zeros((6, 64, 2), dtype=np.float32)
        pred[:, :, 0] = np.arange(6)[:, None]
        result = metric(pred, np.zeros((64, 2), dtype=np.float32))
        self.assertEqual(result['candidate_ade_m'], [0, 1, 2, 3, 4, 5])
        self.assertEqual(result['min_ade_m'], 0)
        self.assertEqual(result['ade_mean6_m'], 2.5)

    def test_reject_nonfinite_prediction(self):
        pred = np.zeros((6, 64, 2), dtype=np.float32)
        pred[0, 0, 0] = np.nan
        with self.assertRaises(AssertionError):
            metric(pred, np.zeros((64, 2), dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
