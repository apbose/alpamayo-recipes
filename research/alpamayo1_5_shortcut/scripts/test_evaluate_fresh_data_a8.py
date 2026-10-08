#!/usr/bin/env python3
"""CPU-only report regression tests; never start evaluation or training."""
import unittest
from unittest.mock import patch

from evaluate_fresh_data_a8 import METRICS, merge_results, missing_steps, report_text


def benchmark(values):
    return dict(results={str(step): dict(metrics={key: value for key in METRICS},
        latency_ms=dict(action_expert_diffusion=dict(mean=step * 10.0),
                        end_to_end_model=dict(mean=1000.0 + step * 10)))
        for step, value in values.items()})


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.reference = benchmark({10: 1.0, 5: 1.2, 4: 1.3, 2: 1.5})

    def test_pending_does_not_invent_results(self):
        text, rows = report_text({}, self.reference, state="evaluating", output="test")
        self.assertEqual(rows, [])
        self.assertIn("0/4", text)
        self.assertEqual(text.count("Pending"), 12)

    def test_same_step_and_own_baseline_are_distinct(self):
        result = benchmark({10: 0.8, 5: 0.9, 4: 1.0, 2: 1.2})
        text, rows = report_text(result, self.reference, state="complete", output="test")
        two = next(row for row in rows if row["steps"] == 2 and row["metric"] == "min_ade")
        self.assertAlmostEqual(two["a8_vs_a6_percent"], -20.0)
        self.assertAlmostEqual(two["a8_vs_own_10_percent"], 50.0)
        self.assertIn("5.00x", text)
        self.assertIn("No cross-run A6/A8 latency speedup", text)

    def test_partial_sweep_keeps_missing_cells_pending(self):
        text, rows = report_text(benchmark({10: 0.8}), self.reference, state="evaluating", output="test")
        self.assertEqual(len(rows), 3)
        self.assertIn("1/4", text)
        self.assertEqual(text.count("Pending"), 9)

    def test_cross_run_timing_is_not_a_speedup(self):
        result = benchmark({10: 0.8, 5: 0.9, 4: 1.0, 2: 1.2})
        result["step_sources"] = {step: dict(path="old.json" if step != "2" else "new.json")
                                  for step in result["results"]}
        text, rows = report_text(result, self.reference, state="complete", output="test")
        self.assertNotIn("5.00x", text)
        self.assertIn("2.00x", text)
        self.assertIn("| 2 | 20.00 | 1.020 | — |", text)
        self.assertAlmostEqual(rows[-3]["a8_vs_own_10_percent"], 50.0)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.previous = benchmark({10: 0.8, 5: 0.9, 4: 1.0})
        self.current = benchmark({2: 1.2})
        self.previous["configuration"] = dict(seed=42, inference_steps=[10, 5, 4, 2])
        self.current["configuration"] = dict(seed=42, inference_steps=[2])
        self.previous["step_sources"] = {step: dict(path="old.json", sha256="old-hash")
                                        for step in self.previous["results"]}

    def test_only_missing_two_step_is_requested(self):
        self.assertEqual(missing_steps(self.previous), [2])
        self.assertEqual(missing_steps({}), [10, 5, 4, 2])
        self.assertEqual(missing_steps(benchmark({s: 1.0 for s in (10, 5, 4, 2)})), [])

    def test_unexpected_step_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            missing_steps(benchmark({1: 1.0}))

    @patch("evaluate_fresh_data_a8.digest", return_value="new-hash")
    def test_merge_preserves_previous_results_and_provenance(self, _digest):
        result = merge_results(self.previous, self.current, "new.json")
        self.assertEqual(set(result["results"]), {"10", "5", "4", "2"})
        self.assertEqual(result["results"]["10"], self.previous["results"]["10"])
        self.assertEqual(result["step_sources"]["2"], dict(path="new.json", sha256="new-hash"))
        self.assertEqual(result["step_sources"]["10"], self.previous["step_sources"]["10"])
        self.assertTrue(result["merged_across_runs"])
        self.assertIsNone(result["results"]["2"]["comparison_to_10_step"])
        self.assertNotIn("2", self.previous["results"])
        self.assertNotIn("comparison_to_10_step", self.current["results"]["2"])

    def test_configuration_drift_rejected(self):
        self.current["configuration"]["seed"] = 43
        with self.assertRaisesRegex(ValueError, "configuration changed: seed"):
            merge_results(self.previous, self.current, "new.json")

    def test_overwriting_completed_results_rejected(self):
        with self.assertRaisesRegex(ValueError, "overwrite completed"):
            merge_results(self.previous, self.previous, "new.json")


if __name__ == "__main__":
    unittest.main()
