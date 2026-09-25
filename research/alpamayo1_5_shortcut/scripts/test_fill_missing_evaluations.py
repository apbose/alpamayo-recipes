"""CPU-only regression checks for missing-cell evaluation orchestration."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("missing_evaluations", HERE / "fill_missing_evaluations.py")
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)
tree = ast.parse((HERE / "benchmark_inference_steps.py").read_text())
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "add_ten_step_comparisons")
scope = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), "benchmark_helper", "exec"), scope)
compare = scope["add_ten_step_comparisons"]


def result(error=1.0, milliseconds=100.0):
    return dict(metrics=dict(min_ade=error, ade=error * 2, corner_distance=error),
        latency_ms=dict(end_to_end_model=dict(mean=milliseconds), action_expert_diffusion=dict(mean=milliseconds / 10)),
        per_sample=[dict(sample_index=i) for i in range(128)])


class MissingEvaluationTests(unittest.TestCase):
    def test_existing_ten_step_comparisons_unchanged(self):
        values = {"10": result(1, 200), "5": result(1.25, 100)}
        compare(values)
        self.assertEqual(values["5"]["comparison_to_10_step"], dict(end_to_end_speedup=2.0, action_expert_speedup=2.0, min_ade_delta=0.25))
        self.assertEqual(values["10"]["comparison_to_10_step"]["min_ade_delta"], 0)

    def test_missing_ten_step_has_no_invented_comparison(self):
        values = {"5": result(), "2": result()}
        compare(values)
        self.assertIsNone(values["5"]["comparison_to_10_step"])
        self.assertIsNone(values["2"]["comparison_to_10_step"])

    def test_only_six_requested_cells(self):
        self.assertEqual([(job["id"], job["steps"]) for job in suite.JOBS],
            [("A2", [5]), ("A3", [5]), ("A4", [5]), ("A5", [4, 2]), ("A6", [2])])
        for job in suite.JOBS:
            command = suite.command_for(job, job["steps"], Path("/tmp/example"))
            self.assertIn("benchmark_inference_steps.py", " ".join(command))
            self.assertNotIn("--timeout", command)
            self.assertNotIn("train_paper_ema.py", " ".join(command))

    def make_benchmark(self, checkpoint):
        cfg = dict(validation_manifest_sha256=suite.VAL_SHA, validation_samples=128,
            validation_unique_clips=128, num_traj_samples=6, seed_reset_for_each_step_count=42,
            warmup_samples_per_step_count=1, attention_backend="eager", evaluation_split="val",
            zero_step_size_adapter=False, step_size_adapter_scale=1.0,
            recipe_config="test", gpu="NVIDIA B300 SXM6 AC", torch_version="2.8.0+cu129", torch_cuda_version="12.9",
            checkpoint_config_sha256=suite.digest(checkpoint / "config.json"), checkpoint=str(checkpoint),
            shortcut_runtime=dict(inference_weights="ema", updates=662))
        job = dict(id="test", checkpoint=checkpoint, recipe="test", weights="ema", updates=662)
        return job, dict(configuration=cfg, results={"5": result()})

    def test_protocol_and_ema_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            (checkpoint / "config.json").write_text("{}")
            job, benchmark = self.make_benchmark(checkpoint)
            suite.validate_benchmark(benchmark, job)
            for field, wrong in [("validation_manifest_sha256", "wrong"), ("num_traj_samples", 1), ("seed_reset_for_each_step_count", 7)]:
                changed = copy.deepcopy(benchmark)
                changed["configuration"][field] = wrong
                with self.assertRaises(ValueError):
                    suite.validate_benchmark(changed, job)
            benchmark["configuration"]["shortcut_runtime"]["inference_weights"] = "student"
            with self.assertRaises(ValueError):
                suite.validate_benchmark(benchmark, job)

    def test_incomplete_nonfinite_and_reordered_results_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            (checkpoint / "config.json").write_text("{}")
            job, benchmark = self.make_benchmark(checkpoint)
            bad = copy.deepcopy(benchmark)
            bad["results"]["5"]["per_sample"].pop()
            with self.assertRaises(ValueError):
                suite.validate_benchmark(bad, job)
            bad = copy.deepcopy(benchmark)
            bad["results"]["5"]["per_sample"].reverse()
            with self.assertRaises(ValueError):
                suite.validate_benchmark(bad, job)
            benchmark["results"]["5"]["metrics"]["min_ade"] = float("nan")
            with self.assertRaises(ValueError):
                suite.validate_benchmark(benchmark, job)

    def test_partial_solver_results_are_collected_for_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            destination = output / "A5/attempt_001"
            destination.mkdir(parents=True)
            suite.write_json(destination / "benchmark_results.json", dict(configuration={}, results={"4": result()}))
            historical = {job["id"]: {} for job in suite.JOBS}
            with patch.object(suite, "validate_benchmark"):
                merged = suite.collect_results(output, historical)
            self.assertIn("4", merged["A5"])
            self.assertNotIn("2", merged["A5"])
            self.assertEqual(historical["A5"], {})

    def test_report_labels_quality_and_timing_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            historical = {label: {"10": dict(result=result(), source="historical.json", origin="historical")} for label in suite.LABELS}
            historical["A2"]["5"] = dict(result=result(1.1), source="new.json", origin="new")
            count = suite.write_report(output, output / "DOC.md", historical, dict(status="running", jobs={}))
            self.assertEqual(count, 1)
            report = (output / "REPORT.md").read_text()
            self.assertIn("1.1000 (+10.00%)", report)
            self.assertIn("Not trained", report)
            self.assertIn("do not pool with historical timings", report)
            self.assertFalse(json.loads((output / "summary.json").read_text())["cross_run_latency_speedups_computed"])


if __name__ == "__main__":
    unittest.main()
