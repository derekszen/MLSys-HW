"""Adversarial checks against a copy of the retained HW1 observations."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_hw1_evidence.py"
spec = importlib.util.spec_from_file_location("verify_hw1_evidence", SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class HW1EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.documents = [json.loads((ROOT / "homework-1/results" / name).read_text())
                          for name in ("model-reference.json", "model-validation.json", "benchmark.json")]
        self.reference, self.validation, self.benchmark = self.documents

    def check(self):
        return checker.verify(*self.documents)

    def test_retained_observations_recompute(self):
        result = self.check()
        self.assertTrue(result["passed"])
        self.assertEqual(result["validation_cases"], 3)
        self.assertEqual(result["state_isolation_cases"], 3)
        self.assertEqual(result["benchmark_samples"], 12)

    def test_missing_second_batch_request_fails_despite_pass_flags(self):
        self.validation["state_isolation"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete state-isolation"):
            self.check()

    def test_duplicate_isolation_request_fails(self):
        self.validation["state_isolation"][-1] = copy.deepcopy(self.validation["state_isolation"][-2])
        with self.assertRaisesRegex(ValueError, "duplicate state-isolation"):
            self.check()

    def test_missing_reference_logprob_cannot_be_hidden_by_zip(self):
        self.reference["cases"][0]["selected_logprobs"].pop()
        with self.assertRaisesRegex(ValueError, "reference logprobs"):
            self.check()

    def test_logprobs_must_follow_generated_token_ids(self):
        self.validation["cases"][0]["output"]["meta_info"]["output_token_logprobs"][0][1] += 1
        with self.assertRaisesRegex(ValueError, "token alignment"):
            self.check()

    def test_nonfinite_and_boolean_logprobs_fail(self):
        for value in (float("nan"), float("inf"), True):
            with self.subTest(value=value):
                self.validation["cases"][0]["output"]["meta_info"]["output_token_logprobs"][0][0] = value
                with self.assertRaises(ValueError):
                    self.check()

    def test_recomputed_error_overrides_reported_pass(self):
        self.validation["cases"][0]["output"]["meta_info"]["output_token_logprobs"][0][0] -= 1
        with self.assertRaisesRegex(ValueError, "tolerance exceeded"):
            self.check()

    def test_wrong_reported_error_fails(self):
        self.validation["cases"][0]["max_selected_logprob_error"] = 0
        with self.assertRaisesRegex(ValueError, "reported error"):
            self.check()

    def test_changed_model_path_fails(self):
        self.reference["model"] += "-different-checkpoint"
        with self.assertRaisesRegex(ValueError, "model or precision"):
            self.check()

    def test_missing_and_duplicate_benchmark_samples_fail(self):
        original = copy.deepcopy(self.benchmark["samples"])
        for samples in (original[:-1], original[:-1] + [original[0]]):
            with self.subTest(samples=len(samples)):
                self.benchmark["samples"] = samples
                with self.assertRaisesRegex(ValueError, "benchmark samples"):
                    self.check()

    def test_cached_benchmark_fails(self):
        self.benchmark["samples"][0]["cached_tokens"] = 1
        with self.assertRaisesRegex(ValueError, "cached tokens"):
            self.check()

    def test_wrong_benchmark_output_fails(self):
        self.benchmark["samples"][0]["output_ids"][0] += 1
        with self.assertRaisesRegex(ValueError, "benchmark output"):
            self.check()

    def test_corrupt_timing_and_median_fail(self):
        original = copy.deepcopy(self.benchmark)
        for target in ("ttft_ms", "tpot_ms", "total_ms", "median"):
            with self.subTest(target=target):
                self.documents[2] = copy.deepcopy(original)
                if target == "median":
                    self.documents[2]["summary"][0]["ttft_ms"]["median"] = 1
                else:
                    self.documents[2]["samples"][0][target] = 1
                with self.assertRaises(ValueError):
                    self.check()

    def test_reversed_arrivals_fail(self):
        self.benchmark["samples"][0]["arrivals"].reverse()
        with self.assertRaisesRegex(ValueError, "arrival order"):
            self.check()

    def test_coalesced_first_token_fails(self):
        self.benchmark["samples"][0]["arrivals"].pop(0)
        with self.assertRaisesRegex(ValueError, "first/final token"):
            self.check()

    def test_changed_benchmark_config_fails(self):
        self.benchmark["config"]["dtype"] = "float16"
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            self.check()

    def test_cli_records_hashes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "verification.json"
            command = [sys.executable, str(SCRIPT), "--output", str(output)]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            raw = output.read_bytes()
            self.assertEqual(len(json.loads(raw)["inputs"]), 3)
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(second.returncode, 2)
            self.assertEqual(output.read_bytes(), raw)

    def test_cli_bad_input_is_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text('{"passed": true}')
            result = subprocess.run([sys.executable, str(SCRIPT), "--validation", str(bad)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("malformed HW1 evidence", result.stderr)


if __name__ == "__main__":
    unittest.main()
