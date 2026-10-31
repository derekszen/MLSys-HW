"""CPU-only stdlib comparison tests; not accelerator or full-logit evidence."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_reference.py"
spec = importlib.util.spec_from_file_location("compare_reference", SCRIPT)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def fixtures():
    cases = {"new_tokens": 16, "cases": [
        {"name": "smoke", "input_ids": [1, 2]},
        {"name": "length_32", "input_ids": list(range(32))}]}
    digest = hashlib.sha256(json.dumps(cases).encode()).hexdigest()
    reference = {
        "reference": "transformers.Qwen3_5ForConditionalGeneration, unmodified eager attention",
        "dtype": "bfloat16", "model": "/test/model", "transformers_version": "test",
        "sampling": {"do_sample": False, "new_tokens": 16, "ignore_eos": True},
        "cases": [{**copy.deepcopy(case), "output_ids": [7] * 16,
                   "selected_logprobs": [-1.] * 16,
                   "top5": [{"ids": [7, 8, 9, 10, 11], "logprobs": [-1., -2., -3., -4., -5.]}
                            for _ in range(16)]} for case in cases["cases"]]}
    run = {"status": "generation_completed_not_yet_compared", "mode": "correctness", "tp": 2,
           "physical_devices_requested": [0, 1], "concurrency": 1, "cases_sha256": digest,
           "manifest_sha256": "model manifest", "config_sha256": "model config",
           "config": {**dict.fromkeys(comparison.TP_FIELDS, 2), "ep_size": 1,
                      "pipeline_parallel_size": 1, "dtype": "bfloat16", "model": "/test/model",
                      "sampling_backend": "greedy", "enable_output_logprobs": True},
           "runs": [{"case": case["name"], "repeat": 0, "input_tokens": len(case["input_ids"]),
                     "output_tokens_per_request": 16,
                     "outputs": [{"output_ids": [7] * 16, "meta_info": {
                         "prompt_tokens": len(case["input_ids"]), "completion_tokens": 16,
                         "cached_tokens": 0, "output_token_logprobs": [[-1., 7, None] for _ in range(16)]}}]}
                    for case in cases["cases"]]}
    return reference, run, cases, digest


class ReferenceComparisonTests(unittest.TestCase):
    def setUp(self):
        self.reference, self.run, self.cases, self.digest = fixtures()

    def compare(self, **kwargs):
        return comparison.compare(self.reference, self.run, self.cases,
                                  cases_sha256=self.digest, atol=kwargs.pop("atol", .15), **kwargs)

    def tp1(self):
        result = copy.deepcopy(self.run)
        result.update(tp=1, physical_devices_requested=[0])
        result["config"].update(dict.fromkeys(comparison.TP_FIELDS, 1))
        return result

    def test_valid_and_tolerance(self):
        self.run["runs"][0]["outputs"][0]["meta_info"]["output_token_logprobs"][0][0] = -1.125
        result = self.compare()
        self.assertTrue(result["passed"])
        self.assertEqual(result["summary"], {"request_count": 2, "exact_token_requests": 2,
                                          "tokens_checked": 32, "max_selected_logprob_error": .125})
        self.assertTrue(self.compare(atol=.125)["passed"])
        self.assertFalse(self.compare(atol=.124)["passed"])
        self.assertNotIn("tp1_diagnostics", result)
        self.assertFalse(result["provenance"]["reference_manifest_fingerprint_available"])
        self.assertIn("not full logits or course approval", result["scope"])
        self.assertIn("lacks model manifest/config fingerprints", result["provenance"]["limit"])

    def test_different_input_ids_rejected_even_with_same_length(self):
        self.reference["cases"][0]["input_ids"][0] = 99
        with self.assertRaisesRegex(ValueError, "input IDs"):
            self.compare()

    def test_run_input_provenance_requires_exact_cases_file_fingerprint(self):
        self.run["cases_sha256"] = "different canonical bytes"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.compare()

    def test_different_output_ids_never_compare_logprobs(self):
        output = self.run["runs"][0]["outputs"][0]
        output["output_ids"][3] = 8
        output["meta_info"]["output_token_logprobs"][3] = [-100., 8, None]
        result = self.compare()
        self.assertFalse(result["passed"])
        self.assertIsNone(result["cases"][0]["max_selected_logprob_error"])
        self.assertIsNone(result["summary"]["max_selected_logprob_error"])
        self.assertIn("different tokens/histories", result["cases"][0]["logprob_comparison"])
        self.assertTrue(result["cases"][1]["passed"])

    def test_logprobs_must_be_tied_to_run_output_ids(self):
        self.run["runs"][0]["outputs"][0]["meta_info"]["output_token_logprobs"][0][1] = 8
        with self.assertRaisesRegex(ValueError, "token IDs"):
            self.compare()

    def test_nan_infinite_and_missing_logprobs_rejected_on_either_side(self):
        for side in ("reference", "run"):
            for value in (float("nan"), float("inf"), -float("inf"), None, True):
                with self.subTest(side=side, value=value):
                    self.setUp()
                    if side == "reference":
                        self.reference["cases"][0]["selected_logprobs"][0] = value
                    else:
                        self.run["runs"][0]["outputs"][0]["meta_info"]["output_token_logprobs"][0][0] = value
                    with self.assertRaisesRegex(ValueError, "finite"):
                        self.compare()

    def test_invalid_tolerances(self):
        for atol in (-1, float("nan"), float("inf"), True):
            with self.subTest(atol=atol), self.assertRaises(ValueError):
                self.compare(atol=atol)

    def test_missing_extra_and_reordered_cases(self):
        for side in ("reference", "run"):
            for change in ("missing", "extra", "reverse"):
                with self.subTest(side=side, change=change):
                    self.setUp()
                    rows = self.reference["cases"] if side == "reference" else self.run["runs"]
                    if change == "missing":
                        rows.pop()
                    elif change == "extra":
                        rows.append(copy.deepcopy(rows[0]))
                    else:
                        rows.reverse()
                    with self.assertRaises(ValueError):
                        self.compare()

    def test_mismatched_case_names_lengths_and_repeat_order(self):
        for key, value in (("case", "other"), ("input_tokens", 3), ("repeat", 1),
                           ("output_tokens_per_request", 15)):
            with self.subTest(key=key):
                self.setUp()
                self.run["runs"][0][key] = value
                with self.assertRaises(ValueError):
                    self.compare()

    def test_exactly_16_outputs_and_logprobs_required(self):
        for field in ("reference_ids", "reference_logs", "run_ids", "run_logs", "request_count"):
            with self.subTest(field=field):
                self.setUp()
                output = self.run["runs"][0]["outputs"][0]
                target = {"reference_ids": self.reference["cases"][0]["output_ids"],
                          "reference_logs": self.reference["cases"][0]["selected_logprobs"],
                          "run_ids": output["output_ids"],
                          "run_logs": output["meta_info"]["output_token_logprobs"],
                          "request_count": self.run["runs"][0]["outputs"]}[field]
                target.pop()
                with self.assertRaises(ValueError):
                    self.compare()

    def test_configuration_mismatch_rejected(self):
        changes = [("run", "tp", 1), ("run", "status", "failed"), ("run", "mode", "benchmark"),
                   ("run", "physical_devices_requested", [0, 0]), ("reference", "dtype", "float16"),
                   ("reference", "model", "/different/model"), ("config", "dtype", "float16"),
                   ("config", "ep_size", 2), ("config", "pipeline_parallel_size", 2),
                   ("config", "sampling_backend", "random"), ("config", "enable_output_logprobs", False)]
        changes += [("config", key, 1) for key in comparison.TP_FIELDS]
        for side, key, value in changes:
            with self.subTest(side=side, key=key):
                self.setUp()
                target = {"run": self.run, "reference": self.reference, "config": self.run["config"]}[side]
                target[key] = value
                with self.assertRaises(ValueError):
                    self.compare()

    def test_resolved_paths_match(self):
        self.run["config"]["model"] = "/test/unused/../model"
        self.assertTrue(self.compare()["passed"])

    def test_reference_generation_settings_must_match(self):
        for key, value in (("do_sample", True), ("new_tokens", 15), ("ignore_eos", False)):
            with self.subTest(key=key):
                self.setUp()
                self.reference["sampling"][key] = value
                with self.assertRaises(ValueError):
                    self.compare()

    def test_all_repeats_and_concurrent_requests_are_checked(self):
        self.run["concurrency"] = 2
        rows = []
        for row in self.run["runs"]:
            for repeat in range(2):
                expanded = copy.deepcopy(row)
                expanded["repeat"] = repeat
                expanded["outputs"].append(copy.deepcopy(expanded["outputs"][0]))
                rows.append(expanded)
        self.run["runs"] = rows
        result = self.compare()
        self.assertTrue(result["passed"])
        self.assertEqual(result["summary"]["request_count"], 8)
        self.assertEqual(result["summary"]["tokens_checked"], 128)

    def test_tp1_first_divergence_with_reference_top5_and_margin(self):
        tp1 = self.tp1()
        output = tp1["runs"][0]["outputs"][0]
        for step in (3, 5):
            output["output_ids"][step] = 8
            output["meta_info"]["output_token_logprobs"][step][1] = 8
        result = self.compare(tp1=tp1)
        self.assertTrue(result["passed"])  # TP1 diagnostics cannot weaken TP2/reference verdict.
        self.assertEqual(len(result["tp1_diagnostics"]), 1)
        item = result["tp1_diagnostics"][0]
        self.assertEqual((item["step_index"], item["token_number"]), (3, 4))
        self.assertEqual((item["tp1_token_id"], item["tp2_token_id"]), (8, 7))
        self.assertTrue(item["reference_shares_preceding_history"])
        self.assertEqual(item["reference_top5"], self.reference["cases"][0]["top5"][3])
        self.assertEqual(item["reference_top1_minus_top2_logprob_margin"], 1.)

    def test_tp1_reference_top5_omitted_for_different_preceding_history(self):
        tp1 = self.tp1()
        tp1["runs"][0]["outputs"][0]["output_ids"][3] = 8
        tp1["runs"][0]["outputs"][0]["meta_info"]["output_token_logprobs"][3][1] = 8
        self.reference["cases"][0]["output_ids"][1] = 9
        item = self.compare(tp1=tp1)["tp1_diagnostics"][0]
        self.assertFalse(item["reference_shares_preceding_history"])
        self.assertNotIn("reference_top5", item)
        self.assertNotIn("reference_top1_minus_top2_logprob_margin", item)

    def test_unrelated_tp1_evidence_rejected(self):
        tp1 = self.tp1()
        tp1["manifest_sha256"] = "other checkpoint"
        with self.assertRaises(ValueError):
            self.compare(tp1=tp1)

    def test_cli_explicit_arguments_and_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {name: root / f"{name}.json" for name in ("reference", "run", "cases", "output")}
            for name, document in (("reference", self.reference), ("run", self.run), ("cases", self.cases)):
                paths[name].write_text(json.dumps(document))
            command = [sys.executable, "-B", str(SCRIPT), "--atol", ".15"]
            for name, path in paths.items():
                command.extend([f"--{name}", str(path)])
            success = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(success.returncode, 0, success.stderr)
            result = json.loads(paths["output"].read_text())
            self.assertTrue(result["passed"])
            self.assertNotIn("tp1", result["inputs"])
            self.assertEqual(result["inputs"]["cases"]["sha256"], self.digest)
            output = self.run["runs"][0]["outputs"][0]
            output["output_ids"][0] = 8
            output["meta_info"]["output_token_logprobs"][0][1] = 8
            paths["run"].write_text(json.dumps(self.run))
            failure = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(failure.returncode, 1, failure.stderr)
            self.assertFalse(json.loads(paths["output"].read_text())["passed"])
            self.run["config"]["world_size"] = 1
            paths["run"].write_text(json.dumps(self.run))
            paths["output"].unlink()
            invalid = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(invalid.returncode, 2, invalid.stderr)
            self.assertIn("invalid comparison evidence", invalid.stderr)
            self.assertFalse(paths["output"].exists())


if __name__ == "__main__":
    unittest.main()
