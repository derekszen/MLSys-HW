"""Bounded 27B repeatability and child-PID-only HBM evidence tests (no NPU)."""
import copy
import json
import math
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import summarize_large_run as large
from test_benchmark_summary import encoded, fixture


def large_fixture():
    benchmarks, references, anchors = fixture()
    benchmark, reference = benchmarks[(2, 1)], references[(2, 1)]
    # Keep deliberate independent reference failure, separate from repeatability.
    comparison = {"passed": False, "inputs": {
        "run": {"sha256": large.sha256(encoded(reference))},
        "cases": {"sha256": large.sha256(anchors["cases_bytes"])},
    }, "cases": [{"case": r["case"], "repeat": 0, "request": 0,
                   "passed": r["input_tokens"] != 32} for r in reference["runs"]]}
    return benchmark, reference, anchors, comparison


def hbm_fixture():
    result = {"status": large.COMPLETED, "tp": 2, "process_snapshot_after_load": {
        "stage": "after_engine_startup", "captured_at_unix_seconds": 123,
        "parent_pid": 100, "parent": {"pid": 100}, "children": [
            {"pid": 101, "ppid": 100, "name": "python"},
            {"pid": 102, "ppid": 100, "name": "tokenspeed::scheduler_tp0"},
            {"pid": 103, "ppid": 100, "name": "tokenspeed::scheduler_tp1"},
        ]}}
    text = """| NPU Chip | Process id | Process name | Process memory(MB) |
| 0 0 | 100 | python | 112 |
| 0 0 | 102 | tokenspeedsch | 2000 |
| 0 0 | 103 | tokenspeedsch | 112 |
| 0 1 | 103 | tokenspeedsch | 2100 |
| 0 0 | 999 | tokenspeedsch | 40000 |
| 1 0 | 998 | python | 50000 |
"""
    return result, text


class LargeSummaryTests(unittest.TestCase):
    def setUp(self):
        self.benchmark, self.reference, self.anchors, self.comparison = large_fixture()

    def summarize(self):
        return large.summarize(encoded(self.benchmark), encoded(self.reference),
                               comparison_bytes=encoded(self.comparison), **self.anchors)

    def test_medians_and_failed_accuracy_verdict_are_separate(self):
        result = self.summarize()
        self.assertTrue(result["repeatability_validation_passed"])
        self.assertFalse(result["strict_reference_passed"])
        self.assertFalse(result["accuracy_certified_performance"])
        self.assertNotIn("passed", result)
        self.assertEqual(result["strict_reference_failed_cases"], ["length_32"])
        self.assertEqual(result["verified_benchmark_requests"], 6)
        self.assertEqual([r["median_batch_seconds"] for r in result["rows"]], [2.5, 20])
        self.assertEqual([r["median_aggregate_output_tokens_per_second"] for r in result["rows"]], [6.4, 0.8])

    def test_each_measured_output_is_checked(self):
        for index in range(6):
            with self.subTest(index=index):
                self.setUp()
                self.benchmark["runs"][index]["outputs"][0]["output_ids"][0] += 1
                with self.assertRaisesRegex(ValueError, "benchmark output IDs differ"):
                    self.summarize()

    def test_fail_closed_provenance_config_status_and_shapes(self):
        mutations = [
            lambda b: b.update(manifest_sha256="0" * 64),
            lambda b: b.update(config_sha256="0" * 64),
            lambda b: b.update(cases_sha256="0" * 64),
            lambda b: b["config"].update(gpu_memory_utilization=0.7),
            lambda b: b["config"].update(world_size=1),
            lambda b: b.update(tp=True),
            lambda b: b.update(concurrency=2),
            lambda b: b.update(status="failed"),
            lambda b: b.update(error="interrupted"),
            lambda b: b.update(mode="profile"),
            lambda b: b["reproducibility"]["source_trees"].pop("python"),
            lambda b: b["reproducibility"]["source_trees"]["python"].update(sha256="0" * 64),
            lambda b: b["reproducibility"]["runtime_versions"].update(torch="other"),
            lambda b: b["reproducibility"]["environment"].update(HCCL_NPU_SOCKET_PORT_RANGE="other"),
            lambda b: b.update(physical_devices_requested=[1, 0]),
            lambda b: b["runs"].pop(),
            lambda b: b["runs"].append(copy.deepcopy(b["runs"][0])),
            lambda b: b["runs"][0].update(repeat=True),
            lambda b: b["runs"][0].update(input_tokens=5),
            lambda b: b["runs"][0]["outputs"].append(copy.deepcopy(b["runs"][0]["outputs"][0])),
            lambda b: b["runs"][0]["outputs"][0]["output_ids"].pop(),
            lambda b: b["runs"][0]["outputs"][0]["output_ids"].__setitem__(0, True),
            lambda b: b["runs"][0]["outputs"][0]["meta_info"].update(completion_tokens=15),
            lambda b: b["runs"][0]["outputs"][0]["meta_info"].update(finish_reason={"type": "stop"}),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                self.setUp()
                mutate(self.benchmark)
                with self.assertRaises(ValueError):
                    self.summarize()

    def test_consistent_but_different_source_manifest_rejected(self):
        tree = self.benchmark["reproducibility"]["source_trees"]["python"]
        tree["files"]["module.py"] = "b" * 64
        tree["sha256"] = large.sha256(encoded(tree["files"]))
        with self.assertRaisesRegex(ValueError, "source fingerprints differ"):
            self.summarize()

    def test_missing_reference_and_stale_or_passed_verdict_rejected(self):
        changes = [lambda: self.reference["runs"].pop(),
                   lambda: self.reference["runs"][0]["outputs"][0]["meta_info"].pop("output_token_logprobs"),
                   lambda: self.comparison.update(passed=True),
                   lambda: self.comparison["cases"].pop(),
                   lambda: self.comparison["cases"][1].update(passed=True),
                   lambda: self.comparison["inputs"]["run"].update(sha256="0" * 64)]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                self.setUp()
                change()
                with self.assertRaises(ValueError):
                    self.summarize()

    def test_invalid_timings_rejected(self):
        for field in ("batch_seconds", "aggregate_output_tokens_per_second"):
            for value in (0, -1, True, "1", math.nan, math.inf, 123):
                with self.subTest(field=field, value=value):
                    self.setUp()
                    self.benchmark["runs"][0][field] = value
                    with self.assertRaises(ValueError):
                        self.summarize()

    def test_port_and_source_root_relocation_allowed(self):
        expected = self.summarize()
        self.benchmark["config"]["port"] += 1
        self.benchmark["reproducibility"]["source_trees"]["python"]["root"] = "/relocated"
        self.benchmark["runs"].reverse()
        self.assertEqual(self.summarize(), expected)

    def test_retained_local_evidence_reproduces_summaries(self):
        root = SCRIPTS.parent
        results, metadata = root / "results", root / "metadata/Qwen3.5-27B"
        result = large.summarize(
            (results / "27B-benchmark-tp2-c1-retry/result.json").read_bytes(),
            (results / "27B-correctness-tp2-c1/result.json").read_bytes(),
            comparison_bytes=(results / "27B-reference-comparison.json").read_bytes(),
            cases_bytes=(root / "cases.json").read_bytes(),
            manifest_bytes=(metadata / "source-manifest.json").read_bytes(),
            config_bytes=(metadata / "config.json").read_bytes())
        saved = json.loads((results / "27B-benchmark-summary.json").read_bytes())
        del saved["inputs"]
        self.assertEqual(result, saved)
        self.assertEqual(large.collect_hbm(results), json.loads((results / "owned-hbm-summary.json").read_bytes()))


class OwnedHbmTests(unittest.TestCase):
    def test_only_exact_child_pids_with_cross_chip_allocations(self):
        result, text = hbm_fixture()
        owned = large.owned_hbm(result, text)
        self.assertTrue(owned["all_scheduler_pids_matched"])
        self.assertEqual(owned["total_owned_child_process_memory_mb"], 4212)
        self.assertEqual([r["matched_child_process_memory_mb"] for r in owned["per_chip"]], [2112, 2100])
        self.assertEqual(owned["matched_child_pids"], [102, 103])
        self.assertEqual(owned["unmatched_child_pids"], [101])

    def test_missing_scheduler_is_unknown_not_substituted_or_zero(self):
        result, text = hbm_fixture()
        for missing in ((102,), (102, 103)):
            with self.subTest(missing=missing):
                filtered = "\n".join(line for line in text.splitlines()
                                     if not any(f"| {pid} |" in line for pid in missing))
                owned = large.owned_hbm(result, filtered)
                self.assertFalse(owned["all_scheduler_pids_matched"])
                self.assertIsNone(owned["total_owned_child_process_memory_mb"])
                self.assertEqual(owned["missing_scheduler_child_pids"], list(missing))
                self.assertTrue(all(row["pid"] in (102, 103) for row in owned["rows"]))

    def test_malformed_ownership_or_process_table_rejected(self):
        changes = [lambda r: r.pop("process_snapshot_after_load"),
                   lambda r: r.update(status="failed"),
                   lambda r: r["process_snapshot_after_load"].update(children=[]),
                   lambda r: r["process_snapshot_after_load"]["children"][0].update(pid=100),
                   lambda r: r["process_snapshot_after_load"]["children"][0].update(ppid=999),
                   lambda r: r["process_snapshot_after_load"]["children"][0].update(ppid=101)]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                result, text = hbm_fixture()
                change(result)
                with self.assertRaises(ValueError):
                    large.owned_hbm(result, text)
        result, text = hbm_fixture()
        for bad in ("npu-smi failed", text + "| 0 0 | 102 | tokenspeedsch | 2000 |\n",
                    text.replace("| 2000 |", "| unknown |")):
            with self.assertRaises(ValueError):
                large.owned_hbm(result, bad)


if __name__ == "__main__":
    unittest.main()
