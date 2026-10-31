"""Stdlib behavioral tests; synthetic evidence only, no final-run dependency."""
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import summarize_validation_run as summary
from test_benchmark_summary import encoded, fixture


def hbm_fixture():
    result = {"status": summary.COMPLETED, "tp": 2, "process_snapshot_after_load": {
        "stage": "after_engine_startup", "captured_at_unix_seconds": 123,
        "parent_pid": 100, "parent": {"pid": 100}, "children": [
            {"pid": 101, "ppid": 100, "name": "python"},
            {"pid": 102, "ppid": 100, "name": "tokenspeed::scheduler_tp0"},
            {"pid": 103, "ppid": 100, "name": "tokenspeed::scheduler_tp1"},
        ]}}
    text = """| NPU Chip | Process id | Process name | Process memory(MB) |
| 0 0 | 100 | python | 9999 |
| 0 0 | 102 | scheduler | 2000 |
| 0 0 | 103 | scheduler | 112 |
| 0 1 | 103 | scheduler | 2100 |
| 1 0 | 999 | unrelated | 50000 |
"""
    return result, text


class PairTests(unittest.TestCase):
    def setUp(self):
        benchmarks, correctness, self.anchors = fixture()
        self.tp1, self.tp2 = correctness[(1, 2)], correctness[(2, 2)]
        self.benchmark = benchmarks[(2, 2)]

    def summarize(self, benchmark=True, atol=0.15):
        return summary.summarize_pair(self.tp1, self.tp2, atol=atol,
                                      benchmark=self.benchmark if benchmark else None, **self.anchors)

    def test_strict_verdict_and_benchmark_medians_separate(self):
        result = self.summarize()
        self.assertNotIn("passed", result)
        self.assertTrue(result["strict_tp_comparison"]["passed"])
        self.assertEqual(result["strict_tp_comparison"]["selected_logprob_atol"], 0.15)
        benchmark = result["benchmark"]
        self.assertTrue(benchmark["repeatability_validation_passed"])
        self.assertEqual(benchmark["verified_benchmark_requests"], 12)
        self.assertEqual([row["median_batch_seconds"] for row in benchmark["rows"]], [5, 40])
        self.assertEqual([row["median_aggregate_output_tokens_per_second"] for row in benchmark["rows"]], [6.4, 0.8])
        self.assertNotIn("benchmark", self.summarize(benchmark=False))

    def test_strict_logprob_and_greedy_failures_not_hidden_by_repeatability(self):
        self.tp2["runs"][1]["outputs"][0]["meta_info"]["output_token_logprobs"][0][0] -= 0.2
        result = self.summarize()
        self.assertFalse(result["strict_tp_comparison"]["passed"])
        self.assertTrue(result["benchmark"]["repeatability_validation_passed"])
        self.assertEqual(len([c for c in result["strict_tp_comparison"]["cases"] if not c["passed"]]), 1)
        self.setUp()
        output = self.tp2["runs"][0]["outputs"][1]
        output["output_ids"][0] += 1
        output["meta_info"]["output_token_logprobs"][0][1] += 1
        result = self.summarize()
        failed = [c for c in result["strict_tp_comparison"]["cases"] if not c["passed"]]
        self.assertEqual(len(failed), 1)
        self.assertFalse(failed[0]["greedy_exact"])
        self.assertIsNone(failed[0]["max_selected_logprob_error"])
        self.assertTrue(result["benchmark"]["repeatability_validation_passed"])

    def test_every_benchmark_repeat_and_request_checked(self):
        for index in range(6):
            for request in range(2):
                with self.subTest(index=index, request=request):
                    self.setUp()
                    self.benchmark["runs"][index]["outputs"][request]["output_ids"][0] += 1
                    with self.assertRaisesRegex(ValueError, "benchmark output IDs differ"):
                        self.summarize()

    def test_fail_closed_provenance_config_and_shapes_for_all_inputs(self):
        changes = [
            lambda r: r.update(manifest_sha256="0" * 64),
            lambda r: r.update(config_sha256="0" * 64),
            lambda r: r.update(cases_sha256="0" * 64),
            lambda r: r.update(status="failed"),
            lambda r: r.update(error="interrupted"),
            lambda r: r.update(tp=True),
            lambda r: r.update(concurrency=True),
            lambda r: r["config"].update(world_size=0),
            lambda r: r["config"].update(seed=17),
            lambda r: r["reproducibility"]["source_trees"].pop("python"),
            lambda r: r["reproducibility"]["source_trees"]["python"].update(sha256="0" * 64),
            lambda r: r["reproducibility"]["runtime_versions"].update(torch="other"),
            lambda r: r["reproducibility"]["environment"].update(TOKENIZERS_PARALLELISM="true"),
            lambda r: r["reproducibility"]["environment"].update(ASCEND_RT_VISIBLE_DEVICES="9"),
            lambda r: r.update(physical_devices_requested=[True]),
            lambda r: r["runs"].pop(),
            lambda r: r["runs"].append(copy.deepcopy(r["runs"][0])),
            lambda r: r["runs"][0].update(repeat=True),
            lambda r: r["runs"][0].update(input_tokens=999),
            lambda r: r["runs"][0]["outputs"][0]["output_ids"].pop(),
            lambda r: r["runs"][0]["outputs"][0]["output_ids"].__setitem__(0, True),
            lambda r: r["runs"][0]["outputs"][0]["meta_info"].update(finish_reason={"type": "eos"}),
        ]
        for target in ("tp1", "tp2", "benchmark"):
            for index, change in enumerate(changes):
                with self.subTest(target=target, change=index):
                    self.setUp()
                    change(getattr(self, target))
                    with self.assertRaises(ValueError):
                        self.summarize()

    def test_consistent_wrong_hashes_and_sources_rejected(self):
        for run in (self.tp1, self.tp2, self.benchmark):
            run["manifest_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            self.summarize()
        self.setUp()
        tree = self.benchmark["reproducibility"]["source_trees"]["python"]
        tree["files"]["module.py"] = "b" * 64
        tree["sha256"] = summary.sha256(encoded(tree["files"]))
        with self.assertRaisesRegex(ValueError, "source fingerprints differ"):
            self.summarize()
        self.setUp()
        self.anchors["config_bytes"] = encoded({"text_config": {"vocab_size": 99}})
        with self.assertRaisesRegex(ValueError, "config differs from manifest"):
            self.summarize()

    def test_timing_and_strict_tolerance_rejections(self):
        for field in ("batch_seconds", "aggregate_output_tokens_per_second"):
            for value in (0, -1, True, "1", math.nan, math.inf, 123):
                with self.subTest(field=field, value=value):
                    self.setUp()
                    self.benchmark["runs"][0][field] = value
                    with self.assertRaises(ValueError):
                        self.summarize()
        for atol in (0, 0.16, True, math.nan, math.inf):
            with self.assertRaises(ValueError):
                self.summarize(atol=atol)

    def test_only_port_tp_visibility_and_source_root_may_change(self):
        expected = self.summarize()
        self.tp2["config"]["port"] += 1
        self.tp2["physical_devices_requested"] = [4, 5]
        self.benchmark["physical_devices_requested"] = [4, 5]
        for run in (self.tp2, self.benchmark):
            run["reproducibility"]["environment"]["ASCEND_RT_VISIBLE_DEVICES"] = "4,5"
            run["reproducibility"]["source_trees"]["python"]["root"] = "/relocated"
        self.benchmark["runs"].reverse()
        self.assertEqual(self.summarize(), expected)
        self.benchmark["physical_devices_requested"] = [5, 4]
        self.benchmark["reproducibility"]["environment"]["ASCEND_RT_VISIBLE_DEVICES"] = "5,4"
        with self.assertRaisesRegex(ValueError, "devices differ"):
            self.summarize()


class HbmAndCliTests(unittest.TestCase):
    def populate_hbm(self, directory):
        directory.mkdir()
        result, text = hbm_fixture()
        (directory / "result.json").write_bytes(encoded(result))
        for name in summary.SNAPSHOTS:
            (directory / name).write_text(text)
        return text

    def invoke(self, argv):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return summary.main([str(arg) for arg in argv])

    def test_explicit_runs_all_three_snapshots_and_exact_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "first", root / "nested" / "second"
            self.populate_hbm(first)
            second.parent.mkdir()
            self.populate_hbm(second)
            # A malformed unselected sibling must never be discovered.
            (root / "newest").mkdir()
            (root / "newest" / "result.json").write_text("not json")
            result = summary.collect_hbm([first, second])
            self.assertNotIn("passed", result)
            self.assertEqual(len(result["rows"]), 6)
            self.assertEqual(len(result["inputs"]), 8)
            self.assertEqual({r["snapshot"] for r in result["rows"]}, set(summary.SNAPSHOTS))
            self.assertTrue(all(r["total_owned_child_process_memory_mb"] == 4212 for r in result["rows"]))
            self.assertEqual(result["rows"][0]["unmatched_child_pids"], [101])
            self.assertEqual(len(result["rows"][0]["rows"]), 3)
            for evidence in result["inputs"]:
                self.assertEqual(evidence["sha256"], summary.sha256(Path(evidence["path"]).read_bytes()))
            for directories in ([], [first, first / "."]):
                with self.assertRaises(ValueError):
                    summary.collect_hbm(directories)

    def test_missing_pid_unknown_and_missing_snapshot_rejected_without_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            text = self.populate_hbm(run)
            snapshot = run / "hbm-length_32.txt"
            snapshot.write_text("\n".join(line for line in text.splitlines() if "| 102 |" not in line))
            result = summary.collect_hbm([run])
            row = result["rows"][1]
            self.assertFalse(row["all_scheduler_pids_matched"])
            self.assertIsNone(row["total_owned_child_process_memory_mb"])
            self.assertEqual(row["missing_scheduler_child_pids"], [102])
            output = root / "only-output.json"
            self.assertEqual(self.invoke(["hbm", "--runs", run, "--output", output]), 0)
            self.assertIsNone(json.loads(output.read_bytes())["rows"][1]["total_owned_child_process_memory_mb"])
            snapshot.unlink()
            output.write_text("previous output")
            with self.assertRaises(SystemExit) as error:
                self.invoke(["hbm", "--runs", run, "--output", output])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(output.read_text(), "previous output")

    def test_pair_cli_hashes_failed_verdict_exit_and_input_overwrite_guard(self):
        benchmarks, references, anchors = fixture()
        references[(2, 1)]["runs"][1]["outputs"][0]["meta_info"]["output_token_logprobs"][0][0] -= 0.2
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, document in (("tp1", references[(1, 1)]), ("tp2", references[(2, 1)]),
                                   ("benchmark", benchmarks[(2, 1)])):
                (root / name).mkdir()
                (root / name / "result.json").write_bytes(encoded(document))
            metadata = root / "metadata"
            metadata.mkdir()
            (metadata / "source-manifest.json").write_bytes(anchors["manifest_bytes"])
            (metadata / "config.json").write_bytes(anchors["config_bytes"])
            (root / "cases.json").write_bytes(anchors["cases_bytes"])
            argv = ["pair", "--tp1", root / "tp1", "--tp2", root / "tp2",
                    "--benchmark", root / "benchmark", "--cases", root / "cases.json",
                    "--metadata", metadata, "--atol", "0.15"]
            before = {str(p.relative_to(root)) for p in root.rglob("*")}
            output = root / "summary.json"
            self.assertEqual(self.invoke(argv + ["--output", output]), 1)
            result = json.loads(output.read_bytes())
            self.assertFalse(result["strict_tp_comparison"]["passed"])
            self.assertTrue(result["benchmark"]["repeatability_validation_passed"])
            self.assertNotIn("passed", result)
            self.assertEqual(len(result["inputs"]), 6)
            for item in result["inputs"]:
                self.assertEqual(item["sha256"], summary.sha256(Path(item["path"]).read_bytes()))
            self.assertEqual({str(p.relative_to(root)) for p in root.rglob("*")} - before, {"summary.json"})
            original = (root / "tp1/result.json").read_bytes()
            with self.assertRaises(SystemExit) as error:
                self.invoke(argv + ["--output", root / "tp1/result.json"])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual((root / "tp1/result.json").read_bytes(), original)
            with self.assertRaises(SystemExit):
                self.invoke(["pair", "--output", output])


if __name__ == "__main__":
    unittest.main()
