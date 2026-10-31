"""Stdlib behavioral tests of fail-closed benchmark validation and medians."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("summarize_benchmarks", SCRIPT)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def fixture():
    config_bytes = encoded({"text_config": {"vocab_size": 100}})
    manifest_bytes = encoded({"files": [{"path": "config.json", "sha256": summary.sha256(config_bytes)}]})
    cases_bytes = encoded({"new_tokens": 16, "cases": [
        {"name": "smoke", "input_ids": [1] * 5},
        {"name": "length_32", "input_ids": [1] * 32},
        {"name": "length_256", "input_ids": [1] * 256},
    ]})
    anchors = dict(cases_bytes=cases_bytes, manifest_bytes=manifest_bytes, config_bytes=config_bytes)
    files = {"module.py": "a" * 64}
    tree = {"root": "/original/root", "files": files, "sha256": summary.sha256(encoded(files))}
    benchmarks, correctness = {}, {}
    for tp, concurrency in sorted(summary.GRID):
        config = {key: tp for key in summary.TP_FIELDS}
        config.update(port=30000 + tp, dtype="bfloat16", seed=42)
        common = {
            "tp": tp, "concurrency": concurrency, "status": summary.COMPLETED,
            "config": config, "physical_devices_requested": list(range(tp)),
            "manifest_sha256": summary.sha256(manifest_bytes),
            "config_sha256": summary.sha256(config_bytes), "cases_sha256": summary.sha256(cases_bytes),
            "reproducibility": {
                "source_trees": {name: copy.deepcopy(tree) for name in summary.SOURCE_TREES},
                "runtime_versions": {"torch": "2.9", "torch_npu": "2.9", "triton": "3.2"},
                "environment": {"ASCEND_RT_VISIBLE_DEVICES": ",".join(map(str, range(tp))),
                                "TOKENIZERS_PARALLELISM": "false"},
            },
        }
        for mode, matrix in (("benchmark", benchmarks), ("correctness", correctness)):
            result = copy.deepcopy(common)
            result.update(mode=mode, runs=[])
            for length in ((32, 256) if mode == "benchmark" else (5, 32, 256)):
                for repeat in (range(3) if mode == "benchmark" else range(1)):
                    tokens = [length % 100] * 16
                    outputs = [{"output_ids": tokens[:], "meta_info": {
                        "prompt_tokens": length, "completion_tokens": 16,
                        "finish_reason": {"type": "length", "length": 16},
                        **({"output_token_logprobs": [[-0.5, token] for token in tokens]}
                           if mode == "correctness" else {}),
                    }} for _ in range(concurrency)]
                    seconds = (9, 1, 5)[repeat] * length / 32 * concurrency / tp
                    result["runs"].append({
                        "case": "smoke" if length == 5 else f"length_{length}",
                        "repeat": repeat, "input_tokens": length, "output_tokens_per_request": 16,
                        "outputs": outputs, "batch_seconds": seconds,
                        "aggregate_output_tokens_per_second": 16 * concurrency / seconds,
                    })
            matrix[(tp, concurrency)] = result
    return benchmarks, correctness, anchors


class BenchmarkSummaryTests(unittest.TestCase):
    def setUp(self):
        self.benchmarks, self.correctness, self.anchors = fixture()

    def summarize(self):
        return summary.summarize(self.benchmarks, self.correctness, **self.anchors)

    def reject(self, change):
        change()
        with self.assertRaises(ValueError):
            self.summarize()

    def test_medians_not_means_and_speedup_direction(self):
        result = self.summarize()
        self.assertEqual(len(result["rows"]), 8)
        self.assertEqual(result["verified_benchmark_requests"], 36)
        self.assertEqual(len(result["output_id_checks"]), 36)
        for row in result["rows"]:
            expected = 5 * row["input_tokens"] / 32 * row["concurrency"] / row["tp"]
            self.assertEqual(row["median_batch_seconds"], expected)
            self.assertEqual(row["median_aggregate_output_tokens_per_second"], 16 * row["concurrency"] / expected)
        self.assertEqual([r["tp1_latency_divided_by_tp2_latency"] for r in result["tp2_speedups"]], [2] * 4)

    def test_order_independent_and_deterministic(self):
        before = self.summarize()
        for matrix in (self.benchmarks, self.correctness):
            for result in matrix.values():
                result["runs"].reverse()
        self.assertEqual(self.summarize(), before)
        self.assertEqual(self.summarize(), before)

    def test_every_repeat_and_request_is_compared(self):
        for cell in sorted(summary.GRID):
            for run_index in range(6):
                for request in range(cell[1]):
                    with self.subTest(cell=cell, run=run_index, request=request):
                        benchmarks, correctness, anchors = fixture()
                        benchmarks[cell]["runs"][run_index]["outputs"][request]["output_ids"][0] += 1
                        with self.assertRaisesRegex(ValueError, "benchmark output IDs differ"):
                            summary.summarize(benchmarks, correctness, **anchors)

    def test_incomplete_or_extra_grid_rejected(self):
        for kind in ("cell", "repeat", "case", "duplicate", "extra_repeat", "missing_reference", "duplicate_reference"):
            with self.subTest(kind=kind):
                self.setUp()
                target = self.benchmarks[(2, 2)]["runs"]
                if kind == "cell":
                    del self.benchmarks[(2, 1)]
                elif kind == "repeat":
                    target.pop()
                elif kind == "case":
                    target[:] = [r for r in target if r["input_tokens"] != 256]
                elif kind == "duplicate":
                    target[-1] = copy.deepcopy(target[0])
                elif kind == "extra_repeat":
                    extra = copy.deepcopy(target[-1])
                    extra["repeat"] = 3
                    target.append(extra)
                elif kind == "missing_reference":
                    self.correctness[(1, 2)]["runs"].pop()
                else:
                    self.correctness[(1, 2)]["runs"].append(copy.deepcopy(self.correctness[(1, 2)]["runs"][0]))
                with self.assertRaises(ValueError):
                    self.summarize()

    def test_provenance_and_configuration_rejections(self):
        changes = [
            lambda r: r.update(manifest_sha256="b" * 64),
            lambda r: r.update(config_sha256="b" * 64),
            lambda r: r.update(cases_sha256="b" * 64),
            lambda r: r["config"].update(seed=17),
            lambda r: r["config"].update(world_size=1),
            lambda r: r.update(tp=1),
            lambda r: r.update(concurrency=1),
            lambda r: r.update(mode="correctness"),
            lambda r: r.update(status="failed"),
            lambda r: r.update(error="interrupted"),
            lambda r: r["reproducibility"]["source_trees"].pop("python"),
            lambda r: r["reproducibility"]["source_trees"]["python"].update(sha256="b" * 64),
            lambda r: r["reproducibility"]["source_trees"]["python"]["files"].update({"module.py": "b" * 64}),
            lambda r: r["reproducibility"]["runtime_versions"].update(torch="different"),
            lambda r: r["reproducibility"]["environment"].update(TOKENIZERS_PARALLELISM="true"),
            lambda r: r["reproducibility"]["environment"].update(ASCEND_RT_VISIBLE_DEVICES="0"),
            lambda r: r.update(physical_devices_requested=[0, 0]),
        ]
        for index, change in enumerate(changes):
            with self.subTest(change=index):
                self.setUp()
                self.reject(lambda: change(self.benchmarks[(2, 2)]))

    def test_self_consistent_but_different_source_rejected(self):
        tree = self.benchmarks[(2, 2)]["reproducibility"]["source_trees"]["python"]
        tree["files"]["module.py"] = "b" * 64
        tree["sha256"] = summary.sha256(encoded(tree["files"]))
        with self.assertRaisesRegex(ValueError, "source fingerprints differ"):
            self.summarize()

    def test_anchor_metadata_not_just_agreement_between_runs(self):
        for matrix in (self.benchmarks, self.correctness):
            for result in matrix.values():
                result["cases_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            self.summarize()

    def test_relocated_source_root_and_different_port_allowed(self):
        result = self.benchmarks[(2, 2)]
        result["config"]["port"] += 100
        result["reproducibility"]["source_trees"]["python"]["root"] = "/relocated"
        self.assertTrue(self.summarize()["passed"])

    def test_bad_measurements_rejected(self):
        for value in (0, -1, math.nan, math.inf, True, "1"):
            for field in ("batch_seconds", "aggregate_output_tokens_per_second"):
                with self.subTest(value=value, field=field):
                    self.setUp()
                    self.reject(lambda: self.benchmarks[(1, 1)]["runs"][0].update({field: value}))
        self.setUp()
        self.reject(lambda: self.benchmarks[(1, 1)]["runs"][0].update(aggregate_output_tokens_per_second=123))

    def test_output_shapes_types_and_metadata_rejected(self):
        changes = [
            lambda r: r["outputs"].pop(),
            lambda r: r["outputs"][1]["output_ids"].pop(),
            lambda r: r["outputs"][1]["output_ids"].__setitem__(0, True),
            lambda r: r["outputs"][1]["output_ids"].__setitem__(0, -1),
            lambda r: r["outputs"][1]["meta_info"].update(prompt_tokens=256),
            lambda r: r["outputs"][1]["meta_info"].update(completion_tokens=15),
            lambda r: r["outputs"][1]["meta_info"].update(finish_reason={"type": "stop"}),
            lambda r: r.update(input_tokens=256),
            lambda r: r.update(output_tokens_per_request=15),
            lambda r: r.update(repeat=True),
        ]
        for index, change in enumerate(changes):
            with self.subTest(change=index):
                self.setUp()
                self.reject(lambda: change(self.benchmarks[(2, 2)]["runs"][2]))

    def test_reference_disagreement_or_missing_logprobs_rejected(self):
        output = self.correctness[(2, 2)]["runs"][1]["outputs"][1]
        output["output_ids"][0] += 1
        output["meta_info"]["output_token_logprobs"][0][1] += 1
        with self.assertRaisesRegex(ValueError, "correctness output IDs differ"):
            self.summarize()
        self.setUp()
        del self.correctness[(1, 1)]["runs"][1]["outputs"][0]["meta_info"]["output_token_logprobs"]
        with self.assertRaises(ValueError):
            self.summarize()


if __name__ == "__main__":
    unittest.main()
