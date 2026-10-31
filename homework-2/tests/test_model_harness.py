"""CPU-only harness tests; these are explicitly not distributed/NPU evidence."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = load("run_model")
comparison = load("compare_tp")


def config(tp):
    return runner.build_config("/test/model", tp=tp, devices=list(range(tp)),
                               memory_fraction=0.12, concurrency=1, port=31889)


def artifact(tp):
    return {"tp": tp, "status": "generation_completed_not_yet_compared", "mode": "correctness",
            "manifest_sha256": "same manifest", "config_sha256": "same config",
            "cases_sha256": "same inputs", "concurrency": 1, "config": config(tp),
            "runs": [{"case": "short", "repeat": 0, "input_tokens": 32,
                      "output_tokens_per_request": 2, "outputs": [{"output_ids": [7, 8],
                          "meta_info": {"output_token_logprobs": [[-1., 7, None], [-2., 8, None]],
                                        "prompt_tokens": 32, "completion_tokens": 2, "cached_tokens": 0}}]}]}


class ModelHarnessTests(unittest.TestCase):
    def test_environment_is_allowlisted_and_reports_unset_values(self):
        env = {"ASCEND_RT_VISIBLE_DEVICES": "2,3", "TOKENIZERS_PARALLELISM": "false",
               "HCCL_NPU_SOCKET_PORT_RANGE": "41000-41031", "API_TOKEN": "secret",
               "PYTHONPATH": "/unrelated/private/path", "OTHER": "not recorded"}
        expected = {key: env[key] for key in (
            "ASCEND_RT_VISIBLE_DEVICES", "TOKENIZERS_PARALLELISM", "HCCL_NPU_SOCKET_PORT_RANGE")}
        self.assertEqual(runner.selected_environment(env), expected)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(runner.selected_environment(), expected)
        self.assertEqual(runner.selected_environment({}), dict.fromkeys(expected))

    def test_python_fingerprint_is_deterministic_and_checkout_independent(self):
        sources = {"z.py": b"VALUE = 1\n", "package/a.py": b"# source\n"}
        with tempfile.TemporaryDirectory() as tmp:
            roots = [Path(tmp) / "first", Path(tmp) / "second"]
            for index, root in enumerate(roots):
                items = list(sources.items())
                if index:
                    items.reverse()
                for relative, content in items:
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content)
                    os.utime(path, (100 + index, 100 + index))
            first, second = [runner.fingerprint_python_tree(root) for root in roots]
            expected_files = {name: hashlib.sha256(content).hexdigest()
                              for name, content in sources.items()}
            self.assertEqual(first["files"], expected_files)
            self.assertEqual(list(first["files"]), sorted(sources))
            self.assertEqual(first["files"], second["files"])
            self.assertNotEqual(first["root"], second["root"])
            self.assertEqual(first["sha256"], second["sha256"])
            canonical = json.dumps(expected_files, sort_keys=True, separators=(",", ":")).encode()
            self.assertEqual(first["sha256"], hashlib.sha256(canonical).hexdigest())

    def test_python_fingerprint_changes_with_content_paths_additions_and_deletions(self):
        for change in ("content", "rename", "add", "delete"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = root / "source.py"
                path.write_text("original\n")
                before = runner.fingerprint_python_tree(root)["sha256"]
                if change == "content":
                    path.write_text("modified\n")
                elif change == "rename":
                    path.rename(root / "renamed.py")
                elif change == "add":
                    (root / "added.py").write_text("original\n")
                else:
                    path.unlink()
                self.assertNotEqual(before, runner.fingerprint_python_tree(root)["sha256"])

    def test_python_fingerprint_excludes_caches_and_non_python_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "source.py").write_text("source\n")
            before = runner.fingerprint_python_tree(root)
            # Even a .py file inside a cache directory must be excluded.
            for relative in ("__pycache__/cached.py", "pkg/__pycache__/nested.py",
                             "source.pyc", "config.json", "extension.so"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("ignored\n")
            self.assertEqual(before, runner.fingerprint_python_tree(root))
            for path in root.rglob("*"):
                if path.is_file() and path.name != "source.py":
                    path.write_text("changed ignored file\n")
            self.assertEqual(before, runner.fingerprint_python_tree(root))

    def test_project_fingerprints_cover_exactly_the_three_pythonpath_trees(self):
        relatives = ("python", "tokenspeed-kernel/python", "tokenspeed-kernel-npu/python")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "homework-1" / "tokenspeed"
            for relative in relatives:
                tree = project / relative
                tree.mkdir(parents=True)
                (tree / "source.py").write_text(relative)
            fake_runner = root / "homework-2" / "scripts" / "run_model.py"
            with mock.patch.object(runner, "__file__", str(fake_runner)):
                trees = runner.project_source_fingerprints()
                self.assertEqual(set(trees), set(relatives))
                for relative in relatives:
                    self.assertEqual(trees[relative],
                                     runner.fingerprint_python_tree(project / relative))
                (project / relatives[1] / "source.py").unlink()
                (project / relatives[1]).rmdir()
                missing = runner.project_source_fingerprints()[relatives[1]]
                self.assertIn("FileNotFoundError", missing["error"])
                self.assertNotIn("sha256", missing)

    def test_runtime_versions_prefer_loaded_runtime_and_fall_back_to_metadata(self):
        modules = {"torch": SimpleNamespace(__version__="2.9.0+runtime"),
                   "torch_npu": SimpleNamespace(), "triton": SimpleNamespace()}
        def version(name):
            if name == "torch_npu":
                return "2.9.0"
            raise runner.metadata.PackageNotFoundError(name)
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.object(runner.metadata, "version", side_effect=version) as lookup:
            self.assertEqual(runner.runtime_versions(), {
                "torch": "2.9.0+runtime", "torch_npu": "2.9.0", "triton": "unavailable"})
        self.assertEqual(lookup.call_args_list, [mock.call("torch_npu"), mock.call("triton")])

    def test_process_snapshot_only_reads_runner_and_recursive_children(self):
        def process(pid, ppid, name):
            return SimpleNamespace(pid=pid, ppid=mock.Mock(return_value=ppid),
                                   name=mock.Mock(return_value=name))
        parent = process(100, 90, "runner")
        child = process(101, 100, "scheduler")
        grandchild = process(102, 101, "worker")
        parent.children = mock.Mock(return_value=[grandchild, child])
        psutil = SimpleNamespace(Process=mock.Mock(return_value=parent), Error=RuntimeError)
        with mock.patch.dict(sys.modules, {"psutil": psutil}), \
                mock.patch.object(runner.os, "getpid", return_value=100):
            result = runner.process_snapshot()
        psutil.Process.assert_called_once_with(100)
        parent.children.assert_called_once_with(recursive=True)
        self.assertEqual(result["stage"], "after_engine_startup")
        self.assertEqual(result["parent_pid"], 100)
        self.assertEqual(result["parent"], {"pid": 100, "ppid": 90, "name": "runner"})
        self.assertEqual(result["children"], [
            {"pid": 101, "ppid": 100, "name": "scheduler"},
            {"pid": 102, "ppid": 101, "name": "worker"}])
        self.assertIn("no inferred rank assignment", result["scope"])
        self.assertNotIn("argv", json.dumps(result))

    def test_process_snapshot_reports_child_exit_without_failing_run(self):
        child = SimpleNamespace(pid=101, ppid=mock.Mock(side_effect=RuntimeError("exited")))
        parent = SimpleNamespace(pid=100, ppid=lambda: 90, name=lambda: "runner",
                                 children=mock.Mock(return_value=[child]))
        psutil = SimpleNamespace(Process=mock.Mock(return_value=parent), Error=RuntimeError)
        with mock.patch.dict(sys.modules, {"psutil": psutil}):
            result = runner.process_snapshot()
        self.assertEqual(result["children"], [
            {"pid": 101, "ppid": None, "name": None, "error": "RuntimeError"}])

    def test_main_saves_process_metadata_after_load_outside_generation_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            config_bytes = b'{"text_config": {"vocab_size": 100}}'
            (model / "config.json").write_bytes(config_bytes)
            (model / "source-manifest.json").write_text(json.dumps({"files": [
                {"path": "config.json", "sha256": hashlib.sha256(config_bytes).hexdigest()}]}))
            cases = root / "cases.json"
            cases.write_text(json.dumps({"cases": [{"name": "short", "input_ids": [1, 2]}]}))
            output = root / "results"
            events = []
            engine = SimpleNamespace(
                generate=mock.Mock(return_value=[{"output_ids": [7] * 16, "meta_info": {
                    "completion_tokens": 16, "prompt_tokens": 2, "cached_tokens": 0}}]),
                shutdown=mock.Mock())
            def start_engine(**kwargs):
                self.assertEqual(kwargs, config(1) | {"model": str(model), "dtype": "float16", "kv_cache_dtype": "float16", "lm_head_compute_dtype": "float32"})
                events.append("engine_started")
                return engine
            process_info = {"parent_pid": 100, "children": []}
            def processes():
                self.assertEqual(events[-1], "engine_started")
                events.append("process_snapshot")
                return process_info
            def device_snapshot(path):
                events.append(path.name)
                if path.name == "hbm-loaded.txt":
                    saved = json.loads((output / "result.json").read_text())
                    self.assertEqual(saved["process_snapshot_after_load"], process_info)
                    engine.generate.assert_not_called()
            modules = {"torch": SimpleNamespace(npu=SimpleNamespace(
                is_available=lambda: True, device_count=lambda: 1)),
                "torch_npu": SimpleNamespace(),
                "tokenspeed.runtime.entrypoints.engine": SimpleNamespace(Engine=start_engine)}
            argv = ["run_model.py", "--tp", "1", "--physical-devices", "0",
                    "--model", str(model), "--dtype", "float16", "--lm-head-compute-dtype", "float32",
                    "--cases", str(cases), "--output-dir", str(output),
                    "--memory-fraction", "0.12", "--concurrency", "1", "--mode", "correctness",
                    "--port", "31889", "--repeats", "1"]
            versions = {name: "test-version" for name in ("torch", "torch_npu", "triton")}
            with mock.patch.dict(sys.modules, modules), \
                    mock.patch.dict(os.environ, {"HCCL_NPU_SOCKET_PORT_RANGE": "41000-41031"}), \
                    mock.patch.object(sys, "argv", argv), \
                    mock.patch.multiple(runner, snapshot=device_snapshot,
                                        process_snapshot=processes,
                                        project_source_fingerprints=lambda: {"test": "fingerprint"},
                                        runtime_versions=lambda: versions), \
                    mock.patch.object(runner.time, "perf_counter", side_effect=[10., 12.]):
                runner.main()
            saved = json.loads((output / "result.json").read_text())
            self.assertEqual(events, ["hbm-before.txt", "engine_started", "process_snapshot",
                                      "hbm-loaded.txt", "hbm-short.txt"])
            self.assertEqual(saved["status"], "generation_completed_not_yet_compared")
            self.assertEqual(saved["config"]["dtype"], "float16")
            self.assertEqual(saved["config"]["kv_cache_dtype"], "float16")
            self.assertEqual(saved["config"]["lm_head_compute_dtype"], "float32")
            self.assertEqual(saved["config"]["mamba_ssm_dtype"], "float32")
            self.assertEqual(saved["reproducibility"]["runtime_versions"], versions)
            self.assertEqual(saved["reproducibility"]["source_trees"], {"test": "fingerprint"})
            self.assertEqual(saved["reproducibility"]["environment"]["HCCL_NPU_SOCKET_PORT_RANGE"],
                             "41000-41031")
            self.assertEqual(saved["runs"][0]["batch_seconds"], 2.)
            self.assertEqual(saved["runs"][0]["aggregate_output_tokens_per_second"], 8.)
            self.assertEqual(engine.generate.call_count, 2)  # One warmup, one timed generation.
            engine.shutdown.assert_called_once_with()

    def test_tp_degree_and_visibility_rejected_on_mismatch(self):
        for tp, devices in ((2, [0]), (1, [0, 1]), (2, [0, 0]), (2, [-1, 1])):
            with self.subTest(tp=tp, devices=devices), self.assertRaises(ValueError):
                runner.build_config('/model', tp=tp, devices=devices,
                                    memory_fraction=.12, concurrency=1, port=31889)

    def test_bounded_memory(self):
        for value in (0, -.1, .90001, 1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.build_config('/model', tp=2, devices=[0, 1],
                                    memory_fraction=value, concurrency=1, port=31889)

    def test_explicit_large_model_budget_preserved(self):
        config = runner.build_config('/model', tp=1, devices=[1],
                                     memory_fraction=.9, concurrency=1, port=31889)
        self.assertEqual(config['gpu_memory_utilization'], .9)
        self.assertEqual(config['max_total_tokens'], 16384)

    def test_device_parser(self):
        self.assertEqual(runner.parse_devices('2,3'), [2, 3])
        for text in ('', '0,0', '0,-1', '1,', 'a,2'):
            with self.subTest(text=text), self.assertRaises(Exception):
                runner.parse_devices(text)

    def test_completed_output_contract(self):
        outputs = artifact(1)['runs'][0]['outputs']
        runner.validate_outputs(outputs, batch_size=1, new_tokens=2, input_tokens=32)
        outputs[0]['meta_info']['cached_tokens'] = 1
        with self.assertRaises(ValueError):
            runner.validate_outputs(outputs, batch_size=1, new_tokens=2, input_tokens=32)

    def test_identical_comparison_and_tolerance(self):
        a, b = artifact(1), artifact(2)
        b['runs'][0]['outputs'][0]['meta_info']['output_token_logprobs'][0][0] += .1
        self.assertTrue(comparison.compare(a, b, atol=.15)['passed'])
        self.assertFalse(comparison.compare(a, b, atol=.05)['passed'])

    def test_different_tokens_are_not_conditionally_compared(self):
        a, b = artifact(1), artifact(2)
        b['runs'][0]['outputs'][0]['output_ids'][1] = 9
        b['runs'][0]['outputs'][0]['meta_info']['output_token_logprobs'][1][1] = 9
        result = comparison.compare(a, b, atol=.15)
        self.assertFalse(result['passed'])
        self.assertIsNone(result['cases'][0]['max_selected_logprob_error'])

    def test_incompatible_experiments_rejected(self):
        a = artifact(1)
        variants = []
        for key, value in (('manifest_sha256', 'other'), ('cases_sha256', 'other'),
                           ('concurrency', 2), ('mode', 'benchmark'), ('status', 'failed')):
            b = artifact(2)
            b[key] = value
            variants.append(b)
        b = artifact(2)
        b['config']['dtype'] = 'float16'
        variants.append(b)
        b = artifact(2)
        b['config']['world_size'] = 1
        variants.append(b)
        for b in variants:
            with self.subTest(b=b), self.assertRaises(ValueError):
                comparison.compare(a, b, atol=.15)

    def test_missing_or_corrupt_logprobs_fail(self):
        for value in ([], [[float('nan'), 7], [-2., 8]], [[-1., 999], [-2., 8]]):
            a, b = artifact(1), artifact(2)
            b['runs'][0]['outputs'][0]['meta_info']['output_token_logprobs'] = copy.deepcopy(value)
            with self.assertRaises(ValueError):
                comparison.compare(a, b, atol=.15)

    def test_missing_requests_or_runs_fail(self):
        for mutate in (lambda b: b.update(runs=[]),
                       lambda b: b['runs'][0].update(outputs=[]),
                       lambda b: b['runs'][0]['outputs'][0].update(output_ids=[7])):
            a, b = artifact(1), artifact(2)
            mutate(b)
            with self.assertRaises(ValueError):
                comparison.compare(a, b, atol=.15)


if __name__ == '__main__':
    unittest.main()
