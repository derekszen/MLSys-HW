"""Offline synthetic integrity tests; no accelerator imports or mocked devices."""
import copy
import gzip
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import analyze_decode_probe as a


def save_json(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    Path(path).write_bytes(raw)
    return a.sha256(raw)


def npy_bytes(array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def summaries(logits):
    row = logits[0]
    lp = a.logsoftmax(row)
    top = np.argsort(row, kind="stable")[-10:][::-1]
    return {"top10": [{"token": int(t), "logit": float(row[t]), "logprob": float(lp[t])} for t in top],
            "max_tied_ids": np.flatnonzero(row == row.max()).tolist(),
            "candidates": {str(t): {"logit": float(row[t]), "logprob": float(lp[t])} for t in a.CANDIDATES}}


def fixture(root, divergent=True):
    """Full shapes, tiny file pool; hardlinks avoid writing 96 MiB per test."""
    root = Path(root)
    cases = {"cases": [{"name": name, "input_ids": list(range(length))}
                       for name, length in (("smoke", 5), ("length_32", 32), ("length_256", 256))]}
    case_hash = save_json(root / "cases.json", cases)
    pool = root / "pool"
    pool.mkdir()
    arrays, templates, logprobs = {}, {}, {}
    for variant, winner in (("common", 1118), ("tp1-div", 20943), ("tp2-div", 760)):
        logits = np.full((1, a.VOCAB_SIZE), -10, dtype=np.float32)
        logits[0, 20:30] = np.arange(10, dtype=np.float32) / 2
        logits[0, winner] = 14
        if variant == "tp2-div":
            logits[0, 20943] = 14  # Exact tie: ID 760 wins, irrespective of topk ordering.
        elif variant == "tp1-div":
            logits[0, 760] = 13.9921875
        arrays[variant] = logits
        (pool / f"{variant}.npy").write_bytes(npy_bytes(logits))
        templates[variant] = summaries(logits)
        logprobs[variant] = float(a.logsoftmax(logits)[winner])
    hidden = np.zeros((1, a.HIDDEN_SIZE), dtype=np.float32)
    (pool / "hidden.npy").write_bytes(npy_bytes(hidden))
    source_files = {"example.py": "b" * 64}
    tree = {"root": "/captured/source", "files": source_files,
            "sha256": a.sha256(json.dumps(source_files, sort_keys=True, separators=(",", ":")).encode())}
    for tp in (1, 2):
        directory = root / f"tp{tp}"
        directory.mkdir()
        config = {k: tp for k in a.TP_CONFIG}
        config.update(model="/captured/Qwen3.5-4B", device="npu", dtype="float16", kv_cache_dtype="float16",
                      pipeline_parallel_size=1, sampling_backend="greedy", enable_prefix_caching=False,
                      enable_output_logprobs=True, port=10000 + tp)
        devices = list(range(tp))
        repro = {"source_trees": {name: copy.deepcopy(tree) for name in a.TREE_NAMES},
                 "runtime_versions": {name: "synthetic-offline" for name in ("torch", "torch_npu", "triton")},
                 "environment": {"ASCEND_RT_VISIBLE_DEVICES": ",".join(map(str, devices)),
                                 "TOKENIZERS_PARALLELISM": "false", "HCCL_NPU_SOCKET_PORT_RANGE": "41000-41031"}}
        hashes = {"manifest_sha256": "c" * 64, "config_sha256": "d" * 64, "cases_sha256": case_hash}
        outputs = {}
        for case in cases["cases"]:
            ids = [1118] * 16
            if case["name"] == "length_32" and divergent:
                ids[1] = 20943 if tp == 1 else 760
            outputs[case["name"]] = {"output_ids": ids, "meta_info": {
                "prompt_tokens": len(case["input_ids"]), "completion_tokens": 16, "cached_tokens": 0,
                "finish_reason": {"type": "length", "length": 16},
                "output_token_logprobs": [[logprobs[f"tp{tp}-div" if divergent and i == 1 and case["name"] == "length_32"
                                                     else "common"], t, None] for i, t in enumerate(ids)]}}
        baseline = {"status": "generation_completed_not_yet_compared", "mode": "correctness", "concurrency": 1,
                    "tp": tp, "config": config, "physical_devices_requested": devices,
                    "reproducibility": copy.deepcopy(repro), **hashes,
                    "runs": [{"case": c["name"], "repeat": 0, "outputs": [copy.deepcopy(outputs[c["name"]])]}
                             for c in cases["cases"]]}
        base_path = root / f"baseline-tp{tp}.json"
        baseline_hash = save_json(base_path, baseline)
        plan = {"schema": 1, "output_dir": f"/captured/tp{tp}", "baseline": f"/captured/baseline-tp{tp}.json",
                "baseline_sha256": baseline_hash, **hashes, "labels": a.LABELS, "candidates": a.CANDIDATES,
                "capture_input_ids": cases["cases"][1]["input_ids"]}
        plan_hash = save_json(directory / "probe-plan.json", plan)
        repro["diagnostic_scripts"] = {name: a.sha256((SCRIPTS / name).read_bytes())
                                         for name in ("audit_decode.py", "decode_probe.py", "run_model.py")}
        result = {"status": "diagnostic_completed_not_compared", "tp": tp, "config": config,
                  "physical_devices_requested": devices, "reproducibility": repro, **hashes,
                  "baseline_sha256": baseline_hash, "all_baseline_output_ids_exact": True,
                  "runs": [{"case": case, "repeat": repeat, "outputs": [copy.deepcopy(outputs[case])],
                            "baseline_output_ids_exact": True}
                           for case, repeat in (("smoke", 0), ("length_32", 0), ("length_32", 1), ("length_256", 0))],
                  "process_snapshot_after_load": {"stage": "after_engine_startup", "parent_pid": 100,
                      "parent": {"pid": 100}, "children": [{"pid": 101 + rank, "ppid": 100,
                          "name": f"tokenspeed::scheduler_tp{rank}"} for rank in range(tp)]}}
        save_json(directory / "result.json", result)
        for repeat, label in enumerate(a.LABELS):
            for rank in range(tp):
                folder = directory / f"{label}-rank{rank}"
                folder.mkdir()
                for step in range(16):
                    stem = f"step{step:02d}"
                    variant = f"tp{tp}-div" if divergent and step == 1 else "common"
                    tensors = {}
                    for name, source in (("logits", pool / f"{variant}.npy"), ("hidden", pool / "hidden.npy")):
                        target = folder / f"{stem}-{name}.npy"
                        os.link(source, target)
                        tensors[name] = {"path": target.name, "sha256": a.sha256(source.read_bytes()),
                                         "shape": [1, a.VOCAB_SIZE if name == "logits" else a.HIDDEN_SIZE],
                                         "storage_dtype": "float32"}
                    shard = {"start": rank * a.VOCAB_SIZE // tp, "end": (rank + 1) * a.VOCAB_SIZE // tp}
                    record = {"schema": 1, "label": label, "step_index": step, "rank": rank, "pid": 101 + rank,
                              "plan_sha256": plan_hash, "forward_mode": "EXTEND" if step == 0 else "DECODE",
                              "input_ids": plan["capture_input_ids"] if step == 0 else [outputs["length_32"]["output_ids"][step - 1]],
                              "num_extends": 1 if step == 0 else 0, "input_num_tokens": 32 if step == 0 else 1,
                              "native_logits_dtype": "torch.float16", "hidden_dtype": "torch.float16",
                              "lm_head_shard": shard, "tensors": tensors, **templates[variant],
                              "candidate_dot_fp64": {str(t): float(arrays[variant][0, t]) + (0.00003 if t == 20943 else 0)
                                  for t in a.CANDIDATES if shard["start"] <= t < shard["end"]}}
                    save_json(folder / f"{stem}.json", record)
    return root


class DecodeProbeAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = fixture(self.temp.name)

    def analyze(self):
        return a.analyze(self.root / "tp1", self.root / "tp2", self.root / "baseline-tp1.json",
                         self.root / "baseline-tp2.json", self.root / "cases.json")

    def step_path(self, tp=1, repeat=0, rank=0, step=0):
        return self.root / f"tp{tp}" / f"length_32-repeat{repeat}-rank{rank}" / f"step{step:02d}.json"

    def change(self, path, mutate):
        value = json.loads(Path(path).read_bytes())
        mutate(value)
        save_json(path, value)

    def change_step(self, mutate, **kwargs):
        self.change(self.step_path(**kwargs), mutate)

    def replace_tensor(self, array, name="logits", **kwargs):
        path = self.step_path(**kwargs)
        record = json.loads(path.read_bytes())
        tensor = path.parent / record["tensors"][name]["path"]
        tensor.unlink()  # Never mutate the hardlink pool/sibling evidence.
        tensor.write_bytes(npy_bytes(array))
        record["tensors"][name]["sha256"] = a.sha256(tensor.read_bytes())
        if name == "logits" and array.shape == (1, a.VOCAB_SIZE) and np.isfinite(array).all():
            record.update(summaries(array))
        save_json(path, record)

    def test_success_first_divergence_included_never_compare_after_reconvergence(self):
        report = self.analyze()
        for repeat in report["repetitions"]:
            self.assertEqual([s["step_index"] for s in repeat["compared_steps"]], [0, 1])
            self.assertEqual(repeat["excluded_after_divergence"], list(range(2, 16)))
            div = repeat["first_divergence"]
            self.assertEqual(div["step_index"], 1)
            self.assertEqual(div["tp2"]["max_tied_ids"], [760, 20943])
            self.assertGreater(div["tp2"]["dot_margin_tp1_choice_minus_tp2_choice"], 0)
            self.assertEqual(set(div["tp2"]["diagnostic_pair_dot_fp64"]), {"760", "20943"})
        self.assertTrue(report["all_three_cases_match_original_same_tp_baseline"])
        self.assertFalse(report["scope"]["correctness_threshold_changed"])
        self.assertEqual(report["scope"]["existing_logprob_threshold"], 0.15)
        tensors = [p for p in report["provenance"] if p["kind"] == "tensor"]
        self.assertEqual(len(tensors), 192)
        self.assertTrue(all(p["selected_sha256"] == p["raw_sha256"] for p in tensors))

    def test_no_divergence_compares_all16_without_inventing_a_candidate_pair(self):
        equal = self.root / "equal-history"
        equal.mkdir()
        fixture(equal, divergent=False)
        report = a.analyze(equal / "tp1", equal / "tp2", equal / "baseline-tp1.json",
                           equal / "baseline-tp2.json", equal / "cases.json")
        for repeat in report["repetitions"]:
            self.assertIsNone(repeat["first_divergence"])
            self.assertEqual(len(repeat["compared_steps"]), 16)
            self.assertEqual(repeat["excluded_after_divergence"], [])
        self.assertEqual(report["common_history_summary"]["compared_step_pairs"], 32)
        self.assertEqual(report["common_history_summary"]["max_abs_raw_logit_error"], 0)
        self.assertEqual(report["common_history_summary"]["max_abs_full_vocab_logprob_error"], 0)

    def test_full_vocabulary_float64_metrics_not_topk_or_unrounded_dots(self):
        left = np.full(a.VOCAB_SIZE, 10000, dtype=np.float32)
        right = left.copy()
        right[-1] += 1
        errors = a.distribution_errors(left, right)
        lp = a.logsoftmax(left)
        lq = a.logsoftmax(right)
        self.assertEqual(lp.dtype, np.float64)
        self.assertAlmostEqual(np.exp(lp).sum(), 1)
        self.assertEqual(errors["max_abs_raw_logit_error"], 1)
        self.assertAlmostEqual(errors["max_abs_full_vocab_logprob_error"], float(np.max(abs(lp - lq))))
        self.assertGreater(errors["total_variation"], 0)
        self.assertGreater(errors["kl_tp1_to_tp2"], 0)

    def test_invalid_raw_hash_rejected(self):
        self.change_step(lambda d: d["tensors"]["logits"].update(sha256="0" * 64))
        with self.assertRaisesRegex(a.EvidenceError, "raw tensor hash"):
            self.analyze()

    def test_malformed_digest_rejected(self):
        self.change_step(lambda d: d.update(plan_sha256="not-a-sha"))
        with self.assertRaisesRegex(a.EvidenceError, "malformed SHA256"):
            self.analyze()

    def test_nan_even_with_matching_hash_rejected(self):
        for name, width in (("logits", a.VOCAB_SIZE), ("hidden", a.HIDDEN_SIZE)):
            with self.subTest(name=name):
                path = self.step_path()
                metadata_bytes = path.read_bytes()
                tensor = path.parent / f"step00-{name}.npy"
                tensor_bytes = tensor.read_bytes()
                array = np.zeros((1, width), dtype=np.float32)
                array[0, -1] = np.nan
                self.replace_tensor(array, name=name)
                with self.assertRaisesRegex(a.EvidenceError, "nonfinite tensor"):
                    self.analyze()
                tensor.write_bytes(tensor_bytes)
                path.write_bytes(metadata_bytes)

    def test_exact_raw_shapes_dtype_and_no_trailing_payload(self):
        path = self.step_path()
        record = json.loads(path.read_bytes())
        tensor = path.parent / record["tensors"]["logits"]["path"]
        for raw, reason in ((npy_bytes(np.zeros((a.VOCAB_SIZE,), np.float32)), "shape/dtype"),
                            (npy_bytes(np.zeros((1, a.VOCAB_SIZE), np.float64)), "shape/dtype"),
                            (tensor.read_bytes() + b"trailing", "trailing")):
            with self.subTest(reason=reason):
                tensor.unlink()
                tensor.write_bytes(raw)
                record["tensors"]["logits"]["sha256"] = a.sha256(raw)
                save_json(path, record)
                with self.assertRaisesRegex(a.EvidenceError, reason):
                    self.analyze()

    def test_history_prefill_and_later_generated_input(self):
        for step in (0, 1, 15):
            path = self.step_path(step=step)
            original = path.read_bytes()
            self.change(path, lambda d: d["input_ids"].__setitem__(0, 42))
            with self.subTest(step=step), self.assertRaisesRegex(a.EvidenceError, "input history"):
                self.analyze()
            path.write_bytes(original)

    def test_warmup_tail_anchor_rejected(self):
        self.change_step(lambda d: d.update(forward_mode="DECODE", num_extends=0, input_num_tokens=1))
        with self.assertRaisesRegex(a.EvidenceError, "anchor"):
            self.analyze()

    def test_wrong_candidate_and_dot_ownership(self):
        for mutate, reason in ((lambda d: d["candidates"]["760"].update(logit=123), "raw logit"),
                               (lambda d: d["candidates"].pop("760"), "candidate set"),
                               (lambda d: d["candidate_dot_fp64"].update({"7": 0}), "dot ownership")):
            path = self.step_path()
            original = path.read_bytes()
            self.change(path, mutate)
            with self.subTest(reason=reason), self.assertRaisesRegex(a.EvidenceError, reason):
                self.analyze()
            path.write_bytes(original)

    def test_ties_metadata_and_actual_argmax(self):
        path = self.step_path(tp=2, step=1)
        original = path.read_bytes()
        self.change(path, lambda d: d.update(max_tied_ids=[20943]))
        with self.assertRaisesRegex(a.EvidenceError, "tie metadata"):
            self.analyze()
        path.write_bytes(original)
        result = json.loads((self.root / "tp2/result.json").read_bytes())
        output = result["runs"][1]["outputs"][0]
        output["output_ids"][1] = 20943  # A maximum, but not native argmax's tie winner.
        plan_hash = a.sha256((self.root / "tp2/probe-plan.json").read_bytes())
        with self.assertRaisesRegex(a.EvidenceError, "native argmax"):
            a.validate_step(path.parent, a.LABELS[0], 0, 1, plan_hash, output, list(range(32)), {101, 102}, 2, [])

    def test_tp2_rank_full_logits_match_not_just_candidates(self):
        path = self.step_path(tp=2, rank=1)
        record = json.loads(path.read_bytes())
        array = np.load(path.parent / record["tensors"]["logits"]["path"], allow_pickle=False)
        array[0, -1] -= 1
        self.replace_tensor(array, tp=2, rank=1)
        with self.assertRaisesRegex(a.EvidenceError, "TP2 rank logits/inputs"):
            self.analyze()

    def test_hash_provenance_and_only_owned_pids(self):
        for mutate, reason in ((lambda d: d.update(plan_sha256="f" * 64), "plan hash"),
                               (lambda d: d.update(pid=100), "unowned scheduler"),
                               (lambda d: d["tensors"]["hidden"].update(path="../hidden.npy"), "tensor path")):
            path = self.step_path()
            original = path.read_bytes()
            self.change(path, mutate)
            with self.subTest(reason=reason), self.assertRaisesRegex(a.EvidenceError, reason):
                self.analyze()
            path.write_bytes(original)

    def test_process_snapshot_does_not_make_foreign_scheduler_owned(self):
        self.change(self.root / "tp1/result.json", lambda d: d["process_snapshot_after_load"]["children"][0].update(ppid=999))
        with self.assertRaisesRegex(a.EvidenceError, "unowned/cyclic"):
            self.analyze()

    def test_status_baseline_cases_source_config_and_runtime_fail_closed(self):
        changes = [
            ("tp1/result.json", lambda d: d.update(status="failed"), "did not complete"),
            ("tp1/result.json", lambda d: d.update(baseline_sha256="0" * 64), "baseline hash"),
            ("tp1/result.json", lambda d: d["config"].update(seed=9), "configuration mismatch"),
            ("tp1/result.json", lambda d: d["reproducibility"]["runtime_versions"].update(torch="changed"), "runtime/source"),
            ("tp1/result.json", lambda d: d["reproducibility"]["source_trees"]["python"]["files"].update({"example.py": "e" * 64}), "source-tree hash"),
            ("tp1/result.json", lambda d: d["reproducibility"]["diagnostic_scripts"].update({"decode_probe.py": "0" * 64}), "anchored observer"),
            ("cases.json", lambda d: d["cases"][1]["input_ids"].__setitem__(0, 88), "prompt mismatch"),
        ]
        for relative, mutate, reason in changes:
            path = self.root / relative
            original = path.read_bytes()
            self.change(path, mutate)
            with self.subTest(reason=reason), self.assertRaisesRegex(a.EvidenceError, reason):
                self.analyze()
            path.write_bytes(original)

    def test_all_three_baseline_cases_checked_not_just_length32_flag(self):
        for case_index in (0, 1, 3):
            path = self.root / "tp1/result.json"
            original = path.read_bytes()
            def mutate(d):
                out = d["runs"][case_index]["outputs"][0]
                out["output_ids"][0] = 12
                out["meta_info"]["output_token_logprobs"][0][1] = 12
            self.change(path, mutate)
            with self.subTest(case=case_index), self.assertRaisesRegex(a.EvidenceError, "original baseline"):
                self.analyze()
            path.write_bytes(original)

    def test_two_repeats_exactly16_steps_each_rank_and_no_tails(self):
        path = self.step_path(tp=2, rank=1, repeat=1, step=15)
        original = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(a.EvidenceError, "exactly 16"):
            self.analyze()
        path.write_bytes(original)
        extra = path.with_name("step16.json")
        extra.write_bytes(original)
        with self.assertRaisesRegex(a.EvidenceError, "exactly 16"):
            self.analyze()
        extra.unlink()
        self.change(self.root / "tp1/probe-plan.json", lambda d: d.update(labels=[a.LABELS[0]] * 2))
        with self.assertRaisesRegex(a.EvidenceError, "plan labels"):
            self.analyze()

    def test_untrusted_huge_npy_header_is_rejected_before_allocation(self):
        path = self.step_path()
        record = json.loads(path.read_bytes())
        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(stream, {
            "descr": "<f4", "fortran_order": False, "shape": (1, 10**12)})
        raw = stream.getvalue()
        target = path.parent / record["tensors"]["logits"]["path"]
        target.unlink()
        target.write_bytes(raw)
        record["tensors"]["logits"]["sha256"] = a.sha256(raw)
        save_json(path, record)
        with self.assertRaisesRegex(a.EvidenceError, "shape/dtype"):
            self.analyze()

    def test_json_duplicates_nonfinite_scalars_and_symlinks_rejected(self):
        path = self.step_path()
        original = path.read_bytes()
        for raw in (b'{"schema":1,"schema":1}', b'{"x":NaN}', b'{"x":1e999}'):
            path.write_bytes(raw)
            with self.subTest(raw=raw), self.assertRaises(a.EvidenceError):
                self.analyze()
        path.unlink()
        borrowed = self.root / "borrowed-step.json"
        borrowed.write_bytes(original)
        path.symlink_to(borrowed)
        with self.assertRaisesRegex(a.EvidenceError, "unowned/unexpected capture file"):
            self.analyze()

    def test_gzip_preferred_raw_hash_and_selected_compressed_provenance(self):
        path = self.step_path()
        record = json.loads(path.read_bytes())
        raw_path = path.parent / record["tensors"]["logits"]["path"]
        raw = raw_path.read_bytes()
        archive = raw_path.with_name(raw_path.name + ".gz")
        compressed = gzip.compress(raw, mtime=0)
        archive.write_bytes(compressed)
        raw_path.unlink()
        raw_path.write_bytes(b"ignored raw sibling")
        report = self.analyze()
        provenance = next(p for p in report["provenance"] if p["path"] == str(archive))
        self.assertEqual(provenance["selected_sha256"], a.sha256(compressed))
        self.assertEqual(provenance["raw_sha256"], a.sha256(raw))
        self.assertEqual(provenance["compression"], "gzip")
        raw_path.unlink()  # Portable archive-only inputs work too.
        self.analyze()

    def test_invalid_gzip_never_falls_back_to_good_raw(self):
        path = self.step_path()
        record = json.loads(path.read_bytes())
        raw = path.parent / record["tensors"]["logits"]["path"]
        archive = raw.with_name(raw.name + ".gz")
        for compressed in (b"not gzip", gzip.compress(raw.read_bytes(), mtime=0)[:-6]):
            archive.write_bytes(compressed)
            with self.assertRaisesRegex(a.EvidenceError, "invalid gzip"):
                self.analyze()

    def test_gzip_decompression_bounded_and_decompressed_hash_checked(self):
        path = self.step_path()
        record = json.loads(path.read_bytes())
        raw = path.parent / record["tensors"]["logits"]["path"]
        archive = raw.with_name(raw.name + ".gz")
        archive.write_bytes(gzip.compress(b"x" * (a.MAX_BYTES + 1), mtime=0))
        with self.assertRaisesRegex(a.EvidenceError, "decompressed tensor exceeds"):
            self.analyze()
        archive.write_bytes(gzip.compress(b"incorrect raw payload", mtime=0))
        with self.assertRaisesRegex(a.EvidenceError, "raw tensor hash"):
            self.analyze()

    def test_cli_writes_only_selected_output_and_never_overwrites(self):
        output = self.root / "analysis.json"
        args = ["--tp1", str(self.root / "tp1"), "--tp2", str(self.root / "tp2"),
                "--baseline-tp1", str(self.root / "baseline-tp1.json"), "--baseline-tp2", str(self.root / "baseline-tp2.json"),
                "--cases", str(self.root / "cases.json"), "--output", str(output)]
        before = {p: a.sha256(p.read_bytes()) for p in self.root.rglob("*") if p.is_file()}
        process = subprocess.run([sys.executable, str(SCRIPTS / "analyze_decode_probe.py"), *args], capture_output=True)
        self.assertEqual(process.returncode, 0, process.stderr.decode())
        self.assertEqual(process.stdout, b"")
        self.assertTrue(output.is_file())
        self.assertEqual({p: a.sha256(p.read_bytes()) for p in before}, before)
        self.assertEqual({p for p in self.root.rglob("*") if p.is_file()}, set(before) | {output})
        saved = output.read_bytes()
        with self.assertRaises(SystemExit) as exc:
            a.main(args)
        self.assertEqual(exc.exception.code, 2)
        self.assertEqual(output.read_bytes(), saved)
        for target in (self.root / "cases.json", self.root / "tp1/new-analysis.json"):
            args[-1] = str(target)
            with self.assertRaises(SystemExit):
                a.main(args)
        self.assertFalse((self.root / "tp1/new-analysis.json").exists())
        output.unlink()
        self.change_step(lambda d: d.update(pid=999))
        args[-1] = str(output)
        with self.assertRaises(SystemExit):
            a.main(args)
        self.assertFalse(output.exists())


class RealEvidenceSmoke(unittest.TestCase):
    def test_bounded_retry_observations_when_available(self):
        hw = SCRIPTS.parent
        results = hw / "results"
        tp1 = results / "4B-decode-probe-fp16-tp1-retry"
        tp2 = results / "4B-decode-probe-fp16-tp2-retry"
        baseline1 = results / "4B-fp16-cachefix-correctness-tp1-c1/result.json"
        baseline2 = results / "4B-fp16-cachefix-correctness-tp2-c1/result.json"
        required = [tp1 / "result.json", tp2 / "result.json", baseline1, baseline2, hw / "cases.json"]
        for root in (tp1, tp2):
            for rank in range(1 if root == tp1 else 2):
                for label in a.LABELS:
                    for step in range(16):
                        for name in ("logits", "hidden"):
                            path = root / f"{label}-rank{rank}" / f"step{step:02d}-{name}.npy"
                            if not path.exists() and not path.with_name(path.name + ".gz").exists():
                                self.skipTest("bounded real tensors not installed; synthetic coverage remains self-contained")
        if not all(p.exists() for p in required):
            self.skipTest("real evidence not installed")
        report = a.analyze(tp1, tp2, baseline1, baseline2, hw / "cases.json")
        for repeat in report["repetitions"]:
            div = repeat["first_divergence"]
            self.assertEqual(div["step_index"], 9)
            self.assertEqual(len(repeat["compared_steps"]), 10)
            self.assertEqual(repeat["excluded_after_divergence"], list(range(10, 16)))
            self.assertEqual(div["tp1"]["actual_output_id"], 20943)
            self.assertEqual(div["tp2"]["actual_output_id"], 760)
            self.assertEqual(div["tp1"]["diagnostic_pair_raw_logits"], {"20943": 14.296875, "760": 14.2890625})
            self.assertEqual(div["tp2"]["diagnostic_pair_raw_logits"], {"20943": 14.2890625, "760": 14.2890625})
            self.assertEqual(div["tp2"]["max_tied_ids"], [760, 20943])
            self.assertAlmostEqual(div["tp2"]["dot_margin_tp1_choice_minus_tp2_choice"], 3.465818736e-5, places=12)
            self.assertGreater(div["tp1"]["dot_margin_tp1_choice_minus_tp2_choice"], 0)
            self.assertFalse(repeat["compared_steps"][-1]["hidden_exact_equal"])
        self.assertTrue(report["tp2_rank_logits_and_inputs_exact"])
        for value in report["repeat_reproducibility"].values():
            self.assertTrue(all(value.values()))


if __name__ == "__main__":
    unittest.main()
