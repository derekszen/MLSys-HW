"""Stdlib-only interval, attribution, clock, and fail-closed provenance tests."""
import copy
from decimal import Decimal as D
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import analyze_npu_traces as a

BASE = D("1792817202870000.125")


def fixture():
    events = []

    def meta(kind, pid, tid, **args):
        events.append(dict(name=kind, pid=pid, tid=tid, ph="M", args=args))

    def task(name, pid, tid, start, dur, **args):
        e = dict(name=name, pid=pid, tid=tid, ph="X", ts=str(BASE + D(start)), dur=D(dur), args=args)
        events.append(e)
        return e

    for pid, name in ((100, "Ascend Hardware"), (200, "Communication"), (300, "CANN")):
        meta("process_name", pid, 0, name=name)
    for pid in (100, 200):
        meta("process_labels", pid, 0, labels="NPU 4")
    meta("thread_name", 300, 11, name="Thread 11")
    meta("thread_name", 200, 0, name="Group example Communication")
    meta("thread_name", 200, 1, name="Plane 0")
    index = 0

    def hw(name, tid, start, dur, kind, connection=None):
        nonlocal index
        index += 1
        return task(name, 100, tid, start, dur, **{"Task Type": kind, "Physic Stream Id": tid,
                    "Task Id": index, "connection_id": connection or index})

    hw("PROFILING_ENABLE", 1, "0", "0.02", "PROFILING_ENABLE")
    hw("PROFILING_DISABLE", 1, "25", "0.02", "PROFILING_DISABLE")
    hw("matmul", 50, "4", "3", "AI_CORE")
    hw("local_ReduceSum", 50, "6", "3", "MIX_AIV")
    for n, (op, kernel, start, dur, dtype, count) in enumerate((
            ("allReduce", "allreduceAicpuKernel", 5, 5, "BFP16", 1024),
            ("allGather", "allgatherAicpuKernel", 12, 3, "BFP16", 124160),
            ("broadcast", "broadcastAicpuKernel", 17, 2, "INT32", 6))):
        task(f"hcom_{op}__909_{n}_1", 200, 0, start, dur,
             connection_id=1000 + n, count=count, data_type=dtype)
        h = hw("NOTIFY_RECORD", 60, str(start), "0.1", "NOTIFY_RECORD")
        task("Notify_Record", 200, 1, start, "0.1", **{"src rank": 0, "dst rank": 1,
             "stream id": 60, "task id": h["args"]["Task Id"], "task type": "Notify_Record", "size(Byte)": 0})
        connection = 2000 + n
        hw("NOTIFY_WAIT", 77, str(D(start) + D(".2")), ".1", "NOTIFY_WAIT", connection)
        hw(kernel, 77, str(D(start) + D(".5")), "1", "AI_CPU", connection)
        task("Node@launch", 300, 11, start, ".2", connection_id=connection, item_id=kernel)
    # A long host API is deliberately NOT a device communication interval.
    task("HcclAllReduce", 300, 11, "1", "20")
    run = dict(start_unix_ns=int((BASE + 4) * 1000), end_unix_ns=int((BASE + 16) * 1000),
               start_monotonic_ns=1000, end_monotonic_ns=13000)
    result = dict(physical_devices_requested=[4, 5], concurrency=1, new_tokens=2, runs=[run],
                  process_snapshot_after_load=dict(parent_pid=10, children=[
                      dict(pid=11, ppid=10, name="tokenspeed::scheduler_tp0"),
                      dict(pid=12, ppid=10, name="tokenspeed::scheduler_tp1")]))
    return events, result


def grid_fixture():
    config = dict(device="npu", dtype="bfloat16", world_size=2, base_gpu_id=0, gpu_id_step=1)
    trees = {"python": {"sha256": "new", "files": {"tokenspeed/runtime/engine/request_handler.py": "new"}}}
    repro = dict(source_trees=trees, environment={"ASCEND_RT_VISIBLE_DEVICES": "4,5"}, runtime_versions={"torch": "test"})
    base = dict(status="two_worker_device_traces_collected_not_yet_analyzed", concurrency=1, tp=2,
                new_tokens=2, input_lengths=[32], physical_devices_requested=[4, 5], profile_activities=["NPU"],
                config=config, reproducibility=repro, manifest_sha256="manifest", config_sha256="config", cases_sha256="cases")
    results = {}
    for c in (1, 2):
        r = copy.deepcopy(base)
        r["concurrency"] = c
        r["runs"] = [dict(case="length_32", input_tokens=32, profile_id=f"hw2-npu-c{c}-input32",
                          traces={"TP0": {}, "TP1": {}}, warmup_output_ids=[[9064, 11988]] * c,
                          outputs=[dict(output_ids=[9064, 11988], meta_info=dict(prompt_tokens=32,
                                        completion_tokens=2, cached_tokens=0)) for _ in range(c)])]
        results[c] = r
    canonical = copy.deepcopy(base)
    canonical["runs"] = [dict(case="length_32", outputs=[dict(output_ids=[9064, 11988, 3])])]
    canonical["reproducibility"]["source_trees"]["python"]["files"]["tokenspeed/runtime/engine/request_handler.py"] = "old"
    return results, canonical


class IntervalTests(unittest.TestCase):
    def test_union_nested_adjacent_disjoint_and_zero(self):
        self.assertEqual(a.union([(1, 5), (2, 3), (5, 8), (10, 12), (9, 9)]), [(1, 8), (10, 12)])
        self.assertEqual(a.duration([(1, 5), (2, 3), (5, 8)]), 7)
        self.assertEqual(a.union([]), [])
        with self.assertRaises(ValueError):
            a.union([(3, 2)])

    def test_intersection_and_exact_epoch_clipping(self):
        b = BASE
        left = [(b, b + 7), (b + 5, b + 10)]
        right = [(b + 2, b + 4), (b + 6, b + 12)]
        self.assertEqual(a.intersection(left, right), [(b + 2, b + 4), (b + 6, b + 10)])
        self.assertEqual(a.clip(left, (b + D(".001"), b + D(".003"))),
                         [(b + D(".001"), b + D(".003"))])
        self.assertEqual(a.duration(a.clip(left, (b + D(".001"), b + D(".003")))), D(".002"))
        self.assertEqual(a.intersection([(1, 2)], [(2, 3)]), [])
        with self.assertRaises(ValueError):
            a.clip(left, (b, b))

    def test_bad_numeric_fields(self):
        for value in (float(BASE), "NaN", "Infinity", "not a number", True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                a.number(value)
        with self.assertRaises(ValueError):
            a.interval(dict(ts="1", dur="-1"))

    def test_overlapping_nested_and_duplicate_parents_rejected(self):
        events, _ = fixture()
        p = next(e for e in events if e["name"].startswith("hcom_"))
        for start, dur in (("7", "4"), ("6", "1"), ("5", "5")):
            q = copy.deepcopy(p)
            q.update(ts=str(BASE + D(start)), dur=D(dur))
            with self.subTest(start=start, dur=dur), self.assertRaisesRegex(ValueError, "overlapping/nested"):
                a.validate_parents([p, q])


class DeviceEvidenceTests(unittest.TestCase):
    def analyze(self, events, result):
        return a.analyze_events(events, result, 0, "example-TP0.trace.json")

    def test_both_document_layouts_and_device_metrics(self):
        events, result = fixture()
        for document in (events, {"traceEvents": events}):
            # Round trip through the same Decimal parser as actual CANN files.
            parsed = json.loads(json.dumps(document, default=str), parse_float=D)
            row = self.analyze(a.event_array(parsed), result)
            m = row["request_window"]
            self.assertEqual(m["compute_us"], 5)
            self.assertEqual(m["communication_envelopes_us"], 8)
            self.assertEqual(m["overlap_us"], 4)
            self.assertEqual(m["compute_only_us"], 1)
            self.assertEqual(m["communication_only_us"], 4)
            self.assertEqual(m["unclassified_gap_us"], 3)
            self.assertEqual(m["compute_tasks"], 2)  # local ReduceSum retained; three HCCL kernels excluded
            self.assertEqual(m["collective_count"], 2)  # broadcast after request excluded
            self.assertEqual(m["communication_us_per_returned_output_token"], 4)
            self.assertEqual(row["diagnostics"]["host_HcclAllReduce_API_events_not_device_ops"], 1)
            self.assertEqual(row["diagnostics"]["hccl_subtasks_identically_mapped"], 3)
            self.assertEqual(row["diagnostics"]["hccl_kernels_excluded_from_compute"], 3)
            self.assertEqual(m["payloads"][1]["input_bytes_per_operation"], 2048)

    def test_clock_fallback_and_partial_collective_clipping(self):
        events, result = fixture()
        result["runs"][0].update(start_unix_ns=int((BASE + 6) * 1000), end_unix_ns=int((BASE + 14) * 1000),
                                  start_monotonic_ns=1000, end_monotonic_ns=9000)
        row = self.analyze(events, result)
        self.assertEqual(row["request_window"]["communication_envelopes_us"], 6)
        self.assertEqual(row["request_window"]["partially_clipped_collectives"], 2)
        result["runs"][0]["end_unix_ns"] += 10**12
        row = self.analyze(events, result)
        self.assertIsNone(row["request_window"])
        self.assertFalse(row["clock"]["same_epoch_bracket_verified"])
        self.assertNotIn("communication_us_per_returned_output_token", row["device_active_span"])

    def test_malformed_or_name_only_device_evidence_rejected(self):
        mutations = [
            lambda es, r: es.__setitem__(slice(None), [e for e in es if e.get("pid") != 100]),
            lambda es, r: next(e for e in es if e["name"] == "matmul")["args"].pop("Task Type"),
            lambda es, r: next(e for e in es if e["name"] == "matmul")["args"].update({"Task Type": "FAKE"}),
            lambda es, r: next(e for e in es if e["name"] == "Notify_Record")["args"].update({"task id": 999}),
            lambda es, r: next(e for e in es if e["name"] == "Node@launch")["args"].update(connection_id=-1),
            lambda es, r: next(e for e in es if e["name"].startswith("hcom_"))["args"].update(data_type="UNKNOWN"),
            lambda es, r: next(e for e in es if e["name"] == "Notify_Record")["args"].update({"src rank": 1}),
            lambda es, r: next(e for e in es if e["name"] == "process_labels")["args"].update(labels="NPU 0"),
            lambda es, r: r["process_snapshot_after_load"]["children"][0].update(pid=111),
        ]
        for index, mutate in enumerate(mutations):
            events, result = fixture()
            mutate(events, result)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.analyze(events, result)
        for document in ([], {}, {"traceEvents": "not array"}, ["not event"]):
            with self.subTest(document=document), self.assertRaises(ValueError):
                a.event_array(document)

    def test_filename_and_metadata_conflicts_rejected(self):
        events, result = fixture()
        with self.assertRaisesRegex(ValueError, "filename"):
            a.ownership(events, result, 0, "example-TP1.trace.json")
        events.append(dict(name="process_name", pid=100, tid=0, ph="M", args=dict(name="CPU")))
        with self.assertRaisesRegex(ValueError, "conflicting"):
            self.analyze(events, result)

    def test_card_chip_evidence(self):
        text = "| 2 Ascend910 | OK |\n| 0 4 | 0000:95:00.0 |\n| 2 0 | 11 | worker |\n"
        self.assertEqual(a.card_evidence(text, 11, 4)["card"], 2)
        with self.assertRaises(ValueError):
            a.card_evidence(text, 12, 4)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.events, self.result = fixture()
        self.content = json.dumps({"traceEvents": self.events}, default=str).encode()
        self.recorded = {"bytes": len(self.content), "sha256": hashlib.sha256(self.content).hexdigest(),
                         "events": len(self.events)}
        self.raw = self.root / "hw2-npu-c1-input32-TP0.trace.json"
        self.archive = self.raw.with_name(self.raw.name + ".gz")
        self.raw.write_bytes(self.content)
        self.archive.write_bytes(gzip.compress(self.content, mtime=0))

    def test_raw_gzip_equivalence_and_selected_file_provenance(self):
        rows = []
        for path in (self.raw, self.archive):
            events, fp = a.load_trace(path, self.root, self.recorded)
            rows.append(a.analyze_events(events, self.result, 0, path.name))
            self.assertEqual(a.load(path), {"traceEvents": events})
            self.assertEqual(fp["sha256"], self.recorded["sha256"])
            self.assertEqual(fp["bytes"], len(self.content))
            self.assertEqual(fp["path"], path.name)
            self.assertEqual(fp["stored_file"], {"path": path.name, "bytes": path.stat().st_size,
                                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            self.assertEqual(fp["compression"], "gzip" if path == self.archive else "none")
        self.assertEqual(rows[0], rows[1])
        self.assertNotEqual(hashlib.sha256(self.archive.read_bytes()).hexdigest(), self.recorded["sha256"])
        with self.assertRaisesRegex(ValueError, "filename"):
            a.ownership(self.events, self.result, 0, "example-TP1.trace.json.gz")

    def test_expected_uncompressed_fingerprint_and_event_count_required(self):
        for path in (self.raw, self.archive):
            for field, wrong in (("sha256", "0" * 64), ("bytes", len(self.content) + 1),
                                 ("events", len(self.events) + 1)):
                with self.subTest(path=path.name, field=field), self.assertRaisesRegex(ValueError, "mismatch"):
                    a.load_trace(path, self.root, {**self.recorded, field: wrong})

    def test_trace_grid_raw_gzip_mixed_and_duplicates(self):
        peer = self.root / "hw2-npu-c1-input32-TP1.trace.json"
        peer.write_bytes(self.content)
        self.assertEqual(a.trace_paths(self.root, 1), {0: self.archive, 1: peer})
        self.raw.unlink()
        self.assertEqual(a.trace_paths(self.root, 1), {0: self.archive, 1: peer})
        peer_archive = peer.with_name(peer.name + ".gz")
        peer_archive.write_bytes(self.archive.read_bytes())
        peer.unlink()
        self.assertEqual(a.trace_paths(self.root, 1), {0: self.archive, 1: peer_archive})
        self.raw.write_bytes(self.content)
        peer.write_bytes(self.content)
        self.archive.unlink()
        peer_archive.unlink()
        self.assertEqual(a.trace_paths(self.root, 1), {0: self.raw, 1: peer})
        peer.unlink()
        with self.assertRaisesRegex(ValueError, "trace file grid"):
            a.trace_paths(self.root, 1)

    def test_trace_grid_rejects_unknown_extra_files(self):
        (self.root / "hw2-npu-c1-input32-TP1.trace.json").write_bytes(self.content)
        for name in ("extra.trace.json", "extra.trace.json.gz", "hw2-npu-c1-input32-TP2.trace.json.gz",
                     "hw2-npu-c1-input32-TP0.trace.json.gz.gz", "hw2-npu-c1-input32-TP0.trace.json.bak"):
            extra = self.root / name
            extra.write_bytes(b"unexpected")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "trace file grid"):
                a.trace_paths(self.root, 1)
            extra.unlink()

    def test_gzip_preferred_without_reading_unselected_raw_and_no_fallback(self):
        (self.root / "hw2-npu-c1-input32-TP1.trace.json").write_bytes(self.content)
        self.raw.write_bytes(b"unselected raw deliberately differs")
        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            self.assertNotEqual(path, self.raw, "unselected raw must not be read or hashed")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open):
            selected = a.trace_paths(self.root, 1)[0]
            self.assertEqual(selected, self.archive)
            a.load_trace(selected, self.root, self.recorded)
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                a.load_trace(selected, self.root, {**self.recorded, "sha256": "0" * 64})
        # A bad preferred archive fails closed even if raw is valid.
        self.raw.write_bytes(self.content)
        self.archive.write_bytes(gzip.compress(self.content + b" ", mtime=0))
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            a.load_trace(a.trace_paths(self.root, 1)[0], self.root, self.recorded)

    def test_truncated_corrupt_and_invalid_gzip_rejected(self):
        stored = self.archive.read_bytes()
        corrupt = bytearray(stored)
        corrupt[-8] ^= 1  # CRC, not JSON validity
        for data in (stored[:-1], stored[:-8], stored[:len(stored) // 2], bytes(corrupt), b"not gzip"):
            self.archive.write_bytes(data)
            with self.subTest(length=len(data)), self.assertRaisesRegex(ValueError, "invalid/truncated gzip"):
                a.load_trace(self.archive, self.root, self.recorded)

    def test_inflate_bound_exact_limit_and_concatenated_members(self):
        self.assertEqual(a.MAX_FILE_BYTES, 80_000_000)
        limit = 8192
        payload = b'"' + b"x" * (limit - 2) + b'"'
        self.archive.write_bytes(gzip.compress(payload, mtime=0))
        with patch.object(a, "MAX_FILE_BYTES", limit):
            self.assertEqual(a.load(self.archive), "x" * (limit - 2))
            for data in (gzip.compress(payload + b" ", mtime=0),
                         gzip.compress(payload, mtime=0) + gzip.compress(b" ", mtime=0)):
                self.archive.write_bytes(data)
                with patch.object(a.json, "loads") as parse:
                    with self.assertRaisesRegex(ValueError, "decompressed file exceeds analysis bound"):
                        a.load(self.archive)
                    parse.assert_not_called()
            self.raw.write_bytes(payload + b" ")
            with self.assertRaisesRegex(ValueError, "file exceeds analysis bound"):
                a.load(self.raw)
            self.archive.write_bytes(b"x" * (limit + 1))
            with self.assertRaisesRegex(ValueError, "file exceeds analysis bound"):
                a.load(self.archive)


class GridTests(unittest.TestCase):
    def validate(self, results, canonical):
        return a.validate_grid(results, canonical, "cases", "manifest", "config")

    def test_only_prior_profile_hook_file_difference_allowed(self):
        results, canonical = grid_fixture()
        checks = self.validate(results, canonical)
        self.assertEqual(checks["canonical_prefix_ids"], [9064, 11988])
        self.assertEqual(len(checks["canonical_source_differences"]), 1)
        canonical["reproducibility"]["source_trees"]["python"]["files"]["model.py"] = "old"
        with self.assertRaisesRegex(ValueError, "unrelated canonical source"):
            self.validate(results, canonical)

    def test_mismatched_grid_source_model_case_and_outputs_rejected(self):
        mutations = [
            lambda rs: rs.pop(2),
            lambda rs: rs[2].update(concurrency=1),
            lambda rs: rs[2].update(physical_devices_requested=[0, 1]),
            lambda rs: rs[2]["config"].update(dtype="float16"),
            lambda rs: rs[2].update(manifest_sha256="other-model"),
            lambda rs: rs[2].update(cases_sha256="other-cases"),
            lambda rs: rs[2]["reproducibility"]["source_trees"]["python"]["files"].update(model="changed"),
            lambda rs: rs[2]["runs"][0]["traces"].pop("TP1"),
            lambda rs: rs[2]["runs"][0]["outputs"][0].update(output_ids=[9064, 0]),
            lambda rs: rs[2]["runs"][0]["outputs"][0]["meta_info"].update(cached_tokens=32),
        ]
        for index, mutate in enumerate(mutations):
            results, canonical = grid_fixture()
            mutate(results)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.validate(results, canonical)


if __name__ == "__main__":
    unittest.main()
