"""Synthetic clock/stream tests: no NPU execution or hardware performance claims."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("run_model")
with mock.patch.dict(sys.modules, {"run_model": runner}):
    verifier = load("summarize_streaming")


def event(index, seconds, ids, count, *, length=32, finish=None):
    return {"index": index, "seconds": seconds, "output_ids": ids,
            "meta_info": {"id": f"request-{index}", "completion_tokens": count,
                          "prompt_tokens": length, "cached_tokens": 0, "finish_reason": finish}}


def events(*, length=32, count=3, batch=2):
    rows = []
    for token in range(count):
        for index in reversed(range(batch)):
            rows.append(event(index, 0.1 + len(rows) * 0.1, [7 + token + index], token + 1,
                              length=length, finish={"type": "length", "length": count}
                              if token == count - 1 else None))
    return rows


def replay(rows, **kwargs):
    return runner.summarize_stream_events(rows, batch_size=2, new_tokens=3,
                                          input_tokens=32, batch_seconds=1, **kwargs)


def artifact():
    runs = []
    for length in (32, 256):
        rows = events(length=length, count=16)
        computed = runner.summarize_stream_events(rows, batch_size=2, new_tokens=16,
                                                  input_tokens=length, batch_seconds=4)
        runs.append({"case": f"length_{length}", "repeat": 0, "input_tokens": length,
                     "output_tokens_per_request": 16, "batch_seconds": 4,
                     "aggregate_output_tokens_per_second": 8,
                     "stream_events": rows, "stream_requests": computed["requests"],
                     "outputs": computed["outputs"],
                     "warmup_output_ids": [o["output_ids"] for o in computed["outputs"]]})
    return {"mode": "streaming", "stream_schema_version": 1,
            "requested_repeats": 1,
            "status": "generation_completed_not_yet_compared", "tp": 2, "concurrency": 2,
            "config": {"stream_interval": 1, "enable_prefix_caching": False,
                       "disable_kvstore": True, "enable_output_logprobs": False}, "runs": runs}


class StreamingMetricTests(unittest.TestCase):
    def test_interleaved_requests_are_demultiplexed_by_index(self):
        result = replay(events())
        self.assertEqual([o["output_ids"] for o in result["outputs"]], [[7, 8, 9], [8, 9, 10]])
        self.assertAlmostEqual(result["requests"][0]["ttft_ms"], 200)
        self.assertAlmostEqual(result["requests"][1]["ttft_ms"], 100)
        self.assertAlmostEqual(result["requests"][0]["tpot_ms"], 200)

    def test_empty_terminal_frame_does_not_move_last_token(self):
        rows = events()
        rows[-1]["meta_info"]["finish_reason"] = None
        rows.append(event(0, .9, [], 3, finish={"type": "length", "length": 3}))
        request = replay(rows)["requests"][0]
        self.assertAlmostEqual(request["last_token_ms"], 600)
        self.assertAlmostEqual(request["request_completion_ms"], 900)

    def test_coalesced_first_chunk_does_not_fabricate_ttft(self):
        rows = [event(i, .1, [7, 8], 2) for i in (0, 1)]
        rows += [event(i, .5, [9], 3, finish={"type": "length", "length": 3}) for i in (0, 1)]
        for request in replay(rows)["requests"]:
            self.assertIsNone(request["ttft_ms"])
            self.assertIsNone(request["tpot_ms"])
            self.assertIsNone(request["inter_token_ms"])
            self.assertEqual(request["first_chunk_ms"], 100)

    def test_later_coalescing_preserves_endpoint_mean_but_not_intertoken_times(self):
        rows = [event(i, .1, [7], 1) for i in (0, 1)]
        rows += [event(i, .5, [8, 9], 3, finish={"type": "length", "length": 3}) for i in (0, 1)]
        request = replay(rows)["requests"][0]
        self.assertEqual(request["ttft_ms"], 100)
        self.assertEqual(request["tpot_ms"], 200)
        self.assertIsNone(request["inter_token_ms"])

    def test_one_output_has_no_decode_interval(self):
        result = runner.summarize_stream_events(events(count=1, batch=1), batch_size=1,
                                                new_tokens=1, input_tokens=32, batch_seconds=1)
        self.assertIsNone(result["requests"][0]["tpot_ms"])
        self.assertEqual(result["requests"][0]["inter_token_ms"], [])

    def test_simultaneous_client_arrivals_are_valid(self):
        rows = events()
        for row in rows:
            row["seconds"] = .1
        self.assertEqual(replay(rows)["requests"][0]["tpot_ms"], 0)

    def test_repeated_token_ids_are_deltas_not_duplicates(self):
        rows = events()
        for row in rows:
            row["output_ids"] = [7]
        self.assertEqual(replay(rows)["outputs"][0]["output_ids"], [7, 7, 7])

    def test_rejects_bad_times(self):
        for value in (float("nan"), float("inf"), -.1, 1.1, True):
            with self.subTest(value=value):
                rows = events()
                rows[1]["seconds"] = value
                with self.assertRaises(ValueError):
                    replay(rows)

    def test_rejects_time_reversal(self):
        rows = events()
        rows[-1]["seconds"] = .01
        with self.assertRaises(ValueError):
            replay(rows)

    def test_rejects_missing_truncated_and_postfinish_requests(self):
        for rows in (events()[:-1], [], [e for e in events() if e["index"] == 0],
                     events() + [event(0, .9, [], 3)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                replay(rows)

    def test_rejects_invalid_index(self):
        for value in (-1, 2, True, "0", None):
            rows = events()
            rows[0]["index"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                replay(rows)

    def test_rejects_identity_changes_or_reuse(self):
        for position, identity in ((2, "different"), (1, "request-1")):
            rows = events()
            rows[position]["meta_info"]["id"] = identity
            with self.assertRaises(ValueError):
                replay(rows)

    def test_rejects_cache_counts_and_abort(self):
        for key, value in (("cached_tokens", 1), ("prompt_tokens", 31),
                           ("completion_tokens", 2), ("finish_reason", {"type": "abort"})):
            rows = events()
            rows[0]["meta_info"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                replay(rows)

    def test_rejects_cumulative_payload_or_invalid_ids(self):
        for ids in ([8, 9], [-1], [True], [7.0], "7"):
            rows = events()
            rows[2]["output_ids"] = ids
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                replay(rows)

    def test_measurement_matches_warmup_and_closes_stream(self):
        rows = events()
        chunks = [{k: v for k, v in e.items() if k != "seconds"} for e in rows]
        stream = mock.MagicMock()
        stream.__iter__.return_value = iter(chunks)
        engine = SimpleNamespace(generate=mock.Mock(return_value=stream))
        expected = replay(rows)["outputs"]
        measured = runner.measure_stream_batch(engine, batch=[[1] * 32] * 2,
                    params={"max_new_tokens": 3}, expected_outputs=expected,
                    clock=mock.Mock(side_effect=[0] + [e["seconds"] for e in rows] + [1]))
        self.assertEqual(measured["outputs"], expected)
        self.assertEqual(measured["aggregate_output_tokens_per_second"], 6)
        stream.close.assert_called_once()
        self.assertTrue(engine.generate.call_args.kwargs["stream"])
        self.assertFalse(engine.generate.call_args.kwargs["return_logprob"])

    def test_measurement_rejects_warmup_mismatch(self):
        rows = events()
        engine = SimpleNamespace(generate=lambda **kw: iter(rows))
        with self.assertRaisesRegex(ValueError, "warmup"):
            runner.measure_stream_batch(engine, batch=[[1] * 32] * 2,
                params={"max_new_tokens": 3}, expected_outputs=[{"output_ids": [0]}] * 2,
                clock=mock.Mock(side_effect=[0] + [e["seconds"] for e in rows] + [1]))

    def test_capture_detaches_reused_producer_buffers(self):
        rows = events(batch=1)
        def produce():
            chunk = copy.deepcopy(rows[0])
            for row in rows:
                chunk["output_ids"][:] = row["output_ids"]
                chunk["meta_info"].update(row["meta_info"])
                yield chunk
        measured = runner.measure_stream_batch(
            SimpleNamespace(generate=lambda **kw: produce()), batch=[[1] * 32],
            params={"max_new_tokens": 3}, expected_outputs=[{"output_ids": [7, 8, 9]}],
            clock=mock.Mock(side_effect=[0, .1, .2, .3, 1]))
        self.assertEqual([r["output_ids"] for r in measured["stream_events"]], [[7], [8], [9]])
        self.assertEqual([r["meta_info"]["completion_tokens"] for r in measured["stream_events"]], [1, 2, 3])

    def test_stream_exception_propagates_and_closes_iterator(self):
        stream = mock.MagicMock()
        stream.__iter__.side_effect = RuntimeError("producer failed")
        with self.assertRaisesRegex(RuntimeError, "producer failed"):
            runner.measure_stream_batch(SimpleNamespace(generate=lambda **kw: stream),
                batch=[[1] * 32], params={"max_new_tokens": 3}, expected_outputs=[], clock=lambda: 0)
        stream.close.assert_called_once()

    def test_streaming_cli_wires_engine_config_warmup_and_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            model_bytes = b'{"text_config": {"vocab_size": 100}}'
            (model / "config.json").write_bytes(model_bytes)
            (model / "source-manifest.json").write_text(json.dumps({"files": [
                {"path": "config.json", "sha256": hashlib.sha256(model_bytes).hexdigest()}]}))
            cases = root / "cases.json"
            cases.write_text(json.dumps({"cases": [
                {"name": f"length_{length}", "input_ids": [1] * length} for length in (32, 256)]}))
            output = root / "results"
            def generate(**kwargs):
                length = len(kwargs["input_ids"][0])
                rows = events(length=length, count=16)
                if kwargs.get("stream"):
                    return iter(rows)
                return runner.summarize_stream_events(rows, batch_size=2, new_tokens=16,
                    input_tokens=length, batch_seconds=4)["outputs"]
            engine = SimpleNamespace(generate=mock.Mock(side_effect=generate), shutdown=mock.Mock())
            factory = mock.Mock(return_value=engine)
            modules = {"torch": SimpleNamespace(npu=SimpleNamespace(
                is_available=lambda: True, device_count=lambda: 2)),
                "torch_npu": SimpleNamespace(),
                "tokenspeed.runtime.entrypoints.engine": SimpleNamespace(Engine=factory)}
            argv = ["run_model.py", "--tp", "2", "--physical-devices", "0,1", "--model", str(model),
                    "--dtype", "bfloat16", "--cases", str(cases), "--output-dir", str(output),
                    "--memory-fraction", "0.12", "--concurrency", "2", "--mode", "streaming",
                    "--port", "31889", "--repeats", "1"]
            with mock.patch.dict(sys.modules, modules), mock.patch.object(sys, "argv", argv), \
                 mock.patch.multiple(runner, snapshot=lambda p: None, process_snapshot=lambda: {},
                    project_source_fingerprints=lambda: {}, runtime_versions=lambda: {}), \
                 mock.patch.object(runner.time, "perf_counter", side_effect=[i / 10 for i in range(68)]):
                runner.main()
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(factory.call_args.kwargs["stream_interval"], 1)
            self.assertFalse(factory.call_args.kwargs["enable_output_logprobs"])
            self.assertEqual(result["mode"], "streaming")
            self.assertEqual(len(result["reproducibility"]["runner_sha256"]), 64)
            self.assertEqual(verifier.summarize(result)["verified_requests"], 4)
            self.assertEqual(engine.generate.call_count, 4)
            engine.shutdown.assert_called_once()


class StreamingReplayTests(unittest.TestCase):
    def test_replays_complete_grid(self):
        result = verifier.summarize(artifact())
        self.assertTrue(result["passed"])
        self.assertEqual(result["verified_requests"], 4)
        self.assertEqual(result["summary"][0]["metrics"]["ttft_ms"]["median"], 150)

    def test_rejects_legacy_failed_or_wrong_configuration(self):
        for key, value in (("mode", "benchmark"), ("status", "failed"), ("stream_schema_version", 2)):
            result = artifact()
            result[key] = value
            with self.assertRaises(ValueError):
                verifier.summarize(result)
        result = artifact()
        result["config"]["enable_prefix_caching"] = True
        with self.assertRaises(ValueError):
            verifier.summarize(result)

    def test_detects_altered_metrics_outputs_and_rate(self):
        for kind in ("timing", "output", "warmup", "rate"):
            result = artifact()
            run = result["runs"][0]
            if kind == "timing":
                run["stream_requests"][0]["ttft_ms"] = 0
            elif kind == "output":
                run["outputs"] = []
            elif kind == "warmup":
                run["warmup_output_ids"] = []
            else:
                run["aggregate_output_tokens_per_second"] = float("nan")
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                verifier.summarize(result)

    def test_rejects_missing_duplicate_and_noncontiguous_samples(self):
        for kind in ("missing", "duplicate", "gap", "missing_final_repeat"):
            result = artifact()
            if kind == "missing":
                result["runs"].pop()
            elif kind == "duplicate":
                result["runs"].append(copy.deepcopy(result["runs"][0]))
            elif kind == "gap":
                result["runs"][0]["repeat"] = 1
            else:
                result["requested_repeats"] = 2
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                verifier.summarize(result)

    def test_raw_events_survive_json_round_trip(self):
        self.assertTrue(verifier.summarize(json.loads(json.dumps(artifact())))["passed"])

    def test_partial_ttft_coverage_does_not_report_selective_median(self):
        result = artifact()
        row = result["runs"][0]
        events_ = row["stream_events"]
        # Coalesce request 1's first two tokens at the second arrival time.
        events_[2]["output_ids"] = events_[0]["output_ids"] + events_[2]["output_ids"]
        del events_[0]
        replayed = runner.summarize_stream_events(events_, batch_size=2, new_tokens=16,
                                                  input_tokens=32, batch_seconds=4)
        row["stream_requests"] = replayed["requests"]
        row["outputs"] = replayed["outputs"]
        metric = verifier.summarize(result)["summary"][0]["metrics"]["ttft_ms"]
        self.assertEqual(metric["observed_requests"], 1)
        self.assertEqual(metric["total_requests"], 2)
        self.assertIsNone(metric["median"])


if __name__ == "__main__":
    unittest.main()
