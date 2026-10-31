"""Bounded warmed requests using the explicit per-worker NPU profiler API.

Profiled latency is not benchmark latency. Success requires two nonempty device
traces; communication attribution and coverage are checked separately offline.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import run_model as harness


def trace_events(path):
    document = json.loads(path.read_text())
    events = document.get("traceEvents") if isinstance(document, dict) else document
    if not isinstance(events, list) or not events:
        raise ValueError("missing Chrome event array")
    device = [e for e in events if e.get("ph") == "X" and
              (e.get("cat", "").lower() == "kernel" or
               e.get("args", {}).get("Task Type") in {"AI_CORE", "AI_VECTOR_CORE", "MIX_AIC"})
              and float(e.get("dur", 0)) > 0]
    if not device:
        raise ValueError("trace has no actual device compute events")
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size, "events": len(events), "device_compute_events": len(device)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--physical-devices", type=harness.parse_devices, required=True)
    p.add_argument("--memory-fraction", type=float, required=True)
    p.add_argument("--concurrency", type=int, choices=(1, 2), required=True)
    p.add_argument("--new-tokens", type=int, choices=(2, 4, 8, 16), required=True,
                   help="explicit bounded profile window; not benchmark output length")
    p.add_argument("--input-lengths", choices=("32", "256", "32,256"), required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    config = harness.build_config(args.model, tp=2, devices=args.physical_devices,
                                  memory_fraction=args.memory_fraction, concurrency=args.concurrency, port=args.port)
    manifest_bytes = (args.model / "source-manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    config_bytes = (args.model / "config.json").read_bytes()
    expected = [e for e in manifest["files"] if e["path"] == "config.json"]
    if len(expected) != 1 or expected[0]["sha256"] != hashlib.sha256(config_bytes).hexdigest():
        raise ValueError("model config/manifest mismatch")
    cases_bytes = args.cases.read_bytes()
    lengths = [int(value) for value in args.input_lengths.split(",")]
    cases = [c for c in json.loads(cases_bytes)["cases"] if len(c["input_ids"]) in lengths]
    if sorted(len(c["input_ids"]) for c in cases) != lengths:
        raise ValueError("requires exactly the requested input-length cases")
    vocab = json.loads(config_bytes)["text_config"]["vocab_size"]
    if any(type(i) is not int or not 0 <= i < vocab for c in cases for i in c["input_ids"]):
        raise ValueError("invalid input token ID")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, args.physical_devices))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    result = {"status": "starting", "config": config, "tp": 2, "concurrency": args.concurrency,
              "physical_devices_requested": args.physical_devices,
              "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
              "cases_sha256": hashlib.sha256(cases_bytes).hexdigest(),
              "profile_activities": ["NPU"], "new_tokens": args.new_tokens, "input_lengths": lengths, "runs": [],
              "reproducibility": {"environment": harness.selected_environment(),
                                  "source_trees": harness.project_source_fingerprints()},
              "scope": "Two-worker device trace collection, not numerical certification or benchmark latency."}
    def save():
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    save()
    engine = None
    try:
        from tokenspeed.runtime.entrypoints.engine import Engine
        harness.snapshot(args.output_dir / "hbm-before.txt")
        engine = Engine(**config)
        result["reproducibility"]["runtime_versions"] = harness.runtime_versions()
        result["process_snapshot_after_load"] = harness.process_snapshot()
        harness.snapshot(args.output_dir / "hbm-loaded.txt")
        params = {"temperature": 0, "max_new_tokens": args.new_tokens, "ignore_eos": True}
        for case in cases:
            batch = [case["input_ids"][:] for _ in range(args.concurrency)]
            warmup = engine.generate(input_ids=batch, sampling_params=params)
            harness.validate_outputs(warmup, batch_size=args.concurrency, new_tokens=args.new_tokens,
                                     input_tokens=len(case["input_ids"]))
            profile_id = f"hw2-npu-c{args.concurrency}-input{len(case['input_ids'])}"
            engine.llm.run(engine.tokenizer_manager.start_profile(
                output_dir=str(args.output_dir), activities=["NPU"], with_stack=False,
                record_shapes=False, profile_id=profile_id))
            row = {"case": case["name"], "input_tokens": len(case["input_ids"]),
                   "profile_id": profile_id, "warmup_output_ids": [o["output_ids"] for o in warmup],
                   "start_unix_ns": time.time_ns(), "start_monotonic_ns": time.monotonic_ns()}
            result["runs"].append(row)
            try:
                row["outputs"] = engine.generate(input_ids=batch, sampling_params=params)
                row.update(end_monotonic_ns=time.monotonic_ns(), end_unix_ns=time.time_ns())
                harness.validate_outputs(row["outputs"], batch_size=args.concurrency, new_tokens=args.new_tokens,
                                         input_tokens=len(case["input_ids"]))
                if [o["output_ids"] for o in row["outputs"]] != row["warmup_output_ids"]:
                    raise ValueError("profiled output differs from warmed output")
            finally:
                save()
                engine.stop_profile()
            row["traces"] = {f"TP{rank}": trace_events(args.output_dir / f"{profile_id}-TP{rank}.trace.json")
                             for rank in range(2)}
            harness.snapshot(args.output_dir / f"hbm-input{len(case['input_ids'])}.txt")
            save()
        result["status"] = "two_worker_device_traces_collected_not_yet_analyzed"
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            if engine is not None:
                engine.shutdown()
        finally:
            save()


if __name__ == "__main__":
    main()
