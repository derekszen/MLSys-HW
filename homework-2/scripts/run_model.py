"""Prepared HW2 TP=1/2 correctness/throughput runner; real NPU execution required."""
import argparse
import copy
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def selected_environment(environ=None):
    """Record only this explicit, nonsecret allowlist; unset values are null."""
    if environ is None:
        environ = os.environ
    return {key: environ.get(key) for key in (
        "ASCEND_RT_VISIBLE_DEVICES", "TOKENIZERS_PARALLELISM",
        "HCCL_NPU_SOCKET_PORT_RANGE",
    )}


def fingerprint_python_tree(root):
    """Hash relative paths and contents, independent of checkout location/mtime."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = {}
    for directory, dirs, names in os.walk(root, onerror=_raise_walk_error):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(names):
            if name.endswith(".py"):
                path = Path(directory) / name
                files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"root": str(root.resolve()), "sha256": hashlib.sha256(canonical).hexdigest(),
            "files": dict(sorted(files.items()))}


def _raise_walk_error(error):
    raise error


def project_source_fingerprints():
    # These are the three PYTHONPATH roots in homework-2/README.md, not the
    # runtime package alone. Never hash unrelated PYTHONPATH entries.
    project = Path(__file__).resolve().parents[2] / "homework-1" / "tokenspeed"
    trees = {}
    for relative in ("python", "tokenspeed-kernel/python", "tokenspeed-kernel-npu/python"):
        root = project / relative
        try:
            trees[relative] = fingerprint_python_tree(root)
        except OSError as exc:
            trees[relative] = {"root": str(root), "error": f"{type(exc).__name__}: {exc}"}
    return trees


def runtime_versions():
    """Do not import accelerator packages just to discover their versions."""
    versions = {}
    for name in ("torch", "torch_npu", "triton"):
        version = getattr(sys.modules.get(name), "__version__", None)
        if version is None:
            try:
                version = metadata.version(name)
            except metadata.PackageNotFoundError:
                version = "unavailable"
        versions[name] = str(version)
    return versions


def process_snapshot():
    """Inspect only this runner and its recursive descendants, never full argv."""
    result = {"stage": "after_engine_startup", "captured_at_unix_seconds": time.time(),
              "parent_pid": os.getpid(), "parent": None, "children": [],
              "scope": "runner and recursive descendants only; no inferred rank assignment"}
    try:
        import psutil
    except ImportError:
        result["error"] = "psutil unavailable"
        return result

    def describe(process):
        return {"pid": process.pid, "ppid": process.ppid(), "name": process.name()}

    try:
        parent = psutil.Process(result["parent_pid"])
        result["parent"] = describe(parent)
        children = parent.children(recursive=True)
    except psutil.Error as exc:
        result["error"] = type(exc).__name__
        return result
    for child in sorted(children, key=lambda process: process.pid):
        try:
            result["children"].append(describe(child))
        except psutil.Error as exc:
            # A descendant may exit between enumeration and inspection.
            result["children"].append({"pid": child.pid, "ppid": None, "name": None,
                                       "error": type(exc).__name__})
    return result


def parse_devices(value):
    parts = value.split(",")
    if not parts or any(not p.isascii() or not p.isdecimal() for p in parts):
        raise argparse.ArgumentTypeError("expected comma-separated nonnegative physical IDs")
    ids = [int(p) for p in parts]
    if len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError("physical IDs must be distinct")
    return ids


def build_config(model, *, tp, devices, memory_fraction, concurrency, port):
    if tp not in (1, 2) or len(devices) != tp or len(set(devices)) != tp:
        raise ValueError("TP degree must match one/two distinct visible devices")
    if any(type(d) is not int or d < 0 for d in devices):
        raise ValueError("invalid device ID")
    if not math.isfinite(memory_fraction) or not 0 < memory_fraction <= 0.9:
        raise ValueError("choose a bounded explicit memory fraction in (0, 0.9]")
    if concurrency not in (1, 2) or not 1024 <= port <= 65535:
        raise ValueError("unsupported concurrency or port")
    return dict(
        model=str(Path(model).resolve()), device="npu", dtype="bfloat16",
        language_model_only=True, trust_remote_code=False,
        world_size=tp, attn_tp_size=tp, dense_tp_size=tp, moe_tp_size=tp,
        lm_head_tp_size=tp, ep_size=1, pipeline_parallel_size=1,
        base_gpu_id=0, gpu_id_step=1,
        max_model_len=512, max_num_seqs=2, max_total_tokens=16384,
        chunked_prefill_size=128, max_prefill_tokens=128,
        gpu_memory_utilization=memory_fraction, mamba_ssm_dtype="float32",
        sampling_backend="greedy", enforce_eager=True,
        disable_prefill_graph=True, disable_pdl=True, disable_autotune=True,
        enable_prefix_caching=False, disable_kvstore=True,
        enable_output_logprobs=True, seed=42, log_level="info", port=port,
        distributed_timeout_seconds=60,
    )


def validate_outputs(outputs, *, batch_size, new_tokens, input_tokens):
    if not isinstance(outputs, list) or len(outputs) != batch_size:
        raise ValueError("engine did not return one result per request")
    for output in outputs:
        ids = output["output_ids"]
        meta = output["meta_info"]
        if len(ids) != new_tokens or meta["completion_tokens"] != new_tokens:
            raise ValueError("incomplete generation")
        if meta["prompt_tokens"] != input_tokens or meta["cached_tokens"] != 0:
            raise ValueError("unexpected prompt length or prefix cache reuse")
        if any(type(i) is not int or i < 0 for i in ids):
            raise ValueError("invalid generated token IDs")


def summarize_stream_events(events, *, batch_size, new_tokens, input_tokens, batch_seconds):
    """Recompute client-observed metrics from indexed token-delta arrivals.

    A coalesced first chunk cannot establish TTFT. Empty terminal frames do not
    move the last-token timestamp. Client timing includes scheduling and IPC;
    it is not an isolated prefill/decode kernel or communication measurement.
    """
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def seconds(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0

    require(type(batch_size) is int and batch_size in (1, 2), "invalid batch size")
    require(type(new_tokens) is int and new_tokens > 0, "invalid output count")
    require(type(input_tokens) is int and input_tokens > 0, "invalid input count")
    require(seconds(batch_seconds) and batch_seconds > 0, "invalid batch duration")
    require(isinstance(events, list) and bool(events), "missing streaming events")
    states = [{"output_ids": [], "arrivals": [], "finished": False, "id": None}
              for _ in range(batch_size)]
    previous = 0
    try:
        for event in events:
            index, elapsed = event["index"], event["seconds"]
            require(type(index) is int and 0 <= index < batch_size, "invalid request index")
            require(seconds(elapsed) and previous <= elapsed <= batch_seconds,
                    "invalid or nonmonotonic arrival time")
            previous = elapsed
            state = states[index]
            require(not state["finished"], "event after request completion")
            ids, meta = event["output_ids"], event["meta_info"]
            require(isinstance(ids, list) and all(type(i) is int and i >= 0 for i in ids),
                    "invalid token delta")
            rid = meta["id"]
            require(isinstance(rid, str) and bool(rid), "missing request identity")
            require(state["id"] in (None, rid), "request identity changed")
            require(all(other is state or other["id"] != rid for other in states),
                    "request identity reused across indices")
            state["id"] = rid
            state["output_ids"].extend(ids)
            count = len(state["output_ids"])
            require(count <= new_tokens, "too many tokens or cumulative payload")
            require(type(meta["completion_tokens"]) is int and meta["completion_tokens"] == count,
                    "delta/metadata token count mismatch")
            require(type(meta["prompt_tokens"]) is int and meta["prompt_tokens"] == input_tokens,
                    "prompt count mismatch")
            require(type(meta["cached_tokens"]) is int and meta["cached_tokens"] == 0,
                    "prefix reuse in streaming benchmark")
            if ids:
                state["arrivals"].append({"tokens": count, "delta_tokens": len(ids),
                                          "seconds": elapsed})
            finish = meta["finish_reason"]
            if finish is not None:
                require(finish == {"type": "length", "length": new_tokens}
                        and count == new_tokens, "request did not finish at requested length")
                state["finished"] = True
                state["completion_seconds"] = elapsed
                state["meta_info"] = meta
        outputs, requests = [], []
        for index, state in enumerate(states):
            require(state["finished"] and bool(state["arrivals"]), "truncated or missing request")
            arrivals = state["arrivals"]
            first, last = arrivals[0]["seconds"], arrivals[-1]["seconds"]
            first_is_single = arrivals[0]["delta_tokens"] == 1
            all_single = all(a["delta_tokens"] == 1 for a in arrivals)
            requests.append({
                "index": index, "arrivals": arrivals,
                "first_chunk_ms": first * 1000,
                "ttft_ms": first * 1000 if first_is_single else None,
                "tpot_ms": (last - first) * 1000 / (new_tokens - 1)
                if first_is_single and new_tokens > 1 else None,
                "inter_token_ms": [(b["seconds"] - a["seconds"]) * 1000
                                   for a, b in zip(arrivals, arrivals[1:])]
                if all_single else None,
                "last_token_ms": last * 1000,
                "request_completion_ms": state["completion_seconds"] * 1000,
                "coalesced_chunks": sum(a["delta_tokens"] > 1 for a in arrivals),
                "ttft_unavailable_reason": None if first_is_single else "first_chunk_coalesced",
            })
            outputs.append({"output_ids": state["output_ids"], "meta_info": state["meta_info"]})
        return {"outputs": outputs, "requests": requests}
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError(f"malformed streaming event: {exc}") from exc


def measure_stream_batch(engine, *, batch, params, expected_outputs, clock=None):
    """Time the public synchronous batch stream; retain replayable raw events."""
    if clock is None:
        clock = time.perf_counter
    start = clock()
    events = []
    stream = engine.generate(input_ids=batch, sampling_params=params,
                             return_logprob=False, stream=True)
    try:
        for chunk in stream:
            elapsed = clock() - start
            # Snapshot before advancing a producer that may reuse dictionaries.
            events.append({"seconds": elapsed, "index": chunk["index"],
                           "output_ids": list(chunk["output_ids"]),
                           "meta_info": copy.deepcopy(chunk["meta_info"])})
        duration = clock() - start
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()
    summary = summarize_stream_events(events, batch_size=len(batch),
                                      new_tokens=params["max_new_tokens"],
                                      input_tokens=len(batch[0]), batch_seconds=duration)
    if [o["output_ids"] for o in summary["outputs"]] != [o["output_ids"] for o in expected_outputs]:
        raise ValueError("streamed token IDs disagree with nonstreaming warmup")
    return {"batch_seconds": duration,
            "aggregate_output_tokens_per_second": params["max_new_tokens"] * len(batch) / duration,
            "outputs": summary["outputs"], "stream_requests": summary["requests"],
            "stream_events": events}


def snapshot(path):
    result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=20)
    path.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("npu-smi snapshot failed; device evidence required")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tp", type=int, choices=(1, 2), required=True)
    p.add_argument("--physical-devices", type=parse_devices, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--dtype", choices=("bfloat16", "float16"), required=True,
                   help="explicit weight/activation precision; recurrent state remains FP32")
    p.add_argument("--lm-head-compute-dtype", choices=("float32",),
                   help="optional FP32 final projection; omission preserves native arithmetic")
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--memory-fraction", type=float, required=True)
    p.add_argument("--concurrency", type=int, choices=(1, 2), required=True)
    p.add_argument("--mode", choices=("correctness", "benchmark", "streaming"), required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--repeats", type=int, required=True)
    args = p.parse_args()
    if args.repeats < 1 or args.repeats > 20:
        p.error("repeats must be in 1..20")
    config = build_config(args.model, tp=args.tp, devices=args.physical_devices,
                          memory_fraction=args.memory_fraction,
                          concurrency=args.concurrency, port=args.port)
    config["dtype"] = args.dtype
    if args.mode == "streaming":
        config.update(stream_interval=1, enable_output_logprobs=False)
    if args.lm_head_compute_dtype is not None:
        config["lm_head_compute_dtype"] = args.lm_head_compute_dtype
    if args.dtype == "float16":
        # `auto` is explicitly BF16 upstream; FP16 attention needs matching KV.
        config["kv_cache_dtype"] = "float16"
    manifest_bytes = (args.model / "source-manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    model_config = (args.model / "config.json").read_bytes()
    expected_config = next(e for e in manifest["files"] if e["path"] == "config.json")
    if hashlib.sha256(model_config).hexdigest() != expected_config["sha256"]:
        raise ValueError("model config does not match source manifest")
    cases_bytes = args.cases.read_bytes()
    cases = json.loads(cases_bytes)["cases"]
    vocab = json.loads(model_config)["text_config"]["vocab_size"]
    if not cases or any(not c["input_ids"] or len(c["input_ids"]) + 16 > 512
                        or any(type(i) is not int or not 0 <= i < vocab for i in c["input_ids"])
                        or not c["name"] or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in c["name"])
                        for c in cases):
        raise ValueError("invalid or out-of-capacity cases")
    if len({c["name"] for c in cases}) != len(cases):
        raise ValueError("duplicate case names")
    if args.mode in ("benchmark", "streaming"):
        cases = [c for c in cases if len(c["input_ids"]) in (32, 256)]
        if {len(c["input_ids"]) for c in cases} != {32, 256}:
            raise ValueError("benchmark requires both 32- and 256-token cases")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, args.physical_devices))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    result = {"status": "starting", "tp": args.tp, "physical_devices_requested": args.physical_devices,
              "config": config, "mode": args.mode, "concurrency": args.concurrency,
              "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "config_sha256": expected_config["sha256"],
              "cases_sha256": hashlib.sha256(cases_bytes).hexdigest(), "runs": [],
              "reproducibility": {"environment": selected_environment()},
              "measurement": "batch wall latency/aggregate throughput; not TTFT, TPOT or isolated communication timing"}
    if args.mode == "streaming":
        result["measurement"] = (
            "client-observed indexed token-delta arrivals; TTFT only for a singleton first chunk; "
            "TPOT=(last-first)/(N-1); includes scheduling/IPC/observer overhead, not isolated kernel timing")
        result["stream_schema_version"] = 1
        result["requested_repeats"] = args.repeats
        result["reproducibility"]["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def save():
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    save()
    engine = None
    try:
        result["reproducibility"]["source_trees"] = project_source_fingerprints()
        result["reproducibility"]["runtime_versions"] = runtime_versions()
        save()
        import torch
        import torch_npu  # noqa: F401
        if not torch.npu.is_available() or torch.npu.device_count() != args.tp:
            raise RuntimeError("NPU availability/visible device count mismatch")
        from tokenspeed.runtime.entrypoints.engine import Engine
        snapshot(args.output_dir / "hbm-before.txt")
        engine = Engine(**config)
        result["reproducibility"]["runtime_versions"] = runtime_versions()
        result["process_snapshot_after_load"] = process_snapshot()
        save()
        snapshot(args.output_dir / "hbm-loaded.txt")
        for case in cases:
            batch = [case["input_ids"][:] for _ in range(args.concurrency)]
            params = {"temperature": 0, "max_new_tokens": 16, "ignore_eos": True}
            warmup = engine.generate(input_ids=batch, sampling_params=params)
            validate_outputs(warmup, batch_size=args.concurrency, new_tokens=16,
                             input_tokens=len(case["input_ids"]))
            for repeat in range(args.repeats):
                if args.mode == "streaming":
                    measured = measure_stream_batch(engine, batch=batch, params=params,
                                                    expected_outputs=warmup)
                    result["runs"].append({"case": case["name"], "repeat": repeat,
                                           "input_tokens": len(case["input_ids"]),
                                           "output_tokens_per_request": 16,
                                           "warmup_output_ids": [o["output_ids"] for o in warmup],
                                           **measured})
                    save()
                    continue
                start = time.perf_counter()
                outputs = engine.generate(input_ids=batch, sampling_params=params,
                                          return_logprob=args.mode == "correctness")
                duration = time.perf_counter() - start
                validate_outputs(outputs, batch_size=args.concurrency, new_tokens=16,
                                 input_tokens=len(case["input_ids"]))
                result["runs"].append({"case": case["name"], "repeat": repeat,
                                       "input_tokens": len(case["input_ids"]),
                                       "output_tokens_per_request": 16,
                                       "batch_seconds": duration,
                                       "aggregate_output_tokens_per_second": 16 * args.concurrency / duration,
                                       "outputs": outputs})
                save()
            snapshot(args.output_dir / f"hbm-{case['name']}.txt")
        result["status"] = "generation_completed_not_yet_compared"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if engine is not None:
            engine.shutdown()
        save()


if __name__ == "__main__":
    main()
