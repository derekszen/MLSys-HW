#!/usr/bin/env python3
"""Bounded two-rank HCCL correctness check, not a TP model or benchmark.

Module import and configuration/result helpers require only the standard library.
"""

import argparse
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import traceback
import uuid
from datetime import timedelta


WORLD_SIZE = 2
DTYPES = ("float32", "bfloat16")
COLLECTIVES = ("all_reduce", "all_gather", "reduce_scatter")
MEMORY_KEYS = (
    "allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes"
)


def parse_physical_devices(value):
    parts = value.split(",")
    if len(parts) != WORLD_SIZE or any(
        not part.isascii() or not part.isdecimal() for part in parts
    ):
        raise argparse.ArgumentTypeError("expected exactly two physical IDs, e.g. 0,1")
    devices = tuple(str(int(part)) for part in parts)
    if len(set(devices)) != WORLD_SIZE:
        raise argparse.ArgumentTypeError("physical IDs must be distinct")
    return devices


def parse_port(value):
    if value == "auto":
        return value
    if not value.isascii() or not value.isdecimal() or not 1 <= int(value) <= 65535:
        raise argparse.ArgumentTypeError("port must be 'auto' or an integer in 1..65535")
    return int(value)


def timeout_seconds(value):
    if not value.isascii() or not value.isdecimal() or not 1 <= int(value) <= 1800:
        raise argparse.ArgumentTypeError("timeout must be an integer in 1..1800 seconds")
    return int(value)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-devices", required=True, type=parse_physical_devices)
    parser.add_argument("--port", required=True, type=parse_port,
                        help="required localhost TCP port, or 'auto' to select a free port")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="new directory; existing directories are refused")
    parser.add_argument("--dist-timeout", type=timeout_seconds, default=30)
    parser.add_argument("--run-timeout", type=timeout_seconds, default=120,
                        help="spawned-worker deadline; cleanup adds at most 10 seconds")
    args = parser.parse_args(argv)
    if args.run_timeout < args.dist_timeout + 10:
        parser.error("--run-timeout must be at least --dist-timeout + 10 seconds")
    return args


def choose_port(requested):
    # Probe only loopback. Closing this socket before TCPStore binds has a race;
    # a competing binder causes a failed run, never a silent retry on another port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0 if requested == "auto" else requested))
        return sock.getsockname()[1]


def expected_output(rank, collective):
    if type(rank) is not int or rank not in range(WORLD_SIZE):
        raise ValueError("rank must be 0 or 1")
    if collective == "all_reduce":
        return [4 + 2 * i for i in range(4)]
    if collective == "all_gather":
        return [[4 * source + i for i in range(4)] for source in range(WORLD_SIZE)]
    if collective == "reduce_scatter":
        return [8 + 8 * rank + 2 * i for i in range(4)]
    raise ValueError(f"unknown collective: {collective}")


def exact_numbers(actual, expected):
    """JSON tensor comparison: reject booleans, NaNs, shape errors and tolerances."""
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(exact_numbers(a, e) for a, e in zip(actual, expected)))
    return type(actual) in (int, float) and math.isfinite(actual) and actual == expected


def validate_rank_result(record, rank, devices, run_id):
    """Do not trust a worker's 'passed' flag without validating all six outputs."""
    if not isinstance(record, dict):
        raise ValueError(f"rank {rank}: result must be an object")
    required = {
        "run_id": run_id, "rank": rank, "world_size": WORLD_SIZE,
        "logical_device": rank, "physical_device_requested": devices[rank],
        "visible_devices": ",".join(devices), "backend": "hccl", "passed": True,
    }
    for key, expected in required.items():
        if type(record.get(key)) is not type(expected) or record[key] != expected:
            raise ValueError(f"rank {rank}: invalid {key}")
    if type(record.get("pid")) is not int or record["pid"] <= 0:
        raise ValueError(f"rank {rank}: invalid PID")
    memory = record.get("memory", {})
    if not isinstance(memory, dict) or any(
        type(memory.get(key)) is not int or memory[key] < 0 for key in MEMORY_KEYS
    ):
        raise ValueError(f"rank {rank}: invalid memory metrics")
    if (memory["reserved_bytes"] < memory["allocated_bytes"]
            or memory["peak_allocated_bytes"] < memory["allocated_bytes"]
            or memory["peak_reserved_bytes"] < memory["reserved_bytes"]):
        raise ValueError(f"rank {rank}: inconsistent memory metrics")
    checks = record.get("checks")
    if not isinstance(checks, list) or len(checks) != len(DTYPES) * len(COLLECTIVES):
        raise ValueError(f"rank {rank}: missing collective checks")
    seen = set()
    for check in checks:
        if not isinstance(check, dict):
            raise ValueError(f"rank {rank}: invalid check")
        dtype, collective = check.get("dtype"), check.get("collective")
        if dtype not in DTYPES or collective not in COLLECTIVES:
            raise ValueError(f"rank {rank}: unexpected dtype/collective")
        key = (dtype, collective)
        if key in seen or check.get("verified") is not True:
            raise ValueError(f"rank {rank}: duplicate or unverified check")
        seen.add(key)
        expected = expected_output(rank, collective)
        if not exact_numbers(check.get("expected"), expected) or not exact_numbers(
            check.get("actual"), expected
        ):
            raise ValueError(f"rank {rank}: incorrect {dtype} {collective} output")


def aggregate_results(output_dir, devices, run_id):
    records = []
    for rank in range(WORLD_SIZE):
        path = Path(output_dir) / f"rank-{rank}.json"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"missing or unreadable rank result: {path}") from exc
        validate_rank_result(record, rank, devices, run_id)
        records.append(record)
    if len({record["pid"] for record in records}) != WORLD_SIZE:
        raise ValueError("rank results must come from two distinct child PIDs")
    return records


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def run_collectives(torch, dist, rank):
    checks = []
    device = f"npu:{rank}"
    for dtype_name in DTYPES:
        dtype = getattr(torch, dtype_name)

        def tensor(values):
            return torch.tensor(values, dtype=dtype, device=device)

        def check(collective, actual):
            torch.npu.synchronize(rank)
            expected = expected_output(rank, collective)
            cpu = actual.cpu()
            verified = bool(torch.equal(cpu, torch.tensor(expected, dtype=dtype)))
            checks.append({"dtype": dtype_name, "collective": collective,
                           "verified": verified, "actual": cpu.tolist(), "expected": expected})
            if not verified:
                raise RuntimeError(f"rank {rank}: exact {dtype_name} {collective} check failed")

        # All values and sums are small integers, exactly representable in BF16.
        value = tensor([4 * rank + i for i in range(4)])
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        check("all_reduce", value)

        value = tensor([4 * rank + i for i in range(4)])
        gathered = [torch.empty_like(value) for _ in range(WORLD_SIZE)]
        dist.all_gather(gathered, value)
        check("all_gather", torch.stack(gathered))

        chunks = [tensor([8 * rank + 4 * dest + i for i in range(4)])
                  for dest in range(WORLD_SIZE)]
        reduced = torch.empty_like(value)
        dist.reduce_scatter(reduced, chunks, op=dist.ReduceOp.SUM)
        check("reduce_scatter", reduced)
    return checks


def worker(rank, config):
    # spawn reimports this module, which has no top-level Torch imports.
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(config["devices"])
    record = {
        "run_id": config["run_id"], "rank": rank, "pid": os.getpid(),
        "world_size": WORLD_SIZE, "logical_device": rank,
        "physical_device_requested": config["devices"][rank],
        "visible_devices": os.environ["ASCEND_RT_VISIBLE_DEVICES"], "passed": False,
    }
    dist = None
    owns_group = False
    try:
        import torch
        import torch_npu
        import torch.distributed as dist

        if not torch.npu.is_available() or torch.npu.device_count() != WORLD_SIZE:
            raise RuntimeError("exactly two visible Ascend NPUs are required")
        torch.npu.set_device(rank)
        record["logical_device"] = int(torch.npu.current_device())
        if record["logical_device"] != rank:
            raise RuntimeError("logical device does not match rank")
        torch.npu.synchronize(rank)
        torch.npu.reset_peak_memory_stats(rank)
        dist.init_process_group(
            backend="hccl", init_method=f"tcp://127.0.0.1:{config['port']}",
            rank=rank, world_size=WORLD_SIZE,
            timeout=timedelta(seconds=config["dist_timeout"]),
        )
        owns_group = True
        record.update(backend=str(dist.get_backend()), rank=dist.get_rank(),
                      world_size=dist.get_world_size(),
                      device_name=torch.npu.get_device_name(rank),
                      torch_version=str(torch.__version__),
                      torch_npu_version=str(torch_npu.__version__))
        record["checks"] = run_collectives(torch, dist, rank)
        torch.npu.synchronize(rank)
        record["memory"] = {
            "allocated_bytes": int(torch.npu.memory_allocated(rank)),
            "reserved_bytes": int(torch.npu.memory_reserved(rank)),
            "peak_allocated_bytes": int(torch.npu.max_memory_allocated(rank)),
            "peak_reserved_bytes": int(torch.npu.max_memory_reserved(rank)),
        }
        record["passed"] = True
    except Exception:
        record["error"] = traceback.format_exc()
    finally:
        if owns_group:
            try:
                dist.destroy_process_group()
            except Exception:
                record["passed"] = False
                record["cleanup_error"] = traceback.format_exc()
        write_json(Path(config["output_dir"]) / f"rank-{rank}.json", record)
    if not record["passed"]:
        raise RuntimeError(f"rank {rank} failed; see rank-{rank}.json")


def stop_owned_children(context):
    """Only touch Process objects returned by our own spawn; never process groups."""
    for process in context.processes:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + 5
    for process in context.processes:
        process.join(timeout=max(0, deadline - time.monotonic()))
    for process in context.processes:
        if process.is_alive():
            process.kill()
    deadline = time.monotonic() + 5
    for process in context.processes:
        process.join(timeout=max(0, deadline - time.monotonic()))
    if any(process.is_alive() for process in context.processes):
        raise RuntimeError("owned worker did not exit after terminate/kill")


def wait_for_children(context, deadline):
    # Avoid ProcessContext.join's extra peer-shutdown waits on error: we own the
    # deadline and cleanup. Call it only once every child has already exited.
    while True:
        if any(p.exitcode not in (None, 0) for p in context.processes):
            raise RuntimeError("a spawned rank exited unsuccessfully; inspect rank files/stderr")
        if time.monotonic() >= deadline:
            raise TimeoutError("spawned-worker deadline exceeded")
        if all(p.exitcode == 0 for p in context.processes):
            context.join(timeout=0)
            return
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))


def main(argv=None):
    args = parse_args(argv)
    # Refuse reuse: stale results must never turn a failed run into a pass.
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        print(f"cannot create new output directory: {exc}", file=sys.stderr)
        return 1
    config = {"devices": args.physical_devices, "port": None,
              "dist_timeout": args.dist_timeout, "run_timeout": args.run_timeout,
              "output_dir": str(args.output_dir.resolve()), "run_id": uuid.uuid4().hex}
    summary = {"passed": False, "world_size": WORLD_SIZE, "config": config,
               "same_card_verified": False,
               "same_card_check": "requires external npu-smi topology verification; adjacency is insufficient"}
    context = None
    try:
        config["port"] = choose_port(args.port)
        # Must precede BOTH Torch imports, including torch.multiprocessing.
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(args.physical_devices)
        import torch.multiprocessing as mp

        deadline = time.monotonic() + args.run_timeout
        context = mp.spawn(worker, args=(config,), nprocs=WORLD_SIZE, join=False)
        wait_for_children(context, deadline)
        summary["ranks"] = aggregate_results(args.output_dir, args.physical_devices, config["run_id"])
        summary["passed"] = True
    except (Exception, KeyboardInterrupt):
        summary["error"] = traceback.format_exc()
    finally:
        if context is not None:
            try:
                stop_owned_children(context)
            except Exception:
                summary["passed"] = False
                summary["cleanup_error"] = traceback.format_exc()
        write_json(args.output_dir / "result.json", summary)
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
