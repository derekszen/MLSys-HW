"""Bounded, owned-worker-only msprof collection; not a profiling validation pass."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import time

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
import run_model as harness


def scheduler_name(name):
    return isinstance(name, str) and (name in ("tokenspeed::sch", "tokenspeed::scheduler")
                                     or re.fullmatch(r"tokenspeed::scheduler_tp[0-9]+", name) is not None)


def select_workers(captured):
    """Pure selection from one captured runner/recursive-child snapshot, not a PID scan."""
    root = captured["parent_pid"]
    if type(root) is not int or root <= 0 or captured.get("error"):
        raise ValueError("invalid owned process snapshot")
    records = captured["children"]
    by_pid = {r["pid"]: r for r in records}
    if len(by_pid) != len(records) or root in by_pid:
        raise ValueError("ambiguous/duplicate child PIDs")

    def owned(record):
        seen = {record["pid"]}
        parent = record.get("ppid")
        while parent != root:
            if parent in seen or parent not in by_pid:
                return False
            seen.add(parent)
            parent = by_pid[parent].get("ppid")
        return True

    connected = [r for r in records if owned(r)]
    if any(r.get("error") or not isinstance(r.get("name"), str) or not r["name"]
           for r in connected):
        raise ValueError("ambiguous unreadable owned child name")
    workers = [r for r in connected if scheduler_name(r["name"])]
    if len(workers) != 2:
        raise ValueError(f"expected exactly two owned schedulers, got {len(workers)}")
    for worker in workers:
        created = worker.get("create_time")
        if (worker.get("error") or type(worker["pid"]) is not int or worker["pid"] <= 0
                or type(created) not in (int, float) or not math.isfinite(created) or created <= 0):
            raise ValueError("missing/invalid scheduler identity")
    ranks = [w["name"].removeprefix("tokenspeed::scheduler_tp") for w in workers
             if w["name"].startswith("tokenspeed::scheduler_tp")]
    if any(rank not in ("0", "1") for rank in ranks) or len(set(ranks)) != len(ranks):
        raise ValueError("ambiguous scheduler rank names")
    return sorted((dict(w) for w in workers), key=lambda r: r["pid"])


def capture_children():
    import psutil
    parent = psutil.Process(os.getpid())
    # A disappearing/inaccessible child fails closed; never substitute a global scan.
    return {"parent_pid": parent.pid, "captured_at_unix_ns": time.time_ns(),
            "scope": "runner recursive descendants only; no inferred rank/device mapping",
            "children": [p.as_dict(attrs=["pid", "ppid", "name", "create_time"])
                         for p in parent.children(recursive=True)]}


def collector_command(executable, pid, duration, output):
    """Only dynamic PID attachment. No application launch or global/device selector."""
    if type(pid) is not int or pid <= 0 or type(duration) is not int or not 8 <= duration <= 20:
        raise ValueError("invalid collector PID/duration")
    return [str(executable), "--dynamic=on", f"--pid={pid}", f"--duration={duration}",
            f"--output={output}", "--task-time=on", "--hccl=on", "--runtime-api=off",
            "--ai-core=off", "--storage-limit=200MB", "--export=off"]


@contextmanager
def bounded(seconds, check=lambda: None):
    """Linux main-thread deadline and collector polling during synchronous Engine calls."""
    deadline = time.monotonic() + seconds
    previous = signal.getsignal(signal.SIGALRM)

    def tick(signum, frame):
        check()
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded operation deadline exceeded")

    signal.signal(signal.SIGALRM, tick)
    signal.setitimer(signal.ITIMER_REAL, 0.05, 0.05)
    try:
        tick(None, None)
        yield
        tick(None, None)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@contextmanager
def defer_launch_interrupts():
    # Retain the Popen/identity before handling an interrupt at the spawn boundary.
    # Do not block signals: a child would inherit that blocked mask across exec.
    pending = []
    handlers = {sig: signal.signal(sig, lambda number, frame: pending.append(number))
                for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if pending:
            raise InterruptedError(f"received signal {pending[0]} during collector launch")


class Collector:
    def __init__(self, executable, worker, duration, directory, record):
        raw = directory / "raw"
        raw.mkdir(parents=True, exist_ok=False)
        command = collector_command(executable, worker["pid"], duration, raw)
        record.update(worker=worker, command=command, output_dir=str(raw), returncode=None)
        self.record, self.directory, self.process = record, directory, None

    def start(self):
        import psutil
        self.started = time.monotonic()
        self.record.update(started_unix_ns=time.time_ns(), started_monotonic_ns=time.monotonic_ns())
        # Log files are outside raw/, so collector logs can never satisfy the data check.
        with (self.directory / "collector.log").open("xb") as log:
            self.process = subprocess.Popen(self.record["command"], stdout=log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
        self.record["collector_pid"] = self.process.pid
        try:
            self.record["collector_create_time"] = psutil.Process(self.process.pid).create_time()
        except psutil.NoSuchProcess:
            self.poll()
            if self.process.returncode is None:
                raise

    def poll(self):
        if self.process is None:
            return 0  # Not launched; there is no process group to signal or wait for.
        code = self.process.poll()
        if code is not None:
            self.record["returncode"] = code
            self.record.setdefault("exit_observed_unix_ns", time.time_ns())
            self.record.setdefault("exit_observed_monotonic_ns", time.monotonic_ns())
        return code

    def signal_owned_group(self, sig):
        import psutil
        if self.poll() is not None:
            return
        process = psutil.Process(self.process.pid)
        if (process.create_time() != self.record["collector_create_time"]
                or not process.is_running() or os.getpgid(process.pid) != process.pid
                or os.getsid(process.pid) != process.pid):
            raise RuntimeError("collector identity/process-group verification failed")
        # The unreaped Popen leader pins this PID/group; never signal by name or worker PID.
        if self.poll() is None:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass


def check_collectors(collectors, require_active=False):
    for collector in collectors:
        code = collector.poll()
        if code is not None and (code != 0 or require_active):
            raise RuntimeError(f"collector {collector.process.pid} exited {code}"
                               + (" before workload completed" if require_active else ""))


def stop_collectors(collectors):
    """Try every owned collector even if another fails. Bounded TERM then KILL waits."""
    errors = []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for collector in collectors:
            try:
                collector.signal_owned_group(sig)
            except Exception as exc:
                errors.append(f"{collector.record.get('collector_pid')}: {type(exc).__name__}: {exc}")
        deadline = time.monotonic() + 5
        while any(c.poll() is None for c in collectors) and time.monotonic() < deadline:
            time.sleep(0.05)
    if any(c.poll() is None for c in collectors):
        errors.append("collector still alive; engine shutdown withheld")
    return errors


def artifact_metadata(root):
    """Filesystem metadata only; do not parse traces, follow links, or copy raw data."""
    rows = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(directory) / d).is_symlink())
        for name in sorted(files):
            path = Path(directory) / name
            info = path.lstat()
            if stat.S_ISREG(info.st_mode):
                rows.append({"path": path.relative_to(root).as_posix(), "bytes": info.st_size,
                             "mtime_unix_ns": info.st_mtime_ns})
    return rows


def has_device_data(rows):
    # A necessary filesystem gate, NOT evidence that device events/HCCL were captured.
    for row in rows:
        parts = Path(row["path"]).parts
        if row["bytes"] > 0 and any(re.fullmatch(r"device_[0-9]+", part)
                                      and parts[i + 1] == "data"
                                      for i, part in enumerate(parts[:-2])):
            return True
    return False


def validate_repeat(outputs, warmup, concurrency):
    harness.validate_outputs(outputs, batch_size=concurrency, new_tokens=16, input_tokens=32)
    if [o["output_ids"] for o in outputs] != [o["output_ids"] for o in warmup]:
        raise ValueError("generated IDs differ from the warmed baseline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--case", required=True, help="explicit name of a fixed 32-token case")
    parser.add_argument("--physical-devices", type=harness.parse_devices, required=True)
    parser.add_argument("--memory-fraction", type=float, required=True)
    parser.add_argument("--concurrency", type=int, choices=(1, 2), required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-duration", type=int, choices=range(8, 21), required=True)
    parser.add_argument("--repeats", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--warmup-repeats", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--msprof", default="msprof", help="vendor executable, no wrapper arguments")
    args = parser.parse_args()
    config = harness.build_config(args.model, tp=2, devices=args.physical_devices,
                                  memory_fraction=args.memory_fraction,
                                  concurrency=args.concurrency, port=args.port)
    executable = shutil.which(args.msprof)
    if executable is None:
        parser.error("vendor msprof executable unavailable")
    manifest_bytes = (args.model / "source-manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    config_bytes = (args.model / "config.json").read_bytes()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    expected = [e for e in manifest["files"] if e["path"] == "config.json"]
    if (manifest.get("repo") != "Qwen/Qwen3.5-0.8B" or len(expected) != 1
            or expected[0]["sha256"] != config_hash):
        raise ValueError("requires manifest-linked Qwen3.5-0.8B config")
    cases_bytes = args.cases.read_bytes()
    cases = json.loads(cases_bytes)
    selected = [c for c in cases["cases"] if c["name"] == args.case]
    vocab = json.loads(config_bytes)["text_config"]["vocab_size"]
    if (cases.get("new_tokens") != 16 or len(selected) != 1
            or len(selected[0]["input_ids"]) != 32
            or any(type(i) is not int or not 0 <= i < vocab for i in selected[0]["input_ids"])):
        raise ValueError("requires one explicit valid 32-input/16-output case")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, args.physical_devices))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    result = {"status": "starting", "tp": 2, "config": config,
              "physical_devices_requested": args.physical_devices, "concurrency": args.concurrency,
              "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "config_sha256": config_hash, "cases_sha256": hashlib.sha256(cases_bytes).hexdigest(),
              "case": selected[0], "profile_duration_seconds": args.profile_duration,
              "warmup_repeats": args.warmup_repeats, "repeats": args.repeats,
              "sampling_params": {"temperature": 0, "max_new_tokens": 16, "ignore_eos": True},
              "return_logprob": False, "collectors": [], "warmups": [], "runs": [],
              "measurement": "wall intervals for trace alignment only; no inference performance claim",
              "startup_wait": "approximately 2 seconds; NOT a capture acknowledgement",
              "reproducibility": {"environment": harness.selected_environment()}}

    def save():
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    def interrupted(signum, frame):
        raise InterruptedError(f"received signal {signum}")

    engine, collectors = None, []
    previous_term = signal.signal(signal.SIGTERM, interrupted)
    save()
    try:
        result["reproducibility"].update(source_trees=harness.project_source_fingerprints(),
                                         runtime_versions=harness.runtime_versions())
        import torch
        import torch_npu  # noqa: F401
        if not torch.npu.is_available() or torch.npu.device_count() != 2:
            raise RuntimeError("NPU availability/visible device count mismatch")
        from tokenspeed.runtime.entrypoints.engine import Engine
        harness.snapshot(args.output_dir / "hbm-before.txt")
        with bounded(300):
            engine = Engine(**config)
        result["reproducibility"]["runtime_versions"] = harness.runtime_versions()
        result["process_snapshot_after_load"] = harness.process_snapshot()
        harness.snapshot(args.output_dir / "hbm-loaded.txt")
        batch = [selected[0]["input_ids"][:] for _ in range(args.concurrency)]
        baseline = None

        def generate(destination, repeat, check=lambda: None, seconds=120):
            row = {"repeat": repeat, "start_unix_ns": time.time_ns(),
                   "start_monotonic_ns": time.monotonic_ns()}
            destination.append(row)
            try:
                with bounded(seconds, check):
                    row["outputs"] = engine.generate(input_ids=batch,
                        sampling_params=result["sampling_params"], return_logprob=False)
            finally:
                row.update(end_monotonic_ns=time.monotonic_ns(), end_unix_ns=time.time_ns())
                row["wall_monotonic_ns"] = row["end_monotonic_ns"] - row["start_monotonic_ns"]
                save()
            harness.validate_outputs(row["outputs"], batch_size=args.concurrency,
                                     new_tokens=16, input_tokens=32)
            if baseline is not None:
                validate_repeat(row["outputs"], baseline, args.concurrency)
            return row["outputs"]

        for repeat in range(args.warmup_repeats):
            baseline = generate(result["warmups"], repeat)
        harness.snapshot(args.output_dir / "hbm-warmed.txt")
        result["process_snapshot_after_warmup"] = capture_children()
        workers = select_workers(result["process_snapshot_after_warmup"])
        result["workers"] = workers
        save()
        for worker in workers:
            check_collectors(collectors, require_active=True)
            if select_workers(capture_children()) != workers:
                raise RuntimeError("owned scheduler identity changed before attachment")
            record = {}
            result["collectors"].append(record)
            collector = Collector(executable, worker, args.profile_duration,
                                  args.output_dir / f"worker-{worker['pid']}", record)
            collectors.append(collector)
            with defer_launch_interrupts():
                collector.start()
            save()
        active = lambda: check_collectors(collectors, require_active=True)
        with bounded(3, active):
            time.sleep(2)  # Approximate startup delay, not capture acknowledgement.
        workload_end = min(c.started + args.profile_duration for c in collectors)
        for repeat in range(args.repeats):
            generate(result["runs"], repeat, active, max(0, workload_end - time.monotonic()))
        # No export here. Even collector finalization is bounded well below 120 s each.
        while any(c.poll() is None for c in collectors):
            check_collectors(collectors)
            if any(c.poll() is None and time.monotonic() > c.started + args.profile_duration + 30
                   for c in collectors):
                raise TimeoutError("collector finalization exceeded duration + 30 seconds")
            time.sleep(0.05)
        check_collectors(collectors)
        for collector in collectors:
            rows = artifact_metadata(Path(collector.record["output_dir"]))
            collector.record["artifacts"] = rows
        save()
        if not all(has_device_data(c.record["artifacts"]) for c in collectors):
            raise RuntimeError("missing nonempty device data for one or both workers; no fallback")
        harness.snapshot(args.output_dir / "hbm-collected.txt")
        result["status"] = "collecting_completed_not_yet_validated"
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        # Finish cleanup despite repeated terminal signals; restore handlers afterward.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        previous_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
        # Stop/wait BOTH collectors before even attempting Engine.shutdown().
        errors = stop_collectors(collectors)
        try:
            if engine is not None and all(c.poll() is not None for c in collectors):
                with bounded(30):
                    engine.shutdown()
        except BaseException as exc:
            errors.append(f"engine shutdown: {type(exc).__name__}: {exc}")
        finally:
            for collector in collectors:
                try:
                    collector.record["artifacts"] = artifact_metadata(Path(collector.record["output_dir"]))
                except Exception as exc:
                    errors.append(f"artifact inventory: {type(exc).__name__}: {exc}")
            if errors:
                result.update(status="failed", cleanup_errors=errors)
            try:
                save()
            finally:
                signal.signal(signal.SIGTERM, previous_term)
                signal.signal(signal.SIGINT, previous_int)
        if errors:
            raise RuntimeError("; ".join(errors))


if __name__ == "__main__":
    main()
