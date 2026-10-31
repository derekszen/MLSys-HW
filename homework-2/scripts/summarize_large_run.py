"""Validate the retained 27B retry and extract child-PID-owned HBM snapshots.

Offline, standard library only. These descriptive timings are NOT accuracy-
certified performance: the independent strict length_32 reference check failed.
"""
import argparse
import json
import math
from pathlib import Path
import re
import statistics

from summarize_benchmarks import (
    COMPLETED, TP_FIELDS, output_ids, positive, require, sha256, source_identity,
)

# Keep comparable c1 runs and the later 0.8B c2 correctness run as well.
HBM_RUNS = [f"{model}-correctness-tp2-c1" for model in ("0.8B", "4B", "9B", "27B")] + [
    "0.8B-correctness-tp2-c2",
] + [
    f"0.8B-benchmark-tp{tp}-c{c}" for tp in (1, 2) for c in (1, 2)
]


def summarize(benchmark_bytes, correctness_bytes, *, cases_bytes, manifest_bytes,
              config_bytes, comparison_bytes):
    """Reject incomplete/mismatched evidence, then summarize all six requests."""
    try:
        return _summarize(benchmark_bytes, correctness_bytes, cases_bytes=cases_bytes,
                          manifest_bytes=manifest_bytes, config_bytes=config_bytes,
                          comparison_bytes=comparison_bytes)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as exc:
        raise ValueError(f"missing or malformed large-run evidence: {exc}") from exc


def _summarize(benchmark_bytes, correctness_bytes, *, cases_bytes, manifest_bytes,
               config_bytes, comparison_bytes):
    benchmark, correctness = map(json.loads, (benchmark_bytes, correctness_bytes))
    comparison = json.loads(comparison_bytes)
    hashes = {"manifest_sha256": sha256(manifest_bytes), "config_sha256": sha256(config_bytes),
              "cases_sha256": sha256(cases_bytes)}
    manifest, model_config, case_doc = map(json.loads, (manifest_bytes, config_bytes, cases_bytes))
    entries = [f for f in manifest["files"] if f["path"] == "config.json"]
    require(len(entries) == 1 and entries[0]["sha256"] == hashes["config_sha256"],
            "model config differs from manifest")
    vocab = model_config["text_config"]["vocab_size"]
    require(type(vocab) is int and vocab > 0, "invalid vocabulary")
    require(type(case_doc["new_tokens"]) is int and case_doc["new_tokens"] == 16,
            "expected 16 generated tokens")
    cases = {}
    for case in case_doc["cases"]:
        name, tokens = case["name"], case["input_ids"]
        require(isinstance(name, str) and bool(name) and name not in cases, "invalid/duplicate case")
        require(isinstance(tokens, list) and bool(tokens)
                and all(type(t) is int and 0 <= t < vocab for t in tokens), "invalid input IDs")
        cases[name] = len(tokens)
    require(sorted(cases.values()) == [5, 32, 256], "expected exactly 5/32/256-token cases")
    measured = {n: length for n, length in cases.items() if length in (32, 256)}
    sources = source_identity(correctness)
    config = {k: v for k, v in correctness["config"].items() if k != "port"}
    versions = correctness["reproducibility"]["runtime_versions"]
    require(all(isinstance(versions[k], str) and bool(versions[k])
                and versions[k] != "unavailable" for k in ("torch", "torch_npu", "triton")),
            "missing runtime versions")
    environment = correctness["reproducibility"]["environment"]
    devices = correctness["physical_devices_requested"]
    require(isinstance(devices, list) and len(devices) == 2 and len(set(devices)) == 2
            and all(type(d) is int and d >= 0 for d in devices), "invalid TP2 devices")
    require(environment["ASCEND_RT_VISIBLE_DEVICES"] == ",".join(map(str, devices)),
            "visibility differs from devices")
    indexed = {}
    for mode, result in (("correctness", correctness), ("benchmark", benchmark)):
        require(type(result["tp"]) is int and result["tp"] == 2
                and type(result["concurrency"]) is int and result["concurrency"] == 1,
                "expected TP2 concurrency 1")
        require(result["mode"] == mode and result["status"] == COMPLETED and "error" not in result,
                "run did not complete expected mode")
        require(all(type(result["config"][key]) is int and result["config"][key] == 2
                    for key in TP_FIELDS), "sharding differs from TP2")
        require({k: v for k, v in result["config"].items() if k != "port"} == config,
                "benchmark/correctness configuration differs")
        require(all(result[k] == v for k, v in hashes.items()), "model/cases fingerprint differs")
        require(source_identity(result) == sources, "source fingerprints differ")
        require(result["reproducibility"]["runtime_versions"] == versions, "runtime versions differ")
        require(result["reproducibility"]["environment"] == environment, "environment differs")
        require(result["physical_devices_requested"] == devices, "devices differ")
        selected = cases if mode == "correctness" else measured
        expected = {(name, repeat) for name in selected
                    for repeat in (range(1) if mode == "correctness" else range(3))}
        runs = {}
        for run in result["runs"]:
            key = (run["case"], run["repeat"])
            require(type(run["repeat"]) is int and key in expected and key not in runs,
                    "extra/duplicate case or repeat")
            length = selected[key[0]]
            require(type(run["input_tokens"]) is int and run["input_tokens"] == length
                    and type(run["output_tokens_per_request"]) is int
                    and run["output_tokens_per_request"] == 16, "case shape differs")
            ids = output_ids(run, concurrency=1, length=length, new_tokens=16, vocab=vocab,
                             correctness=mode == "correctness")
            if mode == "benchmark":
                seconds, throughput = run["batch_seconds"], run["aggregate_output_tokens_per_second"]
                require(positive(seconds) and positive(throughput), "invalid timing/throughput")
                require(math.isclose(throughput, 16 / seconds, rel_tol=1e-12, abs_tol=0),
                        "throughput disagrees with duration")
            runs[key] = (run, ids)
        require(set(runs) == expected, "incomplete case/repeat grid")
        indexed[mode] = runs
    # Preserve the independent failed verdict; do not equate repeatability with accuracy.
    require(comparison["inputs"]["run"]["sha256"] == sha256(correctness_bytes)
            and comparison["inputs"]["cases"]["sha256"] == hashes["cases_sha256"],
            "reference comparison is not bound to this correctness run/cases")
    require(comparison["passed"] is False, "expected retained failed strict reference verdict")
    verdicts = comparison["cases"]
    require(len(verdicts) == 3 and {v["case"] for v in verdicts} == set(cases),
            "incomplete reference verdicts")
    require(all(type(v["passed"]) is bool and v["passed"] == (cases[v["case"]] != 32)
                and v["repeat"] == 0 and v["request"] == 0 for v in verdicts),
            "unexpected reference verdicts")
    checks, rows = [], []
    for name, length in sorted(measured.items(), key=lambda item: item[1]):
        samples = []
        for repeat in range(3):
            run, ids = indexed["benchmark"][(name, repeat)]
            require(ids == indexed["correctness"][(name, 0)][1],
                    f"benchmark output IDs differ: {name}, repeat={repeat}, request=0")
            checks.append({"case": name, "repeat": repeat, "request": 0, "exact_output_ids": True})
            samples.append(run)
        rows.append({"tp": 2, "concurrency": 1, "case": name, "input_tokens": length,
                     "output_tokens_per_request": 16, "repeats": 3,
                     "batch_seconds": [s["batch_seconds"] for s in samples],
                     "aggregate_output_tokens_per_second": [s["aggregate_output_tokens_per_second"] for s in samples],
                     "median_batch_seconds": statistics.median(s["batch_seconds"] for s in samples),
                     "median_aggregate_output_tokens_per_second": statistics.median(
                         s["aggregate_output_tokens_per_second"] for s in samples)})
    return {"model": "Qwen3.5-27B", "repeatability_validation_passed": True,
            "strict_reference_passed": False, "accuracy_certified_performance": False,
            "strict_reference_failed_cases": [v["case"] for v in verdicts if not v["passed"]],
            "comparison_scope": "Each benchmark output ID equals the same TP2 correctness run; NOT independent accuracy certification.",
            "measurement": "Descriptive batch wall latency and aggregate output-token throughput; not TTFT/TPOT or communication timing.",
            "limitations": ["Strict independent Transformers length_32 reference comparison failed; two other cases passed.",
                            "Shared server; uncontrolled contention. Three repeats, no confidence interval.",
                            "No TP1 speedup or physical non-fit claim; TP1 failed its configured 0.8 memory budget."],
            "fingerprints": hashes, "config_without_port": config, "runtime_versions": versions,
            "environment": environment, "physical_devices_requested": devices,
            "source_trees": {n: {"sha256": t["sha256"], "file_count": len(t["files"])}
                             for n, t in sorted(sources.items())},
            "verified_benchmark_requests": len(checks), "output_id_checks": checks, "rows": rows}


def owned_hbm(result, text):
    """Use only process-table rows whose PID is in this run's captured children.

    Keep cross-chip entries for the same child PID; never infer rank assignment.
    Parent/runner and unrelated processes are deliberately excluded.
    """
    try:
        require(result["status"] == COMPLETED and "error" not in result, "HBM run incomplete")
        snapshot = result["process_snapshot_after_load"]
        require(snapshot["stage"] == "after_engine_startup", "unexpected process snapshot stage")
        parent = snapshot["parent_pid"]
        require(type(parent) is int and parent > 0 and snapshot["parent"]["pid"] == parent,
                "invalid parent PID")
        children = snapshot["children"]
        require(isinstance(children, list) and bool(children), "missing child PIDs")
        pids = {child["pid"] for child in children}
        require(len(pids) == len(children) and parent not in pids
                and all(type(pid) is int and pid > 0 for pid in pids), "invalid/duplicate child PIDs")
        require(all(type(c["ppid"]) is int and c["ppid"] in pids | {parent} for c in children),
                "child outside captured process tree")
        for pid in pids:
            seen = set()
            current = pid
            parents = {c["pid"]: c["ppid"] for c in children}
            while current != parent:
                require(current not in seen, "cycle in captured process tree")
                seen.add(current)
                current = parents[current]
        schedulers = {c["pid"] for c in children if c["name"].startswith("tokenspeed::scheduler_tp")}
        require(type(result["tp"]) is int and result["tp"] in (1, 2)
                and len(schedulers) == result["tp"], "missing scheduler child PIDs")
        require("Process memory(MB)" in text, "missing npu-smi process table")
        table = text.split("Process memory(MB)", 1)[1]
        rows, keys = [], set()
        for line in table.splitlines():
            if not re.match(r"^\|\s*\d", line):
                continue
            match = re.fullmatch(r"\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|\s*", line)
            require(match is not None, "malformed npu-smi process row")
            card, chip, pid, name, mb = match.groups()
            card, chip, pid, mb = map(int, (card, chip, pid, mb))
            key = (card, chip, pid)
            require(key not in keys, "duplicate npu-smi process row")
            keys.add(key)
            if pid in pids:
                rows.append({"card": card, "chip": chip, "pid": pid,
                             "process_name": name.strip(), "process_memory_mb": mb})
        missing_schedulers = schedulers - {r["pid"] for r in rows}
        complete = not missing_schedulers
        devices = sorted({(r["card"], r["chip"]) for r in rows})
        require(len(devices) <= result["tp"], "owned HBM chip count exceeds TP")
        if complete:
            require(len(devices) == result["tp"], "owned HBM chip count differs from TP")
        return {"child_pids": sorted(pids), "matched_child_pids": sorted({r["pid"] for r in rows}),
                "unmatched_child_pids": sorted(pids - {r["pid"] for r in rows}),
                "all_scheduler_pids_matched": complete,
                "missing_scheduler_child_pids": sorted(missing_schedulers),
                "coverage": "all scheduler child PIDs matched" if complete else "missing/partial: do not interpret matched sums as complete owned memory",
                "process_snapshot_captured_at_unix_seconds": snapshot["captured_at_unix_seconds"],
                "rows": sorted(rows, key=lambda r: (r["card"], r["chip"], r["pid"])),
                "per_chip": [{"card": card, "chip": chip, "matched_child_process_memory_mb": sum(
                    r["process_memory_mb"] for r in rows if (r["card"], r["chip"]) == (card, chip))}
                             for card, chip in devices],
                "total_owned_child_process_memory_mb": sum(r["process_memory_mb"] for r in rows) if complete else None}
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError(f"missing or malformed HBM evidence: {exc}") from exc


def collect_hbm(results):
    rows, inputs = [], []
    for run_name in HBM_RUNS:
        data = (results / run_name / "result.json").read_bytes()
        result = json.loads(data)
        inputs.append({"path": f"{run_name}/result.json", "sha256": sha256(data)})
        for filename in ("hbm-loaded.txt", "hbm-length_256.txt"):
            raw = (results / run_name / filename).read_bytes()
            rows.append({"run": run_name, "snapshot": filename, **owned_hbm(result, raw.decode())})
            inputs.append({"path": f"{run_name}/{filename}", "sha256": sha256(raw)})
    return {"scope": "Child PIDs from each run's process_snapshot_after_load matched exactly to npu-smi process rows. No parent or unrelated process memory included.",
            "units": "MB as labeled by npu-smi; no implicit GiB conversion",
            "limitations": ["Loaded and post-length_256 snapshots, NOT peaks or continuous memory tracking.",
                            "Same child PID can have allocations on multiple chips; all matching rows retained.",
                            "No rank mapping inferred from process order; no whole-device HBM totals used.",
                            "Absent scheduler PIDs yield explicit missing/partial coverage and a null complete total, never a zero-memory claim.",
                            "Named retained model correctness c1 runs, later 0.8B TP2 c2 correctness, and 0.8B benchmark grid; not a glob over older attempts."],
            "inputs": inputs, "rows": rows}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=root / "results")
    parser.add_argument("--cases", type=Path, default=root / "cases.json")
    parser.add_argument("--metadata", type=Path, default=root / "metadata/Qwen3.5-27B")
    parser.add_argument("--output", type=Path, default=root / "results/27B-benchmark-summary.json")
    parser.add_argument("--hbm-output", type=Path, default=root / "results/owned-hbm-summary.json")
    args = parser.parse_args()
    paths = ["27B-benchmark-tp2-c1-retry/result.json", "27B-correctness-tp2-c1/result.json",
             "27B-reference-comparison.json"]
    raw = [(args.results / path).read_bytes() for path in paths]
    reference = (args.results / "27B-transformers-reference.json").read_bytes()
    require(json.loads(raw[2])["inputs"]["reference"]["sha256"] == sha256(reference),
            "reference bytes differ from retained comparison")
    result = summarize(raw[0], raw[1], cases_bytes=args.cases.read_bytes(),
                       manifest_bytes=(args.metadata / "source-manifest.json").read_bytes(),
                       config_bytes=(args.metadata / "config.json").read_bytes(), comparison_bytes=raw[2])
    result["inputs"] = [{"path": path, "sha256": sha256(data)} for path, data in zip(paths, raw)] + [
        {"path": "27B-transformers-reference.json", "sha256": sha256(reference)}]
    hbm = collect_hbm(args.results)  # Validate everything before writing either output.
    for path, document in ((args.output, result), (args.hbm_output, hbm)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"Verified {result['verified_benchmark_requests']} repeatability checks (strict reference FAILED); {len(hbm['rows'])} owned HBM snapshots")


if __name__ == "__main__":
    main()
