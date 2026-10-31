#!/usr/bin/env python3
"""Offline, stdlib-only analysis of the four bounded 0.8B CANN captures.

No accelerator imports, remote access, raw-file changes, or benchmark inference.
Decimal microseconds stay exact; JSON durations/timestamps are decimal strings.
Fail closed on provenance, ownership, grid, or device-classification ambiguity.
"""
import argparse
from bisect import bisect_right
from collections import Counter
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import zlib

D = Decimal
MAX_FILE_BYTES = 80_000_000
COMPUTE = {"AI_CORE", "AI_VECTOR_CORE", "MIX_AIC", "MIX_AIV", "AI_CPU"}
CONTROL = {"EVENT_RECORD", "EVENT_WAIT", "NOTIFY_RECORD", "NOTIFY_WAIT",
           "NOTIFY_RECORD_SQE", "NOTIFY_WAIT_SQE", "WRITE_VALUE_SQE"}
OTHER = {"SDMA_SQE", "PCIE_DMA_SQE", "PROFILING_ENABLE", "PROFILING_DISABLE"}
KERNELS = {"allreduceAicpuKernel": "AllReduce", "allgatherAicpuKernel": "AllGather",
           "broadcastAicpuKernel": "Broadcast"}
WIDTH = {"BFP16": 2, "INT32": 4}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(isinstance(value, (str, int, D)) and not isinstance(value, bool),
            "numeric trace fields must not pass through binary float")
    try:
        result = D(value)
    except InvalidOperation as exc:
        raise ValueError("invalid decimal") from exc
    require(result.is_finite(), "nonfinite trace field")
    return result


def interval(event):
    start, duration = number(event["ts"]), number(event["dur"])
    require(duration >= 0, "negative duration")
    return start, start + duration


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        require(start <= end, "reversed interval")
        if start == end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def intersection(left, right):
    left, right = union(left), union(right)
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        a, b = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if a < b:
            result.append((a, b))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def clip(intervals, window):
    require(window[0] < window[1], "empty/reversed window")
    return intersection(intervals, [window])


def duration(intervals):
    return sum((b - a for a, b in union(intervals)), D(0))


def bounded_bytes(path):
    # Bound stored bytes AND actual inflation, never trust gzip's size trailer.
    require(path.stat().st_size <= MAX_FILE_BYTES, f"file exceeds analysis bound: {path}")
    with path.open("rb") as stream:
        stored = stream.read(MAX_FILE_BYTES + 1)
    require(len(stored) <= MAX_FILE_BYTES, f"file exceeds analysis bound: {path}")
    content = stored
    if path.name.endswith(".gz"):
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(stored), mode="rb") as stream:
                content = stream.read(MAX_FILE_BYTES + 1)
        except (OSError, EOFError, zlib.error) as exc:
            raise ValueError(f"invalid/truncated gzip: {path}") from exc
        require(len(content) <= MAX_FILE_BYTES, f"decompressed file exceeds analysis bound: {path}")
    return content, stored


def load(path):
    # One bounded file at a time, never stringify the full event array.
    content, _ = bounded_bytes(path)
    return json.loads(content, parse_float=D)


def fingerprint(path, root):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def event_array(document):
    events = document.get("traceEvents") if isinstance(document, dict) else document
    require(isinstance(events, list) and 0 < len(events) <= 400_000, "missing/bounded event array")
    require(all(isinstance(e, dict) and isinstance(e.get("args", {}), dict) for e in events),
            "malformed event/args")
    return events


def metadata(events):
    meta = {}
    for e in events:
        if e.get("ph") == "M":
            key = (e["name"], e["pid"], e.get("tid", 0))
            require(key not in meta or meta[key] == e["args"], "conflicting metadata")
            meta[key] = e["args"]
    return meta


def process(meta, name):
    found = [pid for (kind, pid, _), args in meta.items()
             if kind == "process_name" and args.get("name") == name]
    require(len(found) == 1, f"ambiguous/missing {name} device evidence")
    return found[0]


def collective_kind(e):
    match = re.fullmatch(r"hcom_(allReduce|allGather|broadcast)__\d+_\d+_\d+", e["name"])
    require(match is not None, "unknown communication parent")
    return {"allReduce": "AllReduce", "allGather": "AllGather", "broadcast": "Broadcast"}[match[1]]


def validate_parents(parents):
    require(parents, "no communication envelopes")
    ordered = sorted(parents, key=interval)
    for previous, current in zip(ordered, ordered[1:]):
        require(interval(previous)[1] <= interval(current)[0],
                "overlapping/nested communication parents: refusing ambiguous double count")
    for e in ordered:
        collective_kind(e)
        require(interval(e)[0] < interval(e)[1], "empty communication envelope")
        require(type(e["args"].get("count")) is int and e["args"]["count"] > 0
                and e["args"].get("data_type") in WIDTH
                and "connection_id" in e["args"], "missing/unknown HCCL payload evidence")
    return ordered


def ownership(events, result, rank, filename):
    require(filename.removesuffix(".gz").endswith(f"-TP{rank}.trace.json"), "filename TP mismatch")
    meta = metadata(events)
    hardware, communication, host = (process(meta, name) for name in
                                     ("Ascend Hardware", "Communication", "CANN"))
    physical = result["physical_devices_requested"][rank]
    for pid in (hardware, communication):
        require(meta.get(("process_labels", pid, 0), {}).get("labels") == f"NPU {physical}",
                "trace physical-device label mismatch")
    snapshot = result["process_snapshot_after_load"]
    children = snapshot["children"]
    schedulers = [c for c in children if c["name"].startswith("tokenspeed::scheduler")]
    require(len(schedulers) == 2 and len({c["pid"] for c in schedulers}) == 2,
            "ambiguous owned scheduler pair")
    worker = [c for c in schedulers if c["name"] == f"tokenspeed::scheduler_tp{rank}"]
    require(len(worker) == 1 and worker[0]["ppid"] == snapshot["parent_pid"],
            "missing owned rank/PID")
    worker = worker[0]
    require(meta.get(("thread_name", host, worker["pid"]), {}).get("name") == f"Thread {worker['pid']}",
            "CANN main-thread metadata does not match owned scheduler PID")
    require(not any(("thread_name", host, c["pid"]) in meta for c in schedulers if c != worker),
            "trace includes another owned scheduler")
    # These exports have no literal RankId metadata. Actual src-rank task args
    # independently check the TP tag; virtual Chrome PIDs are NOT OS PIDs.
    src = {e["args"]["src rank"] for e in events if e.get("pid") == communication
           and e.get("ph") == "X" and "src rank" in e["args"]}
    require(src == {rank}, "HCCL source rank contradicts file TP tag")
    groups = [(tid, args["name"]) for (kind, pid, tid), args in meta.items()
              if kind == "thread_name" and pid == communication and args["name"].startswith("Group ")]
    require(len(groups) == 1, "ambiguous HCCL group lanes")
    return meta, {"tp_rank": rank, "logical_device": rank, "physical_device": physical,
                  "owned_worker": worker, "runner_pid": snapshot["parent_pid"],
                  "hardware_chrome_pid": hardware, "communication_chrome_pid": communication,
                  "host_chrome_pid": host, "hccl_src_rank": next(iter(src)),
                  "literal_RankId_metadata": False, "group_tid": groups[0][0], "group": groups[0][1]}


def clock_evidence(hardware, run):
    start, end = D(run["start_unix_ns"]) / 1000, D(run["end_unix_ns"]) / 1000
    enables = [interval(e)[0] for e in hardware if e["args"]["Task Type"] == "PROFILING_ENABLE"]
    disables = [interval(e)[1] for e in hardware if e["args"]["Task Type"] == "PROFILING_DISABLE"]
    wall = end - start
    monotonic = D(run["end_monotonic_ns"] - run["start_monotonic_ns"]) / 1000
    # Operational same-epoch check from explicit profiler markers and runner
    # brackets, not epoch magnitude alone. No shift, fit, or latency sum.
    verified = (len(enables) == len(disables) == 1 and enables[0] <= start < end <= disables[0]
                and start - enables[0] < 1_000_000 and disables[0] - end < 1_000_000
                and abs(wall - monotonic) < 100)
    return (start, end), {"request_unix_us": [start, end], "request_wall_us": wall,
                          "request_monotonic_us": monotonic, "enable_us": enables,
                          "disable_us": disables, "same_epoch_bracket_verified": verified,
                          "criterion": "Unix ns/1000 inside CANN enable/disable; each margin <1s; wall/monotonic delta <100us"}


def analyze_events(events, result, rank, filename):
    meta, owner = ownership(events, result, rank, filename)
    hp, cp, host = (owner[k] for k in ("hardware_chrome_pid", "communication_chrome_pid", "host_chrome_pid"))
    hardware = [e for e in events if e.get("pid") == hp and e.get("ph") == "X"]
    require(hardware, "no hardware tasks")
    for e in hardware:
        require(e["args"].get("Task Type") in COMPUTE | CONTROL | OTHER, "unknown/missing device Task Type")
        require(e["args"].get("Physic Stream Id") == e["tid"] and "Task Id" in e["args"],
                "missing device stream/task evidence")
        interval(e)
    comm = [e for e in events if e.get("pid") == cp and e.get("ph") == "X"]
    parents = validate_parents([e for e in comm if e["tid"] == owner["group_tid"]])
    subtasks = [e for e in comm if e["tid"] != owner["group_tid"]]
    require(subtasks, "no device communication subtasks")
    # Every communication subtask must map to a real hardware stream/task with
    # identical interval. Never count this second representation twice.
    hw_by_task = {}
    for e in hardware:
        hw_by_task.setdefault((e["tid"], e["args"]["Task Id"]), []).append(e)
    associated = set()
    for e in subtasks:
        args = e["args"]
        require("task type" in args and "size(Byte)" in args, "malformed HCCL subtask")
        candidates = hw_by_task.get((args.get("stream id"), args.get("task id")), [])
        matches = [h for h in candidates if interval(h) == interval(e)]
        require(len(matches) == 1, "HCCL subtask lacks unique identical hardware task")
        associated.add(id(matches[0]))
    host_events = [e for e in events if e.get("pid") == host and e.get("ph") == "X"]
    launches = {(e["args"].get("connection_id"), e["args"].get("item_id"))
                for e in host_events if e["name"] == "Node@launch"}
    kernels = sorted([e for e in hardware if e["name"] in KERNELS], key=interval)
    require(len(kernels) == len(parents), "HCCL kernel/envelope count mismatch")
    for kernel, parent in zip(kernels, parents):
        args = kernel["args"]
        require(args["Task Type"] == "AI_CPU" and KERNELS[kernel["name"]] == collective_kind(parent)
                and (args["connection_id"], kernel["name"]) in launches, "unverified HCCL kernel association")
        require(any(h["tid"] == kernel["tid"] and h["args"]["Task Type"] == "NOTIFY_WAIT"
                    and h["args"].get("connection_id") == args["connection_id"] for h in hardware),
                "HCCL launch lacks device notify dependency")
        associated.add(id(kernel))
    # Task identities and launch/dependency evidence, not a substring such as
    # 'reduce', decide exclusions. Local ReduceSum remains compute.
    compute = [e for e in hardware if e["args"]["Task Type"] in COMPUTE and id(e) not in associated]
    control = [e for e in hardware if e["args"]["Task Type"] in CONTROL]
    require(compute, "no noncommunication device compute")
    active = [interval(e) for e in hardware if not e["args"]["Task Type"].startswith("PROFILING_")]
    active += [interval(e) for e in parents]
    full_window = min(a for a, _ in active), max(b for _, b in active)
    request, clock = clock_evidence(hardware, result["runs"][0])

    def metrics(window):
        c, m, w = (clip([interval(e) for e in group], window) for group in (compute, parents, control))
        overlap = duration(intersection(c, m))
        cm = duration(c + m)
        covered = duration(c + m + w)
        selected = [e for e in parents if clip([interval(e)], window)]
        payload = Counter((collective_kind(e), e["args"]["data_type"], e["args"]["count"]) for e in selected)
        partial = sum(interval(e)[0] < window[0] or interval(e)[1] > window[1] for e in selected)
        counts = Counter(e["args"]["Task Type"] for e in compute if clip([interval(e)], window))
        return {"window_us": list(window), "span_us": window[1] - window[0],
                "compute_us": duration(c), "communication_envelopes_us": duration(m),
                "overlap_us": overlap, "compute_only_us": duration(c) - overlap,
                "communication_only_us": duration(m) - overlap,
                "explicit_control_union_us": duration(w), "control_outside_compute_comm_us": covered - cm,
                "unclassified_gap_us": window[1] - window[0] - covered,
                "associated_hccl_kernel_outside_envelopes_us": duration(clip([interval(e) for e in kernels], window))
                    - duration(intersection(clip([interval(e) for e in kernels], window), m)),
                "compute_task_counts": dict(sorted(counts.items())), "compute_tasks": sum(counts.values()),
                "collectives": dict(sorted(Counter(collective_kind(e) for e in selected).items())),
                "collective_count": len(selected), "partially_clipped_collectives": partial,
                "communication_by_operation_us": {
                    op: duration(clip([interval(e) for e in selected if collective_kind(e) == op], window))
                    for op in sorted({collective_kind(e) for e in selected})},
                "payloads": [{"operation": op, "dtype": dtype, "input_elements": count,
                              "input_bytes_per_operation": count * WIDTH[dtype], "operations": n,
                              "envelope_union_us": duration(clip([interval(e) for e in selected
                                  if (collective_kind(e), e["args"]["data_type"], e["args"]["count"])
                                  == (op, dtype, count)], window))}
                             for (op, dtype, count), n in sorted(payload.items())]}

    full = metrics(full_window)
    clipped = metrics(request) if clock["same_epoch_bracket_verified"] else None
    if clipped is not None:
        clipped["communication_us_per_returned_output_token"] = (
            clipped["communication_envelopes_us"] / (result["concurrency"] * result["new_tokens"]))
    after = [e for e in parents if interval(e)[0] >= request[1]]
    # Containment is deliberately not assumed: some CANN subtasks/kernels
    # overrun parent envelopes. Expose that instead of silently expanding ops.
    starts = [interval(e)[0] for e in parents]
    overruns = 0
    for e in subtasks:
        i = bisect_right(starts, interval(e)[0]) - 1
        if i < 0 or interval(e)[1] > interval(parents[i])[1]:
            overruns += 1
    clock["active_end_minus_request_end_us"] = full_window[1] - request[1]
    return {"ownership": owner, "clock": clock, "device_active_span": full, "request_window": clipped,
            "representative_device_envelopes": [
                {"name": e["name"], "operation": op, "pid": e["pid"], "tid": e["tid"],
                 "interval_us": list(interval(e)), "args": e["args"]}
                for op in ("AllReduce", "AllGather", "Broadcast")
                for e in [next(p for p in parents if collective_kind(p) == op)]],
            "diagnostics": {"events": len(events), "hardware_tasks": len(hardware),
                            "collective_sequence_sha256": hashlib.sha256(json.dumps([
                                (e["name"], e["args"]["data_type"], e["args"]["count"])
                                for e in parents], separators=(",", ":")).encode()).hexdigest(),
                            "hccl_subtasks_identically_mapped": len(subtasks),
                            "subtasks_not_contained_in_parent": overruns,
                            "hccl_kernels_excluded_from_compute": len(kernels),
                            "hccl_kernels_without_envelope_overlap": sum(not intersection([interval(k)], [interval(p)])
                                                                         for k, p in zip(kernels, parents)),
                            "host_HcclAllReduce_API_events_not_device_ops": sum("hcclallreduce" in e["name"].lower()
                                                                                 for e in host_events),
                            "post_request_collectives": dict(Counter(collective_kind(e) for e in after)),
                            "top_full_span_compute_names": dict(Counter(e["name"] for e in compute).most_common(6))}}


def validate_grid(results, canonical, cases, manifest_sha, config_sha):
    require(set(results) == {1, 2}, "requires exactly c1/c2 experiment grid")
    base = results[1]
    for concurrency, r in results.items():
        require(r["status"] == "two_worker_device_traces_collected_not_yet_analyzed"
                and r["concurrency"] == concurrency and r["tp"] == 2 and r["new_tokens"] == 2
                and r["input_lengths"] == [32] and r["physical_devices_requested"] == [4, 5]
                and r["profile_activities"] == ["NPU"], "mismatched experiment grid")
        require(r["config"] == base["config"] == canonical["config"], "mismatched engine config")
        require(r["config"]["device"] == "npu" and r["config"]["dtype"] == "bfloat16"
                and r["config"]["world_size"] == 2 and r["config"]["base_gpu_id"] == 0
                and r["config"]["gpu_id_step"] == 1, "mismatched logical device grid")
        require(r["reproducibility"] == base["reproducibility"], "profile c1/c2 source/runtime/environment mismatch")
        require(r["reproducibility"]["environment"]["ASCEND_RT_VISIBLE_DEVICES"] == "4,5", "visibility mismatch")
        for key, expected in (("manifest_sha256", manifest_sha), ("config_sha256", config_sha), ("cases_sha256", cases)):
            require(r[key] == canonical[key] == expected, f"mismatched {key}")
        require(len(r["runs"]) == 1, "requires one profiled request batch")
        row = r["runs"][0]
        require(row["case"] == "length_32" and row["input_tokens"] == 32
                and row["profile_id"] == f"hw2-npu-c{concurrency}-input32"
                and set(row["traces"]) == {"TP0", "TP1"}, "mismatched case/trace grid")
        reference = [row for row in canonical["runs"] if row["case"] == "length_32"]
        require(len(reference) == 1 and len(reference[0]["outputs"]) == 1, "ambiguous canonical reference")
        prefix = reference[0]["outputs"][0]["output_ids"][:2]
        require(len(prefix) == 2 and len(row["outputs"]) == concurrency
                and row["warmup_output_ids"] == [prefix] * concurrency, "warmup/canonical mismatch")
        for output in row["outputs"]:
            require(output["output_ids"] == prefix and output["meta_info"]["prompt_tokens"] == 32
                    and output["meta_info"]["completion_tokens"] == 2 and output["meta_info"]["cached_tokens"] == 0,
                    "profile outputs differ from canonical first two IDs")
    old, new = canonical["reproducibility"]["source_trees"], base["reproducibility"]["source_trees"]
    require(set(old) == set(new), "canonical source tree grid mismatch")
    differences = []
    for tree in old:
        before, after = old[tree]["files"], new[tree]["files"]
        for file in sorted(set(before) | set(after)):
            if before.get(file) != after.get(file):
                require(tree == "python" and file == "tokenspeed/runtime/engine/request_handler.py"
                        and file in before and file in after, "unrelated canonical source change")
                differences.append({"tree": tree, "file": file, "canonical_sha256": before[file], "profile_sha256": after[file]})
    return {"profile_c1_c2_config_source_runtime_environment_exact": True,
            "canonical_prefix_ids": prefix, "canonical_source_differences": differences,
            "canonical_source_note": "Canonical run predates the NPU profile-hook change in request_handler.py; only that file may differ. Hash inventories do not reconstruct the old code diff.",
            "source_tree_sha256": {k: v["sha256"] for k, v in new.items()},
            "runtime_versions": base["reproducibility"]["runtime_versions"]}


def card_evidence(text, worker_pid, physical):
    # Read only relevant rows from the already-local snapshot; no live process scan.
    card = None
    chip = None
    for line in text.splitlines():
        match = re.match(r"\|\s*(\d+)\s+Ascend910\s*\|", line)
        if match:
            card = int(match[1])
        match = re.match(r"\|\s*(\d+)\s+(\d+)\s*\|\s*[0-9A-Fa-f]{4}:", line)
        if match and int(match[2]) == physical:
            chip = int(match[1])
            require(card == 2 and chip == physical - 4, "physical device not on card 2")
    require(chip is not None, "missing physical card/chip row")
    rows = re.findall(r"\|\s*(\d+)\s+(\d+)\s*\|\s*" + str(worker_pid) + r"\s*\|", text)
    require(("2", str(chip)) in rows, "owned PID not on expected card/chip")
    return {"card": 2, "chip": chip, "worker_pid_seen_on_card_chip": True}


def trace_paths(folder, concurrency):
    expected = {f"hw2-npu-c{concurrency}-input32-TP{rank}.trace.json" for rank in (0, 1)}
    candidates = {p.name: p for p in folder.glob("*.trace.json*")}
    logical = {name.removesuffix(".gz") for name in candidates}
    require(logical == expected, "unexpected trace file grid")
    # Raw + gzip is one logical trace, not two ranks. Never read unselected raw.
    return {rank: candidates.get(name + ".gz", candidates.get(name))
            for rank in (0, 1)
            for name in [f"hw2-npu-c{concurrency}-input32-TP{rank}.trace.json"]}


def load_trace(path, root, recorded):
    content, stored = bounded_bytes(path)
    stored_file = {"path": str(path.relative_to(root)), "bytes": len(stored),
                   "sha256": hashlib.sha256(stored).hexdigest()}
    # Preserve the original raw-trace hash/size semantics across representations.
    fp = {"path": stored_file["path"], "bytes": len(content),
          "sha256": hashlib.sha256(content).hexdigest(), "stored_file": stored_file,
          "compression": "gzip" if path.name.endswith(".gz") else "none"}
    require(all(fp[k] == recorded[k] for k in ("sha256", "bytes")), "trace fingerprint mismatch")
    events = event_array(json.loads(content, parse_float=D))
    require(len(events) == recorded["events"], "trace event count mismatch")
    return events, fp


def analyze(root):
    results_dir = root / "homework-2/results"
    paths = {c: results_dir / f"0.8B-npu-batch-experiment-c{c}/result.json" for c in (1, 2)}
    results = {c: load(path) for c, path in paths.items()}
    canonical_path = results_dir / "0.8B-correctness-tp2-c1/result.json"
    metadata_dir = root / "homework-2/metadata/Qwen3.5-0.8B"
    provenance_paths = list(paths.values()) + [canonical_path, root / "homework-2/cases.json",
                                              metadata_dir / "config.json", metadata_dir / "source-manifest.json",
                                              root / "homework-2/scripts/profile_engine_npu.py",
                                              root / "homework-2/scripts/analyze_npu_traces.py"]
    provenance = [fingerprint(p, root) for p in provenance_paths]
    sha = {p["path"]: p["sha256"] for p in provenance}
    checks = validate_grid(results, load(canonical_path), sha["homework-2/cases.json"],
                           sha["homework-2/metadata/Qwen3.5-0.8B/source-manifest.json"],
                           sha["homework-2/metadata/Qwen3.5-0.8B/config.json"])
    model_config = load(metadata_dir / "config.json")
    require(model_config["text_config"]["hidden_size"] == 1024, "hidden payload assumption mismatch")
    manifest = load(metadata_dir / "source-manifest.json")
    require(manifest.get("repo") == "Qwen/Qwen3.5-0.8B", "wrong model manifest")
    require(any(e["path"] == "config.json" and e["sha256"] == results[1]["config_sha256"] for e in manifest["files"]),
            "model manifest/config mismatch")
    cases = load(root / "homework-2/cases.json")
    require(sum(c["name"] == "length_32" and len(c["input_ids"]) == 32 for c in cases["cases"]) == 1,
            "missing unique length-32 input case")
    rows = []
    for c, result in results.items():
        folder = paths[c].parent
        selected = trace_paths(folder, c)
        hbm_path = folder / "hbm-loaded.txt"
        provenance.append(fingerprint(hbm_path, root))
        for rank in (0, 1):
            path = selected[rank]
            recorded = result["runs"][0]["traces"][f"TP{rank}"]
            events, fp = load_trace(path, root, recorded)
            provenance.append({**fp["stored_file"], "uncompressed_sha256": fp["sha256"],
                               "uncompressed_bytes": fp["bytes"]})
            row = analyze_events(events, result, rank, path.name)
            del events
            row["concurrency"] = c
            row["trace"] = fp
            row["ownership"].update(card_evidence(hbm_path.read_text(), row["ownership"]["owned_worker"]["pid"],
                                                  row["ownership"]["physical_device"]))
            rows.append(row)
    for c in (1, 2):
        pair = [r for r in rows if r["concurrency"] == c]
        require(pair[0]["diagnostics"]["collective_sequence_sha256"]
                == pair[1]["diagnostics"]["collective_sequence_sha256"],
                "TP peers have mismatched collective sequence/payload grid")
    comparison = []
    for rank in (0, 1):
        pair = [r for r in rows if r["ownership"]["tp_rank"] == rank]
        if all(r["request_window"] is not None for r in pair):
            c1, c2 = (r["request_window"] for r in pair)
            comparison.append({"tp_rank": rank,
                "collectives_c1_c2": [c1["collective_count"], c2["collective_count"]],
                "compute_tasks_c1_c2": [c1["compute_tasks"], c2["compute_tasks"]],
                "compute_union_c2_over_c1": c2["compute_us"] / c1["compute_us"],
                "communication_per_output_token_c2_over_c1": (
                    c2["communication_us_per_returned_output_token"] / c1["communication_us_per_returned_output_token"])})
    return {"schema_version": 1, "status": "validated_bounded_local_device_analysis",
            "request_comparison_per_rank": comparison,
            "units": "microseconds; exact Decimal quantities serialized as strings; payloads are input bytes, NOT wire bytes",
            "provenance": provenance, "validation": checks, "traces": rows,
            "theoretical_hidden_allreduce": {"formula": "1024 * T * 2 = 2048*T input bytes (BF16)",
                                              "decode_c1_bytes": 2048, "decode_c2_bytes": 4096,
                                              "prefill_c1_bytes": 65536, "prefill_c2_bytes": 131072},
            "limitations": ["One warmed profile batch per concurrency, 32 input / 2 returned output tokens; not the separate three-repeat benchmark matrix.",
                            "Same-epoch verification uses profiler bracketing and runner clocks; no sub-microsecond external clock calibration. No offset fitting.",
                            "Device-active first/last spans are not request wall latency. Request metrics use exact saved Unix-ns boundaries when bracketing verifies.",
                            "51 collective envelopes per rank start after request end; no unverified cause (extra forward, drain, etc.) is asserted.",
                            "Parent envelopes, hardware HCCL kernels, and plane subtasks have differing extents. Envelopes alone define communication; auxiliary extents are exposed, not added twice.",
                            "Explicit control intervals can overlap work on other streams and do not measure whole-device stall. Gaps include unclassified/other device work, host scheduling and launch gaps; not pure CPU wait.",
                            "Payload count/dtype gives logical per-operation input size, not link traffic or network bandwidth. Shared-device snapshots do not prove exclusivity or peak HBM.",
                            "No per-rank durations are summed into end-to-end latency; output IDs certify only the two-token prefix here, not general accuracy/performance."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    report = analyze(root)  # On failure, do not overwrite a previously valid report.
    output = args.output or root / "homework-2/results/npu-communication-analysis.json"
    output.write_text(json.dumps(report, indent=2, default=str, allow_nan=False) + "\n")
    for row in report["traces"]:
        m = row["request_window"] or row["device_active_span"]
        print(f"c{row['concurrency']} TP{row['ownership']['tp_rank']}: "
              f"compute={m['compute_us']}us comm={m['communication_envelopes_us']}us "
              f"overlap={m['overlap_us']}us collectives={m['collective_count']}")


if __name__ == "__main__":
    main()
