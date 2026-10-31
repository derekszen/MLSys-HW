"""Summarize host-only Chrome events; never interpret CPU spans as NPU kernel time."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path


def repair_names(text):
    """Repair only unescaped JSON payloads in Torch's record_function name field."""
    lines, repairs = [], 0
    for line in text.splitlines(keepends=True):
        if '"name": "' in line and '", "pid":' in line:
            before, rest = line.split('"name": "', 1)
            name, after = rest.rsplit('", "pid":', 1)
            try:
                json.loads('"' + name + '"')
            except json.JSONDecodeError:
                line = before + '"name": ' + json.dumps(name) + ', "pid":' + after
                repairs += 1
        lines.append(line)
    return json.loads("".join(lines)), repairs


def summarize(events):
    threads = defaultdict(list)
    totals = defaultdict(lambda: {"calls": 0, "inclusive_us": 0.0, "self_us": 0.0})
    phases = defaultdict(lambda: {"calls": 0, "host_span_ms": 0.0})
    for e in events:
        if e.get("ph") != "X":
            continue
        if e.get("cat") == "cpu_op":
            threads[(e["pid"], e["tid"])].append(e)
        if e.get("name", "").startswith("tokenspeed.forward.v1:"):
            mode = json.loads(e["name"].split(":", 1)[1])["mode"]
            phases[mode]["calls"] += 1
            phases[mode]["host_span_ms"] += e["dur"] / 1000
    for spans in threads.values():
        stack = []
        for e in sorted(spans, key=lambda e: (e["ts"], -e["dur"])):
            start, duration = e["ts"], e["dur"]
            end = start + duration
            while stack and start >= stack[-1][0]:
                stack.pop()
            if stack:
                if end > stack[-1][0] + 0.01:
                    raise ValueError("Non-nested CPU spans: cannot calculate exclusive time")
                totals[stack[-1][1]]["self_us"] -= duration
            row = totals[e["name"]]
            row["calls"] += 1
            row["inclusive_us"] += duration
            row["self_us"] += duration
            stack.append((end, e["name"]))
    return {"forward_phases": dict(phases),
            "cpu_ops": [{"name": name, **row} for name, row in sorted(
                totals.items(), key=lambda x: x[1]["self_us"], reverse=True)]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    trace, repairs = repair_names(gzip.open(args.trace, "rt").read())
    result = summarize(trace["traceEvents"])
    result.update(raw_trace_sha256=hashlib.sha256(args.trace.read_bytes()).hexdigest(),
                  repaired_unescaped_names=repairs,
                  scope="CPU-only, profiler-instrumented, includes launch/synchronization costs; not NPU kernel durations or an uninstrumented latency baseline")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"phases": result["forward_phases"], "top_cpu_ops": result["cpu_ops"][:15]}, indent=2))


if __name__ == "__main__":
    main()
