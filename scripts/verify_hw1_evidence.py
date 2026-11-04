"""Recompute the frozen HW1 evidence checks without importing an NPU runtime.

This verifies saved observations, not a new hardware run or full-logit accuracy.
The explicit case grid and tolerance belong to this repository's HW1 baseline.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
CASES = {"smoke": 5, "length_32": 32, "length_256": 256}
METRICS = ("ttft_ms", "tpot_ms", "total_ms")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def equal_number(actual, expected, label):
    require(number(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9),
            f"{label} differs from recomputed value")


def token_ids(value, length):
    return (isinstance(value, list) and len(value) == length
            and all(type(t) is int and t >= 0 for t in value))


def indexed(rows, key, expected, label):
    require(isinstance(rows, list), f"{label} must be a list")
    result = {}
    for row in rows:
        identity = key(row)
        require(identity in expected and identity not in result, f"extra/duplicate {label}: {identity}")
        result[identity] = row
    require(set(result) == set(expected), f"incomplete {label}")
    return result


def verify(reference, validation, benchmark):
    """Reject incomplete/mismatched evidence even when stored passed flags are true."""
    try:
        return _verify(reference, validation, benchmark)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as exc:
        raise ValueError(f"malformed HW1 evidence: {exc}") from exc


def _verify(reference, validation, benchmark):
    require(reference["dtype"] == "bfloat16", "unexpected reference precision")
    require(reference["sampling"] == {"do_sample": False, "new_tokens": 16, "ignore_eos": True},
            "unexpected reference sampling")
    config = validation["config"]
    require(config["model"] == reference["model"] and config["dtype"] == reference["dtype"],
            "reference/validation model or precision differs")
    require(config["device"] == "npu" and config["sampling_backend"] == "greedy"
            and config["mamba_ssm_dtype"] == "float32" and config["language_model_only"] is True,
            "unexpected validation configuration")
    require(validation["criteria"]["selected_logprob_atol"] == 0.15, "unexpected tolerance")
    refs = indexed(reference["cases"], lambda r: r["name"], CASES, "reference cases")
    observed = indexed(validation["cases"], lambda r: r["name"], CASES, "validation cases")
    errors = {}
    for name, length in CASES.items():
        ref, row = refs[name], observed[name]
        require(token_ids(ref["input_ids"], length) and token_ids(ref["output_ids"], 16),
                f"invalid reference token IDs: {name}")
        logs = ref["selected_logprobs"]
        require(isinstance(logs, list) and len(logs) == 16 and all(number(v) for v in logs),
                f"invalid reference logprobs: {name}")
        output = row["output"]
        require(type(row["input_tokens"]) is int and row["input_tokens"] == length,
                f"input length differs: {name}")
        require(token_ids(output["output_ids"], 16) and output["output_ids"] == ref["output_ids"],
                f"generated IDs differ: {name}")
        meta = output["meta_info"]
        require(meta["prompt_tokens"] == length and meta["completion_tokens"] == 16
                and meta["finish_reason"] == {"type": "length", "length": 16},
                f"generation metadata differs: {name}")
        actual = meta["output_token_logprobs"]
        require(isinstance(actual, list) and len(actual) == 16,
                f"incomplete selected logprobs: {name}")
        require(all(isinstance(v, list) and len(v) == 3 and number(v[0])
                    and type(v[1]) is int and v[1] == token
                    for v, token in zip(actual, ref["output_ids"])),
                f"invalid selected logprobs or token alignment: {name}")
        delta = max(abs(a[0] - b) for a, b in zip(actual, logs))
        require(delta <= 0.15, f"numerical tolerance exceeded: {name}")
        equal_number(row["max_selected_logprob_error"], delta, f"reported error: {name}")
        require(row["greedy_exact"] is True and row["passed"] is True,
                f"stored verdict disagrees: {name}")
        errors[name] = delta

    isolation_expected = {("A_after_B", ""): "smoke", ("batch2", "smoke"): "smoke",
                          ("batch2", "length_32"): "length_32"}
    isolation = indexed(validation["state_isolation"], lambda r: (r["kind"], r.get("name", "")),
                        isolation_expected, "state-isolation cases")
    for key, name in isolation_expected.items():
        row = isolation[key]
        require(token_ids(row["output_ids"], 16) and row["output_ids"] == refs[name]["output_ids"]
                and row["passed"] is True, f"state-isolation mismatch: {key}")
    require(validation["passed"] is True, "stored validation verdict is not a pass")

    expected_config = dict(config, enable_prefix_caching=False, disable_kvstore=True,
                           enable_output_logprobs=False, stream_interval=1)
    require(benchmark["config"] == expected_config, "benchmark/validation configuration differs")
    require(type(benchmark["concurrency"]) is int and benchmark["concurrency"] == 1,
            "unexpected benchmark concurrency")
    require(type(benchmark["repeats"]) is int and benchmark["repeats"] == 3,
            "expected three benchmark repeats")
    grid = {(length, count, repeat) for length in (32, 256) for count in (8, 16) for repeat in range(3)}
    samples = indexed(benchmark["samples"],
                      lambda r: (r["input_tokens"], r["output_tokens"], r["repeat"]), grid,
                      "benchmark samples")
    groups = {}
    for (length, count, repeat), row in samples.items():
        require(all(type(row[k]) is int for k in ("input_tokens", "output_tokens", "repeat")),
                "non-integer benchmark grid coordinate")
        require(token_ids(row["output_ids"], count)
                and row["output_ids"] == refs[f"length_{length}"]["output_ids"][:count],
                "benchmark output differs from reference")
        require(type(row["cached_tokens"]) is int and row["cached_tokens"] == 0,
                "benchmark reused cached tokens")
        arrivals = row["arrivals"]
        require(isinstance(arrivals, list) and len(arrivals) >= 2, "missing streaming arrivals")
        previous_count, previous_time = 0, 0
        for arrival in arrivals:
            n, seconds = arrival["tokens"], arrival["seconds"]
            require(type(n) is int and previous_count < n <= count
                    and number(seconds) and seconds > previous_time, "invalid streaming arrival order")
            previous_count, previous_time = n, seconds
        require(arrivals[0]["tokens"] == 1 and arrivals[-1]["tokens"] == count,
                "first/final token arrival missing")
        equal_number(row["ttft_ms"], arrivals[0]["seconds"] * 1000, "TTFT")
        equal_number(row["tpot_ms"], (arrivals[-1]["seconds"] - arrivals[0]["seconds"]) * 1000 / (count - 1),
                     "TPOT")
        require(number(row["total_ms"]) and row["total_ms"] >= arrivals[-1]["seconds"] * 1000,
                "total latency precedes final token")
        groups.setdefault((length, count), []).append(row)
    summaries = indexed(benchmark["summary"], lambda r: (r["input_tokens"], r["output_tokens"]),
                        groups, "benchmark summaries")
    for key, rows in groups.items():
        for metric in METRICS:
            values = [row[metric] for row in rows]
            for stat, func in (("median", statistics.median), ("min", min), ("max", max)):
                equal_number(summaries[key][metric][stat], func(values), f"{key} {metric} {stat}")
    require(benchmark["passed"] is True, "stored benchmark verdict is not a pass")
    return {"passed": True, "validation_cases": len(observed), "state_isolation_cases": len(isolation),
            "benchmark_samples": len(samples), "benchmark_cells": len(groups),
            "max_selected_logprob_error": errors,
            "scope": "Offline recomputation of saved HW1 observations; no hardware inference executed.",
            "limitations": ["Selected logprobs, not full-logit equivalence.",
                            "HW1 records lack checkpoint/source fingerprints; matching paths/configs do not prove provenance.",
                            "Timing checks verify arithmetic, not clock accuracy or uncontended hardware performance."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name, filename in (("reference", "model-reference.json"), ("validation", "model-validation.json"),
                           ("benchmark", "benchmark.json")):
        parser.add_argument(f"--{name}", type=Path, default=ROOT / "homework-1/results" / filename)
    parser.add_argument("--output", type=Path, help="new JSON output path; stdout if omitted")
    args = parser.parse_args(argv)
    try:
        documents, inputs = [], []
        for name in ("reference", "validation", "benchmark"):
            path = getattr(args, name).resolve()
            raw = path.read_bytes()
            documents.append(json.loads(raw))
            inputs.append({"role": name, "path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
        result = verify(*documents)
        result["inputs"] = inputs
        text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as stream:
                stream.write(text)
        else:
            print(text, end="")
    except (OSError, ValueError, UnicodeError) as exc:
        parser.exit(2, f"HW1 evidence verification failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
