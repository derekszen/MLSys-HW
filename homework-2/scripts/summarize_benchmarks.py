"""Fail-closed summary of the existing 0.8B TP/concurrency benchmark matrix.

Only batch wall latency and aggregate output throughput are summarized. Every
measured request must reproduce its matching correctness run's exact output IDs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

GRID = {(tp, concurrency) for tp in (1, 2) for concurrency in (1, 2)}
TP_FIELDS = {"world_size", "attn_tp_size", "dense_tp_size", "moe_tp_size", "lm_head_tp_size"}
SOURCE_TREES = {"python", "tokenspeed-kernel/python", "tokenspeed-kernel-npu/python"}
COMPLETED = "generation_completed_not_yet_compared"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(ch in "0123456789abcdef" for ch in value))


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def source_identity(result):
    trees = result["reproducibility"]["source_trees"]
    require(set(trees) == SOURCE_TREES, "missing/unexpected source trees")
    identity = {}
    for name, tree in trees.items():
        files = tree["files"]
        require("error" not in tree and isinstance(files, dict) and bool(files), "missing source files")
        require(all(isinstance(p, str) and p.endswith(".py") and digest(h)
                    for p, h in files.items()), "invalid source file fingerprint")
        canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        require(digest(tree["sha256"]) and sha256(canonical) == tree["sha256"],
                "source tree fingerprint disagrees with file manifest")
        identity[name] = {"sha256": tree["sha256"], "files": files}
    return identity


def shared_config(result):
    # TP dimensions and rendezvous port are the only allowed config differences.
    return {k: v for k, v in result["config"].items() if k not in TP_FIELDS | {"port"}}


def output_ids(run, *, concurrency, length, new_tokens, vocab, correctness):
    outputs = run["outputs"]
    require(isinstance(outputs, list) and len(outputs) == concurrency, "missing/extra request output")
    ids = []
    for output in outputs:
        tokens = output["output_ids"]
        require(isinstance(tokens, list) and len(tokens) == new_tokens
                and all(type(t) is int and 0 <= t < vocab for t in tokens),
                "truncated/invalid output IDs")
        meta = output["meta_info"]
        require(meta["prompt_tokens"] == length and meta["completion_tokens"] == new_tokens,
                "output metadata token counts differ")
        require(meta["finish_reason"] == {"type": "length", "length": new_tokens},
                "generation did not reach the requested length")
        if correctness:
            logs = meta["output_token_logprobs"]
            require(len(logs) == new_tokens and [entry[1] for entry in logs] == tokens,
                    "correctness selected-logprob IDs differ")
            require(all(type(entry[0]) in (int, float) and math.isfinite(entry[0]) for entry in logs),
                    "invalid correctness selected logprobs")
        ids.append(tokens)
    return ids


def summarize(benchmarks, correctness, *, cases_bytes, manifest_bytes, config_bytes):
    """Validate the complete matrix and return deterministic medians and ratios.

    Inputs are dictionaries keyed by (TP, concurrency), plus exact archived
    cases/model metadata bytes. Malformed or mismatched evidence raises ValueError.
    Source roots may relocate; relative file manifests and their digests may not.
    """
    try:
        return _summarize(benchmarks, correctness, cases_bytes=cases_bytes,
                          manifest_bytes=manifest_bytes, config_bytes=config_bytes)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as exc:
        raise ValueError(f"missing or malformed benchmark evidence: {exc}") from exc


def _summarize(benchmarks, correctness, *, cases_bytes, manifest_bytes, config_bytes):
    require(set(benchmarks) == GRID and set(correctness) == GRID, "incomplete TP/concurrency grid")
    cases_document = json.loads(cases_bytes)
    model_config = json.loads(config_bytes)
    manifest = json.loads(manifest_bytes)
    hashes = {"manifest_sha256": sha256(manifest_bytes), "config_sha256": sha256(config_bytes),
              "cases_sha256": sha256(cases_bytes)}
    config_entries = [entry for entry in manifest["files"] if entry["path"] == "config.json"]
    require(len(config_entries) == 1 and config_entries[0]["sha256"] == hashes["config_sha256"],
            "model config does not match manifest")
    vocab = model_config["text_config"]["vocab_size"]
    require(type(vocab) is int and vocab > 0, "invalid vocabulary size")
    new_tokens = cases_document["new_tokens"]
    require(type(new_tokens) is int and new_tokens == 16, "expected 16 generated tokens")
    cases = {}
    for case in cases_document["cases"]:
        name, tokens = case["name"], case["input_ids"]
        require(isinstance(name, str) and bool(name) and name not in cases, "duplicate/invalid case")
        require(isinstance(tokens, list) and bool(tokens)
                and all(type(t) is int and 0 <= t < vocab for t in tokens), "invalid case input IDs")
        cases[name] = len(tokens)
    measured = {name: length for name, length in cases.items() if length in (32, 256)}
    require(sorted(measured.values()) == [32, 256], "expected exactly one 32- and one 256-token case")
    anchor = correctness[(1, 1)]
    sources = source_identity(anchor)
    config = shared_config(anchor)
    versions = anchor["reproducibility"]["runtime_versions"]
    require(all(isinstance(versions[k], str) and bool(versions[k])
                for k in ("torch", "torch_npu", "triton")), "missing runtime versions")
    expected_environment = {k: v for k, v in anchor["reproducibility"]["environment"].items()
                            if k != "ASCEND_RT_VISIBLE_DEVICES"}
    indexed = {}
    for mode, matrix in (("correctness", correctness), ("benchmark", benchmarks)):
        for (tp, concurrency), result in sorted(matrix.items()):
            require(type(result["tp"]) is int and result["tp"] == tp
                    and type(result["concurrency"]) is int and result["concurrency"] == concurrency,
                    "claimed TP/concurrency differs from matrix cell")
            require(result["mode"] == mode and result["status"] == COMPLETED and "error" not in result,
                    "run did not complete the expected mode")
            require(all(type(result["config"][key]) is int and result["config"][key] == tp
                        for key in TP_FIELDS), "configured sharding differs from TP")
            require(shared_config(result) == config, "non-parallel configuration differs")
            if mode == "benchmark":
                require({k: v for k, v in result["config"].items() if k != "port"}
                        == {k: v for k, v in correctness[(tp, concurrency)]["config"].items() if k != "port"},
                        "benchmark/correctness configuration differs")
            require(all(result[k] == v for k, v in hashes.items()), "manifest/config/cases fingerprint differs")
            require(source_identity(result) == sources, "source fingerprints differ")
            repro = result["reproducibility"]
            require(repro["runtime_versions"] == versions, "runtime versions differ")
            environment = repro["environment"]
            require({k: v for k, v in environment.items() if k != "ASCEND_RT_VISIBLE_DEVICES"}
                    == expected_environment, "non-visibility environment differs")
            devices = result["physical_devices_requested"]
            require(isinstance(devices, list) and len(devices) == tp
                    and all(type(d) is int and d >= 0 for d in devices) and len(set(devices)) == tp,
                    "invalid requested devices")
            require(environment["ASCEND_RT_VISIBLE_DEVICES"] == ",".join(map(str, devices)),
                    "device visibility differs from requested devices")
            require(devices == correctness[(tp, concurrency)]["physical_devices_requested"],
                    "benchmark/correctness devices differ")
            selected_cases = cases if mode == "correctness" else measured
            repeats = range(1) if mode == "correctness" else range(3)
            expected = {(name, repeat) for name in selected_cases for repeat in repeats}
            runs = {}
            for run in result["runs"]:
                key = (run["case"], run["repeat"])
                require(type(run["repeat"]) is int and key in expected and key not in runs,
                        "extra/duplicate case or repeat")
                length = selected_cases[key[0]]
                require(type(run["input_tokens"]) is int and run["input_tokens"] == length
                        and type(run["output_tokens_per_request"]) is int
                        and run["output_tokens_per_request"] == new_tokens, "case shape differs")
                ids = output_ids(run, concurrency=concurrency, length=length,
                                 new_tokens=new_tokens, vocab=vocab, correctness=mode == "correctness")
                if mode == "benchmark":
                    seconds, throughput = run["batch_seconds"], run["aggregate_output_tokens_per_second"]
                    require(positive(seconds) and positive(throughput), "invalid timing/throughput")
                    require(math.isclose(throughput, new_tokens * concurrency / seconds,
                                         rel_tol=1e-12, abs_tol=0), "throughput disagrees with duration")
                runs[key] = {"run": run, "ids": ids}
            require(set(runs) == expected, "incomplete case/repeat grid")
            indexed[(mode, tp, concurrency)] = runs
    # References must agree across TP and concurrency as well as within a batch.
    reference = indexed[("correctness", 1, 1)]
    for tp, concurrency in sorted(GRID):
        for (name, _), item in indexed[("correctness", tp, concurrency)].items():
            require(all(ids == reference[(name, 0)]["ids"][0] for ids in item["ids"]),
                    "correctness output IDs differ across TP/requests")
    rows, checks = [], []
    for tp, concurrency in sorted(GRID):
        runs = indexed[("benchmark", tp, concurrency)]
        refs = indexed[("correctness", tp, concurrency)]
        for name, length in sorted(measured.items(), key=lambda item: item[1]):
            samples = []
            for repeat in range(3):
                item = runs[(name, repeat)]
                for request, ids in enumerate(item["ids"]):
                    require(ids == refs[(name, 0)]["ids"][request],
                            f"benchmark output IDs differ: tp={tp}, c={concurrency}, {name}, repeat={repeat}, request={request}")
                    checks.append({"tp": tp, "concurrency": concurrency, "case": name,
                                   "repeat": repeat, "request": request, "exact_output_ids": True})
                samples.append(item["run"])
            rows.append({"tp": tp, "concurrency": concurrency, "case": name, "input_tokens": length,
                         "repeats": 3, "output_tokens_per_request": new_tokens,
                         "batch_seconds": [s["batch_seconds"] for s in samples],
                         "aggregate_output_tokens_per_second": [s["aggregate_output_tokens_per_second"] for s in samples],
                         "median_batch_seconds": statistics.median(s["batch_seconds"] for s in samples),
                         "median_aggregate_output_tokens_per_second": statistics.median(s["aggregate_output_tokens_per_second"] for s in samples)})
    ratios = []
    for concurrency in (1, 2):
        for length in (32, 256):
            pair = {r["tp"]: r for r in rows if r["concurrency"] == concurrency and r["input_tokens"] == length}
            ratios.append({"concurrency": concurrency, "input_tokens": length,
                           "tp1_latency_divided_by_tp2_latency": pair[1]["median_batch_seconds"] / pair[2]["median_batch_seconds"]})
    return {"passed": True, "model": "Qwen3.5-0.8B", "fingerprints": hashes,
            "source_trees": {name: {"sha256": tree["sha256"], "file_count": len(tree["files"])}
                             for name, tree in sorted(sources.items())},
            "runtime_versions": versions, "shared_config": config,
            "measurement": "batch wall latency and aggregate output-token throughput; not TTFT, TPOT or isolated communication timing",
            "limitations": ["Shared server; uncontrolled contention may affect measurements.",
                            "Three repeats per cell; descriptive medians only, no confidence interval.",
                            "Exact output-ID checks only; no new selected-logprob tolerance or full-logit claim."],
            "speedup_definition": "median TP1 batch seconds / median TP2 batch seconds; greater than 1 means TP2 is faster",
            "verified_benchmark_requests": len(checks), "output_id_checks": checks,
            "rows": rows, "tp2_speedups": ratios}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=root / "results")
    parser.add_argument("--cases", type=Path, default=root / "cases.json")
    parser.add_argument("--metadata", type=Path, default=root / "metadata/Qwen3.5-0.8B")
    parser.add_argument("--output", type=Path, default=root / "results/0.8B-benchmark-summary.json")
    args = parser.parse_args()
    matrices, evidence = {}, []
    for mode in ("benchmark", "correctness"):
        matrices[mode] = {}
        for tp, concurrency in sorted(GRID):
            relative = f"0.8B-{mode}-tp{tp}-c{concurrency}/result.json"
            data = (args.results / relative).read_bytes()
            matrices[mode][(tp, concurrency)] = json.loads(data)
            evidence.append({"path": relative, "sha256": sha256(data)})
    result = summarize(matrices["benchmark"], matrices["correctness"],
                       cases_bytes=args.cases.read_bytes(),
                       manifest_bytes=(args.metadata / "source-manifest.json").read_bytes(),
                       config_bytes=(args.metadata / "config.json").read_bytes())
    result["inputs"] = evidence
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"Verified {result['verified_benchmark_requests']} benchmark requests; wrote {args.output}")


if __name__ == "__main__":
    main()
