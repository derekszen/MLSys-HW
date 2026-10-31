"""Offline, explicit-evidence TP validation and child-PID-owned HBM summaries.

Examples (all input paths and the single output path must be selected explicitly):
  summarize_validation_run.py pair --tp1 RUN1 --tp2 RUN2 --benchmark BENCH2 \
      --cases cases.json --metadata metadata/MODEL --atol 0.15 --output summary.json
  summarize_validation_run.py hbm --runs RUN1 RUN2 --output memory.json

Strict TP comparison and benchmark repeatability are separate judgments. Neither
certifies independent reference accuracy. HBM snapshots are not peak memory.
"""
import argparse
import json
import math
from pathlib import Path
import statistics

from compare_tp import compare
from summarize_benchmarks import (
    COMPLETED, TP_FIELDS, output_ids, positive, require, sha256, shared_config,
    source_identity,
)
from summarize_large_run import owned_hbm

SNAPSHOTS = ("hbm-loaded.txt", "hbm-length_32.txt", "hbm-length_256.txt")
MALFORMED = (KeyError, TypeError, IndexError, AttributeError, OverflowError)


def _metadata(cases_bytes, manifest_bytes, config_bytes):
    hashes = {"manifest_sha256": sha256(manifest_bytes),
              "config_sha256": sha256(config_bytes), "cases_sha256": sha256(cases_bytes)}
    manifest, config, document = map(json.loads, (manifest_bytes, config_bytes, cases_bytes))
    entries = [entry for entry in manifest["files"] if entry["path"] == "config.json"]
    require(len(entries) == 1 and entries[0]["sha256"] == hashes["config_sha256"],
            "model config differs from manifest")
    vocab = config["text_config"]["vocab_size"]
    require(type(vocab) is int and vocab > 0, "invalid vocabulary")
    require(type(document["new_tokens"]) is int and document["new_tokens"] == 16,
            "expected 16 generated tokens")
    cases = {}
    for case in document["cases"]:
        name, tokens = case["name"], case["input_ids"]
        require(isinstance(name, str) and bool(name) and name not in cases,
                "invalid/duplicate case")
        require(isinstance(tokens, list) and bool(tokens)
                and all(type(t) is int and 0 <= t < vocab for t in tokens), "invalid input IDs")
        cases[name] = len(tokens)
    require(bool(cases), "missing cases")
    return hashes, vocab, cases, manifest.get("repo")


def _validate_run(result, anchor, *, tp, mode, hashes, cases, vocab):
    require(type(result["tp"]) is int and result["tp"] == tp, "unexpected TP")
    concurrency = anchor["concurrency"]
    require(type(concurrency) is int and concurrency in (1, 2)
            and type(result["concurrency"]) is int and result["concurrency"] == concurrency,
            "unexpected concurrency")
    require(result["mode"] == mode and result["status"] == COMPLETED and "error" not in result,
            "run did not complete expected mode")
    require(all(type(result["config"][k]) is int and result["config"][k] == tp for k in TP_FIELDS),
            "configured sharding differs from TP")
    require(shared_config(result) == shared_config(anchor), "non-parallel configuration differs")
    require(all(result[k] == v for k, v in hashes.items()), "model/cases fingerprint differs")
    require(source_identity(result) == source_identity(anchor), "source fingerprints differ")
    repro, expected = result["reproducibility"], anchor["reproducibility"]
    versions = repro["runtime_versions"]
    require(all(isinstance(versions[k], str) and bool(versions[k]) and versions[k] != "unavailable"
                for k in ("torch", "torch_npu", "triton")), "missing runtime versions")
    require(versions == expected["runtime_versions"], "runtime versions differ")
    environment = repro["environment"]
    require({k: v for k, v in environment.items() if k != "ASCEND_RT_VISIBLE_DEVICES"}
            == {k: v for k, v in expected["environment"].items() if k != "ASCEND_RT_VISIBLE_DEVICES"},
            "non-visibility environment differs")
    devices = result["physical_devices_requested"]
    require(isinstance(devices, list) and len(devices) == tp
            and all(type(d) is int and d >= 0 for d in devices) and len(set(devices)) == tp,
            "invalid requested devices")
    require(environment["ASCEND_RT_VISIBLE_DEVICES"] == ",".join(map(str, devices)),
            "visibility differs from requested devices")
    selected = cases if mode == "correctness" else {n: size for n, size in cases.items() if size in (32, 256)}
    if mode == "benchmark":
        require(sorted(selected.values()) == [32, 256], "expected one 32- and one 256-token case")
    expected_keys = {(name, repeat) for name in selected
                     for repeat in (range(1) if mode == "correctness" else range(3))}
    indexed = {}
    for run in result["runs"]:
        key = (run["case"], run["repeat"])
        require(type(run["repeat"]) is int and key in expected_keys and key not in indexed,
                "extra/duplicate case or repeat")
        length = selected[key[0]]
        require(type(run["input_tokens"]) is int and run["input_tokens"] == length
                and type(run["output_tokens_per_request"]) is int
                and run["output_tokens_per_request"] == 16, "case shape differs")
        ids = output_ids(run, concurrency=concurrency, length=length, new_tokens=16,
                         vocab=vocab, correctness=mode == "correctness")
        if mode == "benchmark":
            seconds, throughput = run["batch_seconds"], run["aggregate_output_tokens_per_second"]
            require(positive(seconds) and positive(throughput), "invalid timing/throughput")
            require(math.isclose(throughput, 16 * concurrency / seconds, rel_tol=1e-12, abs_tol=0),
                    "throughput disagrees with duration")
        indexed[key] = (run, ids)
    require(set(indexed) == expected_keys, "incomplete case/repeat grid")
    return indexed


def summarize_pair(tp1, tp2, *, cases_bytes, manifest_bytes, config_bytes, atol, benchmark=None):
    """Validate one model/concurrency pair, preserving compare_tp's strict verdict.

    Malformed evidence raises ValueError; numerical failure is returned, not
    discarded. The optional benchmark is compared only with TP2 correctness.
    """
    try:
        require(type(atol) in (int, float) and atol == 0.15, "explicit atol=0.15 required")
        hashes, vocab, cases, model = _metadata(cases_bytes, manifest_bytes, config_bytes)
        indexes = []
        for tp, result in ((1, tp1), (2, tp2)):
            indexes.append(_validate_run(result, tp1, tp=tp, mode="correctness",
                                         hashes=hashes, cases=cases, vocab=vocab))
        strict = compare(tp1, tp2, atol=atol)
        result = {"model": model, "fingerprints": hashes, "strict_tp_comparison": strict,
                  "shared_config": shared_config(tp1),
                  "runtime_versions": tp1["reproducibility"]["runtime_versions"],
                  "source_trees": {name: {"sha256": tree["sha256"], "file_count": len(tree["files"])}
                                   for name, tree in sorted(source_identity(tp1).items())},
                  "concurrency": tp1["concurrency"],
                  "limitations": ["TP agreement is not independent reference accuracy certification.",
                                  "No full-logit comparison; exact greedy IDs and selected logprobs only."]}
        if benchmark is not None:
            measured = _validate_run(benchmark, tp2, tp=2, mode="benchmark", hashes=hashes,
                                     cases=cases, vocab=vocab)
            require(benchmark["physical_devices_requested"] == tp2["physical_devices_requested"],
                    "benchmark/correctness devices differ")
            rows, checks = [], []
            for name, length in sorted(cases.items(), key=lambda item: item[1]):
                if length not in (32, 256):
                    continue
                samples = []
                for repeat in range(3):
                    run, ids = measured[(name, repeat)]
                    for request, tokens in enumerate(ids):
                        require(tokens == indexes[1][(name, 0)][1][request],
                                f"benchmark output IDs differ: {name}, repeat={repeat}, request={request}")
                        checks.append({"case": name, "repeat": repeat, "request": request,
                                       "exact_output_ids": True})
                    samples.append(run)
                seconds = [s["batch_seconds"] for s in samples]
                throughput = [s["aggregate_output_tokens_per_second"] for s in samples]
                rows.append({"tp": 2, "concurrency": tp2["concurrency"], "case": name,
                             "input_tokens": length, "output_tokens_per_request": 16, "repeats": 3,
                             "batch_seconds": seconds, "aggregate_output_tokens_per_second": throughput,
                             "median_batch_seconds": statistics.median(seconds),
                             "median_aggregate_output_tokens_per_second": statistics.median(throughput)})
            result["benchmark"] = {"repeatability_validation_passed": True,
                                   "scope": "Exact outputs match TP2 correctness, not independent accuracy.",
                                   "measurement": "Batch wall latency and aggregate output throughput; not TTFT/TPOT.",
                                   "limitations": ["Shared-server contention; three repeats, descriptive medians only."],
                                   "verified_benchmark_requests": len(checks), "output_id_checks": checks,
                                   "rows": rows}
        return result
    except MALFORMED as exc:
        raise ValueError(f"missing or malformed validation evidence: {exc}") from exc


def _load(path, inputs, role):
    path = Path(path).resolve()
    data = path.read_bytes()
    inputs.append({"role": role, "path": str(path), "sha256": sha256(data)})
    return data


def collect_hbm(run_directories):
    """Summarize exactly the selected directories, never glob or select newest."""
    directories = [Path(p).resolve() for p in run_directories]
    require(bool(directories) and len(set(directories)) == len(directories),
            "select at least one run directory, without duplicates")
    rows, inputs = [], []
    for directory in directories:
        result = json.loads(_load(directory / "result.json", inputs, "run"))
        for filename in SNAPSHOTS:
            raw = _load(directory / filename, inputs, "snapshot")
            rows.append({"run": str(directory), "snapshot": filename,
                         **owned_hbm(result, raw.decode("utf-8"))})
    return {"scope": "Only captured child PIDs, using summarize_large_run.owned_hbm; no parent or unrelated memory.",
            "units": "MB as labeled by npu-smi; no implicit GiB conversion",
            "limitations": ["Loaded, post-length_32 and post-length_256 snapshots; not peaks.",
                            "Missing scheduler PIDs mean unknown complete memory (null), never a pass or zero.",
                            "Cross-chip child allocations retained; no inferred rank mapping or whole-device totals."],
            "inputs": inputs, "rows": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="mode", required=True)
    pair = commands.add_parser("pair", help="one selected-model TP1/TP2 correctness pair")
    for flag in ("tp1", "tp2", "cases", "metadata"):
        pair.add_argument(f"--{flag}", type=Path, required=True)
    pair.add_argument("--benchmark", type=Path, help="optional TP2 benchmark run directory")
    pair.add_argument("--atol", type=float, choices=(0.15,), required=True)
    hbm = commands.add_parser("hbm", help="owned HBM for an explicit list of run directories")
    hbm.add_argument("--runs", type=Path, nargs="+", required=True)
    for command in (pair, hbm):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.mode == "hbm":
            document = collect_hbm(args.runs)
        else:
            inputs = []
            tp1 = json.loads(_load(args.tp1 / "result.json", inputs, "tp1_correctness"))
            tp2 = json.loads(_load(args.tp2 / "result.json", inputs, "tp2_correctness"))
            benchmark = (json.loads(_load(args.benchmark / "result.json", inputs, "tp2_benchmark"))
                         if args.benchmark is not None else None)
            document = summarize_pair(tp1, tp2, benchmark=benchmark, atol=args.atol,
                                      cases_bytes=_load(args.cases, inputs, "cases"),
                                      manifest_bytes=_load(args.metadata / "source-manifest.json", inputs, "manifest"),
                                      config_bytes=_load(args.metadata / "config.json", inputs, "model_config"))
            document["inputs"] = inputs
        require(str(args.output.resolve()) not in {entry["path"] for entry in document["inputs"]},
                "output must not overwrite input evidence")
        serialized = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    except (ValueError, OSError, UnicodeError) as exc:
        parser.exit(2, f"invalid evidence: {exc}\n")
    print(f"Wrote {args.output}")
    return 1 if args.mode == "pair" and not document["strict_tp_comparison"]["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
