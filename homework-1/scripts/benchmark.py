"""Warm single-request streaming baseline; caches disabled, exact generated counts."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from engine_validate import engine_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--physical-device", type=int, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, required=True)
    args = p.parse_args()
    if args.repeats < 1:
        p.error("--repeats must be positive")
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.physical_device)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from tokenspeed.runtime.entrypoints.engine import Engine

    reference = json.loads(args.reference.read_text())
    cases = [c for c in reference["cases"] if c["name"].startswith("length_")]
    config = engine_config(args.model)
    config.update(enable_prefix_caching=False, disable_kvstore=True,
                  enable_output_logprobs=False, stream_interval=1)
    result = {"config": config, "physical_device": args.physical_device,
              "method": "perf_counter at client; TTFT=start to first token; TPOT=(last-first)/(N-1); includes scheduling/IPC, not pure kernel time",
              "concurrency": 1, "repeats": args.repeats, "samples": [], "summary": []}
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    def snapshot(name):
        proc = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=20)
        args.output.with_name(f"hbm-{name}.txt").write_text(proc.stdout + proc.stderr)
    engine = Engine(**config)
    try:
        snapshot("startup")
        for case in cases:
            # Compile both prefill/chunked-prefix and decode shapes before timing.
            warm = engine.generate(input_ids=case["input_ids"], sampling_params={
                "temperature": 0, "max_new_tokens": 16, "ignore_eos": True})
            assert warm["output_ids"] == case["output_ids"], "uncached warmup reference mismatch"
        for case in cases:
            for count in (8, 16):
                for repeat in range(args.repeats):
                    start = time.perf_counter()
                    arrivals = []
                    last_count = 0
                    output_ids = []
                    final = None
                    for chunk in engine.generate(input_ids=case["input_ids"], sampling_params={
                        "temperature": 0, "max_new_tokens": count, "ignore_eos": True}, stream=True):
                        now = time.perf_counter()
                        # Inline-detokenizer streaming emits token deltas.
                        output_ids.extend(chunk["output_ids"])
                        n = len(output_ids)
                        if n > last_count:
                            arrivals.append({"tokens": n, "seconds": now-start})
                            last_count = n
                        final = chunk
                    assert final is not None and last_count == count
                    assert output_ids == case["output_ids"][:count]
                    assert final["meta_info"]["cached_tokens"] == 0
                    assert arrivals[0]["tokens"] == 1, "TTFT unavailable: first chunk coalesced"
                    row = {"input_tokens": len(case["input_ids"]), "output_tokens": count,
                           "repeat": repeat, "ttft_ms": arrivals[0]["seconds"]*1000,
                           "tpot_ms": (arrivals[-1]["seconds"]-arrivals[0]["seconds"])*1000/(count-1),
                           "total_ms": (time.perf_counter()-start)*1000,
                           "arrivals": arrivals, "output_ids": output_ids,
                           "cached_tokens": final["meta_info"]["cached_tokens"]}
                    result["samples"].append(row)
                    save()
                snapshot(f"{len(case['input_ids'])}-{count}")
                rows = result["samples"][-args.repeats:]
                summary = {"input_tokens": len(case["input_ids"]), "output_tokens": count}
                for metric in ("ttft_ms", "tpot_ms", "total_ms"):
                    summary[metric] = {"median": statistics.median(r[metric] for r in rows),
                                       "min": min(r[metric] for r in rows),
                                       "max": max(r[metric] for r in rows)}
                result["summary"].append(summary)
                print(json.dumps(summary), flush=True)
                save()
        result["passed"] = True
        save()
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
