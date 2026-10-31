"""Bounded single-chip TokenSpeed Qwen3.5 bring-up (not a benchmark)."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-device", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.physical_device)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from tokenspeed.runtime.entrypoints.engine import Engine

    config = dict(
        model=str(args.model.resolve()), device="npu", dtype="bfloat16",
        language_model_only=True, trust_remote_code=False,
        max_model_len=512, max_num_seqs=2, max_total_tokens=16384,
        chunked_prefill_size=128, max_prefill_tokens=128,
        gpu_memory_utilization=0.12, mamba_ssm_dtype="float32",
        sampling_backend="greedy", enforce_eager=True,
        disable_prefill_graph=True, disable_pdl=True, disable_autotune=True,
        seed=42, log_level="info", port=31887,
    )
    start = time.perf_counter()
    engine = Engine(**config)
    try:
        loaded = time.perf_counter()
        result = engine.generate(
            prompt="The capital of France is",
            sampling_params={"temperature": 0, "max_new_tokens": 8},
        )
        payload = {"config": config, "physical_device": args.physical_device,
                   "load_seconds": loaded - start,
                   "request_seconds": time.perf_counter() - loaded,
                   "result": result, "scope": "bring-up only; no reference comparison"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        print(json.dumps(payload, indent=2, default=str), flush=True)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
