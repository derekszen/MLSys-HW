"""Compare the real TokenSpeed engine with saved unmodified Transformers outputs."""
import argparse
import json
import os
from pathlib import Path
import time


def engine_config(model):
    return dict(
        model=str(model.resolve()), device="npu", dtype="bfloat16",
        language_model_only=True, trust_remote_code=False,
        max_model_len=512, max_num_seqs=2, max_total_tokens=16384,
        chunked_prefill_size=128, max_prefill_tokens=128,
        gpu_memory_utilization=0.12, mamba_ssm_dtype="float32",
        sampling_backend="greedy", enforce_eager=True,
        disable_prefill_graph=True, disable_pdl=True, disable_autotune=True,
        enable_output_logprobs=True, seed=42, log_level="info", port=31887,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--physical-device", type=int, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.physical_device)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from tokenspeed.runtime.entrypoints.engine import Engine

    reference = json.loads(args.reference.read_text())
    config = engine_config(args.model)
    result = {"config": config, "physical_device": args.physical_device,
              "criteria": {"greedy_ids": "exact for all 16 steps", "selected_logprob_atol": 0.15},
              "cases": [], "state_isolation": []}
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    params = {"temperature": 0, "max_new_tokens": 16, "ignore_eos": True}
    engine = Engine(**config)
    try:
        # Generation starts AsyncLLM's receive loop; control calls before the
        # first generation can otherwise wait forever in this upstream version.
        engine.generate(input_ids=reference["cases"][0]["input_ids"],
                        sampling_params={"temperature": 0, "max_new_tokens": 1})
        for case in reference["cases"]:
            engine.flush_cache()
            start = time.perf_counter()
            output = engine.generate(input_ids=case["input_ids"], sampling_params=params,
                                     return_logprob=True)
            observed = output["meta_info"]["output_token_logprobs"]
            delta = [abs(a[0]-b) for a,b in zip(observed, case["selected_logprobs"])]
            exact = output["output_ids"] == case["output_ids"]
            row = {"name": case["name"], "input_tokens": len(case["input_ids"]),
                   "output": output, "seconds": time.perf_counter()-start,
                   "greedy_exact": exact, "max_selected_logprob_error": max(delta),
                   "passed": exact and len(observed)==16 and max(delta)<=0.15}
            result["cases"].append(row)
            print(json.dumps(row, default=str), flush=True)
            save()
        # Request A, then a different longer B above, then A again without flush.
        a, b = reference["cases"][0], reference["cases"][1]
        again = engine.generate(input_ids=a["input_ids"], sampling_params=params)
        result["state_isolation"].append({"kind": "A_after_B", "output_ids": again["output_ids"],
                                          "passed": again["output_ids"] == a["output_ids"]})
        engine.flush_cache()
        batch = engine.generate(input_ids=[a["input_ids"], b["input_ids"]], sampling_params=params)
        for expected, actual in zip((a,b),batch):
            result["state_isolation"].append({"kind": "batch2", "name": expected["name"],
                                              "output_ids": actual["output_ids"],
                                              "passed": actual["output_ids"] == expected["output_ids"]})
        result["passed"] = all(x["passed"] for x in result["cases"]+result["state_isolation"])
        save()
    finally:
        engine.shutdown()
    if not result["passed"]:
        raise SystemExit("Numerical or isolation criterion failed; inspect recorded evidence")


if __name__ == "__main__":
    main()
