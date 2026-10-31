"""Collect host-side operator attribution through TokenSpeed's existing profiler API."""
import argparse
import json
import os
from pathlib import Path
from engine_validate import engine_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--physical-device", type=int, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.physical_device)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from tokenspeed.runtime.entrypoints.engine import Engine
    config = engine_config(args.model)
    config.update(enable_prefix_caching=False, disable_kvstore=True,
                  enable_output_logprobs=False)
    cases = json.loads(args.reference.read_text())["cases"]
    case = next(c for c in cases if c["name"] == "length_256")
    params = {"temperature": 0, "max_new_tokens": 8, "ignore_eos": True}
    engine = Engine(**config)
    try:
        engine.generate(input_ids=case["input_ids"], sampling_params=params)
        engine.llm.run(engine.tokenizer_manager.start_profile(
            output_dir=str(args.output_dir.resolve()), activities=["CPU"],
            with_stack=False, record_shapes=True, profile_id="hw1-host-256-8"))
        try:
            out = engine.generate(input_ids=case["input_ids"], sampling_params=params)
            assert out["output_ids"] == case["output_ids"][:8]
        finally:
            engine.stop_profile()
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
