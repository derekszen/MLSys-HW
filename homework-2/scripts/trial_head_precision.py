"""Explicit, diagnostic-only FP32 vocabulary projection ablation.

No production files are changed. The final projection uses FP32 copies of the
existing checkpoint weights and hidden states. All transformer/cache arithmetic
is unchanged. This prototype only supports the recorded unquantized dense Qwen
TP1/TP2 configuration, and refuses other logits layouts/features.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ENV = "MLSYS_HW2_HEAD_PRECISION_TRIAL"


def install():
    import torch
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessor
    from tokenspeed.runtime.distributed.comm_ops import all_gather_single

    def fp32_logits(processor, hidden_states, lm_head, logits_metadata,
                    embedding_bias=None, plan=None, *, require_full_vocab):
        if (plan is not None or processor.dp_sampling_enabled or processor.dp_lm_head_tp
                or processor.skip_all_gather or processor.do_argmax
                or processor.logit_scale is not None or processor.final_logit_softcapping
                or embedding_bias is not None or processor.tp_size not in (1, 2)
                or lm_head.weight.dtype not in (torch.float16, torch.bfloat16)
                or lm_head.weight.ndim != 2):
            raise ValueError("FP32-head trial does not support this logits configuration")
        from tokenspeed.runtime.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
        from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod
        if not isinstance(lm_head.quant_method, (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)):
            raise ValueError("FP32-head trial requires unquantized weights")
        logits = torch.nn.functional.linear(hidden_states.float(), lm_head.weight.float())
        if processor.tp_size > 1:
            rows, width = logits.shape
            gathered = torch.empty((processor.tp_size * rows, width), dtype=logits.dtype,
                                   device=logits.device)
            all_gather_single(gathered, logits, processor.tp_group)
            logits = torch.cat(gathered.view(processor.tp_size, rows, width).unbind(0), dim=1)
        return logits[:, :processor.config.vocab_size].contiguous()

    LogitsProcessor._get_logits = fp32_logits


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--head-precision", choices=("float32",), required=True)
    extra, remaining = parser.parse_known_args()
    if "--output-dir" not in remaining:
        parser.error("explicit fresh --output-dir required")
    output = Path(remaining[remaining.index("--output-dir") + 1])
    os.environ[ENV] = extra.head_precision
    plan = {"schema": 1, "mode": "diagnostic_fp32_head_ablation",
            "head_precision": extra.head_precision,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "changes": "FP32 final vocabulary projection only; weights first loaded in model dtype",
            "not_production_implementation": True}
    plan_path = output.with_name(output.name + "-trial-plan.json")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with plan_path.open("x") as stream:
        json.dump(plan, stream, indent=2)
        stream.write("\n")
    import run_model
    sys.argv = [str(Path(run_model.__file__)), *remaining]
    try:
        run_model.main()
    finally:
        result_path = output / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_bytes())
            result["experimental_projection"] = plan
            result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        os.environ.pop(ENV, None)


if __name__ == "__mp_main__":
    if os.environ[ENV] != "float32":
        raise ValueError("unknown explicit head trial")
    install()

if __name__ == "__main__":
    main()
