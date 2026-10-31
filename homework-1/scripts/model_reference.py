"""Independent Transformers reference; no TokenSpeed imports or patched kernels."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--physical-device", type=int, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.physical_device)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    import transformers

    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=True,
    ).to("npu:0").eval()
    source = tokenizer.encode(
        "Machine learning systems combine mathematical models with efficient hardware. "
        "A reliable experiment checks numerical accuracy before measuring performance. " * 40,
        add_special_tokens=False,
    )
    cases = [("smoke", tokenizer.encode("The capital of France is", add_special_tokens=False))]
    cases += [(f"length_{n}", source[:n]) for n in (32, 256)]
    result = {
        "reference": "transformers.Qwen3_5ForConditionalGeneration, unmodified eager attention",
        "transformers_version": transformers.__version__, "dtype": "bfloat16",
        "physical_device": args.physical_device, "model": str(args.model.resolve()),
        "sampling": {"do_sample": False, "new_tokens": 16, "ignore_eos": True},
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, ids in cases:
        x = torch.tensor([ids], device="npu:0", dtype=torch.long)
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                input_ids=x, attention_mask=torch.ones_like(x), do_sample=False,
                min_new_tokens=16, max_new_tokens=16, eos_token_id=None,
                pad_token_id=tokenizer.pad_token_id or 0,
                return_dict_in_generate=True, output_scores=True,
            )
        torch.npu.synchronize()
        output_ids = generated.sequences[0, len(ids):].tolist()
        logprobs = [score.float().log_softmax(-1)[0] for score in generated.scores]
        case = {"name": name, "input_ids": ids, "output_ids": output_ids,
                "text": tokenizer.decode(output_ids),
                "seconds": time.perf_counter() - start,
                "selected_logprobs": [v[t].item() for v,t in zip(logprobs,output_ids)],
                "top5": [{"ids": v.topk(5).indices.tolist(), "logprobs": v.topk(5).values.tolist()} for v in logprobs]}
        result["cases"].append(case)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(case), flush=True)
    result["peak_allocated_bytes"] = torch.npu.max_memory_allocated()
    result["peak_reserved_bytes"] = torch.npu.max_memory_reserved()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
