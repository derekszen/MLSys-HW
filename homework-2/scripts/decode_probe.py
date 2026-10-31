"""Opt-in diagnostic-only observation of the real recurrent decode path.

Installed only by audit_decode.py's spawned workers. No production source is
edited. Native logits and returned objects are never replaced by this observer.
CPU copies synchronize execution, so diagnostic timings are not benchmarks.
"""
import hashlib
import json
import math
import os
from pathlib import Path


def write_json_new(path, document):
    with Path(path).open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")


def install(plan_path):
    import torch
    from tokenspeed.runtime.models.base.causal_lm import BaseCausalLM

    plan_bytes = Path(plan_path).read_bytes()
    plan = json.loads(plan_bytes)
    root = Path(plan["output_dir"])
    gate = root / "capture-gate.json"
    candidates = plan["candidates"]
    counts = {}
    original = BaseCausalLM.exit_logits

    def observe(model, input_ids, hidden_states, aux_hidden_states, ctx):
        output = original(model, input_ids, hidden_states, aux_hidden_states, ctx)
        if not gate.exists():
            return output
        label = json.loads(gate.read_text())["label"]
        if label not in plan["labels"]:
            raise ValueError("unexpected diagnostic request label")
        logits = output.next_token_logits
        if ctx.bs != 1 or logits.shape != (1, model.logits_processor.config.vocab_size):
            raise ValueError("probe requires one full-vocabulary sampled row")
        if model.mapping.pp_size != 1:
            raise ValueError("probe does not support pipeline parallelism")
        rank = model.mapping.rank
        key = (label, rank)
        step = counts.get(key, 0)
        # Overlapped execution can finish an old warmup decode after the
        # frontend returned. Start only at this request's exact prompt, and
        # exclude speculative tail work beyond its 16 returned output tokens.
        if step == 0:
            if ctx.num_extends != 1:
                return output
            if input_ids.detach().cpu().reshape(-1).tolist() != plan["capture_input_ids"]:
                raise ValueError("diagnostic prefill does not match canonical input")
        elif step >= 16:
            return output
        elif ctx.num_extends != 0:
            raise ValueError("unexpected second prefill inside diagnostic request")
        counts[key] = step + 1
        folder = root / f"{label}-rank{rank}"
        folder.mkdir(exist_ok=True)
        raw = logits.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not torch.isfinite(raw).all():
            raise ValueError("nonfinite captured logits")
        # One non-chunked prompt followed by real one-token recurrent decodes.
        # The final prompt row is the only sampled row for this batch size.
        hidden = hidden_states[-1:].detach().to(device="cpu", dtype=torch.float32)
        values, indices = torch.topk(raw[0], 10)
        maximum = raw.max().item()
        ties = torch.nonzero(raw[0] == maximum).flatten().tolist()
        logprobs = torch.log_softmax(raw[0], dim=-1)
        shard = model.lm_head.shard_indices
        high_precision = {}
        # FP64 dot products are diagnostic only: evaluate the very same stored
        # weight/activation values without final FP16/BF16 logit rounding.
        for token in candidates:
            if shard.org_vocab_start_index <= token < shard.org_vocab_end_index:
                local = token - shard.padded_org_vocab_start_index
                weight = model.lm_head.weight[local].detach().cpu().float()
                high_precision[str(token)] = math.fsum(
                    float(a) * float(b) for a, b in zip(hidden[0].tolist(), weight.tolist())
                )
        stem = f"step{step:02d}"
        tensor_files = {}
        for name, tensor in (("logits", raw), ("hidden", hidden)):
            path = folder / f"{stem}-{name}.npy"
            with path.open("xb") as stream:
                import numpy as np
                np.save(stream, tensor.numpy(), allow_pickle=False)
            tensor_files[name] = {"path": path.name,
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "shape": list(tensor.shape), "storage_dtype": "float32"}
        write_json_new(folder / f"{stem}.json", {
            "schema": 1, "label": label, "step_index": step, "rank": rank,
            "pid": os.getpid(), "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "forward_mode": ctx.forward_mode.name,
            "input_ids": input_ids.detach().cpu().reshape(-1).tolist(),
            "num_extends": ctx.num_extends, "input_num_tokens": ctx.input_num_tokens,
            "native_logits_dtype": str(logits.dtype), "hidden_dtype": str(hidden_states.dtype),
            "top10": [{"token": int(t), "logit": float(v), "logprob": float(logprobs[t])}
                      for t, v in zip(indices.tolist(), values.tolist())],
            "max_tied_ids": ties,
            "candidates": {str(t): {"logit": float(raw[0, t]), "logprob": float(logprobs[t])}
                           for t in candidates},
            "candidate_dot_fp64": high_precision,
            "lm_head_shard": {"start": shard.org_vocab_start_index, "end": shard.org_vocab_end_index},
            "tensors": tensor_files,
            "observer": "unmodified native output; CPU copies and FP64 candidate dots only",
        })
        return output

    BaseCausalLM.exit_logits = observe
