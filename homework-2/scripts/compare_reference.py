"""Compare local independent Transformers reference JSON with a HW2 TP2 run.

The reference schema is homework-1/scripts/model_reference.py. It has no model
manifest/config fingerprint: matching resolved paths and BF16 labels cannot prove
identical model bytes. The run records input lengths, not input IDs, so its exact
canonical input IDs are linked by the SHA256 of the supplied cases file.
Only greedy IDs and selected-token logprobs are checked, not full logits or
course approval. Optional TP1 evidence is diagnostic and never changes the verdict.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


TP_FIELDS = ("world_size", "attn_tp_size", "dense_tp_size", "moe_tp_size", "lm_head_tp_size")
NEW_TOKENS = 16


def require(condition, message):
    if not condition:
        raise ValueError(message)


def token_ids(ids, count=None):
    return (isinstance(ids, list) and bool(ids)
            and (count is None or len(ids) == count)
            and all(type(token) is int and token >= 0 for token in ids))


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def resolved_model(value):
    require(isinstance(value, str) and Path(value).is_absolute(),
            "model must be an absolute resolved path")
    return str(Path(value).resolve())


def validate_reference(reference, cases):
    require(reference["reference"] == (
        "transformers.Qwen3_5ForConditionalGeneration, unmodified eager attention"),
        "unexpected independent reference schema")
    require(reference["dtype"] == "bfloat16", "reference must use BF16")
    sampling = reference["sampling"]
    require(sampling["do_sample"] is False and sampling["ignore_eos"] is True
            and type(sampling["new_tokens"]) is int and sampling["new_tokens"] == NEW_TOKENS,
            "reference must use greedy, ignore-EOS, 16-token generation")
    require(len(reference["cases"]) == len(cases), "missing/extra reference case")
    for ref, case in zip(reference["cases"], cases):
        require(ref["name"] == case["name"], "reference case names/order differ")
        require(token_ids(ref["input_ids"]) and ref["input_ids"] == case["input_ids"],
                "reference input IDs differ from canonical cases")
        require(token_ids(ref["output_ids"], NEW_TOKENS), "reference output count/IDs invalid")
        logs = ref["selected_logprobs"]
        require(isinstance(logs, list) and len(logs) == NEW_TOKENS and all(map(finite, logs)),
                "reference selected logprobs must be finite and complete")


def validate_run(run, cases, cases_sha256, model, *, tp):
    require(run["status"] == "generation_completed_not_yet_compared"
            and run["mode"] == "correctness", "run must complete correctness generation")
    config = run["config"]
    require(type(run["tp"]) is int and run["tp"] == tp, f"expected TP={tp}")
    require(all(type(config[key]) is int and config[key] == tp for key in TP_FIELDS),
            "configured sharding/world size disagrees with claimed TP degree")
    require(all(type(config[key]) is int and config[key] == 1
                for key in ("ep_size", "pipeline_parallel_size")),
            "unexpected expert/pipeline parallel configuration")
    devices = run["physical_devices_requested"]
    require(isinstance(devices, list) and len(devices) == tp
            and all(type(device) is int and device >= 0 for device in devices)
            and len(set(devices)) == tp, "physical device count/IDs disagree with TP")
    require(config["dtype"] == "bfloat16", "run must use BF16")
    require(resolved_model(config["model"]) == model, "reference/run model path mismatch")
    require(config["sampling_backend"] == "greedy" and config["enable_output_logprobs"] is True,
            "run requires greedy sampling and selected logprobs")
    require(run["cases_sha256"] == cases_sha256, "run canonical cases fingerprint mismatch")
    concurrency = run["concurrency"]
    require(type(concurrency) is int and concurrency in (1, 2), "invalid run concurrency")
    runs = run["runs"]
    require(isinstance(runs, list) and runs and len(runs) % len(cases) == 0,
            "missing/extra run case")
    repeats = len(runs) // len(cases)
    require(1 <= repeats <= 20, "invalid repeat count")
    expected = [(case, repeat) for case in cases for repeat in range(repeats)]
    for row, (case, repeat) in zip(runs, expected):
        require(row["case"] == case["name"] and type(row["repeat"]) is int
                and row["repeat"] == repeat, "run case names/order/repeats differ")
        require(type(row["input_tokens"]) is int and row["input_tokens"] == len(case["input_ids"])
                and type(row["output_tokens_per_request"]) is int
                and row["output_tokens_per_request"] == NEW_TOKENS, "run case shape differs")
        require(isinstance(row["outputs"], list) and len(row["outputs"]) == concurrency,
                "missing/extra request output")
        for output in row["outputs"]:
            ids, meta = output["output_ids"], output["meta_info"]
            require(token_ids(ids, NEW_TOKENS), "run output count/IDs invalid")
            require(meta["prompt_tokens"] == len(case["input_ids"])
                    and meta["completion_tokens"] == NEW_TOKENS and meta["cached_tokens"] == 0,
                    "run output metadata shape/cache mismatch")
            logs = meta["output_token_logprobs"]
            require(isinstance(logs, list) and len(logs) == NEW_TOKENS,
                    "missing selected-token logprobs")
            for token, entry in zip(ids, logs):
                require(isinstance(entry, list) and len(entry) >= 2 and finite(entry[0]),
                        "run selected logprobs must be finite")
                require(type(entry[1]) is int and entry[1] == token,
                        "logprob token IDs do not match run output IDs")


def tp1_diagnostics(reference, run, tp1):
    diagnostics = []
    refs = {case["name"]: case for case in reference["cases"]}
    require(len(tp1["runs"]) == len(run["runs"]), "TP1/TP2 repeat counts differ")
    for left, right in zip(tp1["runs"], run["runs"]):
        ref = refs[right["case"]]
        for request, (a, b) in enumerate(zip(left["outputs"], right["outputs"])):
            x, y = a["output_ids"], b["output_ids"]
            step = next((i for i in range(NEW_TOKENS) if x[i] != y[i]), None)
            if step is None:
                continue
            shared = ref["output_ids"][:step] == x[:step]  # x/y match before first divergence.
            item = {"case": right["case"], "repeat": right["repeat"], "request": request,
                    "step_index": step, "token_number": step + 1,
                    "tp1_token_id": x[step], "tp2_token_id": y[step],
                    "reference_shares_preceding_history": shared}
            if shared:
                top = ref["top5"][step]
                require(token_ids(top["ids"], 5) and len(set(top["ids"])) == 5
                        and len(top["logprobs"]) == 5 and all(map(finite, top["logprobs"]))
                        and top["logprobs"] == sorted(top["logprobs"], reverse=True),
                        "invalid reference top5 diagnostic")
                item.update(reference_token_id=ref["output_ids"][step], reference_top5=top,
                            reference_top1_minus_top2_logprob_margin=(
                                top["logprobs"][0] - top["logprobs"][1]))
            else:
                item["reference_diagnostic_note"] = "top5 omitted: reference has a different preceding history"
            diagnostics.append(item)
    return diagnostics


def compare(reference, run, cases_document, *, atol, cases_sha256, tp1=None):
    """cases_sha256 must hash the exact bytes of the supplied canonical cases file."""
    require(finite(atol) and atol >= 0, "tolerance must be finite and nonnegative")
    require(type(cases_document["new_tokens"]) is int
            and cases_document["new_tokens"] == NEW_TOKENS, "canonical output count must be 16")
    cases = cases_document["cases"]
    require(isinstance(cases, list) and cases, "missing canonical cases")
    require(all(isinstance(case["name"], str) and case["name"]
                and token_ids(case["input_ids"]) for case in cases), "invalid canonical cases")
    require(len({case["name"] for case in cases}) == len(cases), "duplicate canonical case names")
    validate_reference(reference, cases)
    model = resolved_model(reference["model"])
    validate_run(run, cases, cases_sha256, model, tp=2)
    refs = {case["name"]: case for case in reference["cases"]}
    rows = []
    for row in run["runs"]:
        ref = refs[row["case"]]
        for request, output in enumerate(row["outputs"]):
            exact = ref["output_ids"] == output["output_ids"]
            # Never compare selected logprobs on different token/history sequences.
            error = max(abs(a - b[0]) for a, b in zip(
                ref["selected_logprobs"], output["meta_info"]["output_token_logprobs"])) if exact else None
            rows.append({"case": row["case"], "repeat": row["repeat"], "request": request,
                         "output_tokens": NEW_TOKENS, "greedy_exact": exact,
                         "max_selected_logprob_error": error,
                         "logprob_comparison": "same tokens and histories" if exact else "skipped: different tokens/histories",
                         "passed": exact and error <= atol})
    all_exact = all(row["greedy_exact"] for row in rows)
    result = {"passed": all(row["passed"] for row in rows), "selected_logprob_atol": atol,
              "scope": "Greedy IDs and selected-token logprobs only; not full logits or course approval. Strict TP1/TP2 verdicts are unchanged.",
              "provenance": {
                  "resolved_model_path": model, "dtype": "bfloat16", "cases_sha256": cases_sha256,
                  "run_inputs": "Run stores lengths, not full input IDs; its cases_sha256 matches the exact canonical file bytes. Reference input IDs match directly, in order.",
                  "selected_logprobs": "Reference values are positionally associated with reference output_ids by its schema; run logprob entry IDs are checked against run output_ids.",
                  "reference_manifest_fingerprint_available": False,
                  "limit": "Reference schema lacks model manifest/config fingerprints. Matching resolved model paths and BF16 labels does not establish identical checkpoint bytes or full provenance."},
              "summary": {"request_count": len(rows), "exact_token_requests": sum(row["greedy_exact"] for row in rows),
                          "tokens_checked": NEW_TOKENS * len(rows),
                          "max_selected_logprob_error": max(row["max_selected_logprob_error"] for row in rows) if all_exact else None},
              "cases": rows}
    if tp1 is not None:
        validate_run(tp1, cases, cases_sha256, model, tp=1)
        require(tp1["concurrency"] == run["concurrency"], "TP1/TP2 concurrency differs")
        for key in ("manifest_sha256", "config_sha256"):
            require(tp1[key] == run[key], f"TP1/TP2 {key} differs")
        result["tp1_diagnostics"] = tp1_diagnostics(reference, run, tp1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "run", "cases", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--tp1", type=Path, help="Explicit optional TP1 artifact for first-divergence diagnostics only")
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in ("reference", "run", "cases")}
    if args.tp1 is not None:
        paths["tp1"] = args.tp1
    require(args.output.resolve() not in {path.resolve() for path in paths.values()},
            "output must not overwrite an input artifact")
    raw = {name: path.read_bytes() for name, path in paths.items()}
    documents = {name: json.loads(data) for name, data in raw.items()}
    try:
        result = compare(documents["reference"], documents["run"], documents["cases"],
                         atol=args.atol, cases_sha256=hashlib.sha256(raw["cases"]).hexdigest(),
                         tp1=documents.get("tp1"))
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        parser.error(f"invalid comparison evidence: {exc}")
    result["inputs"] = {name: {"path": str(paths[name].resolve()), "sha256": hashlib.sha256(data).hexdigest()}
                        for name, data in raw.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
