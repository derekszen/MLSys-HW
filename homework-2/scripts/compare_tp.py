"""Compare paired model runs; generation completion alone is not numerical success."""
import argparse
import json
import math
from pathlib import Path

PARALLEL_FIELDS = {"world_size", "attn_tp_size", "dense_tp_size", "moe_tp_size", "lm_head_tp_size", "port"}


def compare(left, right, *, atol):
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if left["tp"] != 1 or right["tp"] != 2:
        raise ValueError("expected TP=1 then TP=2")
    for run in (left, right):
        if run["status"] != "generation_completed_not_yet_compared" or run["mode"] != "correctness":
            raise ValueError("both runs must complete correctness-mode generation")
        if any(run["config"][field] != run["tp"] for field in PARALLEL_FIELDS - {"port"}):
            raise ValueError("configured sharding/world size disagrees with claimed TP degree")
    for key in ("manifest_sha256", "config_sha256", "cases_sha256", "concurrency"):
        if left[key] != right[key]:
            raise ValueError(f"mismatched {key}")
    def shared_config(run):
        return {k: v for k, v in run["config"].items() if k not in PARALLEL_FIELDS}
    if shared_config(left) != shared_config(right):
        raise ValueError("non-parallel configuration differs")
    if not left["runs"] or len(left["runs"]) != len(right["runs"]):
        raise ValueError("missing or unequal case counts")
    rows = []
    for a, b in zip(left["runs"], right["runs"]):
        for key in ("case", "repeat", "input_tokens", "output_tokens_per_request"):
            if a[key] != b[key]:
                raise ValueError("case order/shape differs")
        if len(a["outputs"]) != left["concurrency"] or len(b["outputs"]) != right["concurrency"]:
            raise ValueError("missing request output")
        for index, (x, y) in enumerate(zip(a["outputs"], b["outputs"])):
            expected = a["output_tokens_per_request"]
            if len(x["output_ids"]) != expected or len(y["output_ids"]) != expected:
                raise ValueError("truncated generation")
            lx = x["meta_info"]["output_token_logprobs"]
            ly = y["meta_info"]["output_token_logprobs"]
            if len(lx) != expected or len(ly) != expected:
                raise ValueError("missing selected-token logprobs")
            for output, logs in ((x, lx), (y, ly)):
                if [entry[1] for entry in logs] != output["output_ids"]:
                    raise ValueError("logprob token IDs do not match generation")
                if any(type(entry[0]) not in (float, int) or not math.isfinite(entry[0]) for entry in logs):
                    raise ValueError("nonfinite/missing logprob")
            exact = x["output_ids"] == y["output_ids"]
            # After a divergent token, logprobs condition on different histories.
            error = max(abs(u[0] - v[0]) for u, v in zip(lx, ly)) if exact else None
            rows.append({"case": a["case"], "repeat": a["repeat"], "request": index,
                         "greedy_exact": exact, "max_selected_logprob_error": error,
                         "passed": exact and error <= atol})
    return {"passed": all(r["passed"] for r in rows), "selected_logprob_atol": atol,
            "scope": "greedy IDs and selected logprobs, not full logits; inspect rank/device logs separately",
            "cases": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tp1", type=Path, required=True)
    p.add_argument("--tp2", type=Path, required=True)
    p.add_argument("--atol", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = compare(json.loads(args.tp1.read_text()), json.loads(args.tp2.read_text()), atol=args.atol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
