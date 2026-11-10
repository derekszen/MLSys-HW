"""Replay HW2 streaming timing evidence; never infer metrics from batch latency."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from run_model import summarize_stream_events


def summarize(result):
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    try:
        require(result["mode"] == "streaming" and result["stream_schema_version"] == 1,
                "expected streaming schema version 1; legacy batch results are insufficient")
        require(result["status"] == "generation_completed_not_yet_compared", "run did not complete")
        require(type(result["tp"]) is int and result["tp"] in (1, 2), "invalid TP")
        concurrency = result["concurrency"]
        require(type(concurrency) is int and concurrency in (1, 2), "invalid concurrency")
        config = result["config"]
        require(config["stream_interval"] == 1 and config["enable_prefix_caching"] is False
                and config["disable_kvstore"] is True and config["enable_output_logprobs"] is False,
                "unexpected streaming/cache/logprob settings")
        grouped = {}
        require(isinstance(result["runs"], list) and bool(result["runs"]), "missing samples")
        for run in result["runs"]:
            length, repeat = run["input_tokens"], run["repeat"]
            require(type(length) is int and length in (32, 256), "unexpected input length")
            require(type(repeat) is int and 0 <= repeat < 20, "invalid repeat")
            require(type(run["output_tokens_per_request"]) is int
                    and run["output_tokens_per_request"] == 16, "expected 16 outputs")
            rows = grouped.setdefault(length, {})
            require(repeat not in rows, "duplicate sample")
            replay = summarize_stream_events(
                run["stream_events"], batch_size=concurrency, new_tokens=16,
                input_tokens=length, batch_seconds=run["batch_seconds"])
            require(replay["requests"] == run["stream_requests"], "stored timing differs from event replay")
            require(replay["outputs"] == run["outputs"], "stored outputs differ from event replay")
            require([o["output_ids"] for o in replay["outputs"]] == run["warmup_output_ids"],
                    "stream/warmup output mismatch")
            rate = run["aggregate_output_tokens_per_second"]
            require(type(rate) in (int, float) and math.isfinite(rate)
                    and math.isclose(rate, 16 * concurrency / run["batch_seconds"], rel_tol=1e-12),
                    "throughput arithmetic mismatch")
            rows[repeat] = run
        require(set(grouped) == {32, 256}, "missing input-length condition")
        count = result["requested_repeats"]
        require(type(count) is int and 1 <= count <= 20, "invalid requested repeat count")
        repeats = set(grouped[32])
        require(repeats == set(grouped[256]) == set(range(count)),
                "incomplete or noncontiguous repeat grid")
        summaries = []
        for length, rows in sorted(grouped.items()):
            requests = [request for row in rows.values() for request in row["stream_requests"]]
            metrics = {}
            for metric in ("ttft_ms", "tpot_ms", "first_chunk_ms", "last_token_ms"):
                values = [request[metric] for request in requests if request[metric] is not None]
                # Report coverage; never present a selectively observed median as a full-condition result.
                metrics[metric] = {"observed_requests": len(values), "total_requests": len(requests),
                                   "median": statistics.median(values) if len(values) == len(requests) else None}
            summaries.append({"input_tokens": length, "repeats": len(rows), "metrics": metrics,
                              "batch_seconds_median": statistics.median(r["batch_seconds"] for r in rows.values())})
        return {"passed": True, "tp": result["tp"], "concurrency": concurrency,
                "verified_requests": sum(len(rows) * concurrency for rows in grouped.values()),
                "summary": summaries,
                "scope": "Recorded event/metric consistency and warmup repeatability only; not independent model accuracy or proof of hardware execution. Client time includes scheduling, IPC and observer overhead. No pure prefill/decode/communication attribution."}
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as exc:
        raise ValueError(f"missing or malformed streaming evidence: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.result.read_bytes()
    output = summarize(json.loads(raw))
    output["input_sha256"] = hashlib.sha256(raw).hexdigest()
    output["verifier_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output["runner_sha256"] = hashlib.sha256(Path(__file__).with_name("run_model.py").read_bytes()).hexdigest()
    with args.output.open("x") as stream:
        stream.write(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"Verified {output['verified_requests']} streamed requests; wrote {args.output}")


if __name__ == "__main__":
    main()
