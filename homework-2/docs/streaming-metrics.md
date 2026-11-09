# HW2 streaming latency instrumentation — Steven Walxim (2501112130)

## Contribution and current status

This is a new HW2 implementation, developed with AI assistance in Steven's
workstream on November 9, 2026. It extends the existing TP experiment runner and
adds an offline timing verifier and 25 focused tests. It does not claim authorship
of the group's TP sharding, HCCL backend or earlier hardware results.

The original HW2 runner measures batch wall latency and aggregate throughput.
Those measurements cannot reveal whether TP changes the client's wait for the
first token or the subsequent delivery rate. The new `--mode streaming` measures
these separately at concurrency 1 or 2, for either TP1 or TP2. This directly
supports the assignment's comparison of latency across TP and concurrency.

**Local implementation, synthetic tests and real Ascend TP1 streaming validation
are complete. New TP2 streaming validation is pending a free same-card pair.** Existing performance tables remain measurements of the
old non-streaming mode. No new acceleration or communication-cost result is
claimed. See [local validation](../results/steven-streaming-local-validation.json) and
[hardware verification](../results/steven-streaming-20261109/hardware-verification.json).

## Design and metrics

- Use the existing `Engine.generate(..., stream=True)` public API. The engine
  submits a batch together and tags interleaved responses with `index`; each
  index must retain a unique request ID.
- Set `stream_interval=1`, disable output logprobs and retain disabled prefix
  caching/KV store. A non-streaming warmup precedes measurements. Every streamed
  request must reproduce its corresponding warmup IDs exactly.
- Timestamp each chunk immediately on receipt using `perf_counter`, then copy
  its token delta and metadata. Preserve raw events and the runner's SHA256 so
  derived metrics can be recomputed after the run.
- **TTFT:** batch submission to first token-bearing chunk, only if that chunk
  contains exactly one token. Otherwise save `null`, with the reason
  `first_chunk_coalesced`; retain first-chunk latency separately.
- **TPOT:** `(last token arrival - first token arrival)/(N-1)`, only when the
  first token was separately observed and `N>1`. Later coalescing still permits
  this endpoint mean, but prevents a complete inter-token interval sequence.
- **Inter-token intervals:** available only if every token-bearing chunk is a
  singleton. Equal arrival timestamps are valid client observations. Repeated
  token IDs represent new positions and are never deduplicated.
- Empty finish frames do not move the last-token timestamp. Completion time and
  whole-stream batch time are recorded separately.
- Per-condition medians of TTFT/TPOT require complete observation coverage;
  otherwise the verifier emits a null median and observed/total request counts.

These are **client-observed latency metrics**, including dispatch, scheduling,
IPC and observer overhead. TTFT is not isolated prefill compute time, TPOT is not
isolated decode kernel time, and their TP differences are not pure HCCL time.
Use the existing device-trace analysis for compute/communication attribution.
Streaming can change scheduling and observation overhead, so compare matched
streaming TP1/TP2 runs rather than mixing them with old non-streaming benchmarks.

## Code and verification boundaries

- [run_model.py](../scripts/run_model.py): `summarize_stream_events` validates
  event order, index/identity, token deltas, cumulative metadata counts, cache
  status and length termination; `measure_stream_batch` captures the stream;
  the CLI persists raw events, per-request metrics, warmup IDs and repeat count.
- [summarize_streaming.py](../scripts/summarize_streaming.py): recomputes every
  saved timing and output, checks throughput arithmetic and warmup agreement,
  requires both input lengths and every requested repeat, and refuses to
  overwrite an existing output. It rejects old batch-only evidence.
- [test_streaming_metrics.py](../tests/test_streaming_metrics.py): synthetic
  clock/engine tests, including the complete CLI wiring, reused producer buffers,
  stream exceptions, coalescing, missing requests and deliberate corruptions.

The offline verifier establishes internal consistency and repeatability against
the recorded warmup. It cannot authenticate hardware execution, certify model
accuracy, or prove TP1/TP2 agreement by itself. Retain real launch logs, model and
runtime fingerprints, rank/device evidence, and matching correctness checks.

## Local checks

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s homework-2/tests -p test_streaming_metrics.py -v
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s homework-2/tests -v
```

## TP2 follow-up once a free same-card pair is available

Use the existing [server setup and occupancy checks](../README.md#server-reproduction-templates).
The runtime must include the verified HW2 adaptation; do not overwrite Steven's
separate HW1 reproduction checkout or reapply patches to an already-adapted tree.
Select a permitted same-card pair and run sequentially in new output directories.

For 0.8B, keep model weights, BF16, FP32 state, memory fraction 0.12, cases,
output length 16 and runtime revision fixed. Test the four `(TP, concurrency)`
cells `(1,1), (1,2), (2,1), (2,2)`; each run covers input lengths 32 and 256.
Start with one repeat as a smoke test, then use three repeats in fresh directories.
Compare fresh correctness outputs before interpreting latency. Existing shared
server contention still limits any causal claim or speedup conclusion.

```bash
# After the README's environment setup; choose TP/C/DEVICES from the four cells.
# OUT must be a fresh directory; e.g. runs/hw2-streaming-YYYYMMDD/tp2-c2-smoke.
: "${TP:?select 1 or 2}" "${C:?select 1 or 2}"
: "${DEVICES:?select permitted physical device IDs}" "${OUT:?fresh result directory}"
mkdir -p "$(dirname "$OUT")"
timeout --signal=TERM --kill-after=20s 600s "$PY" homework-2/scripts/run_model.py \
  --tp "$TP" --physical-devices "$DEVICES" \
  --model homework-1/models/Qwen3.5-0.8B --dtype bfloat16 \
  --cases homework-2/cases.json --output-dir "$OUT" \
  --memory-fraction 0.12 --concurrency "$C" --mode streaming \
  --port 31889 --repeats 1 > "$OUT.log" 2>&1
"$PY" homework-2/scripts/summarize_streaming.py \
  --result "$OUT/result.json" --output "$OUT/streaming-summary.json"
```

TP1 hardware validation now covers 18 measured streaming requests plus two smoke
requests, with exact reference IDs; nine fresh correctness requests pass the
selected-logprob tolerance. TP2 remains untested for this new streaming mode.
The new contribution is implemented instrumentation with real TP1 validation,
not a new TP2 speedup or communication measurement. The third member's identity
remains for the group to complete; Steven confirmed there are three members.

## Recheck the recorded hardware evidence locally

All five jobs finished with exit code 0 and released chip 1 and engine port 31891.
Server-side synthetic tests also passed all 25 cases. The fresh runtime source
fingerprints match the deployed 2,972-file HW2 source. Code and model configuration
match between each correctness/streaming pair, except the intentional streaming
interval/logprob controls. Raw output events, job status and lossless compressed
logs are retained with checksums.

From the repository root, using fresh output paths:

```bash
.venv/bin/python homework-2/results/steven-streaming-20261109/verify_hardware.py \
  --output /tmp/hw2-streaming-hardware-recheck.json
.venv/bin/python homework-2/scripts/summarize_streaming.py \
  --result homework-2/results/steven-streaming-20261109/tp1-c2-streaming/result.json \
  --output /tmp/hw2-streaming-c2-recheck.json
```

The hardware checker needs the same adapted source tree to reproduce source
fingerprints. Its reference is Steven's retained November 6 Transformers run;
no fresh Transformers run or full-logit certification is claimed. The three
repetitions are descriptive shared-server measurements, not confidence intervals.
