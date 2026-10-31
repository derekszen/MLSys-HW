# Bounded in-process NPU profile — actual server evidence

**Explicit `activities=["NPU"]` now works in both scheduler workers.** The retained 0.8B TP2 experiment captures input **32 → 2 returned output tokens**, concurrency **1 and 2**, with real device computation and HCCL events on both ranks. [Communication results](communication-results.md) and [analysis JSON](../results/npu-communication-analysis.json) define the exact validated scope; this is not benchmark latency or full-model numerical certification.

## Actual runs and historical failures

The [access-resume check](../results/access-resume-check.txt) succeeded. Original physical 0/1 later became busy/unbalanced; the [4B failure](../results/4B-prefix-audit-run.txt) is preserved. [Fresh topology/occupancy inventory](../results/prefix-retry-inventory.txt) established **physical 4/5 on card 2**, not adjacent-ID guesswork or exclusive reservation. The [profile inventory](../results/profile-success-inventory.txt) and per-run HBM/owned-child metadata corroborate the selected pair.

| Route / window | Actual outcome |
| --- | --- |
| Whole-application `msprof`, input 32/256 → 16 | [Export timed out at 240 s](../results/0.8B-msprof-c1-run.txt); [rank0 offline export](../results/msprof-rank0-export.txt) had empty device data / host-only output |
| Owned-worker dynamic `msprof --pid` | [Failed](../results/0.8B-dynamic-profile-c1-run.txt): no valid PID despite owned workers; collector exit 255; **no global fallback** |
| In-process NPU, long 16-token window / 256-input export | [Matrix attempt](../results/0.8B-npu-profile-matrix-run.txt) and [short-window attempt including 256](../results/0.8B-npu-short-profile-run.txt) retain vendor `CANNTimelineParser` / `RelationParser` / `TraceViewParser` failures |
| In-process NPU, input 32 → 2, c1 and c2 | **Both ranks captured successfully:** [c1 result](../results/0.8B-npu-batch-experiment-c1/result.json), [c2 result](../results/0.8B-npu-batch-experiment-c2/result.json); subsequent offline analysis validates ownership, device/HCCL lanes and request clipping |

The adapter is documented in [TokenSpeed NPU profiling](../../homework-1/tokenspeed/docs/npu-profile.md) and packaged in the current [HW2 patch](../patches/manifest.json). NPU selection is explicit and lazily imports the vendor profiler; omitting activities retains the existing CPU/GPU route. Level1 captures HCCL events without hardware metrics. A nonempty parseable trace is necessary, but only the separate analyzer establishes usable attribution and timeline evidence. Optional CPU activity is not a promise of complete all-thread host correlation.

## Reproduction template

**Collection and the later numerical precision/tie audit are complete:** the [4B fix and full-generation validation](4B-discrepancy.md) close that former next action. This template is for optional reproduction, not an outstanding requirement. If a new short capture is needed, use the prepared course environment and `PYTHONPATH` from [README](../README.md#server-reproduction-templates), fresh inventory, a permitted same-card pair, and sequential runs. Do not kill other jobs, bypass memory balance or reuse retained directories.

```bash
: "${PY:?course NPU Python, e.g. homework-1/.venv/bin/python}"
: "${PAIR:?permitted same-card physical IDs after fresh npu-smi (retained pair was 4,5)}"
export HCCL_NPU_SOCKET_PORT_RANGE=41000-41031
npu-smi info
# Fresh root; each runner output directory below must not already exist.
PROFILE_ROOT=$(mktemp -d "$PWD/homework-2/results/npu-review-XXXXXX")
for C in 1 2; do
  timeout --signal=TERM --kill-after=20s 300s "$PY" \
    homework-2/scripts/profile_engine_npu.py \
    --model homework-1/models/Qwen3.5-0.8B --cases homework-2/cases.json \
    --physical-devices "$PAIR" --memory-fraction 0.12 --concurrency "$C" \
    --new-tokens 2 --input-lengths 32 --port 31889 \
    --output-dir "$PROFILE_ROOT/c$C" > "$PROFILE_ROOT/c$C.log" 2>&1 || exit $?
done
```

The runner fixes eager BF16 TP2, disables prefix reuse, warms the same batch, then records **one** profiled batch per concurrency. Every returned sequence must match warmup; retained IDs `[9064, 11988]` also equal the first two canonical length-32 output IDs. It stops profiling and checks each rank's exported JSON for actual device compute. The outer timeout bounds each invocation (TERM at 300 s, KILL after another 20 s); the runner does not itself impose an aggregate disk quota. Keep this two-token/32-input window: larger exports are not validated. Native driver hangs or abrupt termination can still prevent normal cleanup; never compensate with broad process killing.

## Communication verdict and archives

Per rank/request, c1 and c2 both have **102 collectives**; actual BF16 decode-sized AllReduce input is **2,048→4,096 bytes**, not measured wire traffic. Envelope communication ms/returned-output-token worsens on rank0 and improves on rank1, so **no general communication timing speedup** is established. Selected request-window compute/envelope union overlap is zero, not universal zero overlap; unclassified gaps are not pure CPU wait. **51 post-request tail ops/rank are excluded.** One warmed batch at each concurrency provides no uncertainty estimate or causal attribution.

Four deterministic `.trace.json.gz` archives total **10,952,799 bytes (10.95 MB)**; ignored uncompressed originals are preserved. The archive-aware analyzer reproduces the official metrics without changing retained results:

```bash
review_dir=$(mktemp -d)
PYTHONDONTWRITEBYTECODE=1 python homework-2/scripts/analyze_npu_traces.py \
  --output "$review_dir/npu-communication-analysis.json"
```

Input/decompression is capped at **80,000,000 bytes and 400,000 events per trace**; invalid preferred gzip fails closed, with no raw fallback. See [archive/provenance details](communication-results.md#raw-and-archived-trace-formats).

## Verification history

- **10 real-NPU profiler tests pass:** [npu-profiler-tests-2.txt](../results/npu-profiler-tests-2.txt), including actual work from a pre-existing worker thread, bounded capture and export-failure/retry cases.
- The [full existing-profile attempt](../results/existing-profile-regression-tests.txt) fails because a CUDA-only test patches missing `torch._C._cuda_init` on the CPU+NPU build. The [compatible run](../results/existing-profile-regression-compatible-tests.txt) reports **29 passed, 1 explicitly deselected** — **not a full upstream-suite pass**.
- **Historical local run: 132 unit tests passed:** [local-tests.txt](../results/local-tests.txt). The later [178-test HW2 suite](../results/local-tests-head-fix.txt) includes the completed numerical-diagnostic analysis. These local tests validate harness/analysis behavior; actual NPU generation evidence is separate.

The prefix failures above remain historical evidence. [Later validation](final-validation.md) passes 9B FP16 and 27B BF16 direct TP1/TP2 checks and completes the refreshed matrix HBM. The later [4B diagnosis and FP32 final-projection fix](4B-discrepancy.md) also pass full-generation checks, closing the technical blocker. Student contribution/report review remains pending. None of these updates extends this profile's short-window scope.
