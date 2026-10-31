# Homework 1 evidence index

## Final technical baseline

| Evidence | Result / interpretation |
| --- | --- |
| `integrated-all-tests.txt` | **1,057 passed**, no skips: 663 NPU cases + 394 CPU cases, five integrated suites |
| `integrated-gdn-tests-2.txt` / `integrated-conv-tests.txt` | 82 / 227 NPU-selected passes |
| `runtime-import.txt` / `integrated-gemma-tests.txt` / `integrated-mha-tests-3.txt` | 144 gated norm / 59 Gemma norm / 151 MHA passes |
| `engine-smoke.json`, `engine-smoke-6.txt` | Actual TokenSpeed checkpoint load, prefill, eight-token decode |
| `model-reference.json` | Unmodified Transformers 5.12.0 BF16 reference; exact input IDs, 16 greedy steps, selected logprobs/top5 |
| `model-validation.json`, `model-validation-run-3.txt` | Three inputs pass exact generated IDs + selected-logprob atol 0.15; A-after-B and batch-2 isolation pass |
| `benchmark.json`, `benchmark-run-2.txt` | 32/256 input × 8/16 output, three repeats each; token arrivals, exact output IDs and zero cache hits verified |
| `hbm-startup.txt`, `hbm-{32,256}-{8,16}.txt` | Raw `npu-smi` snapshots. Worker PID 2588245: 2299 MB startup / 2525 MB after workload; frontend PID 2588138: 112 MB. Other PIDs are not charged to this experiment |
| `profile/hw1-host-256-8-TP0.trace.json.gz`, `profile-summary.json` | CPU-only trace for one 256→8 request after warmup; source trace SHA256 in summary; not NPU device-time attribution |
| `profile-summary-tests.txt` | Four local trace-parser/exclusive-time unit tests pass |
| `upstream-mha-tests.txt` | Four existing upstream NPU MHA regression tests pass, separate from the 1,057 new-suite total |
| `runtime-environment.json`, `private-requirements.txt` | Final environment; replaces neither the historical base snapshot nor dependency logs |
| `patch-verification.json` | Patch applies to a clean archive of the pinned upstream commit and reproduces all 22 modified/new submodule files byte-for-byte |
| `model/source-manifest.json`, `model/config.json`, `model-download.txt` | Official ModelScope model, per-file revision/size/SHA256; large checkpoint excluded from repository |

**Limits:** greedy checks are not full-logit equivalence. The long numerical case reused 47 cached tokens; subsequent benchmark warmups/measurements also verify uncached generation. Memory snapshots are not transient peaks. Client TTFT/TPOT include scheduler, host and IPC costs. Shared-chip background processes were present; visibility is not reservation.

The host trace exporter produced ten unescaped JSON payloads inside event names. `scripts/summarize_profile.py` repairs only these name strings in memory, retaining the original compressed trace unchanged. It reports summed exclusive **CPU** operation time by thread and profiled forward spans, not pure device time. Profiling itself increases latency.

## Retained unsuccessful attempts

Earlier numbered engine, dependency, MHA, model-validation and benchmark logs remain for provenance. They include the old Triton/CANN header incompatibility; CUDA stream construction on NPU; missing Gemma dispatch; insufficient initial cache capacity; native attention layout/page-size constraints; first-call cache-flush receive-loop hang; unsupported decode top-k logprob request; and the streaming API's delta-vs-cumulative token distinction. They are **not successful evidence**. The unconstrained dependency attempt was stopped before installing CUDA Torch; the final constrained environment retains torch 2.9.0+cpu and torch_npu 2.9.0.

## Initial prototype evidence (separate scope)

| Check | Result | File |
| --- | --- | --- |
| CPU standalone state tests | 67 passed | `cpu-state-tests.txt` |
| NPU standalone state tests | 14 passed, no skips | `npu-state-tests.txt` |
| Tiny FP32 GEMM smoke | max absolute error 0; 3072 allocator peak bytes | `npu-smoke.json` |
| Initial inherited environment | Before final private dependency overlay | `environment.json` |

These older tests exercise `hw1/state_ops.py`, not TokenSpeed. Their memory numbers and pytest wall times are not model measurements. Shared CANN/package ownership warnings were retained rather than changing shared installation ownership. See [run instructions](../README.md), [operator audit](../docs/operator-audit.md), and [report](../report.md).
