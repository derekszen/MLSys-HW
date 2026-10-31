# HW2 historical run evidence — before the final precision/budget reruns

**Historical snapshot, not the current status page.** See [latest validation](final-validation.md) for passing 9B FP16 and 27B direct TP comparisons, the complete refreshed performance/HBM matrix, the resolved 4B final-projection discrepancy, and the current 15-file patch. The numbers, failed verdicts and then-open work below describe the earlier source/configuration and remain available for traceability.

This index summarizes actual course-server runs using offline-validated evidence. Run completion, strict correctness, repeatability, and profiling completeness are separate verdicts. **HW2 remains incomplete.** The latest [access-resume check](../results/access-resume-check.txt) succeeded. Original physical 0/1 later became busy/unbalanced; the [4B failure](../results/4B-prefix-audit-run.txt) is retained. [Retry inventory](../results/prefix-retry-inventory.txt) verifies **physical 4/5 on card 2** for subsequent TP2 audits/profiles, not a reservation. Server templates and job-local HCCL settings are in [README](../README.md#server-reproduction-templates).

## 1. Generation and numerical status

All four TP2/concurrency-1 correctness runs completed **input 5/32/256 → 16 generated tokens each**, BF16 weights, FP32 recurrent state, greedy sampling, no prefix reuse. Physical devices 0/1 are chips 0/1 of card 0 per [inventory](../results/server-inventory.txt); device visibility is not a reservation. [HCCL](../results/hccl-port-range/result.json) and [project backend](../results/hccl-backend-tests.txt) passed on both ranks. Model rank logs are retained separately from result JSONs.

| Model | Retained generation | Strict comparison | Independent Transformers vs TP2 |
| --- | --- | --- | --- |
| 0.8B | TP1/2, c1/c2 complete | All three cases pass at both c1/c2; maximum selected-logprob error 0.057662129402160645 | Not a new HW2 Transformers claim; HW1 reference evidence remains frozen |
| 4B | TP1/2 c1 complete | **Fails length_32**; smoke/length_256 pass | All three cases pass; maximum selected-logprob error **0.08677434921264648** |
| 9B | TP1/2 c1 complete | **Fails length_32**; smoke/length_256 pass | All three cases pass; maximum selected-logprob error **0.06364011764526367** |
| 27B | TP2 c1 complete; TP1 failed cache budgeting | No completed TP1 baseline | **Fails length_32**; smoke/length_256 pass with errors **0.037874042987823486 / 0.019573017954826355** |

Strict TP comparisons: [0.8B c1](../results/0.8B-tp-comparison-verified-c1.json), [c2](../results/0.8B-tp-comparison-verified-c2.json), [4B](../results/4B-tp-comparison-c1.json), [9B](../results/9B-tp-comparison-c1.json). Independent comparisons: [4B](../results/4B-reference-comparison.json), [9B](../results/9B-reference-comparison.json), [27B](../results/27B-reference-comparison.json). Full generation outputs are in `results/MODEL-correctness-tp2-c1/result.json` (and corresponding TP1/c2 directories). The harness status `generation_completed_not_yet_compared` is not itself a correctness pass.

The strict criterion is exact greedy IDs plus selected-token logprob absolute error ≤0.15 when tokens/histories agree, **not full-logit equivalence**. At the first 4B divergence (token 11), the reference ranks TP2's token 2614 ahead of TP1's 1118 by **0.0625** logprob. At the first 9B divergence (token 15), reference candidates 6762 and 9845 are **tied (margin 0.0)**. These same-prefix observations support a close/tied-token explanation, not a proven root cause. No strict failure has been waived; selected logprobs from already-diverged histories must not be subtracted as like-for-like measurements. Historical independent reference JSONs lack checkpoint manifest/config fingerprints, so they do not certify identical checkpoint bytes or establish course approval.

### Completed bounded full-prefill prefix audit

Each of the **16 reference prefixes of `length_32`** is independently full-prefilled, generating one token with no cache reuse. All required candidate logprobs are compared at unchanged **atol=0.15**:

| Comparison | Verdict | Maximum candidate error |
| --- | --- | --- |
| [4B TP1 vs TP2](../results/4B-prefix-tp-comparison.json) | **Pass** | **0.12479639053344727** |
| [9B TP1 vs TP2](../results/9B-prefix-tp-comparison.json) | **Fail at generated step 8 / index 7** | **0.181732177734375** |
| [27B TP2 vs independent Transformers fresh-prefix](../results/27B-prefix-reference-comparison.json) | **Fail at step 6 / index 5** | **0.18715286254882812** |

The first [27B fresh-prefix reference attempt](../results/27B-prefix-transformers/result.json) OOMed under occupancy; the [retry](../results/27B-prefix-transformers-retry/result.json) completed when room was available. Both are retained. These are bounded top-candidate comparisons, **not full-vocabulary or recurrent-decode equivalence**, and cannot reverse earlier strict greedy failures. [Audit details](prefix-audit.md) document provenance and candidate coverage. Next is numerical precision/tie investigation of the failing prefixes, not tolerance relaxation.

### 27B capacity and retry history

[TP1 failure log](../results/27B-correctness-run.txt): `gpu_memory_utilization=0.8`; the GDN cache budget leaves **−4.23 GiB** after fixed workspaces, unable to hold a null parent plus one usable parent. This is a failure of the recorded runtime/configured budget under shared occupancy, **not proof that 27B cannot physically fit on a single chip**. Transformers subsequently ran ([reference](../results/27B-transformers-reference.json)), but its length-32 output differs from TokenSpeed TP2, so no overall accuracy pass follows.

The [first larger-model benchmark](../results/27B-benchmark-tp2-c1/result.json) failed startup with “The memory capacity is unbalanced. Some GPUs may be occupied by other processes” ([log](../results/27B-reference-benchmark-run.txt)). The [retry](../results/27B-benchmark-tp2-c1-retry/result.json) completed ([log](../results/27B-benchmark-retry-run.txt)). The guard was **not bypassed**; matching source-tree fingerprints tie the successful retry to the same TP2 correctness source. No unrelated process was killed or counted as experiment memory.

## 2. Batch latency and output throughput

Unprofiled measurements only: each case gets a warm-up, then three repeats; output length 16. Values below are descriptive medians on a shared server, without confidence intervals or contention control. They are **batch wall latency and aggregate output-token throughput**, not TTFT, TPOT, per-token latency, or isolated compute/communication timing.

### 0.8B completed matrix

[Existing fail-closed summary](../results/0.8B-benchmark-summary.json) validates 36 measured request outputs against the corresponding correctness runs (including both requests at c2), complete TP1/2 × c1/c2 × inputs 32/256 × 3 repeats.

| TP | Concurrency | Input tokens | Median batch seconds | Median output tok/s |
| --- | --- | --- | --- | --- |
| 1 | 1 | 32 | 1.230699 | 13.000740 |
| 1 | 1 | 256 | 1.974879 | 8.101762 |
| 1 | 2 | 32 | 1.418139 | 22.564790 |
| 1 | 2 | 256 | 3.002901 | 10.656360 |
| 2 | 1 | 32 | 1.468870 | 10.892730 |
| 2 | 1 | 256 | 2.209740 | 7.240670 |
| 2 | 2 | 32 | 1.705192 | 18.766220 |
| 2 | 2 | 256 | 3.699789 | 8.649142 |

TP1 median latency / TP2 median latency: c1 input 32/256 **0.837855 / 0.893715**; c2 **0.831659 / 0.811641**. TP2 was slower in these observed cells, but the matrix does not establish the cause or a communication-time fraction. Increasing concurrency improves aggregate throughput while also increasing batch latency; that alone does not establish a communication-strategy benefit.

### 27B retry: repeatable but strict reference failed

[New summary](../results/27B-benchmark-summary.json) compares `27B-benchmark-tp2-c1-retry` against **`27B-correctness-tp2-c1`**, not TP1 or the failed benchmark attempt.

| TP | Concurrency | Input tokens | Repeats | Median batch seconds | Median output tok/s |
| --- | --- | --- | --- | --- | --- |
| 2 | 1 | 32 | 3 | **3.5321749600116163** | **4.529786938965045** |
| 2 | 1 | 256 | 3 | **5.610784759977832** | **2.851651005066039** |

The validator checks archived manifest/config/cases SHA256, config-to-manifest binding, vocabulary/token shapes, all three source-tree file manifests and recomputed digests, runtime versions/environment/device identity, and configuration equality except rendezvous port. It requires exactly 32/256 × 3 repeats and **every one of the six output-ID sequences** to equal the same TP2 correctness run. Positive finite batch seconds must agree with throughput `16 / seconds`. The failed independent reference comparison is bound to that correctness file and the case-file hash; the CLI also checks the archived reference's hash.

**Repeatability validation passed; overall strict reference comparison failed; accuracy-certified performance = false.** There is intentionally no ambiguous top-level `passed: true` in this summary, no TP1 speedup, and no full-logit or TTFT/TPOT claim. Raw samples and input-file hashes are retained in JSON.

## 3. Owned HBM: exact child-PID matches only

[Machine-readable extraction](../results/owned-hbm-summary.json) selects each retained model TP2/c1 correctness run, the later 0.8B TP2/c2 correctness run, and the four 0.8B benchmark cells. Including c2 preserves the latest 0.8B correctness snapshot while c1 keeps the four-model comparison consistent. It matches **only `process_snapshot_after_load.children[].pid`** to `npu-smi` process-table rows in `hbm-loaded.txt` and `hbm-length_256.txt`. Parent/runner memory and unrelated processes are excluded. Per-row PID, card/chip, memory, raw snapshot hashes and source result hashes are retained. A child can legitimately have allocations on both chips (e.g. a 112 MB context); both matched rows count, without inferring rank from process ordering.

Units below are **MB as labeled by npu-smi**. These are loaded/post-length-256 snapshots, **not peaks**. `chip0 / chip1` below means card 0's chips, not inferred rank numbers.

| Run | Loaded matched child MB (chip0 / chip1) | Post-length-256 matched child MB (chip0 / chip1) | Scheduler-PID coverage |
| --- | --- | --- | --- |
| 0.8B correctness TP2 c1 | 2288 / 2288 | 2515 / 2515 | Both present |
| 4B correctness TP2 c1 | 6040 / 5928 | 6265 / 6153 | Both present |
| 9B correctness TP2 c1 | 10612 / 10500 | 10839 / 10727 | Both present |
| 27B correctness TP2 c1 | 28166 / 28054 | 28279 / 28279 | Both present |
| 0.8B correctness TP2 c2 (later run) | 2400 / 2288 | 2629 / 2517 | Both present |
| 0.8B benchmark TP1 c1 | **Unknown** | **Unknown** | Child scheduler PID 2675900 absent from both snapshots |
| 0.8B benchmark TP1 c2 | 2305 / — | 2533 / — | Scheduler present; TP1 uses chip0 only |
| 0.8B benchmark TP2 c1 | 2288 / 2288 | 2515 / 2515 | Both present |
| 0.8B benchmark TP2 c2 | **Partial: 112 / 2288** | **Partial: 112 / 2517** | PID 2677021 absent; only PID 2677022 matched on both chips |

For 27B correctness, complete matched-child totals are **56,220 MB loaded / 56,558 MB post-length-256**. For missing/partial benchmark snapshots, the complete total is **JSON null**, not zero; the partial 112 MB is not a replacement for the missing worker's memory. No cause for the absent rows is established. Device-wide totals, stale PIDs, or similarly named unrelated processes are never substituted. Snapshot coverage gaps remain a limitation even though benchmark output validation passed.

## 4. Bounded two-rank NPU profiling and communication

**Both-rank short captures are now validated.** The in-process adapter uses explicit `activities=["NPU"]`; input **32 → 2 output tokens**, concurrency 1 and 2, each has two owned worker device traces with compute and HCCL evidence. [Capture record](npu-profile-run.md) includes commands, bounds and vendor-parser limitations; [communication results](communication-results.md) and [JSON](../results/npu-communication-analysis.json) are authoritative for definitions, attribution, clocks and exact metrics.

- **102 collectives/request/rank** at either concurrency: 98 AllReduce + 2 AllGather + 2 Broadcast. Actual BF16 decode-sized AllReduce input doubles **2,048→4,096 bytes**, logical payload, not wire bytes.
- Envelope communication time per returned token: rank0 **13.039560→22.407875 ms (worse)**; rank1 **31.882620→5.177255 ms (better)**. One warmed batch per concurrency cannot support a general communication-time speedup or causal explanation.
- Compute-union ∩ communication-envelope-union is **zero in the selected request windows**, not a universal overlap claim. Unclassified gaps are **not pure CPU wait**; **51 post-request tail collectives/rank** are excluded from request metrics.
- Four deterministic gzip traces total **10,952,799 bytes** (190,423,566 uncompressed); ignored raw originals remain preserved. Archive-aware analyzer reproduction matches the official metrics, with selected-file provenance differences documented separately.

### Superseded capture failures and remaining limits

The original whole-application `msprof` [generation](../results/0.8B-profile-tp2-c1/result.json) completed, but [export timed out at 240 s](../results/0.8B-msprof-c1-run.txt). [Offline rank0 export](../results/msprof-rank0-export.txt) exited 0 with empty device data and host-only output. Original remote artifacts remain at `~/MLSys-HW/homework-2/results/msprof-c1/` ([inventory](../results/profile-artifact-inventory.txt)). The later [owned-PID dynamic attach](../results/0.8B-dynamic-profile-c1-run.txt) failed with vendor `--pid` reporting no valid PID; no global fallback was used. These are **historical failures**, superseded within the short-capture scope, not erased.

Long **16-token** and **256-input** exports still fail the vendor parser. Successful short captures do not certify those settings. Profile batch times, including the old **34.582075 / 36.621605 s**, are perturbed and must not replace the unprofiled benchmark matrix.

## 5. Packaging, reproducibility and open work

The then-current [archived HW2 patch](../patches/archive/hccl-npu-profile.patch) and [archived manifest](../patches/archive/hccl-npu-profile-manifest.json) included HCCL + NPU profiling: **6 files, 42,916 bytes**. Today's 15-file patch is documented in [latest validation](final-validation.md). [Verification](../results/patch-verification.json) reconstructed base `52bdb9763bb2da3d61336398072671885cc0a33e` + frozen HW1 + HW2, matching **2,968 files' bytes and git modes**. HW2 SHA256: `ea88a840112da0c173ed3fa491efa39853fab33708f583115cd5cb5681c77fcd`; frozen HW1 remains `e68f2f26e49d30180280adbab0bb0ebaf433246e27457788e60277621613b8c6`. Source files/index and HW1 patch were unchanged. The original three-file [archive](../patches/archive/README.md) is historical, not the current patch. Reconstruction verification is not additional hardware correctness.

From the repository root, without remote access or overwriting retained results:

```bash
review_dir=$(mktemp -d)
PYTHONDONTWRITEBYTECODE=1 python homework-2/scripts/analyze_npu_traces.py \
  --output "$review_dir/npu-communication-analysis.json"
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s homework-2/tests -v
```

Retained tests: **132 local passes** ([log](../results/local-tests.txt)); **10 real-NPU profiler passes** ([log](../results/npu-profiler-tests-2.txt)). The [existing-profile full attempt](../results/existing-profile-regression-tests.txt) failed on a CUDA-only test patching missing `torch._C._cuda_init` in this CPU+NPU build; the [compatible run](../results/existing-profile-regression-compatible-tests.txt) has **29 passed / 1 explicitly deselected**, not a full upstream-suite pass. Local harness tests do not certify model numerics.

**Next:** numerical precision/tie audit, prioritizing 9B index 7 and 27B index 5 while preserving earlier strict greedy failures. Bounded profiling/communication analysis is already available within its scope. Missing HBM scheduler rows remain unknown, and reference acceptance/provenance limits and student contribution/report review remain open. No overall HW2 pass is claimed.
