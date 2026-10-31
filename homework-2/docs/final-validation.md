# Latest HW2 validation and performance evidence

**All four models now have passing full-generation TP1/TP2 configurations.** [4B's discrepancy](4B-discrepancy.md) was captured on the real recurrent path, explained as an FP16 rounding tie, and fixed by explicit FP32 final projection. Two fresh fixed launch pairs pass; a same-source native control still fails. [Earlier runs](run-evidence.md) remain historical evidence. See [requirements](../REQUIREMENTS.md) and [technical completion](../COMPLETION.md); nothing has been submitted.

## 1. Precision and numerical validation

All paired runs use identical checkpoint/case fingerprints, matching precision/task settings, FP32 recurrent state, eager execution, no prefix reuse, and 16 generated tokens for each 5/32/256-token input. TP dimensions and rendezvous port are the permitted config differences. No failed comparison was repaired by increasing `atol=0.15`.

| Model | Precision / memory fraction | Latest TP1/TP2 result | Evidence |
| --- | --- | --- | --- |
| 0.8B | BF16 / 0.12 | All three cases pass at concurrency 1 and 2; max selected-logprob error **0.057662129402160645** | [c1](../results/final-matrix/tp-comparison-c1.json), [c2](../results/final-matrix/tp-comparison-c2.json), [full log](../results/final-matrix-run.txt) |
| 4B | FP16 weights/activations, **FP32 final projection** /0.25 | All three cases pass in two fresh launch pairs; max selected-logprob error **0.005949735641479492** | [initial](../results/4B-fp32-head-initial-validation.json), [confirmation](../results/4B-fp32-head-confirmation-validation.json), [full log](../results/4B-fp32-head-final-run.txt) |
| 9B | FP16 / 0.45 | All three cases pass; max selected-logprob error **0.005827903747558594** | [validated pair](../results/9B-fp16-validation-summary.json), [full log](../results/9B-fp16-cachefix-correctness-run.txt) |
| 27B | BF16 / **0.9** | All three cases pass; max selected-logprob error **0.0773688554763794** | [validated pair + benchmark](../results/final-27B/validation-summary.json), [full log](../results/final-27B-run.txt) |

For 4B, earlier native FP16 and BF16 full-generation failures remain historical failures, including the narrower BF16 [same-prefix candidate pass](../results/4B-prefix-tp-comparison.json). The new result is **not** a relabelled prefix audit: the integrated FP32-head configuration passes all 16 greedy IDs in each of the three complete generations, twice. [Diagnosis](4B-discrepancy.md) shows the identical-history step10 rounded tie and its higher-precision candidate scores. FP16 alone was not sufficient; the explicit final-projection precision is part of the passing 4B configuration.

After integrating the option, default/native-head source regressions also pass for [0.8B](../results/head-fix-regression/0.8B-validation.json), [9B](../results/head-fix-regression/9B-validation.json), and [27B](../results/head-fix-regression/27B-validation.json). The performance summaries below remain bound to their original matched source/configuration runs; these newer correctness runs do not relabel those measurements as newly collected.

27B first succeeded at fraction 0.9 in [the budget experiment](../results/27B-budget90-tp-comparison.json), then repeated successfully on the final source tree. The original 0.8-budget failure and independent-reference discrepancy remain archived. A successful direct TokenSpeed TP1/TP2 baseline now exists, so the current 27B comparison does **not** depend on claiming a course-approved alternative reference.

### FP16 implementation fix

The first FP16 4B run failed before generation because Qwen GDN convolution history was hardcoded BF16. The recipe now follows activation dtype; recurrent SSM state remains independently FP32. Explicit `kv_cache_dtype="float16"` is supported in the config resolver and CLI/Python validation; upstream `auto` still means BF16. This is a cache-type correctness fix, not a cast that hides the kernel validator or a tolerance change. See [runtime contract](../../homework-1/tokenspeed/docs/qwen35-cache-precision.md).

The harness requires explicit `--dtype {bfloat16,float16}`, selects FP16 KV when requested, and now accepts optional **`--lm-head-compute-dtype float32`** for the final projection. Omission retains native semantics; no token-specific logic is added. Fraction 0.9 remains an explicit upper allocator budget, with memory guards intact. [New head+cache tests](../results/lm-head-precision-tests.txt): **67 passed** (40+27). Earlier failures remain historical, not success evidence.

## 2. Refreshed unprofiled performance matrix

Three measured repetitions after warm-up per case, 16 outputs/request. **Batch wall latency and aggregate output throughput**, not TTFT/TPOT or isolated kernel time. Shared-server contention is uncontrolled; descriptive medians only.

| Model | TP | Concurrency | Input 32: seconds / output tok/s | Input 256: seconds / output tok/s |
| --- | --- | --- | --- | --- |
| 0.8B | 1 | 1 | 1.232503 / 12.981709 | 1.911833 / 8.368932 |
| 0.8B | 1 | 2 | 1.429913 / 22.378982 | 3.040349 / 10.525109 |
| 0.8B | 2 | 1 | 1.447189 / 11.055917 | 2.135147 / 7.493628 |
| 0.8B | 2 | 2 | 1.606095 / 19.924098 | 3.264206 / 9.803304 |
| 27B | 2 | 1 | **3.638593 / 4.397304** | **6.038791 / 2.649537** |

[0.8B summary](../results/final-matrix/benchmark-summary.json) validates all **36** measured request outputs against matching correctness runs, with checkpoint/config/cases, source-tree hashes, runtime/environment and configuration checks. TP2 is slower in all four observed cells; this alone does not isolate communication as the cause.

[27B summary](../results/final-27B/validation-summary.json) validates the strict TP1/TP2 pair and all **6** TP2 benchmark outputs against its own TP2 correctness run, on the **same final source/configuration at fraction 0.9**. This is TP agreement plus repeatability, not independent full-logit accuracy certification. Do not combine it with the historical fraction-0.8 benchmark as though they were one run.

## 3. Configuration-matched owned HBM

Exact recorded child-PID matches only, excluding parent and unrelated jobs. Both scheduler PIDs appear in all listed TP2 snapshots. Values are **MB as labelled by npu-smi**, not GiB, and **snapshots, not peak usage**. Cross-chip context allocations are included. Card/chip topology is independently shown in raw inventories; rank mapping is not inferred from row order.

| Benchmark | Devices | Loaded chip0 / chip1 MB | After input 32 | After input 256 |
| --- | --- | --- | --- | --- |
| 0.8B TP1 c1 | card0 physical1 only | — / 2305 | — / 2529 | — / 2529 |
| 0.8B TP1 c2 | card0 physical1 only | — / 2305 | — / 2533 | — / 2533 |
| 0.8B TP2 c1 | card0 physical0,1 | 2400 / 2288 | 2625 / 2513 | 2627 / 2515 |
| 0.8B TP2 c2 | card0 physical0,1 | 2400 / 2288 | 2627 / 2515 | 2629 / 2517 |
| 27B TP2 c1 | card2 physical4,5 | 28166 / 28054 | 28391 / 28279 | 28391 / 28279 |

Sources: [complete new matrix HBM](../results/final-matrix/hbm-summary.json), [model HBM](../results/final-model-hbm-summary.json). New 27B TP1 on physical5 uses **53,385 MB loaded / 53,603 MB post-generation**. The successful 4B FP32-head configuration uses [post256 HBM](../results/4B-fp32-head-hbm-summary.json) **8439/8327 MB** on card0 (native-head control 6267/6155 MB); 9B's FP16 snapshot is 10841/10729 MB. [27B inventory](../results/final-27B-inventory.txt) establishes physical4/5 as the two chips of card2.

The **historical** matrix's two missing-worker cells remain unknown in [old HBM evidence](../results/owned-hbm-summary.json). The new full matrix closes the deliverable gap without filling or rewriting those old cells. No peak measurement or device reservation is claimed.

## 4. Reproduction and validation

Use the [server environment/template](../README.md#server-reproduction-templates). Explicit launch choices are in the table above; add `--lm-head-compute-dtype float32` to both 4B settings. Each correctness run has `--repeats 1`, benchmarks `--repeats 3`, port 31889, cases `homework-2/cases.json`, max model length 512 and max total tokens 16384. Runs were sequential. Keep fresh output directories and inspect device occupancy first. [4B-specific offline reproduction](4B-discrepancy.md#5-reproduce-and-test) additionally verifies the trace diagnosis and fixed pair.

Offline reproduction, from repository root:

```bash
review_dir=$(mktemp -d)
python homework-2/scripts/summarize_benchmarks.py \
  --results homework-2/results/final-matrix \
  --output "$review_dir/matrix.json"
python homework-2/scripts/summarize_validation_run.py pair \
  --tp1 homework-2/results/final-27B/27B-correctness-tp1-c1 \
  --tp2 homework-2/results/final-27B/27B-correctness-tp2-c1 \
  --benchmark homework-2/results/final-27B/27B-benchmark-tp2-c1 \
  --cases homework-2/cases.json --metadata homework-2/metadata/Qwen3.5-27B \
  --atol 0.15 --output "$review_dir/27B.json"
python homework-2/scripts/summarize_validation_run.py hbm \
  --runs homework-2/results/final-matrix/0.8B-benchmark-tp1-c1 \
         homework-2/results/final-matrix/0.8B-benchmark-tp1-c2 \
         homework-2/results/final-matrix/0.8B-benchmark-tp2-c1 \
         homework-2/results/final-matrix/0.8B-benchmark-tp2-c2 \
  --output "$review_dir/hbm.json"
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s homework-2/tests -v
```

The selected-model validator rejects source/config/provenance mismatches rather than accepting a passing comparison from another run. HBM selection uses explicit directories, never a newest-file glob. Summaries include input hashes; absolute paths are local provenance and may change on relocation.

[Latest local suite](../results/local-tests-head-fix.txt): **178 passed**. [Head/cache tests](../results/lm-head-precision-tests.txt): **67 passed**; existing logits suite **50 passed/1 skipped** in [regression log](../results/head-fix-native-regression-run.txt). Earlier real-NPU profiler tests 10 passed and compatible profile subset 29 passed/1 deselected remain their stated scope; no full upstream-suite pass is claimed. Current [patch manifest](../patches/manifest.json): **15 files / 89,738 bytes**; [reconstruction](../patches/patch-verification.json): **2,972 files byte/mode-exact**, preserving three executable source modes. Frozen HW1 is unchanged.

## Completion and limitations

Required execution, matched model comparisons, sharding, performance/HBM matrix, larger-model results and bounded communication experiment all have evidence. **The 4B technical blocker is closed** for its explicit mixed-precision configuration. No arbitrary-input or topology-invariance guarantee, isolated timing/peak-HBM claim, or historical-failure rewrite follows. Student review and an accurate contribution statement remain before submission; nothing has been submitted.
