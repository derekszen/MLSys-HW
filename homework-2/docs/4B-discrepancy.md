# 4B discrepancy: observed rounding tie and validated fix

**Resolved for the recorded homework configuration.** FP16 weights/activations with an explicit **FP32 final vocabulary projection** pass the original full-generation TP1/TP2 check for all three inputs, twice in fresh launches. Neither the test inputs, greedy rule nor `atol=0.15` changed. Historical native-precision failures remain failures.

## 1. Observe the actual recurrent path, not a fresh-prefix approximation

The read-only [diagnostic runner](../scripts/audit_decode.py) and [observer](../scripts/decode_probe.py) capture the existing model exit before sampling. They return the original output object unchanged. Workers install the observer only in this explicitly launched diagnostic process; no production source was modified for capture.

Each of two repetitions starts at the exact 32-token prompt and observes its next 15 **real incremental decode** forwards. Full-vocabulary native logits and final hidden rows are copied to CPU. Both TP2 ranks are captured, with PID ownership and source/config/checkpoint/case fingerprints. Diagnostic synchronization invalidates performance interpretation, so these are not benchmark measurements.

The first attempt mistakenly counted an in-flight warmup tail. It is retained as a failed diagnostic and **excluded**. The successful retries anchor on the exact EXTEND prompt and exclude pre-request/post-16-token work. The [offline analyzer](../scripts/analyze_decode_probe.py) verifies every input ID against the actual preceding output, mode, count, tensor hash/dtype/shape, native argmax, both ranks, and baseline outputs. All three diagnostic case outputs reproduce the original same-TP baseline IDs.

**Authoritative analysis:** [4B-decode-discrepancy-analysis.json](../results/4B-decode-discrepancy-analysis.json). Both repetitions are byte-identical for captured logits/hidden rows within each TP setting, and the two TP2 ranks agree. This is observed repeatability, not a universal deterministic guarantee.

## 2. Immediate cause at generated token 10

Both settings have generated the same first nine tokens; the current decode input is token271. They therefore score the same full token history. The two leading candidates are:

| Setting | Native FP16 logit: token20943 | Native FP16 logit: token760 | Native greedy output |
| --- | --- | --- | --- |
| TP1 | **14.296875** | 14.2890625 | 20943 |
| TP2 | **14.2890625** | **14.2890625** | 760, the lower ID in the exact tie |

At this magnitude, one FP16 spacing is **0.0078125**. For additional diagnosis, the observer computes FP64 dot products from the **same stored FP16 head weights and hidden vector** for the named candidates only:

| Setting | Dot for20943 | Dot for760 | Margin20943−760 |
| --- | --- | --- | --- |
| TP1 | 14.29482058884605 | 14.292772969027283 | **0.002047619818768** |
| TP2 | 14.29169215188449 | 14.29165749369713 | **0.000034658187360** |

The higher-precision candidate calculation prefers20943 in **both** settings. Small upstream TP differences exist; those hidden states are **not identical**. Final FP16 rounding preserves the ordering onTP1 but collapses it into an exact tie onTP2, after which the ordinary lowest-ID argmax correctly selects760. The differing token then changes all later conditioning histories.

This identifies the **immediate cause of this observed FP16 first divergence** and a targeted remedy. It does not locate every earlier floating-point difference, prove that the historical BF16 failure has the identical mechanism, or establish topology invariance. The FP64 dots are observer scalars tied to the captured implementation/provenance; head weights are not duplicated in the trace archive for offline dot recomputation.

For all common-history steps0–9 across both repetitions, the analyzer compares all248,320 vocabulary entries: maximum raw-logit error **0.01953125**, maximum full-vocabulary logprob error **0.03100232360368338** using stable FP64 normalization. Steps10–15 after the divergence are deliberately **not** compared as equal-history distributions. These descriptive diagnostics do not replace the strict full-generation test below.

## 3. Minimal explicit precision fix

Launch4B with:

```bash
--dtype float16 --lm-head-compute-dtype float32
```

The model's stored weights/activations and KV cache remainFP16; recurrent state remainsFP32. Only the final hidden→vocabulary projection widens its operands **before** multiplication, producing FP32 logits. Casting already-rounded FP16 logits afterward cannot restore the lost ordering.

Implementation:

- `runtime/utils/server_args.py`: optional explicit CLI/Python precision override, validation; omission retains native arithmetic.
- `runtime/utils/env.py`: publish the resolved option into worker configuration, including clearing an earlier override when omitted.
- `runtime/layers/logits_processor.py`: use FP32 operands with autocast disabled, then the existing gather/layout/scale/softcap/sampling path. No reassignment of stored or tied weights; no sampler bias or token-specific rule.

Quantized/incompatible heads and bitwise envelopes are refused rather than silently ignoring the override. See the [runtime contract](../../homework-1/tokenspeed/docs/lm-head-precision.md). The initial out-of-tree [ablation](../results/4B-fp32-head-trial-comparison.json) was followed by the real integrated implementation; final evidence does **not** depend on the prototype monkeypatch.

## 4. Controlled verification of the full generation

Same model/cases, FP16 transformer, FP32 SSM, fraction0.25, concurrency1, eager execution, no prefix reuse. Compare exact16 greedy IDs and selected-token logprobs at the original **0.15** tolerance. Each pair uses matching source/runtime/configuration except TP dimensions/port.

| Case | Integrated fix: greedy IDs | Maximum selected-logprob error |
| --- | --- | --- |
| Smoke, input5 | Exact16/16 | **0.005949735641479492** |
| Input32 | Exact16/16 | **0.002103090286254883** |
| Input256 | Exact16/16 | **0.0005067288875579834** |

Both [initial](../results/4B-fp32-head-initial-validation.json) and [fresh confirmation](../results/4B-fp32-head-confirmation-validation.json) pass with the same results. [Full log](../results/4B-fp32-head-final-run.txt).

A fresh **native-head control on the same patched source** still fails only input32, at the same first divergence: [control](../results/head-fix-regression/4B-native-validation.json). The override is therefore not merely benefiting from a general code update or an overwritten failure.

Other models were rerun with the option omitted on the final source: [0.8B BF16](../results/head-fix-regression/0.8B-validation.json), [9B FP16](../results/head-fix-regression/9B-validation.json), [27B BF16](../results/head-fix-regression/27B-validation.json) all pass their matched TP1/TP2 comparisons. These are per-model configurations, not a claim that enabling the option fixes every model or every prompt.

### Cost and scope

This accuracy fix is not free: per-forward FP32 weight conversion uses temporary memory and FP32 vocabulary logits double the logical gather payload relative to FP16. Matched child-owned post256 HBM snapshots on card0 change from **9581→14175MB** forTP1, and **6267/6155→8439/8327MB** forTP2. Both independent fixed runs reproduce the latter values; [fixed-run HBM](../results/4B-fp32-head-hbm-summary.json) and [native/fixed comparison snapshots](../results/4B-head-precision-hbm-comparison.json). These are process-memory snapshots including allocator effects, **not pure tensor-size attribution or peaks**. No speedup is claimed for the precision override; the homework's unprofiled performance matrices remain separate.

## 5. Reproduce and test

Offline diagnostic analysis (NumPy, no accelerator), using a new output path. The local tests used NumPy2.4.6, pinned in [requirements-analysis.txt](../requirements-analysis.txt); the existing server environment was not upgraded:

```bash
review_dir=$(mktemp -d)
python homework-2/scripts/analyze_decode_probe.py \
  --tp1 homework-2/results/4B-decode-probe-fp16-tp1-retry \
  --tp2 homework-2/results/4B-decode-probe-fp16-tp2-retry \
  --baseline-tp1 homework-2/results/4B-fp16-cachefix-correctness-tp1-c1/result.json \
  --baseline-tp2 homework-2/results/4B-fp16-cachefix-correctness-tp2-c1/result.json \
  --cases homework-2/cases.json --output "$review_dir/diagnosis.json"
python homework-2/scripts/summarize_validation_run.py pair \
  --tp1 homework-2/results/4B-fp32-head-confirmation-tp1 \
  --tp2 homework-2/results/4B-fp32-head-confirmation-tp2 \
  --cases homework-2/cases.json --metadata homework-2/metadata/Qwen3.5-4B \
  --atol 0.15 --output "$review_dir/fixed-comparison.json"
```

The analyzer prefers deterministic `.npy.gz` archives, checks the original decompressed hashes, limits decompression and never falls back from an invalid gzip. Raw arrays remain locally but are ignored by git. Successful and failed diagnostic captures together have224 archived arrays:112,422,912 raw bytes /63,280,206 compressed bytes; the failed capture is never included in the analysis.

[Tests](../results/local-tests-head-fix.txt): **178 local passes**, including24 diagnostic-analysis tests. Server CPU tensor tests: **40 new LM-head tests +27 cache tests =67 passes** ([log](../results/lm-head-precision-tests.txt)); existing logits-processor suite **50 passed,1 skipped** ([combined regression log](../results/head-fix-native-regression-run.txt)). These are focused suites, not a full upstream-suite claim.

**Conclusion:** the observed4B discrepancy is explained and the recorded full-generation check is fixed. All four homework models now have a passing matched TP1/TP2 configuration, with failures, trade-offs and measurement limits retained. Student review/contribution confirmation and actual submission remain separate from technical completion.
