# Homework 2 — technical completion record

**Technical work is complete for the stored assignment and recorded experimental scope.** All four required models run one sharded model across the two chips of a single910C card and have passing matched TP1/TP2 configurations. The last4B discrepancy was explained and fixed, not waived by changing the tolerance. **Nothing has been submitted.**

Confirmed members: **Zhuoxi Zeng ·2501213387** and **Steven Walxim ·2501112130**. The [report](report.md) records their respective contributions. Steven confirmed the group has three members; only the third member's details remain unconfirmed. The new [streaming latency instrumentation](docs/streaming-metrics.md) is locally implemented/tested, with real NPU validation pending. It does not change the earlier recorded technical results. See the [submission guide](../SUBMISSION.md).

## Requirement coverage

| Requirement | Evidence |
| --- | --- |
| Four separate TP2 model runs, both ranks and same-card mapping | [Latest runs](docs/final-validation.md), original rank logs, inventories and child-owned HBM; final4B and native-model regression logs |
| Matched-weight/precision/task TP1/TP2 correctness | Passing configurations below;27B has a real single-chip baseline at explicit fraction0.9 |
| Typical layer shapes, sharding/replication, state and collectives | [Parallelism analysis](docs/parallelism-plan.md) |
| Two input lengths × two concurrencies, latency/throughput/per-chip HBM; larger-model results | [0.8B matrix](results/final-matrix/benchmark-summary.json), [complete matrix HBM](results/final-matrix/hbm-summary.json), [27B benchmark](results/final-27B/validation-summary.json), [model HBM](results/final-model-hbm-summary.json) |
| Overlap-aware compute/communication analysis and one strategy experiment | [Two-rank batch1→2 experiment](docs/communication-results.md), raw archived traces and reproducible analyzer |

## Passing correctness configurations

Each compares identical weights, precision/task settings and fixed5/32/256-token inputs; each produces16 greedy output tokens. FP32 recurrent state, eager execution, no prefix reuse. Strict project criterion: exact output IDs and selected-logprob absolute error≤**0.15**; the assignment itself does not prescribe this number.

| Model | Transformer / final projection | Memory fraction | Maximum selected-logprob error | Evidence |
| --- | --- | --- | --- | --- |
| 0.8B | BF16 / native |0.12|0.0576621294|[final-source c1](results/head-fix-regression/0.8B-validation.json), [matrix c2](results/final-matrix/tp-comparison-c2.json)|
| 4B | FP16 / **explicit FP32** |0.25|**0.0059497356**|[initial](results/4B-fp32-head-initial-validation.json), [fresh confirmation](results/4B-fp32-head-confirmation-validation.json)|
| 9B | FP16 / native |0.45|0.0058279037|[final-source pair](results/head-fix-regression/9B-validation.json)|
| 27B | BF16 / native |0.9|0.0773688555|[final-source pair](results/head-fix-regression/27B-validation.json)|

For4B, supply **`--dtype float16 --lm-head-compute-dtype float32` to both TP settings**. The observer identified a final FP16 rounding tie at token10; FP32 projection before rounding fixes the complete generation. A same-source native-head control still fails, and original failures remain available. [Diagnosis, controlled comparison and costs](docs/4B-discrepancy.md).

## Verification and delivery

[Final consistency verification](results/completion-verification.json) binds all four passing pairs to the current runtime source, confirms exact 4B outputs/logprobs across fresh launches, and checks the same-source native control, patch hashes/modes and frozen HW1.

- **178 local tests passed** ([log](results/local-tests-head-fix.txt)), including24 trace-analysis tests.
- **67 CPU tensor/config tests passed** on the server:40 new final-head tests +27 cache tests ([log](results/lm-head-precision-tests.txt)). Existing logits suite:50 passed/1 skipped. Real course-server inference, not these unit tests alone, establishes the model comparisons.
- Current [incremental HW2 patch](patches/hccl-torch29.patch): **15 files /89,738bytes**, SHA256 `76071724b56e1f7922dd3fd9a4c2c5b359f3422252ac10c74daa5e5d0f2350f6`. [Reconstruction](patches/patch-verification.json): pinned base + frozen HW1 + HW2 equals **2,972 files byte/mode-exact**.
- Frozen HW1 patch remains SHA256 `e68f2f26e49d30180280adbab0bb0ebaf433246e27457788e60277621613b8c6`; previous HW2 patch versions are archived. [Reproduction instructions](README.md).

## Limits retained, not hidden

Results cover the recorded models/configurations/cases, not arbitrary-input or topology-invariant bitwise guarantees. The FP32 head costs memory and wider vocabulary communication; it is not a speed optimization. Timing is descriptive on a shared server; HBM values are snapshots, not peaks. Device profiling covers the documented short32-input/2-output windows, with unclassified gaps not identified as pure waiting. Longer vendor-parser failures and focused-suite skips remain documented. Earlier benchmark/source bindings are preserved rather than relabelled as new runs.

No unresolved technical blocker remains for these assignment deliverables. Actual platform submission is separate from technical completion; no upload has occurred.
