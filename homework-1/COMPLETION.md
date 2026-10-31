# Homework 1 closure / Homework 2 handoff

**Technical baseline complete; report and submission materials prepared, not uploaded.** Zhuoxi Zeng (2501213387) is confirmed in the [submission guide](../SUBMISSION.md). The report includes the documented coordination role and technical results. See [report](report.md) and [reproduction instructions](README.md).

## Frozen baseline

- Upstream: `52bdb9763bb2da3d61336398072671885cc0a33e`.
- Adaptation: `patches/ascend-qwen3.5.patch`.
- Patch SHA256: `e68f2f26e49d30180280adbab0bb0ebaf433246e27457788e60277621613b8c6`.
- Clean-base application and all 22 changed/new files verified byte-for-byte: `results/patch-verification.json`.
- 1,057 integrated tests passed (663 NPU/394 CPU), plus four existing MHA regression tests.
- Three inputs × 16 greedy steps match Transformers; selected-token logprob checks and state-isolation checks pass. This is not full-logit equivalence.
- 32/256 input × 8/16 output, three repeats, retained TTFT/TPOT, memory snapshots and CPU-only profile. Shared-chip measurement limitations are explicit.

Homework 2 starts from this source baseline. Future HW2 adaptation changes must be recorded separately rather than overwriting this patch or relabeling historical HW1 measurements as new runs. The live VPN session is an operational dependency, not part of the reproducible numerical result; losing that session does not invalidate retained HW1 evidence.
