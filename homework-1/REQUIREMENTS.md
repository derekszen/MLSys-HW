# Homework 1 — original requirements and submission evidence

Authority: [the original assignment](assignment.md). This is an evidence crosswalk, not an invented scoring rubric or a guarantee of full marks.

| Explicit requirement | Report / evidence | Status |
| --- | --- | --- |
| Single 910C chip, Qwen3.5-0.8B, environment/operators/dependencies and launch method | [Report §2](report.md#2-环境固定模型与运行配置), [runtime environment](results/runtime-environment.json), [NPU smoke](results/npu-smoke.json), [README](README.md) | Demonstrated in the recorded course environment; current reproduction paths are checkout-relative. |
| Weight loading, Prefill, Decode, GDN/conv/state; main computation on NPU; reused versus modified components | Report §3 and Appendix A; [operator audit](docs/operator-audit.md); frozen [patch](patches/ascend-qwen3.5.patch) | Source path, logical shapes/dtypes, layouts, state ownership and host metadata synchronization documented. |
| Layered reference checks, typical/boundary cases, multistep state and request isolation, multiple input lengths and stated numerical criteria | Report §4; [1,057-test log](results/integrated-all-tests.txt), [model reference](results/model-reference.json), [model validation](results/model-validation.json) | Independent FP64 operator oracles plus same-NPU Transformers reference; 5/32/256-input ×16-output exact IDs and configured selected-logprob bound. Small/boundary unit geometry is distinguished from model-shaped E2E coverage. |
| Two input lengths × two generation lengths; TTFT, decode time and HBM; phase distinction and evidence-backed bottleneck | Report §5–6; [benchmark](results/benchmark.json), [host profile](results/profile-summary.json), retained HBM snapshots | 32/256 input ×8/16 output, three repeats; client TTFT/TPOT and host-only profile explicitly labelled, snapshots not peaks. |
| Report starts with each member's main contribution | Report opening | The opening table records the student's name, ID and documented coordination role. |
| Submit stage report to teaching platform | [Submission guide](../SUBMISSION.md) | PDF and separate code/evidence package prepared; student upload and receipt check remain. |

Full-logit equivalence, optimization speedup, isolated-device benchmarking, instantaneous peak HBM and a device-kernel timing breakdown are not explicit requirements in the stored assignment. Their absence limits the claims; it is not concealed or described as completed work.

The frozen HW1 patch remains unchanged. Later Homework 2 changes and results are delivered separately.
