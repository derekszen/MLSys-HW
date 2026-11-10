# Homework 1 — Qwen3.5-0.8B on Ascend 910C

**Single-chip technical baseline completed.** The adapted course TokenSpeed fork loads the actual checkpoint and runs text prefill, decode, and continuous inference on one Ascend chip. The [Chinese report](report.md) records **Zhuoxi Zeng (2501213387)**, **Steven Walxim (2501112130)**, and **陈子睿 (2601112045)** with their respective contributions. Steven's fresh results are in the [separate reproduction report](rebuild/report.md); the original technical evidence below remains unchanged. See the [submission guide](../SUBMISSION.md). Nothing has been submitted. [COMPLETION.md](COMPLETION.md) records the frozen baseline handed off to Homework 2.

## Verified results

- **1,057 integrated tests passed:** 663 NPU cases plus 394 CPU reference/contract cases, no skips. Covers GDN, causal convolution, gated/Gemma RMSNorm, D256 MHA and public dispatch. [Combined log](results/integrated-all-tests.txt).
- **End-to-end:** Transformers 5.12.0 reference matches all 16 greedy tokens for inputs of 5, 32 and 256 tokens. Selected-token logprob maximum absolute errors: 0.12556 / 0.04081 / 0.00222, below the configured engineering acceptance bound of 0.15. Repeated-request and batch-2 isolation checks pass. This is not full-logit equivalence. [Evidence](results/model-validation.json).
- **Performance:** 32/256 input × 8/16 generated tokens, three repetitions each, no prefix reuse. Median TTFT **237–245 ms / 979–1009 ms**, TPOT **66.9–67.7 ms/token**. Every timed output is checked against reference IDs. [Samples](results/benchmark.json).
- **Memory:** worker `npu-smi` snapshots: 2299 MB at startup, 2525 MB after workloads, plus 112 MB frontend (tool units). These are snapshots, not transient peaks. Startup Torch allocated/reserved: 1.92/2.06 GiB. Shared-device used memory is not model-only memory.
- **Host profile:** thousands of small eager operations and temporary tensors support targeting recurrent-path launch/allocation overhead first. [Summary](results/profile-summary.json); CPU-only, not NPU kernel timing or bandwidth measurement.

Physical device 1 was exposed as logical `npu:0`, reporting `Ascend910_9362`. Other jobs were present: **device visibility is not a reservation, and these are not uncontended performance measurements**. No shared environment or other jobs were modified.

## Adaptation and scope

See the [operator audit](docs/operator-audit.md) for shapes, dtypes, implementations and test coverage. New Ascend registry paths cover FP32-state GDN, causal conv and gated normalization. Gemma norm now uses a vendor-neutral facade; D256 attention uses native BSH calls instead of unsupported TND. CUDA-only auxiliary stream creation is avoided on Ascend. Existing model, loaders, scheduler/LCM cache, GEMM, RoPE, sampling and supported native attention are reused.

Baseline: BF16, FP32 recurrent state, TP=1, greedy, text-only, eager execution, prefill chunks of 128, max context 512. Graph capture, MTP/speculation, quantization, multimodal inputs, longer contexts, TP=2, exhaustive logits and optimized throughput are **not established**. Metadata `.tolist()` synchronization remains an explicit performance limitation.

## Reproduce source

The submodule pins upstream `52bdb9763bb2da3d61336398072671885cc0a33e`; the adaptation is separately retained in a [patch](patches/ascend-qwen3.5.patch) with a [hash manifest](patches/manifest.json). A dirty submodule alone would not preserve new files in the outer repository.

From the project root, on a **clean pinned submodule**:

```bash
git submodule update --init --recursive
git -C homework-1/tokenspeed apply --check ../patches/ascend-qwen3.5.patch
git -C homework-1/tokenspeed apply ../patches/ascend-qwen3.5.patch
```

Do not apply twice: the current working copy and the HW1 submission package already contain the adaptation. **Do not run the HW1 exporter against the live HW2-adapted submodule:** that would overwrite the frozen baseline with later changes. Keep the retained patch unchanged; use the separate HW2 incremental exporter only when intentionally updating HW2.

## Reproduce environment and model

Actual [runtime environment](results/runtime-environment.json), [private packages](results/private-requirements.txt), and [constraints](scripts/npu-constraints.txt) are retained. This is a private overlay over the course CANN 9.0.0 / torch 2.9.0+cpu / torch_npu 2.9.0 base, **not a portable standalone environment lock**. Triton-Ascend distribution is 3.2.1 although its module reports 3.2.0. Never let pip upgrade this stack to CUDA Torch/Triton.

```bash
# Start at this repository/package root. The retained server checkout was ~/MLSys-HW.
ROOT="$PWD"
source /usr/local/Ascend/cann-9.0.0/set_env.sh
cd "$ROOT/homework-1"
# Only if .venv does not exist:
/opt/miniconda3/envs/aics/bin/python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -c scripts/npu-constraints.txt \
  -r results/private-requirements.txt \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi

# Copy the retained per-file revisions first; do not resolve a new mutable master.
mkdir -p models/Qwen3.5-0.8B
cp results/model/source-manifest.json models/Qwen3.5-0.8B/source-manifest.json
.venv/bin/python scripts/download_model.py --repo Qwen/Qwen3.5-0.8B \
  --output models/Qwen3.5-0.8B
```

Model weights are excluded from Git and submission packages. The downloader verifies SHA256 and resumes interrupted downloads. Reproduction requires authorized course-server access; personal VPN/keyring configuration and credentials are not submission material.

## Run tests and experiments

Choose a permitted physical device explicitly; `1` is the recorded experiment's selection, not an allocation for future runs.

```bash
# Start at this repository/package root, not inside homework-1.
ROOT="$PWD"
source /usr/local/Ascend/cann-9.0.0/set_env.sh
cd "$ROOT/homework-1"
export PYTHONPATH="$PWD/tokenspeed/python:$PWD/tokenspeed/tokenspeed-kernel/python:$PWD/tokenspeed/tokenspeed-kernel-npu/python"
export NPU_PHYSICAL_DEVICE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

ASCEND_RT_VISIBLE_DEVICES="$NPU_PHYSICAL_DEVICE" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
.venv/bin/python -m pytest -q -p no:cacheprovider \
  tokenspeed/tokenspeed-kernel-npu/test/test_gdn.py \
  tokenspeed/tokenspeed-kernel-npu/test/test_causal_conv.py \
  tokenspeed/tokenspeed-kernel-npu/test/test_gated_rmsnorm.py \
  tokenspeed/tokenspeed-kernel-npu/test/test_gemma_layer.py \
  tokenspeed/tokenspeed-kernel-npu/test/test_mha_d256.py

.venv/bin/python scripts/model_reference.py --physical-device "$NPU_PHYSICAL_DEVICE" \
  --model models/Qwen3.5-0.8B --output results/model-reference.json
.venv/bin/python scripts/engine_validate.py --physical-device "$NPU_PHYSICAL_DEVICE" \
  --model models/Qwen3.5-0.8B --reference results/model-reference.json \
  --output results/model-validation.json
.venv/bin/python scripts/benchmark.py --physical-device "$NPU_PHYSICAL_DEVICE" \
  --model models/Qwen3.5-0.8B --reference results/model-reference.json \
  --output results/benchmark.json --repeats 3
```

The numerical run exercises prefix reuse (47 cached tokens in the long case); benchmark warmups and measurements independently verify uncached outputs. `flush_cache()` may report failure while L2 writeback is active, so calling it alone is not proof of zero cached tokens. The benchmark disables reuse and asserts `cached_tokens==0`.

For host profiling, run `scripts/profile_engine.py` with the same model/reference/device arguments plus `--output-dir results/profile`, then `scripts/summarize_profile.py TRACE.json.gz OUTPUT.json`. It preserves the raw trace and narrowly repairs malformed nested JSON names only during parsing. Four parser unit tests run locally without Torch.

The original standalone prototypes and their 67 CPU/14 NPU tests remain as historical groundwork, separate from the integrated evidence. [Results index](results/README.md) distinguishes successful runs from retained failed attempts. Homework 2 is delivered separately; its changes are not part of this frozen HW1 baseline.
