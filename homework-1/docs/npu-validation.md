# Bounded Ascend NPU validation

**These checks exercise standalone `hw1/state_ops.py` prototypes only.** They do
not integrate TokenSpeed, touch its submodule, load model weights, test a full
model or serving stack, or establish a throughput/latency benchmark. A successful
matmul smoke check is not a state-ops correctness pass; neither establishes
full-model correctness. See [state-ops.md](state-ops.md) for the prototype contracts
and CPU mathematical tests.

## Parent-owned execution

Run from `homework-1` in a parent-provisioned environment with matching PyTorch,
`torch_npu`, Ascend runtime/drivers, and pytest. No dependencies, large tensors,
models, remote access, or server runs are provisioned by these files. The parent
owns physical-device selection and server execution. Use the environment's Python
in place of `python` below if needed.

Set `NPU_PHYSICAL_DEVICE` explicitly to the physical device index assigned by the
parent. There is deliberately no default device. For example, after assigning
that variable in the shell:

```sh
: "${NPU_PHYSICAL_DEVICE:?set the parent-assigned physical NPU index}"
PYTHONDONTWRITEBYTECODE=1 python -B scripts/npu_smoke.py \
  --physical-device "$NPU_PHYSICAL_DEVICE"

ASCEND_RT_VISIBLE_DEVICES="$NPU_PHYSICAL_DEVICE" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python -B -m pytest -q -rs -p no:cacheprovider tests/test_state_ops_npu.py
```

The smoke script requires exactly one nonnegative integer `--physical-device`,
overrides `ASCEND_RT_VISIBLE_DEVICES` **before importing either `torch` or
`torch_npu`**, and rejects a runtime reporting anything other than one visible
NPU. The selected physical device is addressed as logical `npu:0`. The pytest
command sets visibility before Python starts; its fixture also requires exactly
one visible NPU. Visibility is not a reservation or concurrency lock: the parent
must select an appropriate device. A runtime that does not honor visibility must
be corrected by the parent, not worked around by guessing another logical index.

## Smoke result

`scripts/npu_smoke.py` constructs one deterministic FP32 tensor:
`x = arange(256).reshape(16,16) / 256`. It computes `x @ x.T` on CPU and `npu:0`,
synchronizes, and prints a JSON object containing:

- Selected physical/logical device and device name.
- PyTorch and `torch_npu` versions, operation, and comparison tolerances.
- Maximum absolute error, peak allocated bytes, and a `passed` boolean.

The comparison uses `atol=rtol=1e-5`; a numerical failure returns status 1.
Missing frameworks, unavailable hardware, and visibility/configuration failures
raise errors rather than claiming success. Nonfinite results also fail (strict
JSON encoding will reject a nonfinite error). The script disables HF32 matmul.
Peak bytes come from `torch.npu.max_memory_allocated(0)` after resetting peak
statistics and synchronizing, and cover the tiny NPU allocation/matmul. They are
allocator-observed bytes, not total device usage, runtime/workspace accounting,
or a model memory estimate. No timing, warmup, speedup, or benchmark is reported.

## Correctness scope (14 parameterized cases)

`tests/test_state_ops_npu.py` uses fixed CPU seeds and transfers only tiny inputs
and explicit caller-owned state to `npu:0`. References consume the **same already
quantized inputs**, not separately generated FP32 approximations. All float64
reference and comparison arithmetic stays on CPU. The module fixture disables
HF32 convolution/matmul to check FP32 compute and restores those flags afterward.
It requires a `torch_npu` version exposing these precision controls; unsupported
kernels or broken runtime initialization are not silently replaced by CPU math.

- **Recurrence (6 cases):** FP32/BF16, grouped heads `HK=2, HV=4, K=3, V=5, T=4`,
  representative `HK=1, HV=2, K=128, V=96, T=3`, and a zero-token case with
  `K=5, V=3`. Each uses the independent float64 affine-transition expansion
  `_transition_reference` imported absolutely from `tests.test_state_ops`, with
  nonzero initial FP32 state. Batch size is one.
- **Convolution (8 cases):** FP32/BF16, `(W,T)=(1,3),(4,2),(3,5),(3,0)`, batch
  size one and three channels. The reference independently applies FP64 grouped
  `F.conv1d` to the full input/history prefix, then SiLU. Nonempty history is
  nonzero; exact returned raw-input history is checked, including `W=1`.
- **State contracts:** both operations check device/dtype, multi-token chunks,
  one-token steps, empty chunks, nonmutation of every input and caller state,
  nonaliasing of nonempty caller state, and rerun/streaming parity after an
  unrelated request using different explicit state. Returned recurrence state
  stays FP32 for BF16 inputs. No full-sized model states are allocated.

Both `atol` and `rtol` are 3e-6 for FP32 outputs and recurrence state, 8e-3 for
BF16 recurrence output, and 2e-2 for BF16 convolution output. Input nonmutation
and convolution-history comparisons are exact. These are bounded-case numerical
checks, not universal tolerances or cross-device bitwise-equality guarantees.

## Interpreting verification

**Skip != NPU pass.** Missing PyTorch/`torch_npu` or an unavailable NPU skips this
module on CPU-only hosts. Review pytest's `-rs` output and require all 14 cases to
actually execute and pass on the assigned NPU before reporting NPU validation.
A skip, CPU-test pass, collection success, or syntax check cannot establish that.
Framework-import/runtime failures other than unavailable prerequisites remain
errors; unsupported BF16 or operator kernels are not waived as passing tests.

Actual server execution is now recorded in [results](../results/README.md):
**67 CPU tests passed; all 14 NPU cases executed and passed (no skips)** on
physical device 1, exposed as logical `npu:0`, reporting `Ascend910_9362`.
The smoke test passed with maximum absolute error 0 and 3,072 peak allocated
bytes. These bytes are not total HBM consumption. The server had other active
jobs; pytest wall times are test-run metadata, not inference performance results.
Local PyTorch remains unavailable; these runtime results came from the course
server's private environment, not the development machine.
