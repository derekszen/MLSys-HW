# Explicit-state PyTorch inference prototypes

**These are standalone homework starters, not an integrated TokenSpeed adaptation.**
They do not modify or call the TokenSpeed submodule, load model weights, wire up
prefill/decode, implement a serving state cache, or establish an end-to-end model
or performance baseline. CPU mathematical tests are **not Ascend/NPU validation**.
The recurrence follows the requested TokenSpeed-compatible `[B,HV,V,K]` layout
and head mapping; there is no claim of tested equivalence to an upstream kernel.

## API and common contract

```python
from hw1 import causal_depthwise_conv1d, gated_delta_recurrence

y, history = causal_depthwise_conv1d(x, weight, history=None, bias=None)
y, state = gated_delta_recurrence(q, k, v, a, b, A_log, dt_bias, state=None)
```

Both functions are inference-only (`torch.no_grad`). Inputs must be strided
PyTorch tensors; non-contiguous views are supported. Activations support FP32,
BF16, and FP16, with one common activation dtype per call. FP64 is used only in
test references, not accepted as an activation dtype by these prototypes.
Shapes, dtypes, devices, and positive channel/head dimensions are validated before
computation, even for zero-token sequences. B and T may be zero; other dimensions
must be positive. Values are assumed finite and reasonably scaled; no expensive
finite-value checks or overflow guards are provided.

All math and allocations stay on the input device. There is no `.cpu()` fallback,
implicit device transfer, global state, or mutation of caller tensors. Returned
state/history does not alias caller-owned history/state, including empty-token
calls. Missing state/history means zeros allocated on the input device. Returned
state is intended to be passed directly to the next chunk or one-token step;
separate requests must receive separate state rather than reusing another
request's returned state.

## Causal depthwise convolution

| Argument | Shape | Dtype |
| --- | --- | --- |
| `x` | `[B,T,C]` | Activation dtype |
| `weight` | `[C,W]` | Same as x |
| `history` (optional) | `[B,C,W-1]` | Same as x |
| `bias` (optional) | `[C]` | Same as x |
| output | `[B,T,C]` | Same as x |
| returned history | `[B,C,W-1]` | Same as x |

History contains **raw inputs, oldest first**, not previous convolution outputs.
Concatenate history with `x.transpose(1,2)` and perform grouped `F.conv1d` with
`groups=C`, stride 1, no padding, then SiLU. PyTorch uses cross-correlation:
`weight[c,0]` multiplies the oldest sample and `weight[c,W-1]` the current sample.
Bias is added before SiLU. Compute convolution and activation in FP32 and cast
output back to x.dtype. Keep the final W-1 raw inputs as the new history, including
any previous history still needed when T < W-1. W=1 returns a zero-width history.
T=0 returns an empty output and an unchanged **copy** of history.

## Gated delta recurrence

| Argument | Shape | Dtype |
| --- | --- | --- |
| `q`, `k` | `[B,T,HK,K]` | Common activation dtype |
| `v` | `[B,T,HV,V]` | Common activation dtype |
| `a`, `b` | `[B,T,HV]` | Common activation dtype |
| `A_log`, `dt_bias` | `[HV]` | Activation dtype or FP32 |
| `state` (optional) | `[B,HV,V,K]` | **FP32** |
| output | `[B,T,HV,V]` | v.dtype |
| returned state | `[B,HV,V,K]` | **FP32** |

All tensors, including parameters and state, must share q's device. HV must be a
positive multiple of HK. K and V can differ. Normalize q and k independently in
FP32 using `vector / max(L2_norm(vector), 1e-12)`; zero vectors stay zero. Repeat
normalized key/query heads consecutively with `repeat_interleave(HV // HK)`.
For HK=2, HV=4 the source-head mapping is `[0,0,1,1]`, **not** `[0,1,0,1]`.

For each time t and value head, with key/query column vectors of size K and
value column vector of size V:

```text
g_t     = -exp(A_log) * softplus(a_t + dt_bias)
d_t     = exp(g_t)
beta_t  = sigmoid(b_t)
D_t     = d_t * S_(t-1)                 # [V,K], decay FIRST
p_t     = D_t @ k_t                    # [V], predict from DECAYED state
delta_t = beta_t * (v_t - p_t)
S_t     = D_t + outer(delta_t, k_t)     # [V,K]
y_t     = (S_t @ q_t) / sqrt(K)         # [V], read UPDATED state
```

All gates, normalization, dot products, and state accumulation are FP32.
Output alone is cast back to v.dtype. In particular, normalized k is not scaled
by `1/sqrt(K)` in the state update; the query readout carries that factor.
There is no output gate, projection, residual, convolution, or RMSNorm folded
into this function. T=0 returns `[B,0,HV,V]` and a copy of the initial FP32 state.
The simple Python time loop prioritizes inspectability, not speed or fusion.

## Verification

From `homework-1`, using the parent-provisioned CPU PyTorch/pytest environment:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_state_ops.py
```

No NumPy dependency is needed. Tests are small and deterministic where a numerical
reference matters. The representative K=128 case uses HK=1, HV=2, V=96, T=3;
it exercises a realistic key dimension and unequal K/V, not a full model shape.

The recurrence reference uses a distinct **float64 affine transition-matrix
expansion**, not a copy of the prediction/delta implementation:

```text
M_t = d_t * (I - beta_t * k_t @ k_t.T)  # [K,K]
U_t = beta_t * v_t @ k_t.T             # [V,K]
S_t = S_initial @ product(M_0 ... M_t)
      + sum_i U_i @ product(M_(i+1) ... M_t)
```

Each prefix is expanded independently with nonzero initial state, exposing state
orientation, decay ordering, and head-mapping mistakes. The convolution reference
uses float64 full-prefix grouped `F.conv1d` followed by SiLU. Both references
consume already-quantized activation inputs rather than comparing different
input quantizations. Additional tests cover grouped heads, K != V, chunk/step
parity, nonzero and default states, request isolation, caller nonmutation,
non-contiguous views, zero vectors, T=0, empty batches, W=1, short sequences,
and malformed shapes/dtypes/devices. Device-mismatch tests use meta tensors to
check rejection without requiring GPU/NPU hardware; they do not execute math
on meta devices.

Numerical bounds (`atol` and `rtol` use the same value):

| Comparison | FP32 | BF16 | FP16 |
| --- | --- | --- | --- |
| Recurrence output vs float64 | 3e-6 | 8e-3 | 1e-3 |
| Recurrence state vs float64 | 3e-6 | 3e-6 | 3e-6 |
| Convolution output vs float64 | 3e-6 | 2e-2 | 3e-3 |

State stays FP32 even with low-precision activations; rounded outputs need wider
bounds. Convolution's BF16 bound accounts for output rounding at the tested
magnitudes. These are small CPU-case correctness tolerances, not universal
long-sequence error guarantees or cross-device bitwise requirements. Chunk/step
output parity uses these same bounds, with 3e-6 for recurrent state and exact
comparison for convolution history. An explicit BF16-versus-FP32 comparison
holds quantized inputs fixed. Non-contiguous recurrence comparison allows 1e-6
reduction-order noise. Actual NPU execution, supported kernels,
precision behavior, synchronization, model wiring, and end-to-end correctness
remain unverified here.
