"""Small, device-local inference references with caller-owned explicit state.

See ``docs/state-ops.md`` for layout, precision, and empty-sequence contracts.
These deliberately simple implementations are not TokenSpeed integrations.
"""

import math

import torch
from torch import Tensor
from torch.nn import functional as F


_INPUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _tensor(name: str, value: Tensor, ndim: int) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got shape {tuple(value.shape)}")
    if value.layout != torch.strided:
        raise ValueError(f"{name} must use a strided layout")


def _like(name: str, value: Tensor, reference: Tensor, dtypes: tuple) -> None:
    if value.device != reference.device:
        raise ValueError(f"{name} must be on {reference.device}, got {value.device}")
    if value.dtype not in dtypes:
        raise TypeError(f"{name} has unsupported dtype {value.dtype}; expected {dtypes}")


@torch.no_grad()
def causal_depthwise_conv1d(
    x: Tensor,
    weight: Tensor,
    history: Tensor | None = None,
    bias: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return SiLU causal depthwise convolution and oldest-first input history.

    Shapes: x [B,T,C], weight [C,W], history [B,C,W-1], bias [C].
    Missing history means zeros. All tensors have x's dtype/device. Computation
    is FP32; output/history have x's dtype. Empty sequences return empty output
    and a copy of history. Neither input tensors nor caller state are mutated.
    """
    _tensor("x", x, 3)
    _like("x", x, x, _INPUT_DTYPES)
    _tensor("weight", weight, 2)
    _like("weight", weight, x, (x.dtype,))
    batch, tokens, channels = x.shape
    if channels < 1 or weight.shape[0] != channels or weight.shape[1] < 1:
        raise ValueError("weight must have shape [C,W] with C > 0 and W > 0")
    width = weight.shape[1]
    if bias is not None:
        _tensor("bias", bias, 1)
        _like("bias", bias, x, (x.dtype,))
        if bias.shape != (channels,):
            raise ValueError("bias must have shape [C]")
    if history is None:
        history = x.new_zeros((batch, channels, width - 1))
    else:
        _tensor("history", history, 3)
        _like("history", history, x, (x.dtype,))
        if history.shape != (batch, channels, width - 1):
            raise ValueError("history must have shape [B,C,W-1]")
    if tokens == 0 or batch == 0:
        return x.new_empty((batch, tokens, channels)), history.clone()

    full_input = torch.cat((history, x.transpose(1, 2)), dim=-1)
    result = F.conv1d(
        full_input.float(),
        weight.float().unsqueeze(1),
        bias=None if bias is None else bias.float(),
        groups=channels,
    )
    # The explicit width==1 branch avoids Python's -0 slice meaning "all".
    new_history = full_input[:, :, -(width - 1):].clone() if width > 1 else full_input[:, :, :0].clone()
    return F.silu(result).transpose(1, 2).to(x.dtype), new_history


@torch.no_grad()
def gated_delta_recurrence(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    state: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Qwen3.5-style gated delta recurrence with state [B,HV,V,K].

    q/k: [B,T,HK,K]; v: [B,T,HV,V]; a/b: [B,T,HV];
    A_log/dt_bias: [HV]. HV must be divisible by HK. q/k are L2-normalized
    in FP32 (epsilon 1e-12) and repeated consecutively to HV. State must be
    FP32 on the input device; missing state means zeros. Output is v.dtype.
    See docs for exact update equations. This is inference-only, not a kernel.
    """
    _tensor("q", q, 4)
    _like("q", q, q, _INPUT_DTYPES)
    for name, value, ndim in (("k", k, 4), ("v", v, 4), ("a", a, 3), ("b", b, 3)):
        _tensor(name, value, ndim)
        _like(name, value, q, (q.dtype,))
    batch, tokens, key_heads, key_dim = q.shape
    if key_heads < 1 or key_dim < 1:
        raise ValueError("HK and K must be positive")
    if k.shape != q.shape:
        raise ValueError("k must have the same shape as q")
    if v.shape[:2] != (batch, tokens):
        raise ValueError("v must have the same B,T dimensions as q")
    value_heads, value_dim = v.shape[2:]
    if value_heads < 1 or value_dim < 1 or value_heads % key_heads:
        raise ValueError("HV and V must be positive, and HV must be divisible by HK")
    for name, value in (("a", a), ("b", b)):
        if value.shape != (batch, tokens, value_heads):
            raise ValueError(f"{name} must have shape [B,T,HV]")
    for name, value in (("A_log", A_log), ("dt_bias", dt_bias)):
        _tensor(name, value, 1)
        _like(name, value, q, (q.dtype, torch.float32))
        if value.shape != (value_heads,):
            raise ValueError(f"{name} must have shape [HV]")
    if state is None:
        current = torch.zeros(
            (batch, value_heads, value_dim, key_dim), dtype=torch.float32, device=q.device
        )
    else:
        _tensor("state", state, 4)
        _like("state", state, q, (torch.float32,))
        if state.shape != (batch, value_heads, value_dim, key_dim):
            raise ValueError("state must have shape [B,HV,V,K]")
        current = state.clone()
    if tokens == 0:
        return v.new_empty(v.shape), current

    repeats = value_heads // key_heads
    queries = F.normalize(q.float(), dim=-1, eps=1e-12).repeat_interleave(repeats, dim=2)
    keys = F.normalize(k.float(), dim=-1, eps=1e-12).repeat_interleave(repeats, dim=2)
    decay = torch.exp(-torch.exp(A_log.float()) * F.softplus(a.float() + dt_bias.float()))
    beta = torch.sigmoid(b.float())
    values = v.float()
    outputs = []
    for index in range(tokens):
        key = keys[:, index]
        current = current * decay[:, index, :, None, None]
        prediction = (current * key.unsqueeze(-2)).sum(dim=-1)
        delta = beta[:, index, :, None] * (values[:, index] - prediction)
        current = current + delta.unsqueeze(-1) * key.unsqueeze(-2)
        output = (current * queries[:, index].unsqueeze(-2)).sum(dim=-1) / math.sqrt(key_dim)
        outputs.append(output)
    return torch.stack(outputs, dim=1).to(v.dtype), current
