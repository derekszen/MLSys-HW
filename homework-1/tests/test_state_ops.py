"""CPU mathematical tests, not TokenSpeed integration or Ascend validation."""

import math

import pytest
import torch
from torch.nn import functional as F

from hw1 import causal_depthwise_conv1d, gated_delta_recurrence


def _inputs(*, dtype=torch.float32, batch=2, tokens=5, hk=2, hv=4, kd=3, vd=5):
    rng = torch.Generator().manual_seed(2026)

    def rand(*shape):
        return torch.randn(shape, generator=rng).to(dtype)

    return dict(
        q=rand(batch, tokens, hk, kd),
        k=rand(batch, tokens, hk, kd),
        v=rand(batch, tokens, hv, vd),
        a=rand(batch, tokens, hv),
        b=rand(batch, tokens, hv),
        A_log=rand(hv).float() - 1,
        dt_bias=rand(hv).float(),
        state=rand(batch, hv, vd, kd).float(),
    )


def _transition_reference(data):
    """Independent affine transition expansion; never uses prediction/delta.

    S_t = S_0 (M_0 ... M_t) + sum_i U_i (M_{i+1} ... M_t),
    M_i = d_i (I - beta_i k_i k_i^T), U_i = beta_i v_i k_i^T.
    Compute each prefix independently in float64, including nonzero S_0.
    """
    q, k, v = (data[name].double() for name in ("q", "k", "v"))
    heads, dim = v.shape[2], q.shape[-1]
    mapping = torch.arange(heads) // (heads // q.shape[2])
    q = (q / q.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-12)).index_select(2, mapping)
    k = (k / k.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-12)).index_select(2, mapping)
    z = data["a"].double() + data["dt_bias"].double()
    decay = (-data["A_log"].double().exp() * torch.logaddexp(z, torch.zeros_like(z))).exp()
    beta = 1 / (1 + (-data["b"].double()).exp())
    matrix = decay[..., None, None] * (
        torch.eye(dim, dtype=torch.float64)
        - beta[..., None, None] * torch.einsum("bthk,bthl->bthkl", k, k)
    )
    source = beta[..., None, None] * torch.einsum("bthv,bthk->bthvk", v, k)
    initial = data["state"].double()
    outputs = []
    final = initial.clone()
    for end in range(q.shape[1]):
        final = initial.clone()
        for index in range(end + 1):
            final = final @ matrix[:, index]
        for start in range(end + 1):
            contribution = source[:, start]
            for index in range(start + 1, end + 1):
                contribution = contribution @ matrix[:, index]
            final = final + contribution
        outputs.append(torch.einsum("bhvk,bhk->bhv", final, q[:, end]) / math.sqrt(dim))
    output = torch.stack(outputs, 1) if outputs else v.new_empty(v.shape)
    return output, final


def _slice(data, start, end, state):
    return {
        **data,
        **{name: data[name][:, start:end] for name in ("q", "k", "v", "a", "b")},
        "state": state,
    }


def _noncontiguous(value):
    # Preserve values/shape while giving every dimension a non-dense stride.
    buffer = torch.empty((*value.shape, 2), dtype=value.dtype, device=value.device)
    buffer[..., 0] = value
    return buffer[..., 0]


@pytest.mark.parametrize("hk,hv,kd,vd,tokens", [(1, 1, 1, 1, 4), (2, 6, 3, 5, 5), (1, 2, 128, 96, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_delta_matches_independent_transition_matrices(hk, hv, kd, vd, tokens, dtype):
    data = _inputs(batch=1, hk=hk, hv=hv, kd=kd, vd=vd, tokens=tokens, dtype=dtype)
    output, state = gated_delta_recurrence(**data)
    expected_output, expected_state = _transition_reference(data)
    assert output.dtype == dtype
    assert state.dtype == torch.float32
    assert output.device == data["q"].device == state.device
    # Reference consumes already-quantized inputs; only output is rounded back.
    tolerance = {torch.float32: 3e-6, torch.bfloat16: 8e-3, torch.float16: 1e-3}[dtype]
    torch.testing.assert_close(output.double(), expected_output, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(state.double(), expected_state, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_delta_chunk_step_parity_and_request_isolation(dtype):
    data = _inputs(dtype=dtype)
    snapshot = {name: value.clone() for name, value in data.items()}
    full_output, full_state = gated_delta_recurrence(**data)
    for cuts in ((0, 2, 5), (0, 1, 2, 3, 4, 5), (0, 0, 3, 5, 5)):
        state, chunks = data["state"], []
        for start, end in zip(cuts, cuts[1:]):
            output, state = gated_delta_recurrence(**_slice(data, start, end, state))
            chunks.append(output)
        tolerance = 3e-6 if dtype == torch.float32 else 8e-3
        torch.testing.assert_close(torch.cat(chunks, 1), full_output, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(state, full_state, atol=3e-6, rtol=3e-6)
    # A separate request with unrelated state must not affect a rerun.
    gated_delta_recurrence(**{**data, "state": torch.full_like(data["state"], 9)})
    again = gated_delta_recurrence(**data)
    for actual, expected in zip(again, (full_output, full_state)):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for name, original in snapshot.items():
        torch.testing.assert_close(data[name], original, atol=0, rtol=0)
    assert full_state.data_ptr() != data["state"].data_ptr()


def test_bfloat16_against_float32_with_identical_quantized_inputs():
    data = _inputs(dtype=torch.bfloat16)
    fp32 = {name: value.float() for name, value in data.items()}
    output, state = gated_delta_recurrence(**data)
    reference_output, reference_state = gated_delta_recurrence(**fp32)
    torch.testing.assert_close(output.float(), reference_output, atol=8e-3, rtol=8e-3)
    torch.testing.assert_close(state, reference_state, atol=0, rtol=0)
    x = data["v"].flatten(2)
    weight = torch.randn(x.shape[-1], 3).bfloat16()
    low_output, low_history = causal_depthwise_conv1d(x, weight)
    ref_output, ref_history = causal_depthwise_conv1d(x.float(), weight.float())
    torch.testing.assert_close(low_output.float(), ref_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(low_history.float(), ref_history, atol=0, rtol=0)


def test_delta_noncontiguous_and_zero_vectors():
    data = _inputs()
    data["q"][:, 1] = 0
    data["k"][:, 2] = 0
    expected = gated_delta_recurrence(**data)
    views = {name: _noncontiguous(value) for name, value in data.items()}
    assert all(not value.is_contiguous() for value in views.values())
    actual = gated_delta_recurrence(**views)
    reference = _transition_reference(data)
    for result, dense, independent in zip(actual, expected, reference):
        torch.testing.assert_close(result, dense, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(result.double(), independent, atol=3e-6, rtol=3e-6)
    assert torch.count_nonzero(actual[0][:, 1]) == 0


@pytest.mark.parametrize("tokens", [0, 3])
def test_delta_default_state_and_empty(tokens):
    data = _inputs(tokens=tokens)
    data["state"].zero_()
    output, state = gated_delta_recurrence(**data)
    no_state = {name: value for name, value in data.items() if name != "state"}
    default_output, default_state = gated_delta_recurrence(**no_state)
    torch.testing.assert_close(output, default_output)
    torch.testing.assert_close(state, default_state)
    if tokens == 0:
        data["state"].fill_(2)
        empty, copied = gated_delta_recurrence(**data)
        assert empty.shape == data["v"].shape
        assert copied.data_ptr() != data["state"].data_ptr()
        torch.testing.assert_close(copied, data["state"])


@pytest.mark.parametrize("width,tokens", [(1, 4), (4, 2), (3, 7), (3, 0)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_conv_full_prefix_reference_chunk_step_and_nonmutation(width, tokens, dtype):
    rng = torch.Generator().manual_seed(17)
    x = torch.randn(2, tokens, 3, generator=rng).to(dtype)
    weight = torch.randn(3, width, generator=rng).to(dtype)
    bias = torch.randn(3, generator=rng).to(dtype)
    history = torch.randn(2, 3, width - 1, generator=rng).to(dtype)
    snapshots = [value.clone() for value in (x, weight, history, bias)]
    output, new_history = causal_depthwise_conv1d(x, weight, history, bias)
    prefix = torch.cat((history, x.transpose(1, 2)), -1)
    tolerance = {torch.float32: 3e-6, torch.bfloat16: 2e-2, torch.float16: 3e-3}[dtype]
    if tokens:
        # Full-prefix grouped cross-correlation with no implicit left padding.
        reference = F.silu(F.conv1d(prefix.double(), weight.double()[:, None], bias.double(), groups=3))
        torch.testing.assert_close(output.double(), reference.transpose(1, 2), atol=tolerance, rtol=tolerance)
    else:
        assert output.shape == (2, 0, 3)
    assert output.dtype == dtype == new_history.dtype
    torch.testing.assert_close(new_history, prefix[:, :, -(width - 1):] if width > 1 else prefix[:, :, :0])
    for cuts in ([0, min(tokens, 2), tokens], list(range(tokens + 1)) if tokens else [0, 0]):
        current, chunks = history, []
        for start, end in zip(cuts, cuts[1:]):
            chunk, current = causal_depthwise_conv1d(x[:, start:end], weight, current, bias)
            chunks.append(chunk)
        torch.testing.assert_close(torch.cat(chunks, 1), output, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(current, new_history, atol=0, rtol=0)
    for original, value in zip(snapshots, (x, weight, history, bias)):
        torch.testing.assert_close(value, original, atol=0, rtol=0)
    if history.numel():
        assert new_history.data_ptr() != history.data_ptr()


def test_conv_zero_default_history_noncontiguous_and_request_isolation():
    x = torch.randn(2, 5, 3)
    weight, bias = torch.randn(3, 4), torch.randn(3)
    history = torch.zeros(2, 3, 3)
    expected = causal_depthwise_conv1d(x, weight, history, bias)
    no_state = causal_depthwise_conv1d(x, weight, bias=bias)
    views = causal_depthwise_conv1d(*(_noncontiguous(t) for t in (x, weight, history, bias)))
    causal_depthwise_conv1d(x, weight, torch.ones_like(history), bias)
    again = causal_depthwise_conv1d(x, weight, history, bias)
    for actual in (no_state, views, again):
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, atol=0, rtol=0)
    # No-bias path, including causal zero-padding across the full prefix.
    output, _ = causal_depthwise_conv1d(x, weight)
    reference = F.silu(F.conv1d(F.pad(x.transpose(1, 2), (3, 0)), weight[:, None], groups=3))
    torch.testing.assert_close(output, reference.transpose(1, 2))


@pytest.mark.parametrize("name,replacement,error", [
    ("q", torch.zeros(2, 5, 2), ValueError),
    ("q", torch.zeros(2, 5, 0, 3), ValueError),
    ("q", torch.zeros(2, 5, 2, 0), ValueError),
    ("k", torch.zeros(2, 5, 1, 3), ValueError),
    ("v", torch.zeros(2, 4, 4, 5), ValueError),
    ("v", torch.zeros(2, 5, 3, 5), ValueError),
    ("v", torch.zeros(2, 5, 4, 0), ValueError),
    ("a", torch.zeros(2, 5, 2), ValueError),
    ("b", torch.zeros(2, 5, 4, 1), ValueError),
    ("A_log", torch.zeros(3), ValueError),
    ("dt_bias", torch.zeros(1, 4), ValueError),
    ("state", torch.zeros(2, 4, 3, 5), ValueError),
    ("state", torch.zeros(2, 4, 5, 3, dtype=torch.bfloat16), TypeError),
    ("q", torch.zeros(2, 5, 2, 3, dtype=torch.int64), TypeError),
    ("k", torch.zeros(2, 5, 2, 3, dtype=torch.float64), TypeError),
    ("a", torch.zeros(2, 5, 4, dtype=torch.bfloat16), TypeError),
    ("A_log", torch.zeros(4, dtype=torch.int32), TypeError),
    ("v", None, TypeError),
])
def test_delta_validation(name, replacement, error):
    data = _inputs()
    data[name] = replacement
    with pytest.raises(error):
        gated_delta_recurrence(**data)


@pytest.mark.parametrize("name", ["k", "v", "a", "b", "A_log", "dt_bias", "state"])
def test_delta_rejects_device_mismatch_without_cpu_fallback(name):
    data = _inputs()
    data[name] = torch.empty_like(data[name], device="meta")
    with pytest.raises(ValueError, match="must be on"):
        gated_delta_recurrence(**data)


@pytest.mark.parametrize("name,replacement,error", [
    ("x", torch.zeros(2, 5), ValueError),
    ("x", torch.zeros(2, 5, 0), ValueError),
    ("x", torch.zeros(2, 5, 3, dtype=torch.int64), TypeError),
    ("weight", torch.zeros(3, 0), ValueError),
    ("weight", torch.zeros(2, 4), ValueError),
    ("weight", torch.zeros(3, 4, dtype=torch.bfloat16), TypeError),
    ("history", torch.zeros(2, 3, 4), ValueError),
    ("history", torch.zeros(2, 3, 3, dtype=torch.float16), TypeError),
    ("bias", torch.zeros(2), ValueError),
    ("bias", torch.zeros(3, dtype=torch.int64), TypeError),
])
def test_conv_validation(name, replacement, error):
    data = dict(x=torch.zeros(2, 5, 3), weight=torch.zeros(3, 4), history=torch.zeros(2, 3, 3), bias=torch.zeros(3))
    data[name] = replacement
    with pytest.raises(error):
        causal_depthwise_conv1d(**data)


@pytest.mark.parametrize("name", ["weight", "history", "bias"])
def test_conv_rejects_device_mismatch_without_cpu_fallback(name):
    data = dict(x=torch.zeros(2, 5, 3), weight=torch.zeros(3, 4), history=torch.zeros(2, 3, 3), bias=torch.zeros(3))
    data[name] = torch.empty_like(data[name], device="meta")
    with pytest.raises(ValueError, match="must be on"):
        causal_depthwise_conv1d(**data)


def test_empty_batch():
    data = _inputs(batch=0)
    output, state = gated_delta_recurrence(**data)
    assert output.shape == data["v"].shape
    assert state.shape == data["state"].shape
    output, history = causal_depthwise_conv1d(torch.empty(0, 3, 2), torch.ones(2, 4))
    assert output.shape == (0, 3, 2)
    assert history.shape == (0, 2, 3)
