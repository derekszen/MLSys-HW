"""Bounded standalone-prototype NPU correctness, not TokenSpeed/model tests."""

import pytest


torch = pytest.importorskip("torch", reason="NPU validation requires PyTorch")
pytest.importorskip("torch_npu", reason="NPU validation requires torch_npu")
if not torch.npu.is_available():
    pytest.skip("no Ascend NPU available; skip is not an NPU pass", allow_module_level=True)

from torch.nn import functional as F

from hw1.state_ops import causal_depthwise_conv1d, gated_delta_recurrence
from tests.test_state_ops import _inputs, _slice, _transition_reference


@pytest.fixture(scope="module")
def npu_device():
    assert torch.npu.device_count() == 1, "select one NPU with ASCEND_RT_VISIBLE_DEVICES"
    torch.npu.set_device(0)
    # Validate the prototypes' FP32 computation, not reduced-precision HF32 math.
    conv_hf32 = torch.npu.conv.allow_hf32
    matmul_hf32 = torch.npu.matmul.allow_hf32
    torch.npu.conv.allow_hf32 = False
    torch.npu.matmul.allow_hf32 = False
    try:
        yield torch.device("npu:0")
    finally:
        torch.npu.conv.allow_hf32 = conv_hf32
        torch.npu.matmul.allow_hf32 = matmul_hf32


def _close(actual, expected, tolerance=0):
    # All comparison/reference arithmetic is CPU float64, never NPU float64.
    torch.testing.assert_close(
        actual.cpu().double(), expected.cpu().double(), atol=tolerance, rtol=tolerance
    )


def _cuts(tokens):
    return ((0, 0, min(2, tokens), tokens, tokens),
            tuple(range(tokens + 1)) if tokens else (0, 0))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("hk,hv,kd,vd,tokens", [
    (2, 4, 3, 5, 4),
    (1, 2, 128, 96, 3),
    (1, 2, 5, 3, 0),
])
def test_delta_reference_streaming_and_isolation(npu_device, dtype, hk, hv, kd, vd, tokens):
    cpu = _inputs(dtype=dtype, batch=1, tokens=tokens, hk=hk, hv=hv, kd=kd, vd=vd)
    expected_output, expected_state = _transition_reference(cpu)
    data = {name: value.to(npu_device) for name, value in cpu.items()}
    output, state = gated_delta_recurrence(**data)
    tolerance = 3e-6 if dtype == torch.float32 else 8e-3
    assert output.device == state.device == npu_device
    assert output.dtype == dtype
    assert state.dtype == torch.float32
    assert state.data_ptr() != data["state"].data_ptr()
    _close(output, expected_output, tolerance)
    _close(state, expected_state, 3e-6)

    # Interleave a separate request with unrelated caller-owned state.
    other_state = torch.full_like(data["state"], 9)
    other_snapshot = other_state.cpu().clone()
    gated_delta_recurrence(**{**data, "state": other_state})
    for cuts in _cuts(tokens):
        current, chunks = data["state"], []
        for start, end in zip(cuts, cuts[1:]):
            chunk, current = gated_delta_recurrence(**_slice(data, start, end, current))
            assert chunk.device == current.device == npu_device
            assert chunk.dtype == dtype and current.dtype == torch.float32
            chunks.append(chunk)
        _close(torch.cat(chunks, 1), output, tolerance)
        _close(current, state, 3e-6)
    again = gated_delta_recurrence(**data)
    _close(again[0], output, tolerance)
    _close(again[1], state, 3e-6)
    for name, original in cpu.items():
        _close(data[name], original)
    _close(other_state, other_snapshot)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("width,tokens", [(1, 3), (4, 2), (3, 5), (3, 0)])
def test_conv_reference_streaming_and_isolation(npu_device, dtype, width, tokens):
    rng = torch.Generator().manual_seed(17)

    def rand(*shape):
        return torch.randn(shape, generator=rng).to(dtype)

    cpu = dict(x=rand(1, tokens, 3), weight=rand(3, width),
               history=rand(1, 3, width - 1), bias=rand(3))
    # Independent full-prefix FP64 grouped cross-correlation, then SiLU.
    prefix = torch.cat((cpu["history"], cpu["x"].transpose(1, 2)), -1).double()
    if tokens:
        expected_output = F.silu(F.conv1d(
            prefix, cpu["weight"].double()[:, None], cpu["bias"].double(), groups=3
        )).transpose(1, 2)
    else:
        expected_output = torch.empty((1, 0, 3), dtype=torch.float64)
    expected_history = prefix[:, :, -(width - 1):] if width > 1 else prefix[:, :, :0]
    data = {name: value.to(npu_device) for name, value in cpu.items()}
    output, history = causal_depthwise_conv1d(**data)
    tolerance = 3e-6 if dtype == torch.float32 else 2e-2
    assert output.device == history.device == npu_device
    assert output.dtype == history.dtype == dtype
    if data["history"].numel():
        assert history.data_ptr() != data["history"].data_ptr()
    _close(output, expected_output, tolerance)
    _close(history, expected_history)

    other_history = torch.full_like(data["history"], 9)
    other_snapshot = other_history.cpu().clone()
    causal_depthwise_conv1d(**{**data, "history": other_history})
    for cuts in _cuts(tokens):
        current, chunks = data["history"], []
        for start, end in zip(cuts, cuts[1:]):
            chunk, current = causal_depthwise_conv1d(
                data["x"][:, start:end], data["weight"], current, data["bias"]
            )
            assert chunk.device == current.device == npu_device
            assert chunk.dtype == current.dtype == dtype
            chunks.append(chunk)
        _close(torch.cat(chunks, 1), output, tolerance)
        _close(current, history)
    again = causal_depthwise_conv1d(**data)
    _close(again[0], output, tolerance)
    _close(again[1], history)
    for name, original in cpu.items():
        _close(data[name], original)
    _close(other_history, other_snapshot)
