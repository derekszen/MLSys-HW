#!/usr/bin/env python3
"""Tiny, explicitly selected Ascend device check; not a benchmark."""

import argparse
import json
import os


def _physical_device(value: str) -> str:
    if not value.isascii() or not value.isdecimal():
        raise argparse.ArgumentTypeError("expected one nonnegative physical device index")
    return str(int(value))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-device", required=True, type=_physical_device)
    args = parser.parse_args()

    # Must precede BOTH framework imports: the selected physical device becomes npu:0.
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = args.physical_device
    import torch
    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is unavailable")
    count = torch.npu.device_count()
    if count != 1:
        raise RuntimeError(f"expected exactly one visible NPU, found {count}")
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats(0)

    cpu_input = torch.arange(256, dtype=torch.float32).reshape(16, 16) / 256
    expected = cpu_input @ cpu_input
    with torch.no_grad():
        npu_input = cpu_input.to("npu:0")
        actual = npu_input @ npu_input
    torch.npu.synchronize()
    peak_bytes = int(torch.npu.max_memory_allocated(0))
    actual_cpu = actual.cpu()
    max_error = float((actual_cpu - expected).abs().max().item())
    passed = bool(torch.allclose(actual_cpu, expected, atol=1e-5, rtol=1e-5))
    print(json.dumps({
        "physical_device": args.physical_device,
        "logical_device": "npu:0",
        "device_name": torch.npu.get_device_name(0),
        "torch_version": str(torch.__version__),
        "torch_npu_version": str(torch_npu.__version__),
        "operation": "16x16 FP32 X @ X.T, X=arange(256).reshape(16,16)/256",
        "max_abs_error": max_error,
        "peak_allocated_bytes": peak_bytes,
        "atol": 1e-5,
        "rtol": 1e-5,
        "passed": passed,
    }, sort_keys=True, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
