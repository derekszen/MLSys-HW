"""Observe 4B's real autoregressive logits without changing production sources.

Explicit single-request diagnostic. Spawned workers install a read-only model
exit observer. An on-disk gate is armed only after warmup. No latency claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import run_model as harness
from decode_probe import write_json_new

PLAN_ENV = "MLSYS_HW2_DECODE_PROBE_PLAN"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, choices=(1, 2), required=True)
    parser.add_argument("--physical-devices", type=harness.parse_devices, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="existing same-TP/precision result.json; diagnostic must reproduce it")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memory-fraction", type=float, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    config = harness.build_config(args.model, tp=args.tp, devices=args.physical_devices,
                                  memory_fraction=args.memory_fraction, concurrency=1, port=args.port)
    config["dtype"] = args.dtype
    if args.dtype == "float16":
        config["kv_cache_dtype"] = "float16"
    cases_bytes = args.cases.read_bytes()
    cases = json.loads(cases_bytes)["cases"]
    if [(c["name"], len(c["input_ids"])) for c in cases] != [
        ("smoke", 5), ("length_32", 32), ("length_256", 256)
    ]:
        raise ValueError("requires the canonical three cases in original order")
    baseline_bytes = args.baseline.read_bytes()
    baseline = json.loads(baseline_bytes)
    if (baseline["status"] != "generation_completed_not_yet_compared"
            or baseline["mode"] != "correctness" or baseline["tp"] != args.tp
            or baseline["concurrency"] != 1):
        raise ValueError("baseline must be completed matching TP/concurrency correctness")
    if ({k: v for k, v in config.items() if k != "port"}
            != {k: v for k, v in baseline["config"].items() if k != "port"}):
        raise ValueError("diagnostic non-port configuration differs from baseline")
    manifest_bytes = (args.model / "source-manifest.json").read_bytes()
    config_bytes = (args.model / "config.json").read_bytes()
    hashes = {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
              "cases_sha256": hashlib.sha256(cases_bytes).hexdigest()}
    if any(baseline[key] != value for key, value in hashes.items()):
        raise ValueError("baseline model/cases fingerprint mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = args.output_dir.resolve()
    plan = {"schema": 1, "output_dir": str(root), "candidates": [760, 20943, 1118, 2614],
            "labels": ["length_32-repeat0", "length_32-repeat1"],
            "capture_input_ids": cases[1]["input_ids"],
            "baseline": str(args.baseline.resolve()),
            "baseline_sha256": hashlib.sha256(baseline_bytes).hexdigest(), **hashes}
    write_json_new(root / "probe-plan.json", plan)
    os.environ[PLAN_ENV] = str(root / "probe-plan.json")
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, args.physical_devices))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    result = {"status": "starting", "tp": args.tp, "config": config,
              "physical_devices_requested": args.physical_devices, "runs": [],
              "measurement": "read-only synchronous diagnostic, not benchmark",
              "baseline_sha256": plan["baseline_sha256"], **hashes,
              "reproducibility": {"environment": harness.selected_environment(),
                  "source_trees": harness.project_source_fingerprints(),
                  "diagnostic_scripts": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                         for name in ("audit_decode.py", "decode_probe.py", "run_model.py")}}}
    engine = None
    gate = root / "capture-gate.json"
    def save():
        (root / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    save()
    try:
        import torch
        import torch_npu  # noqa: F401
        if not torch.npu.is_available() or torch.npu.device_count() != args.tp:
            raise RuntimeError("NPU availability/visibility mismatch")
        from tokenspeed.runtime.entrypoints.engine import Engine
        harness.snapshot(root / "hbm-before.txt")
        engine = Engine(**config)
        result["reproducibility"]["runtime_versions"] = harness.runtime_versions()
        result["process_snapshot_after_load"] = harness.process_snapshot()
        harness.snapshot(root / "hbm-loaded.txt")
        for case in cases:
            params = {"temperature": 0, "max_new_tokens": 16, "ignore_eos": True}
            warmup = engine.generate(input_ids=[case["input_ids"]], sampling_params=params)
            harness.validate_outputs(warmup, batch_size=1, new_tokens=16, input_tokens=len(case["input_ids"]))
            for repeat in range(2 if case["name"] == "length_32" else 1):
                if case["name"] == "length_32":
                    write_json_new(gate, {"label": f"length_32-repeat{repeat}"})
                try:
                    outputs = engine.generate(input_ids=[case["input_ids"]], sampling_params=params,
                                              return_logprob=True)
                finally:
                    gate.unlink(missing_ok=True)
                harness.validate_outputs(outputs, batch_size=1, new_tokens=16, input_tokens=len(case["input_ids"]))
                reference = next(r for r in baseline["runs"] if r["case"] == case["name"] and r["repeat"] == 0)
                exact = outputs[0]["output_ids"] == reference["outputs"][0]["output_ids"]
                result["runs"].append({"case": case["name"], "repeat": repeat, "outputs": outputs,
                                       "baseline_output_ids_exact": exact})
                save()
        for label in plan["labels"]:
            for rank in range(args.tp):
                paths = sorted((root / f"{label}-rank{rank}").glob("step*.json"))
                if len(paths) != 16:
                    raise ValueError(f"missing recurrent capture: {label}, rank={rank}")
        result["status"] = "diagnostic_completed_not_compared"
        result["all_baseline_output_ids_exact"] = all(r["baseline_output_ids_exact"] for r in result["runs"])
        if not result["all_baseline_output_ids_exact"]:
            raise ValueError("observer did not reproduce baseline output IDs")
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        gate.unlink(missing_ok=True)
        if engine is not None:
            engine.shutdown()
        save()
        os.environ.pop(PLAN_ENV, None)


# Engine uses multiprocessing spawn. Its workers re-import this entrypoint;
# only these explicitly launched diagnostic children install the observer.
if __name__ == "__mp_main__":
    from decode_probe import install
    install(Path(os.environ[PLAN_ENV]))

if __name__ == "__main__":
    main()
