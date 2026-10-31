"""Bounded, provenance-linked full-prefill numerical audit (not decode equivalence)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

# Also support importlib-based CPU tests without installing this scripts directory.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
import run_model as harness
import compare_reference as strict

TP_FIELDS = strict.TP_FIELDS
PRECISION = {"dtype": "bfloat16", "enforce_eager": True, "quantization": None,
             "model_modified": False}
SEMANTICS = "independent_full_prefill_one_token_on_reference_history"
PLAN_SCHEMA = "prefix-audit-plan-v1"
RESULT_SCHEMA = "prefix-audit-result-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return sha256(canonical(value))


def hash_string(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def ids(value, count=None, vocab=None):
    return (strict.token_ids(value, count)
            and (vocab is None or all(token < vocab for token in value)))


def logprob(value):
    return strict.finite(value) and value <= 0


def top_values(top, count, vocab=None):
    require(isinstance(top, dict) and ids(top["ids"], count, vocab)
            and len(set(top["ids"])) == count, "invalid topK token IDs")
    logs = top["logprobs"]
    require(isinstance(logs, list) and len(logs) == count and all(map(logprob, logs))
            and logs == sorted(logs, reverse=True), "invalid/nonfinite topK logprobs")
    return dict(zip(top["ids"], logs))


def expected_prefixes(case, trajectories):
    """Never attach a trajectory's token after its history leaves the reference."""
    reference = case["reference_output_ids"]
    rows = []
    for step in range(16):
        candidates = set(case["reference_top5"][step]["ids"])
        candidates.add(reference[step])
        eligible = []
        for trajectory in trajectories:
            if trajectory["output_ids"][:step] == reference[:step]:
                candidates.add(trajectory["output_ids"][step])
                eligible.append(trajectory["label"])
        rows.append({"case": case["name"], "step": step,
                     "input_ids": case["input_ids"] + reference[:step],
                     "reference_token_id": reference[step],
                     "candidate_ids": sorted(candidates), "eligible_trajectories": eligible})
    return rows


def make_plan(documents, sources):
    """Parse the real saved reference/run schemas; no inferred run input IDs."""
    cases_doc = documents["cases"]
    require(type(cases_doc["new_tokens"]) is int and cases_doc["new_tokens"] == 16,
            "canonical cases must specify 16 new tokens")
    cases = cases_doc["cases"]
    require(isinstance(cases, list) and cases
            and all(isinstance(c["name"], str) and c["name"] and ids(c["input_ids"])
                    and len(c["input_ids"]) + 16 <= 512 for c in cases), "invalid canonical cases")
    require(len({c["name"] for c in cases}) == len(cases), "duplicate canonical case")
    reference = documents["reference"]
    strict.validate_reference(reference, cases)
    model_path = strict.resolved_model(reference["model"])
    for name, tp in (("tp1", 1), ("tp2", 2)):
        if name not in documents:
            continue
        run = documents[name]
        strict.validate_run(run, cases, sources["cases"]["sha256"], model_path, tp=tp)
        for key in ("manifest_sha256", "config_sha256"):
            require(hash_string(run[key]) and run[key] == documents["tp2"][key],
                    f"source {name} {key} mismatch/invalid")
    selected = [c for c in reference["cases"] if c["name"] == "length_32"]
    require(len(selected) == 1 and len(selected[0]["input_ids"]) == 32,
            "audit requires the canonical length_32 case")
    ref = selected[0]
    require(isinstance(ref["top5"], list) and len(ref["top5"]) == 16,
            "missing reference top5 positions")
    for step, top in enumerate(ref["top5"]):
        top_values(top, 5)
        require(ref["output_ids"][step] in top["ids"]
                and ref["selected_logprobs"][step] == top["logprobs"][top["ids"].index(ref["output_ids"][step])],
                "reference selected token/top5 disagree")
    case = {"name": ref["name"], "input_ids": ref["input_ids"],
            "reference_output_ids": ref["output_ids"], "reference_top5": ref["top5"]}
    trajectories = []
    for name in ("tp1", "tp2"):
        if name in documents:
            for row in documents[name]["runs"]:
                if row["case"] != case["name"]:
                    continue
                for request, output in enumerate(row["outputs"]):
                    trajectories.append({"label": f"{name}:repeat{row['repeat']}:request{request}",
                                         "output_ids": output["output_ids"]})
    divergences = []
    for trajectory in trajectories:
        step = next((i for i, (a, b) in enumerate(zip(
            trajectory["output_ids"], ref["output_ids"])) if a != b), None)
        divergences.append({"trajectory": trajectory["label"], "first_divergence_step": step,
                            "prefix_already_in_reference_plan": step is not None})
    plan = {"schema": PLAN_SCHEMA, "semantics": SEMANTICS, "precision": PRECISION.copy(),
            "expected_model": {"path": model_path,
                               "manifest_sha256": documents["tp2"]["manifest_sha256"],
                               "config_sha256": documents["tp2"]["config_sha256"]},
            "sources": sources, "planner_scripts_sha256": scripts_fingerprint(),
            "case": case, "trajectories": trajectories,
            "first_divergences": divergences, "prefixes": expected_prefixes(case, trajectories),
            "provenance_limit": "Saved Transformers reference lacks manifest/config hashes. Expected model fingerprints come from TP2 (checked against TP1 when supplied); this does not retroactively fingerprint reference weights."}
    validate_plan(plan)
    return plan


def validate_plan(plan, vocab=None):
    require(plan["schema"] == PLAN_SCHEMA and plan["semantics"] == SEMANTICS
            and plan["precision"] == PRECISION, "invalid plan semantics/precision/schema")
    model = plan["expected_model"]
    require(strict.resolved_model(model["path"]) == model["path"]
            and all(hash_string(model[k]) for k in ("manifest_sha256", "config_sha256")),
            "invalid expected model provenance")
    planner_scripts = plan["planner_scripts_sha256"]
    require(set(planner_scripts) == {"audit_prefix.py", "run_model.py", "compare_reference.py"}
            and all(map(hash_string, planner_scripts.values())), "invalid planner script provenance")
    sources = plan["sources"]
    require(set(sources) in ({"cases", "reference", "tp2"}, {"cases", "reference", "tp1", "tp2"}),
            "invalid plan sources")
    for source in sources.values():
        require(isinstance(source["path"], str) and Path(source["path"]).is_absolute()
                and hash_string(source["sha256"]), "invalid source provenance")
    case = plan["case"]
    require(case["name"] == "length_32" and ids(case["input_ids"], 32, vocab)
            and ids(case["reference_output_ids"], 16, vocab), "invalid reference history/IDs")
    require(len(case["reference_top5"]) == 16, "missing reference top5")
    for step, top in enumerate(case["reference_top5"]):
        values = top_values(top, 5, vocab)
        require(case["reference_output_ids"][step] in values, "reference target absent from top5")
    trajectories = plan["trajectories"]
    require(isinstance(trajectories, list) and trajectories
            and len({t["label"] for t in trajectories}) == len(trajectories)
            and all(isinstance(t["label"], str) and ids(t["output_ids"], 16, vocab)
                    for t in trajectories), "invalid trajectories")
    require(plan["prefixes"] == expected_prefixes(case, trajectories),
            "prefix/candidates differ from reference history; alternate-history evidence forbidden")
    for row in plan["prefixes"]:
        require(type(row["step"]) is int and type(row["reference_token_id"]) is int
                and ids(row["input_ids"], vocab=vocab) and ids(row["candidate_ids"], vocab=vocab),
                "invalid prefix/candidate IDs")
    require(all(len(row["input_ids"]) + 1 <= 512 for row in plan["prefixes"]),
            "prefix exceeds context 512")
    expected = []
    for trajectory in trajectories:
        step = next((i for i in range(16) if trajectory["output_ids"][i]
                     != case["reference_output_ids"][i]), None)
        expected.append({"trajectory": trajectory["label"], "first_divergence_step": step,
                         "prefix_already_in_reference_plan": step is not None})
    require(plan["first_divergences"] == expected, "invalid first-divergence history")


def read_sources(paths):
    raw = {name: Path(path).read_bytes() for name, path in paths.items()}
    return ({name: json.loads(data) for name, data in raw.items()},
            {name: {"path": str(Path(paths[name]).resolve()), "sha256": sha256(data)}
             for name, data in raw.items()})


def verify_sources(plan):
    documents, sources = read_sources({name: source["path"] for name, source in plan["sources"].items()})
    require(sources == plan["sources"], "source fingerprint mismatch")
    require(make_plan(documents, sources) == plan, "plan does not reproduce from hashed source evidence")


def model_fingerprint(model_dir, expected):
    model_dir = Path(model_dir).resolve()
    require(str(model_dir) == expected["path"], "model path differs from plan")
    manifest_raw = (model_dir / "source-manifest.json").read_bytes()
    config_raw = (model_dir / "config.json").read_bytes()
    manifest = json.loads(manifest_raw)
    entries = [entry for entry in manifest["files"] if entry["path"] == "config.json"]
    require(len(entries) == 1 and entries[0]["sha256"] == sha256(config_raw),
            "model config does not match manifest")
    actual = {"path": str(model_dir), "manifest_sha256": sha256(manifest_raw),
              "config_sha256": sha256(config_raw)}
    require(actual == expected, "model manifest/config fingerprint differs from plan")
    config = json.loads(config_raw)
    vocab = config.get("text_config", config)["vocab_size"]
    require(type(vocab) is int and vocab >= 20, "invalid model vocabulary")
    return actual, vocab


def parse_tokenspeed(output, prefix, top_k, vocab):
    harness.validate_outputs([output], batch_size=1, new_tokens=1, input_tokens=len(prefix))
    require(all(type(output["meta_info"][key]) is int for key in
                ("prompt_tokens", "completion_tokens", "cached_tokens")), "invalid token-count metadata")
    entries = output["meta_info"]["output_top_logprobs"]
    require(isinstance(entries, list) and len(entries) == 1
            and isinstance(entries[0], list), "missing one-token topK payload")
    require(all(isinstance(entry, (list, tuple)) and len(entry) == 3 and entry[2] is None
                for entry in entries[0]), "invalid [logprob, tokenID, None] entries")
    top = {"ids": [entry[1] for entry in entries[0]],
           "logprobs": [entry[0] for entry in entries[0]]}
    top_values(top, top_k, vocab)
    require(output["output_ids"][0] in top["ids"], "generated token missing from topK")
    return top


def measured_row(prefix, top, values, generated):
    candidates = prefix["candidate_ids"]
    require(all(type(token) is int and logprob(value) for token, value in values.items()),
            "invalid candidate measurements")
    return {**prefix, "generated_token_id": generated, "top": top,
            "candidate_logprobs": {str(token): values[token] for token in candidates if token in values},
            "missing_candidate_ids": [token for token in candidates if token not in values]}


def scripts_fingerprint():
    return {name: sha256((_SCRIPT_DIR / name).read_bytes())
            for name in ("audit_prefix.py", "run_model.py", "compare_reference.py")}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def execute(args):
    raw_plan = args.plan.read_bytes()
    plan = json.loads(raw_plan)
    validate_plan(plan)
    verify_sources(plan)
    model, vocab = model_fingerprint(args.model, plan["expected_model"])
    validate_plan(plan, vocab)
    config = harness.build_config(args.model, tp=args.tp, devices=args.physical_devices,
                                  memory_fraction=args.memory_fraction, concurrency=1, port=args.port)
    require(args.backend != "transformers" or args.tp == 1,
            "Transformers audit requires one device, not TP2")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, args.physical_devices))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.backend == "transformers":
        config = {"model": model["path"], "dtype": "bfloat16", "attn_implementation": "eager",
                  "trust_remote_code": False, "local_files_only": True, "use_cache": False}
    result = {"schema": RESULT_SCHEMA, "status": "starting", "semantics": SEMANTICS,
              "backend": args.backend, "plan": plan, "plan_sha256": digest(plan),
              "plan_file_sha256": sha256(raw_plan), "model": model, "model_sha256": digest(model),
              "vocab_size": vocab, "precision": PRECISION.copy(), "config": config,
              "tp": args.tp, "physical_devices_requested": args.physical_devices, "top_k": args.top_k,
              "rows": [], "reproducibility": {"scripts_sha256": scripts_fingerprint(),
                  "environment": harness.selected_environment(), "runtime_versions": harness.runtime_versions()},
              "scope": "All required candidates on the same full-prefill reference prefixes only; not recurrent decode or full-vocabulary equality."}
    def save():
        write_json(args.output_dir / "result.json", result)
    save()
    engine = None
    try:
        import torch
        import torch_npu  # noqa: F401
        require(torch.npu.is_available() and torch.npu.device_count() == args.tp,
                "NPU availability/visible count mismatch")
        torch.manual_seed(42)
        harness.snapshot(args.output_dir / "hbm-before.txt")
        if args.backend == "tokenspeed":
            trees = harness.project_source_fingerprints()
            require(all(hash_string(tree.get("sha256")) for tree in trees.values()),
                    "TokenSpeed source fingerprint unavailable")
            result["reproducibility"]["source_trees"] = trees
            save()
            from tokenspeed.runtime.entrypoints.engine import Engine
            engine = Engine(**config)
        else:
            import transformers
            result["reproducibility"]["runtime_versions"]["transformers"] = transformers.__version__
            # Load the unmodified conditional-generation model, as in the saved reference.
            # No offload, quantization, module replacement, or model-code changes.
            model_instance = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
                str(args.model.resolve()), dtype=torch.bfloat16, attn_implementation="eager",
                trust_remote_code=False, local_files_only=True).to("npu:0").eval()
            result["memory_fraction_note"] = "TokenSpeed allocator option only; not applied to Transformers. Single-device BF16 model must fit."
        result["reproducibility"]["runtime_versions"].update(harness.runtime_versions())
        result["process_snapshot_after_load"] = harness.process_snapshot()
        harness.snapshot(args.output_dir / "hbm-loaded.txt")
        save()
        for prefix in plan["prefixes"]:
            if args.backend == "tokenspeed":
                outputs = engine.generate(input_ids=[prefix["input_ids"][:]],
                    sampling_params={"temperature": 0, "max_new_tokens": 1, "ignore_eos": True},
                    return_logprob=True, top_logprobs_num=args.top_k)
                require(isinstance(outputs, list) and len(outputs) == 1, "expected one request result")
                top = parse_tokenspeed(outputs[0], prefix["input_ids"], args.top_k, vocab)
                values = top_values(top, args.top_k, vocab)
                row = measured_row(prefix, top, values, outputs[0]["output_ids"][0])
                row["raw_output"] = outputs[0]
            else:
                with torch.inference_mode():
                    inputs = torch.tensor([prefix["input_ids"]], dtype=torch.long, device="npu:0")
                    output = model_instance(input_ids=inputs, use_cache=False)
                    logs = output.logits[0, -1].float().log_softmax(dim=-1)
                    require(logs.shape[0] == vocab, "model logits vocabulary differs from config")
                    top_logs, top_ids = logs.topk(args.top_k)
                    top = {"ids": top_ids.tolist(), "logprobs": top_logs.tolist()}
                    values = {token: logs[token].item() for token in prefix["candidate_ids"]}
                    top_values(top, args.top_k, vocab)
                    row = measured_row(prefix, top, values, int(logs.argmax().item()))
                    del output, logs, inputs, top_logs, top_ids
            result["rows"].append(row)
            save()
        harness.snapshot(args.output_dir / "hbm-after.txt")
        result["status"] = ("audit_coverage_failed" if any(r["missing_candidate_ids"] for r in result["rows"])
                            else "audit_completed_not_compared")
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if engine is not None:
                engine.shutdown()
        except BaseException as exc:
            result["status"] = "failed"
            result["shutdown_error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            save()
    return 0 if result["status"] == "audit_completed_not_compared" else 1


def validate_result(result):
    require(result["schema"] == RESULT_SCHEMA and result["status"] == "audit_completed_not_compared"
            and result["semantics"] == SEMANTICS, "audit incomplete or candidate coverage failed")
    plan = result["plan"]
    vocab = result["vocab_size"]
    require(type(vocab) is int and vocab >= 20, "invalid vocabulary")
    validate_plan(plan, vocab)
    require(hash_string(result["plan_file_sha256"]) and result["plan_sha256"] == digest(plan),
            "invalid plan fingerprint")
    require(result["model"] == plan["expected_model"] and result["model_sha256"] == digest(result["model"]),
            "invalid model fingerprint")
    require(result["precision"] == PRECISION and type(result["top_k"]) is int
            and result["top_k"] in (10, 20), "unsupported precision/topK")
    versions = result["reproducibility"]["runtime_versions"]
    require(isinstance(versions, dict) and {"torch", "torch_npu", "triton"} <= set(versions)
            and all(isinstance(value, str) and value for value in versions.values()),
            "missing runtime versions")
    devices = result["physical_devices_requested"]
    require(ids(devices) and len(set(devices)) == len(devices), "invalid physical device IDs")
    require(type(result["tp"]) is int and result["tp"] in (1, 2), "invalid TP degree")
    scripts = result["reproducibility"]["scripts_sha256"]
    require(set(scripts) == {"audit_prefix.py", "run_model.py", "compare_reference.py"}
            and all(map(hash_string, scripts.values())) and scripts == plan["planner_scripts_sha256"],
            "missing/mismatched planner/runner script fingerprints")
    config = result["config"]
    if result["backend"] == "tokenspeed":
        expected = harness.build_config(result["model"]["path"], tp=result["tp"],
            devices=result["physical_devices_requested"], memory_fraction=config["gpu_memory_utilization"],
            concurrency=1, port=config["port"])
        require(config == expected, "unexpected TokenSpeed numerical/runtime config")
        trees = result["reproducibility"]["source_trees"]
        require(set(trees) == {"python", "tokenspeed-kernel/python", "tokenspeed-kernel-npu/python"}
                and all(hash_string(tree["sha256"]) and digest(tree["files"]) == tree["sha256"]
                        for tree in trees.values()), "invalid runtime source fingerprints")
    else:
        require(result["backend"] == "transformers" and result["tp"] == 1
                and len(result["physical_devices_requested"]) == 1
                and isinstance(versions.get("transformers"), str) and versions["transformers"],
                "invalid Transformers placement/version")
        require(config == {"model": result["model"]["path"], "dtype": "bfloat16",
            "attn_implementation": "eager", "trust_remote_code": False,
            "local_files_only": True, "use_cache": False}, "unexpected Transformers precision/config")
    require(len(result["rows"]) == len(plan["prefixes"]), "missing/extra measured prefixes")
    for row, prefix in zip(result["rows"], plan["prefixes"]):
        require(type(row["step"]) is int and type(row["reference_token_id"]) is int
                and ids(row["input_ids"], vocab=vocab) and ids(row["candidate_ids"], vocab=vocab),
                "invalid measured prefix/candidate IDs")
        require({key: row[key] for key in prefix} == prefix, "measured prefix/candidate IDs differ")
        require(row["missing_candidate_ids"] == [] and set(row["candidate_logprobs"])
                == {str(token) for token in prefix["candidate_ids"]}, "missing/extra candidate values")
        require(all(map(logprob, row["candidate_logprobs"].values())), "nonfinite candidate values")
        values = top_values(row["top"], result["top_k"], vocab)
        require(type(row["generated_token_id"]) is int and row["generated_token_id"] in values
                and values[row["generated_token_id"]] == max(values.values()), "invalid argmax/topK")
        for token, value in row["candidate_logprobs"].items():
            if int(token) in values:
                require(value == values[int(token)], "candidate/topK measurements disagree")
        if result["backend"] == "tokenspeed":
            require(parse_tokenspeed(row["raw_output"], row["input_ids"], result["top_k"], vocab)
                    == row["top"], "raw TokenSpeed topK differs")
            require(row["raw_output"]["output_ids"] == [row["generated_token_id"]],
                    "raw TokenSpeed argmax differs")
            require(set(prefix["candidate_ids"]) <= set(values), "required candidate absent from topK")


def compare(left, right, atol):
    require(strict.finite(atol) and 0 <= atol <= 0.15, "atol must be finite and in [0, 0.15]")
    for result in (left, right):
        validate_result(result)
    for key in ("plan_sha256", "plan_file_sha256", "model_sha256", "precision", "vocab_size"):
        require(left[key] == right[key], f"comparison {key} mismatch")
    require(left["reproducibility"]["scripts_sha256"] == right["reproducibility"]["scripts_sha256"],
            "audit/helper script versions differ")
    if left["backend"] == right["backend"] == "tokenspeed":
        require({k: v for k, v in left["config"].items() if k not in TP_FIELDS}
                == {k: v for k, v in right["config"].items() if k not in TP_FIELDS},
                "TokenSpeed configs may differ only in parallel degree")
        require({k: v["sha256"] for k, v in left["reproducibility"]["source_trees"].items()}
                == {k: v["sha256"] for k, v in right["reproducibility"]["source_trees"].items()},
                "TokenSpeed source versions differ")
        require(left["reproducibility"]["runtime_versions"] == right["reproducibility"]["runtime_versions"],
                "TokenSpeed runtime versions differ")
    rows = []
    for a, b in zip(left["rows"], right["rows"]):
        require(a["input_ids"] == b["input_ids"] and a["candidate_ids"] == b["candidate_ids"],
                "comparison prefix/candidate IDs differ")
        errors = {token: abs(value - b["candidate_logprobs"][token])
                  for token, value in a["candidate_logprobs"].items()}
        def diagnostic(row):
            logs = row["top"]["logprobs"]
            return {"argmax_token_id": row["generated_token_id"], "top1_minus_top2": logs[0] - logs[1],
                    "exact_top_tie_ids": [token for token, value in zip(row["top"]["ids"], logs)
                                          if value == logs[0]]}
        maximum = max(errors.values())
        rows.append({"case": a["case"], "step": a["step"], "max_candidate_error": maximum,
                     "candidate_errors": errors, "passed": maximum <= atol,
                     "argmax_equal": a["generated_token_id"] == b["generated_token_id"],
                     "left_diagnostics": diagnostic(a), "right_diagnostics": diagnostic(b)})
    return {"status": "audit_compared", "passed": all(row["passed"] for row in rows),
            "atol": atol, "max_candidate_error": max(row["max_candidate_error"] for row in rows),
            "plan_sha256": left["plan_sha256"], "model_sha256": left["model_sha256"], "rows": rows,
            "scope": "Bounded candidate logprobs on identical full-prefill prefixes only. Argmax/ties are diagnostics, not pass criteria. No claim of recurrent decode-path or full-vocabulary equality; old strict failures are unchanged.",
            "config_comparison": ("parallel-degree differences only" if left["backend"] == right["backend"] == "tokenspeed"
                                  else "shared BF16 unmodified eager precision subset; backend-specific configs validated independently")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    plan_parser = modes.add_parser("plan", help="Create an explicit 16-reference-prefix plan")
    for name in ("reference", "tp2", "cases", "output"):
        plan_parser.add_argument(f"--{name}", type=Path, required=True)
    plan_parser.add_argument("--tp1", type=Path)
    run_parser = modes.add_parser("run", help="Run one independent full-prefill request per planned prefix")
    run_parser.add_argument("--backend", choices=("tokenspeed", "transformers"), required=True)
    for name in ("plan", "model", "output-dir"):
        run_parser.add_argument(f"--{name}", type=Path, required=True)
    run_parser.add_argument("--tp", type=int, choices=(1, 2), required=True)
    run_parser.add_argument("--physical-devices", type=harness.parse_devices, required=True)
    run_parser.add_argument("--memory-fraction", type=float, required=True)
    run_parser.add_argument("--port", type=int, required=True)
    run_parser.add_argument("--top-k", type=int, choices=(10, 20), required=True)
    compare_parser = modes.add_parser("compare", help="Explicit bounded-candidate comparison")
    for name in ("left", "right", "output"):
        compare_parser.add_argument(f"--{name}", type=Path, required=True)
    compare_parser.add_argument("--atol", type=float, required=True)
    args = parser.parse_args(argv)
    try:
        if args.mode == "run":
            return execute(args)
        names = ("reference", "tp1", "tp2", "cases") if args.mode == "plan" else ("left", "right")
        paths = {name: getattr(args, name) for name in names if getattr(args, name, None) is not None}
        # New outputs only, including when a run/plan is invalid. Never touch old verdicts.
        require(not args.output.exists() and args.output.resolve() not in {p.resolve() for p in paths.values()},
                "output must be a new file, distinct from all inputs")
        documents, sources = read_sources(paths)
        if args.mode == "plan":
            result = make_plan(documents, sources)
        else:
            result = compare(documents["left"], documents["right"], args.atol)
            result["inputs"] = sources
        write_json(args.output, result)
        print(json.dumps({"output": str(args.output), "status": result.get("status", "plan_created"),
                          "passed": result.get("passed")}, allow_nan=False))
        return 0 if result.get("passed", True) else 1
    except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
        parser.error(f"invalid audit evidence/config: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
