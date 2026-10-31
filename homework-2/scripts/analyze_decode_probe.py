"""Fail-closed, NumPy-only analysis of the bounded FP16 4B decode diagnostic.

No accelerator imports, generation, benchmarks, or new correctness criterion.
Only the two anchored retry repetitions are admissible. Stored FP64 candidate
scalars are observer evidence, not an offline recomputation from saved weights.
"""
import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import zlib

import numpy as np

MAX_BYTES = 4 * 1024 * 1024
VOCAB_SIZE = 248320
HIDDEN_SIZE = 2560
STEPS = 16
LABELS = ["length_32-repeat0", "length_32-repeat1"]
CANDIDATES = [760, 20943, 1118, 2614]
TREE_NAMES = {"python", "tokenspeed-kernel/python", "tokenspeed-kernel-npu/python"}
TP_CONFIG = {"world_size", "attn_tp_size", "dense_tp_size", "moe_tp_size", "lm_head_tp_size"}
HASH_KEYS = ("manifest_sha256", "config_sha256", "cases_sha256")


class EvidenceError(ValueError):
    """Missing, contradictory, malformed, or out-of-scope evidence."""


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
            "malformed SHA256")
    return value


def bounded_read(path):
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    require(len(data) <= MAX_BYTES, f"file exceeds 4 MiB: {path}")
    return data


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _bad_constant(value):
    raise EvidenceError(f"nonfinite JSON value: {value}")


def document(path, provenance):
    path = Path(path)
    require(not path.is_symlink(), f"symlink JSON is not owned evidence: {path}")
    raw = bounded_read(path)
    result = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_bad_constant,
                        parse_float=lambda value: finite(float(value)))
    require(isinstance(result, dict), f"expected JSON object: {path}")
    provenance.append({"path": str(path.resolve()), "sha256": sha256(raw), "kind": "json"})
    return result, sha256(raw)


def token_ids(value, length):
    require(isinstance(value, list) and len(value) == length
            and all(type(t) is int and 0 <= t < VOCAB_SIZE for t in value),
            "invalid token IDs/length")
    return value


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), "nonfinite/invalid scalar")
    return value


def logsoftmax(logits):
    """Stable float64 normalization over the *entire* captured vocabulary."""
    values = logits.astype(np.float64).reshape(-1)
    shifted = values - values.max()
    return shifted - np.log(np.exp(shifted).sum(dtype=np.float64))


def load_tensor(folder, stem, name, metadata, shape, provenance):
    """Prefer gzip; invalid gzip is fatal, even if a valid raw sibling exists."""
    filename = f"{stem}-{name}.npy"
    require(metadata["path"] == filename, "unexpected tensor path")
    require(metadata["shape"] == list(shape)
            and all(type(dim) is int for dim in metadata["shape"])
            and metadata["storage_dtype"] == "float32",
            "tensor metadata shape/dtype mismatch")
    expected = digest(metadata["sha256"])
    raw_path = Path(folder) / filename
    zipped = raw_path.with_name(filename + ".gz")
    selected = zipped if zipped.exists() or zipped.is_symlink() else raw_path
    require(not selected.is_symlink(), "symlink tensor is not owned evidence")
    selected_bytes = bounded_read(selected)
    if selected == zipped:
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(selected_bytes)) as stream:
                raw = stream.read(MAX_BYTES + 1)
        except (OSError, EOFError, zlib.error) as exc:
            raise EvidenceError(f"invalid gzip: {selected}") from exc
        require(len(raw) <= MAX_BYTES, "decompressed tensor exceeds 4 MiB")
    else:
        raw = selected_bytes
    require(sha256(raw) == expected, f"raw tensor hash mismatch: {selected}")
    # Validate the header before np.load can allocate based on an untrusted shape.
    stream = io.BytesIO(raw)
    version = np.lib.format.read_magic(stream)
    require(version in ((1, 0), (2, 0)), "unsupported NPY header version")
    reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
              else np.lib.format.read_array_header_2_0)
    actual_shape, fortran, dtype = reader(stream)
    require(actual_shape == shape and dtype == np.dtype("float32") and not fortran,
            "raw tensor shape/dtype/order mismatch")
    require(len(raw) == stream.tell() + math.prod(shape) * 4, "NPY payload/trailing bytes mismatch")
    array = np.load(io.BytesIO(raw), allow_pickle=False)
    require(np.isfinite(array).all(), "nonfinite tensor")
    require(np.array_equal(array, array.astype(np.float16).astype(np.float32)),
            "tensor not exactly representable in recorded native FP16 dtype")
    provenance.append({"path": str(selected.resolve()), "kind": "tensor",
                       "selected_sha256": sha256(selected_bytes), "raw_sha256": expected,
                       "compression": "gzip" if selected == zipped else "none",
                       "raw_bytes": len(raw), "shape": list(shape), "storage_dtype": "float32"})
    return array


def validate_repro(repro):
    trees = repro["source_trees"]
    require(set(trees) == TREE_NAMES, "missing source-tree provenance")
    for tree in trees.values():
        require("error" not in tree and tree["files"], "failed/empty source fingerprint")
        for path, value in tree["files"].items():
            require(not Path(path).is_absolute() and ".." not in Path(path).parts
                    and path.endswith(".py"), "invalid source fingerprint path")
            digest(value)
        canonical = json.dumps(tree["files"], sort_keys=True, separators=(",", ":")).encode()
        require(digest(tree["sha256"]) == sha256(canonical), "source-tree hash mismatch")
    versions = repro["runtime_versions"]
    require(set(versions) == {"torch", "torch_npu", "triton"}
            and all(isinstance(v, str) and v and v != "unavailable" for v in versions.values()),
            "missing runtime versions")
    require(set(repro["environment"]) == {"ASCEND_RT_VISIBLE_DEVICES", "TOKENIZERS_PARALLELISM",
                                         "HCCL_NPU_SOCKET_PORT_RANGE"}, "missing runtime environment")


def owned_schedulers(result, tp):
    snapshot = result["process_snapshot_after_load"]
    require("error" not in snapshot and snapshot["stage"] == "after_engine_startup",
            "invalid process snapshot")
    parent = snapshot["parent_pid"]
    require(type(parent) is int and parent > 0 and snapshot["parent"]["pid"] == parent,
            "invalid runner PID")
    children = snapshot["children"]
    pids = [p["pid"] for p in children]
    require(all(type(pid) is int and pid > 0 and pid != parent for pid in pids)
            and len(set(pids)) == len(pids), "invalid/duplicate child PID")
    by_pid = {p["pid"]: p for p in children}
    for child in children:
        seen = set()
        ancestor = child["pid"]
        while ancestor != parent:
            require(ancestor in by_pid and ancestor not in seen, "unowned/cyclic child PID")
            seen.add(ancestor)
            ancestor = by_pid[ancestor]["ppid"]
    schedulers = [p for p in children if p["name"].startswith("tokenspeed::scheduler")]
    require(len(schedulers) == tp, "unexpected scheduler count")
    # Rank is recorded by the observer, not inferred from a process-name suffix.
    return {p["pid"] for p in schedulers}


def validate_runs(result, cases, diagnostic):
    expected = [("smoke", 0), ("length_32", 0), ("length_32", 1), ("length_256", 0)]
    if not diagnostic:
        expected.remove(("length_32", 1))
    require([(r["case"], r["repeat"]) for r in result["runs"]] == expected,
            "missing/duplicate/unexpected case repetitions")
    runs = {}
    for run in result["runs"]:
        require(len(run["outputs"]) == 1, "diagnostic/baseline must be single-request")
        output = run["outputs"][0]
        ids = token_ids(output["output_ids"], STEPS)
        meta = output["meta_info"]
        require(meta["prompt_tokens"] == len(cases[run["case"]])
                and meta["completion_tokens"] == STEPS and meta["cached_tokens"] == 0,
                "output token accounting mismatch")
        require(meta["finish_reason"] == {"type": "length", "length": STEPS},
                "unexpected generation finish reason")
        entries = meta["output_token_logprobs"]
        require(len(entries) == STEPS, "missing output logprobs")
        for token, entry in zip(ids, entries):
            require(len(entry) == 3 and entry[1] == token, "output logprob token mismatch")
            finite(entry[0])
        if diagnostic:
            require(run["baseline_output_ids_exact"] is True, "reported baseline mismatch")
        runs[(run["case"], run["repeat"])] = output
    return runs


def validate_step(folder, label, rank, step, plan_hash, output, prompt, schedulers, tp, provenance):
    stem = f"step{step:02d}"
    data, _ = document(folder / f"{stem}.json", provenance)
    require(all(type(data[key]) is int for key in ("schema", "rank", "step_index"))
            and data["schema"] == 1 and data["label"] == label and data["rank"] == rank
            and data["step_index"] == step, "step identity mismatch")
    require(digest(data["plan_sha256"]) == plan_hash, "capture plan hash mismatch")
    require(type(data["pid"]) is int and data["pid"] in schedulers, "unowned scheduler PID")
    expected_input = prompt if step == 0 else [output["output_ids"][step - 1]]
    token_ids(data["input_ids"], len(expected_input))
    require(data["input_ids"] == expected_input, "input history mismatch")
    require(type(data["num_extends"]) is int and type(data["input_num_tokens"]) is int
            and data["forward_mode"] == ("EXTEND" if step == 0 else "DECODE")
            and data["num_extends"] == (1 if step == 0 else 0)
            and data["input_num_tokens"] == (32 if step == 0 else 1),
            "invalid prefill/decode anchor")
    require(data["native_logits_dtype"] == data["hidden_dtype"] == "torch.float16",
            "unexpected native dtype")
    require(set(data["tensors"]) == {"logits", "hidden"}, "missing/unexpected tensors")
    logits = load_tensor(folder, stem, "logits", data["tensors"]["logits"], (1, VOCAB_SIZE), provenance)[0]
    hidden = load_tensor(folder, stem, "hidden", data["tensors"]["hidden"], (1, HIDDEN_SIZE), provenance)[0]
    ties = np.flatnonzero(logits == logits.max()).tolist()
    require(isinstance(data["max_tied_ids"], list)
            and all(type(t) is int for t in data["max_tied_ids"])
            and data["max_tied_ids"] == ties, "native maximum tie metadata mismatch")
    selected = int(np.argmax(logits))  # Native greedy argmax: lowest ID on exact ties.
    require(selected == output["output_ids"][step], "native argmax differs from actual output ID")
    require(set(data["candidates"]) == {str(t) for t in CANDIDATES}, "wrong candidate set")
    top = data["top10"]
    require(len(top) == 10 and len({t["token"] for t in top}) == 10, "invalid top10")
    token_ids([t["token"] for t in top], 10)
    require([t["logit"] for t in top] == sorted([t["logit"] for t in top], reverse=True)
            and top[-1]["logit"] == float(np.partition(logits, -10)[-10]), "wrong top10 cutoff/order")
    normalization_offsets = []
    for token, item in [(int(t), v) for t, v in data["candidates"].items()] + [(t["token"], t) for t in top]:
        require(finite(item["logit"]) == float(logits[token]), "candidate/top10 raw logit mismatch")
        native_lp = finite(item["logprob"])
        require(native_lp <= 0, "positive native logprob")
        normalization_offsets.append(item["logit"] - native_lp)
    # The observer's CPU float32 reduction is not the analysis normalization.
    # Check internal summary consistency, not agreement with a new metric cutoff.
    offset_scale = max(1.0, max(abs(v) for v in normalization_offsets))
    require(max(normalization_offsets) - min(normalization_offsets)
            <= 8 * np.finfo(np.float32).eps * offset_scale,
            "candidate/top10 logprobs have inconsistent normalization")
    shard = {"start": rank * VOCAB_SIZE // tp, "end": (rank + 1) * VOCAB_SIZE // tp}
    require(data["lm_head_shard"] == shard, "LM-head shard mismatch")
    dots = data["candidate_dot_fp64"]
    require(set(dots) == {str(t) for t in CANDIDATES if shard["start"] <= t < shard["end"]},
            "wrong candidate dot ownership/set")
    for value in dots.values():
        finite(value)
    return {"metadata": data, "logits": logits, "hidden": hidden}


def load_evidence(root, baseline_path, tp, cases, cases_hash, provenance):
    root = Path(root)
    result, result_hash = document(root / "result.json", provenance)
    plan, plan_hash = document(root / "probe-plan.json", provenance)
    baseline, baseline_hash = document(baseline_path, provenance)
    require(result["status"] == "diagnostic_completed_not_compared" and result["tp"] == tp
            and "error" not in result, "raw result did not complete successfully")
    require(baseline["status"] == "generation_completed_not_yet_compared"
            and baseline["mode"] == "correctness" and baseline["tp"] == tp
            and baseline["concurrency"] == 1 and "error" not in baseline, "invalid original baseline")
    require(result["all_baseline_output_ids_exact"] is True, "diagnostic reports baseline mismatch")
    require(plan["schema"] == 1 and plan["labels"] == LABELS and plan["candidates"] == CANDIDATES,
            "invalid capture plan labels/candidates")
    require(plan["capture_input_ids"] == cases["length_32"], "plan canonical prompt mismatch")
    # Original absolute paths are retained, not required to exist on this offline machine.
    require(Path(plan["output_dir"]).is_absolute() and Path(plan["baseline"]).is_absolute(),
            "missing original capture paths")
    require(digest(plan["baseline_sha256"]) == digest(result["baseline_sha256"]) == baseline_hash,
            "baseline hash mismatch")
    for key in HASH_KEYS:
        require(digest(result[key]) == digest(plan[key]) == digest(baseline[key]),
                f"model/cases provenance mismatch: {key}")
    require(result["cases_sha256"] == cases_hash, "canonical cases hash mismatch")
    config = result["config"]
    require({k: v for k, v in config.items() if k != "port"}
            == {k: v for k, v in baseline["config"].items() if k != "port"},
            "baseline non-port configuration mismatch")
    require(all(config[k] == tp for k in TP_CONFIG) and config["device"] == "npu"
            and config["dtype"] == config["kv_cache_dtype"] == "float16"
            and config["pipeline_parallel_size"] == 1 and config["sampling_backend"] == "greedy"
            and config["enable_prefix_caching"] is False and config["enable_output_logprobs"] is True,
            "unexpected diagnostic configuration")
    repro = result["reproducibility"]
    validate_repro(repro)
    validate_repro(baseline["reproducibility"])
    for key in ("source_trees", "runtime_versions", "environment"):
        require(repro[key] == baseline["reproducibility"][key], f"baseline runtime/source mismatch: {key}")
    scripts = repro["diagnostic_scripts"]
    require(set(scripts) == {"audit_decode.py", "decode_probe.py", "run_model.py"},
            "missing diagnostic script provenance")
    for name, value in scripts.items():
        digest(value)
        if name != "run_model.py":
            require(value == sha256(bounded_read(Path(__file__).with_name(name))),
                    f"capture not from the inspected anchored observer: {name}")
    devices = result["physical_devices_requested"]
    require(isinstance(devices, list) and len(devices) == tp and len(set(devices)) == tp
            and all(type(d) is int and d >= 0 for d in devices)
            and devices == baseline["physical_devices_requested"], "physical-device mismatch")
    require(repro["environment"]["ASCEND_RT_VISIBLE_DEVICES"] == ",".join(map(str, devices))
            and repro["environment"]["TOKENIZERS_PARALLELISM"] == "false", "runtime visibility mismatch")
    runs = validate_runs(result, cases, True)
    references = validate_runs(baseline, cases, False)
    for (case, _), output in runs.items():
        require(output["output_ids"] == references[(case, 0)]["output_ids"],
                "diagnostic output IDs differ from original baseline")
    schedulers = owned_schedulers(result, tp)
    folders = {f"{label}-rank{rank}" for label in LABELS for rank in range(tp)}
    require({p.name for p in root.iterdir() if p.is_dir()} == folders, "unexpected/missing capture folders")
    captures = {}
    rank_pids = {}
    for repeat, label in enumerate(LABELS):
        for rank in range(tp):
            folder = root / f"{label}-rank{rank}"
            require(not folder.is_symlink(), "symlink capture folder")
            require({p.name for p in folder.glob("*.json")} == {f"step{s:02d}.json" for s in range(STEPS)},
                    "requires exactly 16 steps per label/rank")
            allowed = {f"step{s:02d}-{name}.npy{suffix}" for s in range(STEPS)
                       for name in ("logits", "hidden") for suffix in ("", ".gz")}
            allowed |= {f"step{s:02d}.json" for s in range(STEPS)}
            require(all(p.name in allowed and not p.is_symlink() and p.is_file()
                        for p in folder.iterdir()), "unowned/unexpected capture file")
            records = []
            for step in range(STEPS):
                record = validate_step(folder, label, rank, step, plan_hash, runs[("length_32", repeat)],
                                       cases["length_32"], schedulers, tp, provenance)
                pid = record["metadata"]["pid"]
                require(rank not in rank_pids or rank_pids[rank] == pid, "rank changed scheduler PID")
                rank_pids[rank] = pid
                records.append(record)
            captures[(repeat, rank)] = records
    require(set(rank_pids.values()) == schedulers, "captures do not cover distinct owned scheduler PIDs")
    if tp == 2:
        for repeat in range(2):
            for left, right in zip(captures[(repeat, 0)], captures[(repeat, 1)]):
                require(left["metadata"]["input_ids"] == right["metadata"]["input_ids"]
                        and np.array_equal(left["logits"], right["logits"]), "TP2 rank logits/inputs mismatch")
    return {"result": result, "plan": plan, "runs": runs, "captures": captures,
            "result_sha256": result_hash, "baseline_sha256": baseline_hash, "plan_sha256": plan_hash,
            "rank_pids": rank_pids}


def distribution_errors(left, right):
    lp, lq = logsoftmax(left), logsoftmax(right)
    p, q = np.exp(lp), np.exp(lq)
    return {"max_abs_raw_logit_error": float(np.max(np.abs(left.astype(np.float64) - right))),
            "max_abs_full_vocab_logprob_error": float(np.max(np.abs(lp - lq))),
            "kl_tp1_to_tp2": float(np.sum(p * (lp - lq), dtype=np.float64)),
            "kl_tp2_to_tp1": float(np.sum(q * (lq - lp), dtype=np.float64)),
            "total_variation": float(np.sum(np.abs(p - q), dtype=np.float64) / 2)}


def divergence_detail(evidence, repeat, step, pair):
    record = evidence["captures"][(repeat, 0)][step]
    data = record["metadata"]
    dots = {}
    for rank in range(evidence["result"]["tp"]):
        dots.update(evidence["captures"][(repeat, rank)][step]["metadata"]["candidate_dot_fp64"])
    require(all(str(t) in dots for t in pair), "divergence tokens missing diagnostic candidate dots")
    return {"actual_output_id": evidence["runs"][("length_32", repeat)]["output_ids"][step],
            "max_tied_ids": data["max_tied_ids"], "candidates": data["candidates"],
            "diagnostic_pair": pair,
            "diagnostic_pair_raw_logits": {str(t): float(record["logits"][t]) for t in pair},
            "diagnostic_pair_dot_fp64": {str(t): dots[str(t)] for t in pair},
            "dot_margin_tp1_choice_minus_tp2_choice": dots[str(pair[0])] - dots[str(pair[1])]}


def analyze(tp1, tp2, baseline_tp1, baseline_tp2, cases_path):
    provenance = []
    case_document, cases_hash = document(cases_path, provenance)
    require([(c["name"], len(c["input_ids"])) for c in case_document["cases"]]
            == [("smoke", 5), ("length_32", 32), ("length_256", 256)], "invalid canonical cases")
    cases = {c["name"]: token_ids(c["input_ids"], len(c["input_ids"])) for c in case_document["cases"]}
    one = load_evidence(tp1, baseline_tp1, 1, cases, cases_hash, provenance)
    two = load_evidence(tp2, baseline_tp2, 2, cases, cases_hash, provenance)
    for key in HASH_KEYS:
        require(one["result"][key] == two["result"][key], f"cross-TP model/cases mismatch: {key}")
    for key in ("source_trees", "runtime_versions", "diagnostic_scripts"):
        require(one["result"]["reproducibility"][key] == two["result"]["reproducibility"][key],
                f"cross-TP source/runtime mismatch: {key}")
    for key in ("TOKENIZERS_PARALLELISM", "HCCL_NPU_SOCKET_PORT_RANGE"):
        require(one["result"]["reproducibility"]["environment"][key]
                == two["result"]["reproducibility"]["environment"][key], "cross-TP environment mismatch")
    normalized = lambda e: {k: v for k, v in e["result"]["config"].items() if k not in TP_CONFIG | {"port"}}
    require(normalized(one) == normalized(two), "cross-TP non-topology configuration mismatch")
    repetitions = []
    for repeat, label in enumerate(LABELS):
        ids1 = one["runs"][("length_32", repeat)]["output_ids"]
        ids2 = two["runs"][("length_32", repeat)]["output_ids"]
        divergence = next((s for s in range(STEPS) if ids1[s] != ids2[s]), None)
        compared = []
        # Full *preceding* output history, not merely the current one-token input.
        for step in range(STEPS):
            if ids1[:step] != ids2[:step]:
                break
            left = one["captures"][(repeat, 0)][step]
            right = two["captures"][(repeat, 0)][step]
            require(left["metadata"]["input_ids"] == right["metadata"]["input_ids"], "common-history input mismatch")
            compared.append({"step_index": step, "preceding_history_equal": True,
                             **distribution_errors(left["logits"], right["logits"]),
                             "hidden_exact_equal": bool(np.array_equal(left["hidden"], right["hidden"])),
                             "max_abs_hidden_error": float(np.max(np.abs(left["hidden"].astype(np.float64) - right["hidden"]))),
                             "tp1_output_id": ids1[step], "tp2_output_id": ids2[step]})
        detail = None
        if divergence is not None:
            pair = [ids1[divergence], ids2[divergence]]
            detail = {"step_index": divergence, "generated_token_number": divergence + 1,
                      "tp1": divergence_detail(one, repeat, divergence, pair),
                      "tp2": divergence_detail(two, repeat, divergence, pair)}
            preserves_preference = all(detail[name]["dot_margin_tp1_choice_minus_tp2_choice"] > 0
                                       for name in ("tp1", "tp2"))
            rounded_tie = set(detail["tp2"]["max_tied_ids"]) == set(pair)
            detail["pair_fp64_preserves_tp1_preference_on_both"] = preserves_preference
            detail["native_argmax_policy"] = "lowest vocabulary ID among exact native maxima"
            detail["interpretation"] = (
                "Stored FP64 candidate dots keep the TP1 choice ahead on both TP settings; "
                "small TP numerical differences plus final FP16 rounding yield a TP2 tie "
                "resolved by native argmax. This does not imply identical upstream hidden states."
                if preserves_preference and rounded_tie else
                "Native outputs differ on equal preceding history; candidate dots are diagnostic only.")
        repetitions.append({"label": label, "first_divergence": detail, "compared_steps": compared,
                            "excluded_after_divergence": list(range(len(compared), STEPS))})
    reproducibility = {}
    for name, evidence in (("tp1", one), ("tp2", two)):
        reproducibility[name] = {
            "output_ids_exact_across_two_repetitions": evidence["runs"][("length_32", 0)]["output_ids"]
                == evidence["runs"][("length_32", 1)]["output_ids"],
            "raw_logits_exact_across_two_repetitions": all(np.array_equal(a["logits"], b["logits"])
                for rank in range(evidence["result"]["tp"])
                for a, b in zip(evidence["captures"][(0, rank)], evidence["captures"][(1, rank)])),
            "hidden_exact_across_two_repetitions": all(np.array_equal(a["hidden"], b["hidden"])
                for rank in range(evidence["result"]["tp"])
                for a, b in zip(evidence["captures"][(0, rank)], evidence["captures"][(1, rank)]))}
    all_compared = [step for repeat in repetitions for step in repeat["compared_steps"]]
    common_history_summary = {
        "compared_step_pairs": len(all_compared),
        "max_abs_raw_logit_error": max(step["max_abs_raw_logit_error"] for step in all_compared),
        "max_abs_full_vocab_logprob_error": max(step["max_abs_full_vocab_logprob_error"] for step in all_compared),
    }
    return {"schema": 1, "status": "validated_offline_diagnostic_analysis", "repetitions": repetitions,
            "common_history_summary": common_history_summary,
            "all_three_cases_match_original_same_tp_baseline": True, "tp2_rank_logits_and_inputs_exact": True,
            "repeat_reproducibility": reproducibility,
            "scope": {"distribution_normalization": "stable float64 full-vocabulary log-softmax of captured rounded raw logits",
                      "comparison_history": "equal full preceding history; includes first divergence; excludes all later steps",
                      "correctness_threshold_changed": False, "existing_logprob_threshold": 0.15,
                      "metrics": "descriptive only; no new pass criterion; does not establish task correctness",
                      "fp64_dots": "observer-recorded pair only, from stored FP16 weights/hidden; not a whole unrounded vocabulary; weights not archived for offline recomputation",
                      "source_validation": "captured source manifests are internally hashed and match baselines; audit/observer scripts match inspected local files; captured run_model hashes must match across TP, not a later local revision",
                      "interpretation": "small TP numerical differences plus final logit rounding can produce an exact tie; upstream hidden states are not assumed identical",
                      "limitations": "two independent repeat labels are observations, not universal reproducibility proof; diagnostic timings are not benchmark timings; original non-retry warmup-tail captures excluded"},
            "bindings": {name: {k: evidence[k] for k in ("result_sha256", "baseline_sha256", "plan_sha256", "rank_pids")}
                         for name, evidence in (("tp1", one), ("tp2", two))},
            "provenance": provenance}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tp1", "tp2", "baseline-tp1", "baseline-tp2", "cases", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        output = args.output.resolve()
        require(not args.output.exists() and not args.output.is_symlink(), "output already exists; no overwrites")
        require(not any(output.is_relative_to(root.resolve()) for root in (args.tp1, args.tp2)),
                "output must be outside read-only evidence directories")
        report = analyze(args.tp1, args.tp2, args.baseline_tp1, args.baseline_tp2, args.cases)
        serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
        with args.output.open("x") as stream:
            stream.write(serialized)
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError) as exc:
        parser.exit(2, f"decode probe analysis rejected: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
