"""CPU numerical/provenance tests, not accelerator or decode-equivalence evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prefix_audit", ROOT / "scripts/audit_prefix.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def real_sources(size="4B", tp1=True):
    paths = {"cases": ROOT / "cases.json",
             "reference": ROOT / f"results/{size}-transformers-reference.json",
             "tp2": ROOT / f"results/{size}-correctness-tp2-c1/result.json"}
    if tp1:
        paths["tp1"] = ROOT / f"results/{size}-correctness-tp1-c1/result.json"
    return audit.read_sources(paths)


def real_plan(size="4B", tp1=True):
    return audit.make_plan(*real_sources(size, tp1))


def numeric_artifact(plan, backend="transformers", tp=1):
    """Pure data fixture with known numerical values, NOT a simulated engine run."""
    model = plan["expected_model"]
    devices = list(range(tp))
    if backend == "tokenspeed":
        config = audit.harness.build_config(model["path"], tp=tp, devices=devices,
            memory_fraction=0.6, concurrency=1, port=31889)
    else:
        config = {"model": model["path"], "dtype": "bfloat16", "attn_implementation": "eager",
                  "trust_remote_code": False, "local_files_only": True, "use_cache": False}
    result = {"schema": audit.RESULT_SCHEMA, "status": "audit_completed_not_compared",
        "semantics": audit.SEMANTICS, "backend": backend, "plan": copy.deepcopy(plan),
        "plan_sha256": audit.digest(plan), "plan_file_sha256": audit.sha256(audit.canonical(plan)),
        "model": copy.deepcopy(model), "model_sha256": audit.digest(model),
        "precision": audit.PRECISION.copy(), "config": config, "tp": tp,
        "physical_devices_requested": devices, "vocab_size": 300000, "top_k": 10,
        "reproducibility": {"scripts_sha256": audit.scripts_fingerprint(),
            "runtime_versions": {"torch": "test", "torch_npu": "test", "triton": "test",
                                 "transformers": "test"}}, "rows": []}
    files = {"fixture.py": audit.sha256(b"numerical fixture, not runtime evidence")}
    result["reproducibility"]["source_trees"] = {
        name: {"files": files.copy(), "sha256": audit.digest(files)}
        for name in ("python", "tokenspeed-kernel/python", "tokenspeed-kernel-npu/python")}
    for prefix in plan["prefixes"]:
        tokens = prefix["candidate_ids"][:]
        tokens += [i for i in range(100) if i not in tokens][:10 - len(tokens)]
        top = {"ids": tokens, "logprobs": [-(i + 1) * 0.25 for i in range(10)]}
        values = dict(zip(top["ids"], top["logprobs"]))
        row = audit.measured_row(copy.deepcopy(prefix), top, values, tokens[0])
        if backend == "tokenspeed":
            row["raw_output"] = {"output_ids": [tokens[0]], "meta_info": {
                "prompt_tokens": len(prefix["input_ids"]), "completion_tokens": 1, "cached_tokens": 0,
                "output_top_logprobs": [[[value, token, None] for token, value in values.items()]]}}
        result["rows"].append(row)
    return result


def perturb(result, delta):
    for row in result["rows"]:
        row["candidate_logprobs"] = {key: value - delta for key, value in row["candidate_logprobs"].items()}
        row["top"]["logprobs"] = [value - delta for value in row["top"]["logprobs"]]
        if "raw_output" in row:
            for entry in row["raw_output"]["meta_info"]["output_top_logprobs"][0]:
                entry[0] -= delta


class PrefixPlanTests(unittest.TestCase):
    def test_real_4b_9b_27b_saved_schemas_and_divergences(self):
        for size, tp1, first_step in (("4B", True, 10), ("9B", True, 14), ("27B", False, 13)):
            with self.subTest(size=size):
                plan = real_plan(size, tp1)
                self.assertEqual(len(plan["prefixes"]), 16)
                self.assertEqual(plan["first_divergences"][0]["first_divergence_step"], first_step)
                self.assertEqual(plan["prefixes"][15]["input_ids"],
                    plan["case"]["input_ids"] + plan["case"]["reference_output_ids"][:15])
                self.assertEqual(len({tuple(p["input_ids"]) for p in plan["prefixes"]}), 16)
                audit.verify_sources(plan)
                audit.validate_plan(plan, 300000)

    def test_optional_tp1_failed_real_27b_evidence_rejected(self):
        with self.assertRaisesRegex(ValueError, "complete correctness"):
            real_plan("27B", tp1=True)

    def test_history_eligibility_stops_after_first_divergence(self):
        plan = real_plan()
        label = "tp1:repeat0:request0"
        self.assertIn(label, plan["prefixes"][10]["eligible_trajectories"])
        for row in plan["prefixes"][11:]:
            self.assertNotIn(label, row["eligible_trajectories"])
        # Even a later reconvergence of output token IDs cannot restore history.
        trajectory = plan["trajectories"][0]
        trajectory["output_ids"][11:] = plan["case"]["reference_output_ids"][11:]
        rows = audit.expected_prefixes(plan["case"], plan["trajectories"])
        self.assertNotIn(label, rows[12]["eligible_trajectories"])

    def test_explicit_alternative_at_common_prefix_only(self):
        plan = real_plan()
        plan["trajectories"][0]["output_ids"][0] = 299999
        plan["trajectories"][0]["output_ids"][1] = 299998
        rows = audit.expected_prefixes(plan["case"], plan["trajectories"])
        self.assertIn(299999, rows[0]["candidate_ids"])
        self.assertNotIn(299998, rows[1]["candidate_ids"])

    def test_invalid_real_source_provenance_and_canonical_history(self):
        docs, sources = real_sources()
        mutations = (
            lambda d: d["tp2"].update(cases_sha256="0" * 64),
            lambda d: d["tp1"].update(manifest_sha256="0" * 64),
            lambda d: d["reference"]["cases"][1]["input_ids"].__setitem__(0, 1),
            lambda d: d["tp2"]["config"].update(dtype="float16"),
            lambda d: d["tp2"]["runs"][1]["outputs"][0]["meta_info"].update(cached_tokens=1),
            lambda d: d["reference"]["cases"][1]["top5"][0]["logprobs"].__setitem__(0, float("nan")),
        )
        for mutation in mutations:
            changed = copy.deepcopy(docs)
            mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                audit.make_plan(changed, sources)

    def test_invalid_plan_ids_context_and_alternate_history(self):
        for mutation in (
            lambda p: p["case"]["input_ids"].__setitem__(0, -1),
            lambda p: p["case"]["input_ids"].__setitem__(0, True),
            lambda p: p["case"].update(input_ids=[1] * 513),
            lambda p: p["prefixes"][3]["input_ids"].__setitem__(-1, 299999),
            lambda p: p["prefixes"][11]["candidate_ids"].append(299999),
            lambda p: p["prefixes"][0]["candidate_ids"].__setitem__(0, False),
            lambda p: p["case"]["reference_top5"][0]["ids"].__setitem__(0, 300000),
            lambda p: p["expected_model"].update(manifest_sha256="unknown"),
            lambda p: p["first_divergences"][0].update(first_divergence_step=11),
        ):
            plan = copy.deepcopy(real_plan())
            mutation(plan)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                audit.validate_plan(plan, 300000)

    def test_source_bytes_must_still_match_and_plan_must_reproduce(self):
        plan = real_plan()
        plan["sources"]["reference"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source fingerprint"):
            audit.verify_sources(plan)
        plan = real_plan()
        plan["provenance_limit"] = "silently remove missing reference provenance warning"
        with self.assertRaisesRegex(ValueError, "does not reproduce"):
            audit.verify_sources(plan)

    def test_model_manifest_and_config_exact_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = b'{"text_config":{"vocab_size":300000}}'
            (root / "config.json").write_bytes(raw)
            manifest = {"files": [{"path": "config.json", "sha256": audit.sha256(raw)}]}
            manifest_raw = audit.canonical(manifest)
            (root / "source-manifest.json").write_bytes(manifest_raw)
            expected = {"path": str(root.resolve()), "manifest_sha256": audit.sha256(manifest_raw),
                        "config_sha256": audit.sha256(raw)}
            self.assertEqual(audit.model_fingerprint(root, expected), (expected, 300000))
            (root / "config.json").write_bytes(raw + b"\n")
            with self.assertRaisesRegex(ValueError, "does not match manifest"):
                audit.model_fingerprint(root, expected)
            (root / "config.json").write_bytes(raw)
            (root / "source-manifest.json").write_bytes(manifest_raw + b"\n")
            with self.assertRaisesRegex(ValueError, "differs from plan"):
                audit.model_fingerprint(root, expected)


class PrefixMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.plan = real_plan()
        self.left = numeric_artifact(self.plan)

    def test_topk_protocol_and_coverage_failure_not_fabricated(self):
        ts = numeric_artifact(self.plan, "tokenspeed")
        row = ts["rows"][0]
        self.assertEqual(audit.parse_tokenspeed(row["raw_output"], row["input_ids"], 10, 300000), row["top"])
        missing = row["candidate_ids"][-1]
        values = audit.top_values(row["top"], 10)
        del values[missing]
        measurement = audit.measured_row(self.plan["prefixes"][0], row["top"], values, row["generated_token_id"])
        self.assertEqual(measurement["missing_candidate_ids"], [missing])
        self.assertNotIn(str(missing), measurement["candidate_logprobs"])
        ts["status"] = "audit_coverage_failed"
        with self.assertRaisesRegex(ValueError, "coverage failed"):
            audit.compare(ts, self.left, 0.15)

    def test_in_memory_engine_tuple_entries(self):
        row = numeric_artifact(self.plan, "tokenspeed")["rows"][0]
        output = copy.deepcopy(row["raw_output"])
        output["meta_info"]["output_top_logprobs"][0] = [
            tuple(entry) for entry in output["meta_info"]["output_top_logprobs"][0]
        ]
        self.assertEqual(audit.parse_tokenspeed(output, row["input_ids"], 10, 300000), row["top"])

    def test_real_old_run_payload_cannot_be_misread_as_topk_audit(self):
        documents, _ = real_sources()
        output = documents["tp2"]["runs"][1]["outputs"][0]
        with self.assertRaisesRegex(ValueError, "incomplete generation"):
            audit.parse_tokenspeed(output, self.plan["case"]["input_ids"], 10, 300000)

    def test_topk_payload_invalid_entries_fail_closed(self):
        original = numeric_artifact(self.plan, "tokenspeed")["rows"][0]
        for mutation in (
            lambda o: o["meta_info"].pop("output_top_logprobs"),
            lambda o: o["meta_info"]["output_top_logprobs"][0].pop(),
            lambda o: o["meta_info"]["output_top_logprobs"][0][0].__setitem__(0, float("nan")),
            lambda o: o["meta_info"]["output_top_logprobs"][0][0].__setitem__(1, -1),
            lambda o: o["meta_info"]["output_top_logprobs"][0][0].__setitem__(2, "decoded"),
            lambda o: o["meta_info"].update(cached_tokens=1),
        ):
            output = copy.deepcopy(original["raw_output"])
            mutation(output)
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, KeyError)):
                audit.parse_tokenspeed(output, original["input_ids"], 10, 300000)

    def test_all_candidates_not_just_selected_are_compared(self):
        right = copy.deepcopy(self.left)
        perturb(right, 0.125)
        result = audit.compare(self.left, right, 0.15)
        self.assertTrue(result["passed"])
        self.assertEqual(result["max_candidate_error"], 0.125)
        self.assertEqual(sum(len(row["candidate_errors"]) for row in result["rows"]),
                         sum(len(row["candidate_ids"]) for row in self.plan["prefixes"]))
        self.assertFalse(audit.compare(self.left, right, 0.1)["passed"])
        perturb(right, 0.125)
        self.assertFalse(audit.compare(self.left, right, 0.15)["passed"])
        # An error ONLY on a non-argmax candidate must also fail.
        right = copy.deepcopy(self.left)
        row = right["rows"][0]
        token = row["candidate_ids"][-1]
        self.assertNotEqual(token, row["generated_token_id"])
        row["candidate_logprobs"][str(token)] -= 0.25
        row["top"]["logprobs"][row["top"]["ids"].index(token)] -= 0.25
        self.assertFalse(audit.compare(self.left, right, 0.15)["passed"])
        for atol in (-1, 0.150001, float("nan"), float("inf"), True):
            with self.subTest(atol=atol), self.assertRaises(ValueError):
                audit.compare(self.left, right, atol)

    def test_nan_infinity_missing_candidate_and_forged_provenance_rejected(self):
        for mutation in (
            lambda r: r["rows"][0]["candidate_logprobs"].update({str(r["rows"][0]["candidate_ids"][0]): float("nan")}),
            lambda r: r["rows"][0]["candidate_logprobs"].update({str(r["rows"][0]["candidate_ids"][0]): float("inf")}),
            lambda r: r["rows"][0]["candidate_logprobs"].pop(str(r["rows"][0]["candidate_ids"][0])),
            lambda r: r.update(plan_sha256="0" * 64),
            lambda r: r.update(model_sha256="0" * 64),
            lambda r: r.update(plan_file_sha256="0" * 64),
            lambda r: r["precision"].update(dtype="float16"),
            lambda r: r["rows"][0]["input_ids"].__setitem__(0, 1),
            lambda r: r["rows"][0]["candidate_ids"].append(299999),
            lambda r: r["config"].update(use_cache=True),
            lambda r: r["reproducibility"]["scripts_sha256"].update({"audit_prefix.py": "0" * 64}),
        ):
            right = copy.deepcopy(self.left)
            mutation(right)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                audit.compare(self.left, right, 0.15)

    def test_argmax_and_exact_ties_are_diagnostics_only(self):
        left = copy.deepcopy(self.left)
        row = left["rows"][0]
        row["top"]["logprobs"][1] = row["top"]["logprobs"][0]
        row["candidate_logprobs"][str(row["top"]["ids"][1])] = row["top"]["logprobs"][0]
        right = copy.deepcopy(left)
        right["rows"][0]["generated_token_id"] = row["top"]["ids"][1]
        result = audit.compare(left, right, 0)
        self.assertTrue(result["passed"])
        self.assertFalse(result["rows"][0]["argmax_equal"])
        self.assertEqual(result["rows"][0]["left_diagnostics"]["top1_minus_top2"], 0)
        self.assertEqual(len(result["rows"][0]["left_diagnostics"]["exact_top_tie_ids"]), 2)

    def test_tp1_tp2_only_parallel_config_difference(self):
        tp1 = numeric_artifact(self.plan, "tokenspeed", 1)
        tp2 = numeric_artifact(self.plan, "tokenspeed", 2)
        self.assertTrue(audit.compare(tp1, tp2, 0.15)["passed"])
        for key, value in (("gpu_memory_utilization", 0.5), ("port", 31890),
                           ("mamba_ssm_dtype", "bfloat16"), ("enable_prefix_caching", True)):
            changed = copy.deepcopy(tp2)
            changed["config"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit.compare(tp1, changed, 0.15)

    def test_transformers_shared_subset_no_tp2_requirement(self):
        tp1 = numeric_artifact(self.plan, "tokenspeed", 1)
        self.assertTrue(audit.compare(tp1, self.left, 0.15)["passed"])
        self.assertTrue(audit.compare(self.left, self.left, 0)["passed"])
        # Direct Transformers candidate values may be outside its stored topK.
        row = self.left["rows"][0]
        token = row["candidate_ids"][-1]
        row["top"]["ids"][row["top"]["ids"].index(token)] = 299999
        self.assertTrue(audit.compare(self.left, self.left, 0)["passed"])

    def test_missing_required_tokenspeed_topk_candidate_rejected(self):
        ts = numeric_artifact(self.plan, "tokenspeed")
        row = ts["rows"][0]
        token = row["candidate_ids"][-1]
        index = row["top"]["ids"].index(token)
        row["top"]["ids"][index] = 299999
        row["raw_output"]["meta_info"]["output_top_logprobs"][0][index][1] = 299999
        with self.assertRaisesRegex(ValueError, "absent from topK"):
            audit.compare(ts, self.left, 0.15)

    def test_real_cli_plan_and_compare_new_outputs_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            docs, sources = real_sources()
            command = [sys.executable, str(ROOT / "scripts/audit_prefix.py"), "plan"]
            for name, source in sources.items():
                command.extend([f"--{name}", source["path"]])
            command.extend(["--output", str(root / "plan.json")])
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads((root / "plan.json").read_text()), self.plan)
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            audit.write_json(root / "left.json", self.left)
            right = copy.deepcopy(self.left)
            perturb(right, 0.25)
            audit.write_json(root / "right.json", right)
            command = [sys.executable, str(ROOT / "scripts/audit_prefix.py"), "compare",
                "--left", str(root / "left.json"), "--right", str(root / "right.json"),
                "--atol", "0.15", "--output", str(root / "comparison.json")]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 1, completed.stderr)
            self.assertFalse(json.loads((root / "comparison.json").read_text())["passed"])
            self.assertEqual(json.loads((root / "left.json").read_text()), self.left)


if __name__ == "__main__":
    unittest.main()
