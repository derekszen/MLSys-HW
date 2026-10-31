"""Bounded, standard-library-only patch byte/mode regressions (no Torch)."""
import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/export_patch.py"
SPEC = importlib.util.spec_from_file_location("hw2_patch_export", SCRIPT)
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)


class PatchExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="test-hw2-patch-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.work = self.root / "work"
        self.work.mkdir()
        self.repo = self.root / "repo"
        self.repo.mkdir()
        export.git(self.repo, "init", "--quiet")
        export.git(self.repo, "config", "core.filemode", "true")

    def assert_permissions(self, path, mode):
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), int(mode, 8) & 0o777)

    def apply_patch(self, name, old_entry, new_entry):
        if old_entry is not None:
            export.write_regular(self.repo / name, old_entry)
            export.git(self.repo, "add", "--", name)
        patch = export.file_patch(self.work, name, old_entry, new_entry)
        self.assert_permissions(self.work / "new" / name, new_entry[0])
        if old_entry is not None:
            self.assert_permissions(self.work / "old" / name, old_entry[0])
        export.git(self.repo, "apply", "--check", "--whitespace=error", "-", input=patch)
        export.git(self.repo, "apply", "--whitespace=error", "-", input=patch)
        self.assertEqual(export.snapshot(self.repo)[name], new_entry)
        self.assert_permissions(self.repo / name, new_entry[0])
        return patch

    def test_assigned_files_and_expected_modes(self):
        expected = {
            "docs/hccl-torch29.md",
            "docs/lm-head-precision.md",
            "docs/npu-profile.md",
            "docs/qwen35-cache-precision.md",
            "python/tokenspeed/runtime/distributed/comm_backend/hccl.py",
            "python/tokenspeed/runtime/engine/request_handler.py",
            "python/tokenspeed/runtime/layers/attention/configs/base.py",
            "python/tokenspeed/runtime/layers/attention/kv_cache/recipes/qwen35.py",
            "python/tokenspeed/runtime/layers/logits_processor.py",
            "python/tokenspeed/runtime/utils/env.py",
            "python/tokenspeed/runtime/utils/server_args.py",
            "test/runtime/test_hccl_torch29.py",
            "test/runtime/test_lm_head_precision.py",
            "test/runtime/test_npu_profile.py",
            "test/runtime/test_qwen35_cache_precision.py",
        }
        self.assertEqual(list(export.HW2_FILES), sorted(expected))
        modes = {name: "100644" for name in expected}
        for name in (
            "python/tokenspeed/runtime/layers/logits_processor.py",
            "python/tokenspeed/runtime/utils/env.py",
            "python/tokenspeed/runtime/utils/server_args.py",
        ):
            modes[name] = "100755"
        self.assertEqual(export.HW2_MODES, modes)

    def test_assigned_executables_keep_modes_when_applied(self):
        for name, mode in export.HW2_MODES.items():
            if mode != "100755":
                continue
            with self.subTest(name=name):
                patch = self.apply_patch(name, (mode, b"old\n"), (mode, b"new\n"))
                self.assertIn(b" 100755\n", patch)
                self.assertNotIn(b"old mode ", patch)
                self.assertNotIn(b"new mode ", patch)

    def test_packaged_patch_reconstructs_source_bytes_and_modes(self):
        hw2 = SCRIPT.parents[1]
        source = hw2.parent / "homework-1/tokenspeed"
        frozen = (hw2.parent / "homework-1/patches/ascend-qwen3.5.patch").read_bytes()
        patch = (hw2 / "patches/hccl-torch29.patch").read_bytes()
        manifest = json.loads((hw2 / "patches/manifest.json").read_text())
        verification = json.loads((hw2 / "patches/patch-verification.json").read_text())
        self.assertEqual(export.sha256(frozen), export.HW1_SHA256)
        self.assertEqual(manifest["sha256"], export.sha256(patch))
        self.assertEqual(verification["hw2_sha256"], export.sha256(patch))
        self.assertEqual(verification["hw2_bytes"], len(patch))
        self.assertEqual(manifest["base_commit"], export.BASE)
        self.assertEqual(verification["base_commit"], export.BASE)
        self.assertEqual(manifest["prerequisite"]["sha256"], export.HW1_SHA256)
        self.assertEqual(verification["hw1_sha256"], export.HW1_SHA256)
        clean = self.root / "clean"
        export.git(self.root, "clone", "--quiet", "--shared", "--no-checkout", str(source), str(clean))
        export.git(clean, "checkout", "--quiet", "--detach", export.BASE)
        export.git(clean, "apply", "--check", "--whitespace=error", "-", input=frozen)
        export.git(clean, "apply", "--whitespace=error", "-", input=frozen)
        baseline = export.snapshot(clean)
        export.git(clean, "apply", "--check", "--whitespace=error", "-", input=patch)
        export.git(clean, "apply", "--whitespace=error", "-", input=patch)
        reconstructed = export.snapshot(clean)
        self.assertEqual(reconstructed, export.snapshot(source))
        changed = sorted(name for name in baseline.keys() | reconstructed.keys()
                         if baseline.get(name) != reconstructed.get(name))
        self.assertEqual(changed, list(export.HW2_FILES))
        records = export.inventory({name: reconstructed[name] for name in changed})
        self.assertEqual(manifest["files"], records)
        self.assertEqual({name: record["mode"] for name, record in records.items()}, export.HW2_MODES)
        self.assertEqual(manifest["new_files"], [name for name in changed if name not in baseline])
        self.assertTrue(verification["passed"])
        self.assertEqual(verification["incremental_files"], changed)
        self.assertEqual(verification["incremental_files_verified"], records)
        self.assertEqual(verification["verified_file_count"], len(reconstructed))
        tree_json = json.dumps(export.inventory(reconstructed), sort_keys=True, separators=(",", ":"))
        self.assertEqual(verification["verified_tree_sha256"], export.sha256(tree_json.encode()))

    def test_materialize_exact_modes_despite_umask_or_existing_permissions(self):
        for mask in (0o077, 0o000):
            previous = os.umask(mask)
            try:
                for mode in ("100644", "100755", "100644"):
                    with self.subTest(umask=oct(mask), mode=mode):
                        path = self.work / f"mask-{mask}" / "file.py"
                        export.write_regular(path, (mode, b"contents\n"))
                        self.assert_permissions(path, mode)
                        self.assertEqual(path.read_bytes(), b"contents\n")
            finally:
                os.umask(previous)

    def test_unsupported_modes_are_rejected_before_writing(self):
        for mode in ("120000", "160000", "100600", "100777"):
            with self.subTest(mode=mode):
                path = self.work / mode
                with self.assertRaisesRegex(ValueError, "unsupported regular-file mode"):
                    export.write_regular(path, (mode, b"not a regular snapshot entry"))
                self.assertFalse(path.exists())

    def test_chmod_failure_is_not_suppressed(self):
        with mock.patch.object(Path, "chmod", side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):
                export.write_regular(self.work / "file.py", ("100755", b"contents\n"))

    def test_ineffective_chmod_is_detected(self):
        path = self.work / "file.py"
        path.write_bytes(b"old\n")
        path.chmod(0o644)
        with mock.patch.object(Path, "chmod", return_value=None):
            with self.assertRaisesRegex(ValueError, "temporary file mode differs from 100755"):
                export.write_regular(path, ("100755", b"new\n"))

    def test_content_changes_preserve_both_temporary_modes_and_applied_modes(self):
        for mode in ("100644", "100755"):
            with self.subTest(mode=mode):
                name = f"python/module-{mode}.py"
                patch = self.apply_patch(name, (mode, b"old\n"), (mode, b"new\n"))
                self.assertIn(f" {mode}\n".encode(), patch)
                self.assertNotIn(b"old mode ", patch)
                self.assertNotIn(b"new mode ", patch)
                self.assertIn(f"diff --git a/{name} b/{name}\n".encode(), patch)
                self.assertNotIn(b"a/old/", patch)
                self.assertNotIn(b"b/new/", patch)

    def test_new_files_preserve_executable_and_nonexecutable_modes(self):
        for mode in ("100644", "100755"):
            with self.subTest(mode=mode):
                name = f"test/new-{mode}.py"
                patch = self.apply_patch(name, None, (mode, b"new\n"))
                self.assertIn(f"new file mode {mode}\n".encode(), patch)
                self.assertIn(b"--- /dev/null\n", patch)

    def test_mode_only_changes_are_not_lost(self):
        for old_mode, new_mode in (("100644", "100755"), ("100755", "100644")):
            with self.subTest(old_mode=old_mode, new_mode=new_mode):
                name = f"python/mode-{new_mode}.py"
                patch = self.apply_patch(name, (old_mode, b"same\n"), (new_mode, b"same\n"))
                self.assertIn(f"old mode {old_mode}\n".encode(), patch)
                self.assertIn(f"new mode {new_mode}\n".encode(), patch)
                self.assertNotIn(b"@@", patch)

    def test_snapshot_and_inventory_distinguish_same_bytes_different_modes(self):
        path = self.repo / "file.py"
        export.write_regular(path, ("100644", b"same\n"))
        before = export.snapshot(self.repo)
        path.chmod(0o755)
        after = export.snapshot(self.repo)
        self.assertNotEqual(before, after)
        self.assertEqual(before["file.py"], ("100644", b"same\n"))
        self.assertEqual(after["file.py"], ("100755", b"same\n"))
        before_record = export.inventory(before)["file.py"]
        after_record = export.inventory(after)["file.py"]
        self.assertEqual(before_record["sha256"], after_record["sha256"])
        self.assertNotEqual(before_record["mode"], after_record["mode"])


if __name__ == "__main__":
    unittest.main()
