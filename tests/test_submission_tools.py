"""Focused offline regressions for submission boundaries and integrity checks."""
import copy
import hashlib
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_submissions as build
import verify_submission as verify


class SubmissionToolsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="submission-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def fixture(self, *, changes=None, content=b"verified evidence\n", extras=(), duplicate=False,
                member_mode=stat.S_IFREG | 0o644, manifest_mode=stat.S_IFREG | 0o644):
        record = {"bytes": len(b"verified evidence\n"), "mode": "0644",
                  "sha256": hashlib.sha256(b"verified evidence\n").hexdigest()}
        manifest = {"format_version": 1, "assignment": 2, "files": {"report.txt": record}}
        if changes:
            changes(manifest)
        path = self.root / "package.zip"
        with zipfile.ZipFile(path, "w") as archive:
            entries = [("report.txt", content, member_mode),
                       ("MANIFEST.json", json.dumps(manifest).encode(), manifest_mode), *extras]
            if duplicate:
                entries.append(("report.txt", content, member_mode))
            for name, data, mode in entries:
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = mode << 16
                archive.writestr(info, data)
        return path

    def test_safe_paths_preserve_normal_source_names(self):
        for name in ("report.md", "homework-1/tokenspeed/.gitignore", "中文/报告.md"):
            self.assertEqual(verify.safe_name(name), name)

    def test_unsafe_paths_are_rejected(self):
        for name in ("", ".", "../x", "/x", "x/../y", "x//y", "x/./y", "x/",
                     "C:/x", "x\\y", "x\x00y", "x\ny", ".git/config", "x/.venv/a",
                     "x/__pycache__/a"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                verify.safe_name(name)

    def test_valid_archive_verifies_without_extracting(self):
        result = verify.verify_archive(self.fixture())
        self.assertTrue(result["passed"])
        self.assertEqual(result["verified_files"], 1)
        self.assertFalse((self.root / "report.txt").exists())

    def test_modified_same_size_content_rejected(self):
        with self.assertRaisesRegex(ValueError, "checksum"):
            verify.verify_archive(self.fixture(content=b"tampered evidence\n"))

    def test_unlisted_extra_member_rejected(self):
        with self.assertRaisesRegex(ValueError, "membership"):
            verify.verify_archive(self.fixture(extras=(("extra.txt", b"x", stat.S_IFREG | 0o644),)))

    def test_missing_listed_member_rejected(self):
        def change(document):
            document["files"]["missing.txt"] = document["files"]["report.txt"]
        with self.assertRaisesRegex(ValueError, "membership"):
            verify.verify_archive(self.fixture(changes=change))

    def test_duplicate_member_rejected(self):
        with self.assertWarns(UserWarning):
            path = self.fixture(duplicate=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            verify.verify_archive(path)

    def test_symlink_mode_rejected(self):
        with self.assertRaisesRegex(ValueError, "mode"):
            verify.verify_archive(self.fixture(member_mode=stat.S_IFLNK | 0o777))

    def test_executable_mode_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "mode"):
            verify.verify_archive(self.fixture(member_mode=stat.S_IFREG | 0o755))

    def test_bad_record_hash_or_size_rejected(self):
        for key, value in (("bytes", True), ("bytes", 999), ("mode", "0777"), ("sha256", "not-a-digest")):
            with self.subTest(key=key):
                def change(document):
                    document["files"]["report.txt"][key] = value
                with self.assertRaises(ValueError):
                    verify.verify_archive(self.fixture(changes=change))

    def test_unsupported_manifest_rejected(self):
        for key, value in (("format_version", 99), ("assignment", 3), ("files", [])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify.verify_archive(self.fixture(changes=lambda doc: doc.update({key: value})))

    def test_traversal_archive_member_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsafe"):
            verify.verify_archive(self.fixture(extras=(("../escape", b"x", stat.S_IFREG | 0o644),)))

    def test_uncompressed_file_and_total_bounds(self):
        path = self.fixture()
        with mock.patch.object(verify, "MAX_FILE_BYTES", 5), self.assertRaises(ValueError):
            verify.verify_archive(path)
        with mock.patch.object(verify, "MAX_TOTAL_BYTES", 5), self.assertRaises(ValueError):
            verify.verify_archive(path)
        with mock.patch.object(verify, "MAX_MANIFEST_BYTES", 5), self.assertRaises(ValueError):
            verify.verify_archive(path)

    def test_relative_evidence_link_discovery(self):
        ast = {"blocks": [{"t": "Link", "c": [["", [], []], [], ["docs/evidence.md#section", ""]]},
                          {"t": "Link", "c": [["", [], []], [], ["https://example.org", ""]]},
                          {"t": "Link", "c": [["", [], []], [], ["#section", ""]]}]}
        self.assertEqual(list(build.relative_links(ast)), ["docs/evidence.md"])

    def test_pdf_removes_file_links_without_changing_source_ast(self):
        original = {"blocks": [{"t": "Link", "c": [["", [], []], [{"t": "Str", "c": "evidence"}],
                                                     ["results/result.json", ""]]},
                               {"t": "Link", "c": [["", [], []], [], ["https://example.org", ""]]}]}
        saved = copy.deepcopy(original)
        transformed = build.pdf_ast(original)
        self.assertEqual(original, saved)
        self.assertEqual(transformed["blocks"][0]["t"], "Span")
        self.assertEqual(transformed["blocks"][1]["t"], "Link")
        self.assertEqual(transformed["blocks"][0]["c"][1], [{"t": "Str", "c": "evidence"}])

    def test_outer_selection_excludes_operational_notes_and_gitlink(self):
        report = self.root / "homework-1/report.md"
        report.parent.mkdir()
        report.write_text("report")
        names = b"homework-1/report.md\0homework-1/tokenspeed\0homework-1/docs/vpn-access.md\0homework-1/ISSUES.md\0"
        with mock.patch.object(build, "ROOT", self.root), mock.patch.object(build, "run", return_value=names):
            self.assertEqual(build.selected_files(1), ["homework-1/report.md"])

    def test_outer_selection_fails_on_unapproved_files(self):
        for name in ("credentials.txt", "secret.json", "raw.npy", "capture.trace.json", "unexpected.bin"):
            path = self.root / "homework-1" / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"x")
            with self.subTest(name=name), mock.patch.object(build, "ROOT", self.root), \
                    mock.patch.object(build, "run", return_value=f"homework-1/{name}\0".encode()), \
                    self.assertRaises(ValueError):
                build.selected_files(1)

    def test_package_writer_preserves_executable_modes(self):
        stage = self.root / "stage"
        stage.mkdir()
        build.write_file(stage, "run.py", b"print('ok')\n", 0o755)
        build.write_file(stage, "report.md", b"report\n")
        archive = self.root / "written.zip"
        result = build.make_archive(stage, archive, 1, {"base_commit": build.BASE})
        self.assertEqual(result["verified_files"], 2)
        with zipfile.ZipFile(archive) as zipped:
            self.assertEqual(stat.S_IMODE(zipped.getinfo("run.py").external_attr >> 16), 0o755)

    def test_write_rejects_path_traversal(self):
        with self.assertRaises(ValueError):
            build.write_file(self.root, "../outside", b"x")

    def report_text(self, homework):
        common = "小组成员主要贡献 Zhuoxi Zeng 2501213387 FP32 "
        return common + ("附录 A 1,057 TTFT TPOT" if homework == 1 else
                         "0.0059497356 0.0773688555 Rank 0 控制独占 271.903950")

    def test_reports_keep_identity_results_and_hardware_terms(self):
        for homework in (1, 2):
            text = self.report_text(homework) + " AI Core / AI_CORE / AI_CPU"
            build.validate_report_text(text, homework)
            self.assertEqual(build.boilerplate_markers(text), [])

    def test_reports_still_require_identity_and_result_markers(self):
        for homework in (1, 2):
            for marker in ("2501213387", "FP32", "Zhuoxi Zeng"):
                with self.subTest(homework=homework, marker=marker), self.assertRaises(ValueError):
                    build.validate_report_text(self.report_text(homework).replace(marker, ""), homework)

    def test_reports_reject_relocated_notes_including_pdf_line_breaks(self):
        for note in ("AI 辅助披露", "AI 编程助手", "AI-assistance disclosure", "AI-assisted work",
                     "AI-use\npolicy", "本仓库未代为提交教学网", "提交前本人审核", "供学生审核后提交"):
            with self.subTest(note=note), self.assertRaisesRegex(ValueError, "boilerplate"):
                build.validate_report_text(self.report_text(1) + " " + note, 1)

    def test_packaged_notes_are_clean_without_rewriting_upstream_attribution(self):
        for homework in (1, 2):
            self.assertEqual(build.boilerplate_markers(build.package_guide(homework)), [])
        stage = self.root / "notes"
        stage.mkdir()
        build.write_file(stage, "report.md", b"AI-assistance disclosure")
        with self.assertRaisesRegex(ValueError, "boilerplate"):
            build.make_archive(stage, self.root / "bad.zip", 1, {})
        build.write_file(stage, "report.md", b"technical report")
        upstream = "homework-1/tokenspeed/docs/upstream.md"
        build.write_file(stage, upstream, b"upstream AI-assisted tooling attribution")
        archive = self.root / "good.zip"
        build.make_archive(stage, archive, 1, {})
        with zipfile.ZipFile(archive) as zipped:
            self.assertEqual(zipped.read(upstream), b"upstream AI-assisted tooling attribution")

    def test_stage_guides_distinguish_baselines_and_upload_status(self):
        first, second = build.package_guide(1), build.package_guide(2)
        self.assertIn("base + HW1 only", first)
        self.assertIn("base + HW1 + HW2", second)
        self.assertIn("not uploaded", first)
        self.assertIn("not uploaded", second)
        self.assertIn("Git-less ZIP", second)


if __name__ == "__main__":
    unittest.main()
