"""Build separate HW1/HW2 PDFs and verified code/evidence ZIPs; never upload.

Requires Git, Pandoc, headless Chromium, Poppler (pdfinfo/pdftotext), and CJK fonts.
Only temporary clones receive patches. No model, credential, or network access.
"""
import argparse
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile

from verify_submission import safe_name, verify_archive

ROOT = Path(__file__).resolve().parents[1]
BASE = "52bdb9763bb2da3d61336398072671885cc0a33e"
HW1_SHA = "e68f2f26e49d30180280adbab0bb0ebaf433246e27457788e60277621613b8c6"
PREFIX = "2501213387_Zhuoxi_Zeng"
EXCLUDED = {
    "homework-1/ISSUES.md", "homework-1/docs/vpn-access.md", "homework-2/ISSUES.md",
}
SUFFIXES = {".md", ".py", ".json", ".txt", ".gz", ".patch"}
REBUILD_COMMANDS = {
    f"homework-1/rebuild/results/2026-11-06/{run}/command.sh"
    for run in ("operator-tests", "engine-validation", "benchmark")
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def run(command, *, cwd=ROOT, data=None, timeout=180):
    completed = subprocess.run(command, cwd=cwd, input=data, capture_output=True,
                               timeout=timeout, check=False,
                               env={**os.environ, "GIT_OPTIONAL_LOCKS": "0",
                                    "PYTHONDONTWRITEBYTECODE": "1"})
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {command}\n"
                           + completed.stderr.decode(errors="replace")[-4000:])
    return completed.stdout


def json_bytes(document):
    return (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode()


def relative_links(ast):
    if isinstance(ast, dict):
        if ast.get("t") in ("Link", "Image"):
            target = ast["c"][-1][0]
            if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) and not target.startswith("#"):
                yield target.split("#", 1)[0]
        for child in ast.values():
            yield from relative_links(child)
    elif isinstance(ast, list):
        for child in ast:
            yield from relative_links(child)


def pdf_ast(value):
    """Keep external/internal links, but never bake temporary file URLs into PDF."""
    if isinstance(value, dict):
        if value.get("t") == "Link":
            attributes, content, target = value["c"]
            if not target[0].startswith(("https://", "http://", "#")):
                return {"t": "Span", "c": [attributes, pdf_ast(content)]}
        return {key: pdf_ast(child) for key, child in value.items()}
    if isinstance(value, list):
        return [pdf_ast(child) for child in value]
    return value


def selected_files(homework):
    """Outer-repo allowlist; ignore rules remove weights, venvs and duplicate raw captures."""
    directories = ["homework-1"] + (["homework-2"] if homework == 2 else [])
    output = run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z",
                  "--", *directories])
    selected = []
    for name in sorted(set(os.fsdecode(item) for item in output.split(b"\0") if item)):
        if name in EXCLUDED or name == "homework-1/tokenspeed":
            continue
        safe_name(name)
        path = ROOT / name
        if (path.is_symlink() or not path.is_file()
                or (path.suffix not in SUFFIXES and name not in REBUILD_COMMANDS)
                or "models" in path.relative_to(ROOT).parts
                or path.name.startswith(".env")
                or re.search(r"credential|password|secret|campus.auth", path.name, re.I)):
            raise ValueError(f"unapproved outer-repository file: {name}")
        # Never silently package a newly added uncompressed diagnostic duplicate.
        if path.suffix == ".npy" or path.name.endswith(".trace.json"):
            raise ValueError(f"raw diagnostic duplicate: {name}")
        selected.append(name)
    if f"homework-{homework}/report.md" not in selected:
        raise ValueError("report missing from selected materials")
    return selected


def write_file(root, name, data, mode=0o644):
    safe_name(name)
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(mode)


def package_guide(homework):
    return f"""# Homework {homework} — Zhuoxi Zeng / 2501213387

**Prepared for student review and submission; not uploaded.**

Start with `{PREFIX}_HW{homework}_Report.pdf`. The same report is available as
`homework-{homework}/report.md` and `homework-{homework}/report.html`; the HTML retains
working relative evidence links after extraction. The PDF is self-contained for
required tables and results; detailed logs, code and analysis are attachments.

## Contents and frozen source

- Original assignment and requirement/evidence crosswalk under `homework-{homework}/`.
- Complete corresponding source under `homework-1/tokenspeed/`, already patched.
- Base: `{BASE}`; frozen HW1 patch SHA256: `{HW1_SHA}`.
- HW{homework} source is base + HW1{(' + HW2 incremental patch' if homework == 2 else ' only')}. **Do not reapply patches to this extracted source.**
- Tests, numerical results, HBM snapshots and compressed profiling/diagnostic evidence.
- `MANIFEST.json` hashes every other file and records Git-equivalent source modes.
- No weights, virtualenv, Git internals, credentials or personal VPN setup.

{('HW1 files in this HW2 package are prerequisite evidence. Use the separate HW1 package to reproduce its frozen source; do not relabel HW2 source as the HW1 baseline.' if homework == 2 else 'Later HW2 runtime changes are deliberately absent from this package.')}

## Verify and reproduce

Before extraction, use the copy of `verify_submission.py` with the ZIP path:

```bash
python verify_submission.py /path/to/{PREFIX}_HW{homework}_Submission.zip
```

Extract each homework to its own directory, preserving file permissions. Read
`homework-{homework}/README.md` for the course-server environment, model downloader,
explicit device selection and reproduction commands. Existing results are evidence;
use fresh output paths for reruns. No NPU is needed to inspect reports/results.

The ZIP has no `.git`: offline summary commands work directly, but the complete
repository suite's Git patch-reconstruction test requires a normal pinned Git
checkout. Full local-suite results were verified in the working repository, not
misrepresented as tests of a Git-less ZIP. Runtime tensor tests and fresh model
inference require the course NPU environment and separately downloaded weights.

## Report and attachments

The PDF is the main report; this ZIP supplies corresponding source and evidence.
The stored HW1 assignment requests a report on the teaching platform, while HW2's
stored assignment does not specify an upload method. Follow the actual assignment
entry for permitted attachments and file-size limits.
"""


def render_report(stage, homework, chromium, work):
    report = stage / f"homework-{homework}/report.md"
    ast = json.loads(run(["pandoc", "--from=gfm", "--to=json", str(report)]))
    for link in relative_links(ast):
        if not link or not (report.parent / link).resolve().is_relative_to(stage.resolve()):
            raise ValueError(f"report link escapes package: {link}")
        if not (report.parent / link).exists():
            raise ValueError(f"report evidence missing from package: {link}")
    css = (ROOT / "scripts/submission.css").read_text()
    title = f"Homework {homework} — Zhuoxi Zeng — 2501213387"
    note = ("附件说明：正文已包含主要结论、表格和原题要求。详细证据的相对链接请在 ZIP 中的 "
            f"homework-{homework}/report.html 或 report.md 打开；独立 PDF 不依赖临时本地文件链接。")
    def document(tree):
        body = run(["pandoc", "--from=json", "--to=html5", "--wrap=none", "--syntax-highlighting=none"],
                   data=json_bytes(tree)).decode()
        return (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
                f'<meta name="viewport" content="width=device-width,initial-scale=1">'
                f'<title>{html.escape(title)}</title><style>{css}</style></head>'
                f'<body class="homework-{homework}"><main>'
                f'{body}<p class="attachment-note">{note}</p></main></body></html>')
    report.with_suffix(".html").write_text(document(ast))
    print_html = work / f"print-hw{homework}.html"
    print_html.write_text(document(pdf_ast(ast)))
    pdf = stage / f"{PREFIX}_HW{homework}_Report.pdf"
    run([chromium, "--headless=new", "--disable-gpu", "--disable-background-networking",
         "--disable-sync", "--disable-extensions", "--no-first-run", "--no-default-browser-check",
         "--host-resolver-rules=MAP * ~NOTFOUND", "--no-pdf-header-footer",
         f"--user-data-dir={work / f'browser-hw{homework}'}", f"--print-to-pdf={pdf}",
         print_html.as_uri()], timeout=180)
    return pdf, verify_pdf(pdf, homework)


def boilerplate_markers(text):
    """Find relocated disclosure/review notes, not hardware terms such as AI Core."""
    compact = re.sub(r"\s+", "", text).casefold()
    markers = (
        "ai辅助", "ai编程助手", "ai使用规定", "aiassistance", "ai-assistance",
        "aiassisted", "ai-assisted", "aiusepolicy", "ai-usepolicy", "aipolicy", "ai-policy",
        "提交前本人审核", "本仓库未代为提交", "供学生审核后提交",
    )
    return [marker for marker in markers if marker in compact]


def validate_report_text(text, homework):
    compact = re.sub(r"\s+", "", text)
    required = ["小组成员主要贡献", "ZhuoxiZeng", "2501213387", "FP32"]
    required += (["附录A", "1057", "TTFT", "TPOT"] if homework == 1
                 else ["0.0059497356", "0.0773688555", "Rank0", "控制独占", "271.903950"])
    # Printed digit group separators are presentation, not a numerical change.
    normalized = compact.replace(",", "")
    if (any(phrase.replace(",", "") not in normalized for phrase in required)
            or "\ufffd" in text or "**" in text):
        raise ValueError("PDF lacks expected Chinese/identity/result text")
    if boilerplate_markers(text):
        raise ValueError("submission workflow boilerplate remains in report")


def verify_pdf(path, homework):
    text = run(["pdftotext", "-layout", str(path), "-"]).decode()
    validate_report_text(text, homework)
    info = run(["pdfinfo", str(path)]).decode()
    match = re.search(r"^Pages:\s+(\d+)\s*$", info, re.M)
    if not match or not 2 <= int(match[1]) <= 40:
        raise ValueError("unexpected PDF page count")
    xml = ET.fromstring(run(["pdftotext", "-bbox", str(path), "-"]))
    pages = [element for element in xml.iter() if element.tag.endswith("}page")]
    words = 0
    for page in pages:
        width, height = float(page.attrib["width"]), float(page.attrib["height"])
        page_words = [element for element in page.iter() if element.tag.endswith("}word")]
        if sum(len(word.text or "") for word in page_words) < 40:
            raise ValueError("nearly empty PDF page")
        for word in page_words:
            coords = [float(word.attrib[k]) for k in ("xMin", "yMin", "xMax", "yMax")]
            if not (0 <= coords[0] <= coords[2] <= width + 0.5
                    and 0 <= coords[1] <= coords[3] <= height + 0.5):
                raise ValueError(f"text outside PDF page: {word.text}")
        words += len(page_words)
    if len(pages) != int(match[1]):
        raise ValueError("PDF page parsers disagree")
    return {"passed": True, "pages": len(pages), "text_words": words,
            "chinese_identity_and_result_markers_present": True,
            "report_workflow_boilerplate_absent": True,
            "text_within_page_bounds": True,
            "scope": "text and layout bounds, not a claim of exhaustive visual review"}


def make_archive(stage, output, homework, source_record):
    files = {}
    for path in sorted(stage.rglob("*")):
        if path.is_dir():
            continue
        name = safe_name(path.relative_to(stage).as_posix())
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"nonregular package file: {name}")
        data = path.read_bytes()
        if (path.suffix in (".md", ".html")
                and not name.startswith("homework-1/tokenspeed/")
                and boilerplate_markers(data.decode("utf-8"))):
            raise ValueError(f"submission workflow boilerplate remains in package document: {name}")
        files[name] = {"bytes": len(data), "sha256": sha256(data),
                       "mode": format(stat.S_IMODE(path.stat().st_mode), "04o")}
    write_file(stage, "MANIFEST.json", json_bytes({
        "format_version": 1, "assignment": homework, "student": {"name": "Zhuoxi Zeng", "id": "2501213387"},
        "source": source_record, "files": files,
        "not_included": ["weights", "virtualenv", "Git internals", "credentials",
                         "personal VPN/operational issue notes", "duplicate raw captures"],
        "submitted": False,
    }))
    with zipfile.ZipFile(output, "w", allowZip64=True) as archive:
        for name in sorted([*files, "MANIFEST.json"]):
            path = stage / name
            entry = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = (stat.S_IFREG | stat.S_IMODE(path.stat().st_mode)) << 16
            method = zipfile.ZIP_STORED if path.suffix in (".gz", ".png", ".jpg", ".pdf") else zipfile.ZIP_DEFLATED
            archive.writestr(entry, path.read_bytes(), compress_type=method, compresslevel=6)
    return verify_archive(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_chromium = ("/usr/lib/chromium/chromium" if Path("/usr/lib/chromium/chromium").exists()
                        else shutil.which("chromium"))
    parser.add_argument("--chromium", default=default_chromium)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submissions")
    args = parser.parse_args()
    for command in ("git", "pandoc", "pdfinfo", "pdftotext"):
        if shutil.which(command) is None:
            parser.error(f"missing local dependency: {command}")
    if not args.chromium:
        parser.error("headless Chromium not found; pass --chromium")
    spec = importlib.util.spec_from_file_location("hw2_export_readonly", ROOT / "homework-2/scripts/export_patch.py")
    export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export)
    source = ROOT / "homework-1/tokenspeed"
    before = export.snapshot(source)
    index_path = Path(os.fsdecode(export.git(source, "rev-parse", "--git-path", "index").strip()))
    if not index_path.is_absolute():
        index_path = source / index_path
    index_before = index_path.read_bytes()
    hw1 = (ROOT / "homework-1/patches/ascend-qwen3.5.patch").read_bytes()
    hw2 = (ROOT / "homework-2/patches/hccl-torch29.patch").read_bytes()
    hw2_verification = json.loads((ROOT / "homework-2/patches/patch-verification.json").read_text())
    if sha256(hw1) != HW1_SHA or sha256(hw2) != hw2_verification["hw2_sha256"]:
        raise ValueError("retained patch hash changed")
    manifest = {"student": {"name": "Zhuoxi Zeng", "id": "2501213387"},
                "submitted": False, "artifacts": {}, "checks": {},
                "build_tools": {"pandoc": run(["pandoc", "--version"]).decode().splitlines()[0],
                                "chromium": run([args.chromium, "--version"]).decode().strip()}}
    with tempfile.TemporaryDirectory(prefix="mlsys-submission-") as temporary:
        work = Path(temporary)
        clean = work / "source"
        export.git(work, "clone", "--quiet", "--shared", "--no-checkout", str(source), str(clean))
        export.git(clean, "checkout", "--quiet", "--detach", BASE)
        deliverables = []
        for homework, patch in ((1, hw1), (2, hw2)):
            export.git(clean, "apply", "--check", "--whitespace=error", "-", input=patch)
            export.git(clean, "apply", "--whitespace=error", "-", input=patch)
            snapshot = export.snapshot(clean)
            records = export.inventory(snapshot)
            tree_sha = sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode())
            if homework == 2 and (snapshot != before or tree_sha != hw2_verification["verified_tree_sha256"]):
                raise ValueError("final source differs from retained complete-tree verification")
            stage = work / f"hw{homework}"
            stage.mkdir()
            for name in selected_files(homework):
                original = ROOT / name
                write_file(stage, name, original.read_bytes(), 0o755 if original.stat().st_mode & 0o111 else 0o644)
            for name, (mode, data) in snapshot.items():
                if mode not in ("100644", "100755"):
                    raise ValueError(f"unsupported source mode: {name}")
                write_file(stage, f"homework-1/tokenspeed/{name}", data, int(mode, 8) & 0o777)
            guide = package_guide(homework).encode()
            write_file(stage, "README.md", guide)
            write_file(stage, "SUBMISSION.md", guide)
            write_file(stage, "verify_submission.py", (ROOT / "scripts/verify_submission.py").read_bytes())
            pdf, pdf_check = render_report(stage, homework, args.chromium, work)
            source_record = {"base_commit": BASE, "hw1_patch_sha256": HW1_SHA,
                             "hw2_patch_sha256": sha256(hw2) if homework == 2 else None,
                             "files": len(snapshot), "tree_sha256": tree_sha,
                             "method": "clean pinned base + sequential retained patches; byte/mode inventory"}
            archive = work / f"{PREFIX}_HW{homework}_Submission.zip"
            zip_check = make_archive(stage, archive, homework, source_record)
            manifest["checks"][f"HW{homework}"] = {"pdf": pdf_check, "archive": zip_check}
            for path in (pdf, archive):
                data = path.read_bytes()
                manifest["artifacts"][path.name] = {"bytes": len(data), "sha256": sha256(data)}
                deliverables.append(path)
            print(f"HW{homework}: {pdf_check['pages']} PDF pages; {zip_check['verified_files']} verified files; "
                  f"ZIP {archive.stat().st_size / 1_000_000:.2f} MB")
        if (export.snapshot(source) != before or index_path.read_bytes() != index_before
                or (ROOT / "homework-1/patches/ascend-qwen3.5.patch").read_bytes() != hw1
                or (ROOT / "homework-2/patches/hccl-torch29.patch").read_bytes() != hw2):
            raise ValueError("source/index/frozen patch changed during packaging")
        manifest["source_worktree_index_and_patches_unchanged"] = True
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for path in deliverables:
            shutil.copyfile(path, args.output_dir / path.name)
        (args.output_dir / "manifest.json").write_bytes(json_bytes(manifest))
        checksums = "".join(f"{item['sha256']}  {name}\n" for name, item in sorted(manifest["artifacts"].items()))
        (args.output_dir / "SHA256SUMS").write_text(checksums)
    print(f"Prepared files in {args.output_dir}; no upload performed.")


if __name__ == "__main__":
    main()
