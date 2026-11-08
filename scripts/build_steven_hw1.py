"""Render Steven's HW1 report locally without changing the group's reports."""
import argparse
import html
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from build_submissions import pdf_ast, relative_links

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "homework-1/rebuild/report.md"


def run(command, *, data=None):
    return subprocess.run(command, input=data, check=True, capture_output=True,
                          timeout=180).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chromium", default=(
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        if Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome").exists()
        else shutil.which("chromium")))
    args = parser.parse_args()
    if not args.chromium:
        parser.error("pass --chromium with a local Chromium executable")
    ast = json.loads(run(["pandoc", "--from=gfm", "--to=json", str(REPORT)]))
    for link in relative_links(ast):
        target = (REPORT.parent / link).resolve()
        if not target.is_relative_to(ROOT) or not target.is_file():
            raise ValueError(f"Missing or out-of-repository evidence: {link}")
    body = run(["pandoc", "--from=json", "--to=html5", "--wrap=none",
                "--syntax-highlighting=none"],
               data=json.dumps(pdf_ast(ast)).encode()).decode()
    css = (ROOT / "scripts/steven_hw1_report.css").read_text()
    title = "Steven Walxim 2501112130 HW1 Report"
    document = (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
                f'<title>{html.escape(title)}</title><style>{css}</style></head>'
                f'<body>{body}</body></html>')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="steven-hw1-report-") as name:
        work = Path(name)
        source = work / "report.html"
        source.write_text(document)
        output = work / "report.pdf"
        run([args.chromium, "--headless=new", "--disable-gpu",
             "--disable-background-networking", "--disable-sync", "--disable-extensions",
             "--no-first-run", "--no-default-browser-check",
             "--host-resolver-rules=MAP * ~NOTFOUND", "--no-pdf-header-footer",
             f"--user-data-dir={work / 'browser'}", f"--print-to-pdf={output}",
             source.as_uri()])
        text = run(["pdftotext", "-layout", str(output), "-"]).decode()
        compact = "".join(text.split())
        for required in ("StevenWalxim", "2501112130", "小组成员主要贡献",
                         "1,058", "0.1255553961", "266.383", "历史", "47"):
            if required not in compact:
                raise ValueError(f"PDF text is missing: {required}")
        if "\ufffd" in text:
            raise ValueError("PDF text contains a replacement character")
        shutil.copyfile(output, args.output)
    print(f"Created {args.output}; no upload performed. Visual review is still required.")


if __name__ == "__main__":
    main()
