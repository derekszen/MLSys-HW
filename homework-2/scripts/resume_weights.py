"""Resume a pinned model manifest using at most three independent downloads.

Reuses the HW1 downloader's immutable revisions, path checks, size checks,
SHA256 verification and HTTP range handling. Never fetches a new manifest.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), required=True)
    args = parser.parse_args()
    manifest = json.loads((args.output / "source-manifest.json").read_text())
    if manifest["source"] != "modelscope.cn":
        raise ValueError("only pinned ModelScope manifests supported")
    repo = manifest["repo"]
    if len(repo.split("/")) != 2 or any(p in ("", ".", "..") for p in repo.split("/")):
        raise ValueError("invalid public repository")
    module_path = Path(__file__).resolve().parents[2] / "homework-1/scripts/download_model.py"
    spec = importlib.util.spec_from_file_location("hw1_download", module_path)
    downloader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(downloader)
    files = manifest["files"]
    if len({entry["path"] for entry in files}) != len(files):
        raise ValueError("duplicate artifact paths")
    for entry in files:
        downloader.validate(entry)

    def fetch(entry):
        # Each thread owns its Session and a distinct artifact/partial file.
        with downloader.requests.Session() as session:
            downloader.download(session, repo, entry, args.output)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fetch, files))
    print(f"Verified all {len(files)} pinned artifacts for {repo}", flush=True)


if __name__ == "__main__":
    main()
