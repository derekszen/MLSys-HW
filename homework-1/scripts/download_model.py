#!/usr/bin/env python3
"""Download public Qwen assets from ModelScope with pinned per-file SHA256 checks.

This is an artifact downloader, not a model implementation. No access token is
used. A saved manifest pins each selected file's revision, size, and SHA256.
Interrupted large downloads resume only when the server honors byte ranges.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
from urllib.parse import quote

import requests


API = "https://modelscope.cn/api/v1/models"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(entry):
    path = PurePosixPath(entry["path"])
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Unsafe model artifact path")
    if not isinstance(entry["size"], int) or entry["size"] < 0:
        raise ValueError("Invalid model artifact size")
    if len(entry["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in entry["sha256"]):
        raise ValueError("Model artifact lacks a SHA256 digest")
    if not entry["revision"]:
        raise ValueError("Model artifact lacks an immutable revision")


def download(session, repo, entry, output):
    validate(entry)
    target = output / entry["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == entry["size"] and sha256(target) == entry["sha256"]:
        print(f"Verified cached {entry['path']}", flush=True)
        return
    part = target.with_name(target.name + ".part")
    size = part.stat().st_size if part.exists() else 0
    if size > entry["size"]:
        raise ValueError("Partial artifact is larger than its manifest size")
    if size < entry["size"]:
        url = f"{API}/{quote(repo, safe='/')}/repo"
        headers = {"Range": f"bytes={size}-"} if size else {}
        with session.get(url, params={"Revision": entry["revision"], "FilePath": entry["path"]},
                         headers=headers, stream=True, timeout=(15, 90)) as response:
            response.raise_for_status()
            if size and response.status_code == 206:
                expected = f"bytes {size}-"
                if not response.headers.get("Content-Range", "").startswith(expected):
                    raise ValueError("Server returned the wrong resume range")
                mode = "ab"
            elif response.status_code == 200:
                size, mode = 0, "wb"
            else:
                raise ValueError("Unexpected artifact download response")
            print(f"Downloading {entry['path']} ({entry['size']} bytes; resume {size})", flush=True)
            last = time.monotonic()
            with part.open(mode) as stream:
                for chunk in response.iter_content(8 * 1024 * 1024):
                    if not chunk:
                        continue
                    stream.write(chunk)
                    size += len(chunk)
                    if size > entry["size"]:
                        raise ValueError("Downloaded artifact exceeds manifest size")
                    if time.monotonic() - last > 30:
                        print(f"{entry['path']}: {size}/{entry['size']} bytes", flush=True)
                        last = time.monotonic()
    if not part.exists() or part.stat().st_size != entry["size"] or sha256(part) != entry["sha256"]:
        raise ValueError(f"Size or SHA256 mismatch for {entry['path']}; partial file retained")
    part.replace(target)
    print(f"SHA256 verified: {entry['path']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-only", action="store_true", help="fetch manifest and non-weight assets only")
    args = parser.parse_args()
    if len(args.repo.split("/")) != 2 or any(p in ("", ".", "..") for p in args.repo.split("/")):
        raise ValueError("Expected a public owner/model repository")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "source-manifest.json"
    with requests.Session() as session:
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["repo"] != args.repo or manifest["source"] != "modelscope.cn":
                raise ValueError("Existing manifest belongs to another source/repository")
        else:
            response = session.get(f"{API}/{quote(args.repo, safe='/')}/repo/files",
                                   params={"Revision": "master", "Recursive": "true"}, timeout=30)
            response.raise_for_status()
            body = response.json()
            if not body.get("Success"):
                raise RuntimeError("ModelScope metadata request failed")
            entries = []
            for file in body["Data"]["Files"]:
                name = file.get("Path", "")
                if not (name.endswith((".json", ".safetensors", ".jinja", ".model")) or name in ("merges.txt", "LICENSE", "LICENSE.txt")):
                    continue
                entry = {"path": name, "revision": file["Revision"], "sha256": file["Sha256"], "size": file["Size"]}
                validate(entry)
                entries.append(entry)
            if not any(e["path"].endswith(".safetensors") for e in entries):
                raise ValueError("No safetensors checkpoint in manifest")
            manifest = {"source": "modelscope.cn", "repo": args.repo, "files": entries}
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        for entry in manifest["files"]:
            if not args.metadata_only or not entry["path"].endswith(".safetensors"):
                download(session, args.repo, entry, args.output)


if __name__ == "__main__":
    main()
