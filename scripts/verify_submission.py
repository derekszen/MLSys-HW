"""Verify a submission ZIP without extracting it (Python standard library only)."""
import argparse
import hashlib
import json
from pathlib import PurePosixPath
import re
import stat
import zipfile

MAX_FILE_BYTES = 100_000_000
MAX_TOTAL_BYTES = 300_000_000
MAX_MANIFEST_BYTES = 5_000_000


def safe_name(name):
    if (not isinstance(name, str) or not name or "\\" in name
            or any(ord(character) < 32 or ord(character) == 127 for character in name)):
        raise ValueError("invalid archive path")
    path = PurePosixPath(name)
    if (not path.parts or path.is_absolute() or path.as_posix() != name
            or any(part in (".", "..", ".git", ".venv", "__pycache__") for part in path.parts)
            or ":" in name or name.endswith("/")):
        raise ValueError(f"unsafe archive path: {name}")
    return name


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [safe_name(entry.filename) for entry in entries]
        if len(names) != len(set(names)) or names.count("MANIFEST.json") != 1:
            raise ValueError("duplicate members or missing manifest")
        total = 0
        for entry in entries:
            mode = entry.external_attr >> 16
            if stat.S_IFMT(mode) != stat.S_IFREG or stat.S_IMODE(mode) not in (0o644, 0o755):
                raise ValueError(f"unsupported archive mode: {entry.filename}")
            if entry.flag_bits & 1 or not 0 <= entry.file_size <= MAX_FILE_BYTES:
                raise ValueError("encrypted or oversized archive member")
            total += entry.file_size
        if total > MAX_TOTAL_BYTES:
            raise ValueError("archive uncompressed size exceeds bound")
        if archive.getinfo("MANIFEST.json").file_size > MAX_MANIFEST_BYTES:
            raise ValueError("oversized manifest")
        manifest = json.loads(archive.read("MANIFEST.json"))
        if manifest.get("format_version") != 1 or manifest.get("assignment") not in (1, 2):
            raise ValueError("unsupported manifest")
        files = manifest.get("files")
        if not isinstance(files, dict) or set(files) != set(names) - {"MANIFEST.json"}:
            raise ValueError("manifest membership differs from archive")
        for name, record in files.items():
            safe_name(name)
            info = archive.getinfo(name)
            if (not isinstance(record, dict) or type(record.get("bytes")) is not int
                    or record["bytes"] != info.file_size
                    or record.get("mode") != format(stat.S_IMODE(info.external_attr >> 16), "04o")
                    or not isinstance(record.get("sha256"), str)
                    or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None):
                raise ValueError(f"invalid size/mode/digest record: {name}")
            digest = hashlib.sha256()
            actual_size = 0
            with archive.open(name) as stream:
                while chunk := stream.read(1024 * 1024):
                    actual_size += len(chunk)
                    if actual_size > info.file_size or actual_size > MAX_FILE_BYTES:
                        raise ValueError(f"inflated size exceeds bound: {name}")
                    digest.update(chunk)
            if actual_size != record["bytes"] or digest.hexdigest() != record["sha256"]:
                raise ValueError(f"content checksum differs: {name}")
        return {"passed": True, "assignment": manifest["assignment"],
                "verified_files": len(files), "uncompressed_bytes": total,
                "source": manifest.get("source")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    args = parser.parse_args()
    try:
        result = verify_archive(args.archive)
    except (ValueError, OSError, zipfile.BadZipFile, KeyError, TypeError) as error:
        parser.exit(1, f"Verification failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
