"""Export HW2 relative to upstream + frozen HW1; never modify the source index.

Run with Python's standard library only. Temporary clones receive both patches;
all tracked and nonignored new files must match the intended working tree.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile

BASE = "52bdb9763bb2da3d61336398072671885cc0a33e"
HW1_SHA256 = "e68f2f26e49d30180280adbab0bb0ebaf433246e27457788e60277621613b8c6"
HW2_FILES = (
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
)
HW2_MODES = {name: "100644" for name in HW2_FILES}
for name in (
    "python/tokenspeed/runtime/layers/logits_processor.py",
    "python/tokenspeed/runtime/utils/env.py",
    "python/tokenspeed/runtime/utils/server_args.py",
):
    HW2_MODES[name] = "100755"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git(repo, *args, input=None, codes=(0,)):
    result = subprocess.run(
        ["git", "-c", "core.quotePath=false", "-c", "diff.mnemonicPrefix=false",
         *args], cwd=repo, input=input, capture_output=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C"},
    )
    if result.returncode not in codes:
        raise RuntimeError(f"git {' '.join(args)}: {result.stderr.decode()}")
    return result.stdout


def snapshot(repo):
    names = git(repo, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    files = {}
    for name in sorted(set(os.fsdecode(p) for p in names.split(b"\0") if p)):
        path = repo / name
        if not path.exists() and not path.is_symlink():
            continue  # Tracked deletions are represented by absence.
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            files[name] = ("120000", os.fsencode(os.readlink(path)))
        elif stat.S_ISREG(mode):
            files[name] = ("100755" if mode & 0o111 else "100644", path.read_bytes())
        else:
            raise ValueError(f"unsupported non-file in export: {name}")
    return files


def inventory(files):
    return {name: {"mode": mode, "sha256": sha256(data)}
            for name, (mode, data) in files.items()}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_regular(path, entry):
    """Materialize a snapshot entry without letting umask lose its Git mode."""
    mode, data = entry
    if mode not in ("100644", "100755"):
        raise ValueError(f"unsupported regular-file mode {mode}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    permissions = int(mode, 8) & 0o777
    path.chmod(permissions)
    if stat.S_IMODE(path.stat().st_mode) != permissions:
        raise ValueError(f"temporary file mode differs from {mode}: {path}")


def file_patch(work, name, old_entry, new_entry):
    """Diff bytes AND modes; preserve executable bits on both temporary sides."""
    write_regular(work / "new" / name, new_entry)
    old_name = "/dev/null"
    if old_entry is not None:
        write_regular(work / "old" / name, old_entry)
        old_name = f"old/{name}"
    chunk = git(work, "diff", "--no-index", "--binary", "--full-index",
                "--no-ext-diff", "--no-textconv", "--no-renames", "--no-color",
                "--diff-algorithm=myers", "--no-indent-heuristic", "--unified=3",
                "--inter-hunk-context=0", "--src-prefix=a/", "--dst-prefix=b/", "--",
                old_name, f"new/{name}", codes=(1,))
    # Normalize only filename headers, never hunk contents or mode headers.
    lines = chunk.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith((b"diff --git ", b"--- ", b"+++ ")):
            lines[i] = line.replace(b"a/old/", b"a/").replace(b"a/new/", b"a/").replace(b"b/new/", b"b/")
    return b"".join(lines)


def main():
    root = Path(__file__).resolve().parents[1]
    project = root.parent
    source = project / "homework-1/tokenspeed"
    frozen_path = project / "homework-1/patches/ascend-qwen3.5.patch"
    frozen = frozen_path.read_bytes()
    manifest = json.loads(frozen_path.with_name("manifest.json").read_text())
    if (sha256(frozen) != HW1_SHA256 or manifest["sha256"] != HW1_SHA256
            or manifest["base_commit"] != BASE or manifest["patch"] != frozen_path.name):
        raise ValueError("frozen HW1 patch/manifest changed")
    if git(source, "rev-parse", "HEAD").decode().strip() != BASE:
        raise ValueError("unexpected source HEAD")
    index_path = Path(os.fsdecode(git(source, "rev-parse", "--git-path", "index").strip()))
    if not index_path.is_absolute():
        index_path = source / index_path
    index_before = index_path.read_bytes()
    intended = snapshot(source)
    with tempfile.TemporaryDirectory(prefix="hw2-patch-") as temporary:
        work = Path(temporary)
        clean = work / "clean"
        git(work, "clone", "--quiet", "--shared", "--no-checkout", str(source), str(clean))
        git(clean, "checkout", "--quiet", "--detach", BASE)
        git(clean, "apply", "--check", "--whitespace=error", "-", input=frozen)
        git(clean, "apply", "--whitespace=error", "-", input=frozen)
        baseline = snapshot(clean)
        changed = sorted(name for name in baseline.keys() | intended.keys()
                         if baseline.get(name) != intended.get(name))
        if changed != list(HW2_FILES):
            raise ValueError(f"HW2 delta is not exactly the assigned files: {changed}")
        chunks = []
        for name in HW2_FILES:
            expected_mode = HW2_MODES[name]
            if name not in intended or intended[name][0] != expected_mode:
                raise ValueError(f"expected HW2 file mode {expected_mode}: {name}")
            if name in baseline and baseline[name][0] != expected_mode:
                raise ValueError(f"expected baseline file mode {expected_mode}: {name}")
            chunks.append(file_patch(work, name, baseline.get(name), intended[name]))
        patch = b"".join(chunks)
        git(clean, "apply", "--check", "--whitespace=error", "-", input=patch)
        git(clean, "apply", "--whitespace=error", "-", input=patch)
        reconstructed = snapshot(clean)
        # Compare actual bytes and git modes for the entire intended file set.
        if reconstructed != intended:
            raise ValueError("clean base + HW1 + HW2 differs from intended working tree")
        if snapshot(source) != intended or index_path.read_bytes() != index_before:
            raise ValueError("source working tree/index changed during export")
        if frozen_path.read_bytes() != frozen:
            raise ValueError("frozen HW1 changed during export")
        records = inventory(intended)
        tree_digest = sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode())
        destination = root / "patches/hccl-torch29.patch"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(patch)
        write_json(destination.with_name("manifest.json"), {
            "base_commit": BASE,
            "prerequisite": {"patch": "homework-1/patches/ascend-qwen3.5.patch", "sha256": HW1_SHA256},
            "patch": destination.name, "sha256": sha256(patch),
            "kind": "incremental_after_frozen_hw1", "files": inventory({n: intended[n] for n in HW2_FILES}),
            "new_files": [n for n in HW2_FILES if n not in baseline],
            "apply_order": ["git apply --check /path/to/homework-1/patches/ascend-qwen3.5.patch",
                            "git apply /path/to/homework-1/patches/ascend-qwen3.5.patch",
                            "git apply --check /path/to/homework-2/patches/hccl-torch29.patch",
                            "git apply /path/to/homework-2/patches/hccl-torch29.patch"],
        })
        write_json(root / "patches/patch-verification.json", {
            "passed": True, "base_commit": BASE, "hw1_sha256": HW1_SHA256,
            "hw2_sha256": sha256(patch), "hw2_bytes": len(patch), "incremental_files": changed,
            "comparison": "byte-exact contents and git modes; all tracked and nonignored new files",
            "verified_file_count": len(intended), "verified_tree_sha256": tree_digest,
            "incremental_files_verified": {n: records[n] for n in HW2_FILES},
            "source_index_unchanged": True, "source_files_unchanged": True,
            "frozen_hw1_unchanged": True,
            "scope": "patch application/byte verification only, not hardware execution",
        })
    print(f"Exported {len(patch)} incremental bytes; verified {len(intended)} files byte-exact")


if __name__ == "__main__":
    main()
