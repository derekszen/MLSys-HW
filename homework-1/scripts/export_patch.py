"""Export dirty TokenSpeed changes, including new files, without changing its index."""
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    repo = root / "tokenspeed"
    def git(*args, check=True):
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, check=check)
    base = git("rev-parse", "HEAD").stdout.decode().strip()
    patch = git("diff", "--binary", "HEAD", "--").stdout
    new = git("ls-files", "--others", "--exclude-standard", "-z").stdout.split(b"\0")
    for raw in sorted(p for p in new if p):
        item = git("diff", "--binary", "--no-index", "--", "/dev/null", raw.decode(), check=False)
        if item.returncode != 1:
            raise RuntimeError(item.stderr.decode())
        patch += item.stdout
    destination = root / "patches" / "ascend-qwen3.5.patch"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(patch)
    (destination.parent / "manifest.json").write_text(json.dumps({
        "base_commit": base,
        "patch": destination.name,
        "sha256": hashlib.sha256(patch).hexdigest(),
        "new_files": [p.decode() for p in sorted(p for p in new if p)],
        "apply": "git -C tokenspeed apply --check ../patches/ascend-qwen3.5.patch && git -C tokenspeed apply ../patches/ascend-qwen3.5.patch",
    }, indent=2) + "\n")
    print(f"Exported {len(patch)} bytes against {base}")


if __name__ == "__main__":
    main()
