"""Fetch public model manifests/configs only; never download checkpoint weights."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

API = "https://modelscope.cn/api/v1/models"
MODELS = ("Qwen3.5-0.8B", "Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.5-27B")


def fetch(url):
    with urllib.request.urlopen(url, timeout=45) as response:
        return response.read()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    catalog = []
    for name in MODELS:
        target = args.output / name
        target.mkdir(parents=True, exist_ok=True)
        manifest_path = target / "source-manifest.json"
        repo = f"Qwen/{name}"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["repo"] != repo or manifest["source"] != "modelscope.cn":
                raise ValueError("Manifest source mismatch")
        else:
            raw = fetch(f"{API}/{repo}/repo/files?Revision=master&Recursive=true")
            body = json.loads(raw)
            if not body.get("Success"):
                raise RuntimeError("ModelScope listing failed")
            entries = []
            for file in body["Data"]["Files"]:
                path = file.get("Path", "")
                if path.endswith((".json", ".safetensors", ".jinja", ".model")) or path in ("merges.txt", "LICENSE", "LICENSE.txt"):
                    entries.append({"path": path, "revision": file["Revision"],
                                    "sha256": file["Sha256"], "size": file["Size"]})
            manifest = {"source": "modelscope.cn", "repo": repo, "files": entries}
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        config_entry = next(e for e in manifest["files"] if e["path"] == "config.json")
        query = urllib.parse.urlencode({"Revision": config_entry["revision"], "FilePath": "config.json"})
        raw_config = fetch(f"{API}/{repo}/repo?{query}")
        if len(raw_config) != config_entry["size"] or hashlib.sha256(raw_config).hexdigest() != config_entry["sha256"]:
            raise ValueError("Config size/hash mismatch")
        (target / "config.json").write_bytes(raw_config)
        c = json.loads(raw_config)["text_config"]
        weights = [e for e in manifest["files"] if e["path"].endswith(".safetensors")]
        if not weights:
            raise ValueError("No checkpoint files in manifest")
        row = {"model": name, "repo": repo, "checkpoint_bytes": sum(e["size"] for e in weights),
               "shards": len(weights), "config_sha256": config_entry["sha256"],
               "hidden": c["hidden_size"], "layers": c["num_hidden_layers"],
               "intermediate": c["intermediate_size"], "q_heads": c["num_attention_heads"],
               "kv_heads": c["num_key_value_heads"], "head_dim": c["head_dim"],
               "gdn_k_heads": c["linear_num_key_heads"], "gdn_v_heads": c["linear_num_value_heads"],
               "gdn_k_dim": c["linear_key_head_dim"], "gdn_v_dim": c["linear_value_head_dim"],
               "vocab": c["vocab_size"], "weights_downloaded": False}
        catalog.append(row)
        print(json.dumps(row), flush=True)
    (args.output / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")


if __name__ == "__main__":
    main()
