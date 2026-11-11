"""Download registry models at their pinned commits into the Hugging Face cache under $SNAPME_ROOT.

    snapme-eval download --model qwen3-5-9b [--model ...]      (HF_HOME defaults to $SNAPME_ROOT/models/hf)

Only weights, configs, tokenizers, templates and model code are fetched; duplicate weight sets (Mistral
`consolidated`, `original/`), GGUF and ONNX files are skipped unless a model file names its own `files`.
Writes one JSON receipt per model to $SNAPME_ROOT/models/receipts/. Needs only the standard library and
huggingface_hub, so it also runs from a small separate environment on a transfer node.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import time
import tomllib

from huggingface_hub import snapshot_download

ALLOW = ["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "*.tiktoken", "*.py", "tekken.json", "*.md"]
IGNORE = ["consolidated*", "original/*", "*.gguf", "*.onnx", "onnx/*", "*.bin", "*.pth", "*.pt", "*.h5",
          "*.msgpack", "*.ot", "*.mlmodel*", "coreml/*", "openvino/*"]


def download(paths: list[str]) -> None:
    """Download each model file's weights at its pinned revision; skip models with a matching receipt."""
    root = pathlib.Path(os.environ["SNAPME_ROOT"])
    receipts = root / "models" / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    for p in paths:
        m = tomllib.loads(pathlib.Path(p).read_text())
        receipt = receipts / f"{m['id']}.json"
        if receipt.exists() and json.loads(receipt.read_text()).get("revision") == m["revision"]:
            print(f"skip {m['id']} (receipt present)", flush=True)
            continue
        t0 = time.time()
        print(f"get {m['hf_repo']}@{m['revision'][:12]}", flush=True)
        # A model file may name its own files (e.g. a single PyTorch checkpoint) with `files = [...]`.
        allow, ignore = (m["files"], None) if m.get("files") else (ALLOW, IGNORE)
        local = snapshot_download(m["hf_repo"], revision=m["revision"], allow_patterns=allow,
                                  ignore_patterns=ignore, max_workers=16)
        # Refresh access times: the 90-day purge goes by atime and the cache keeps upstream mtimes.
        subprocess.run(["find", "-L", local, "-type", "f", "-exec", "touch", "-a", "-c", "{}", "+"], check=False)
        size = sum(f.stat().st_size for f in pathlib.Path(local).rglob("*") if f.is_file())
        receipt.write_text(json.dumps({
            "id": m["id"], "hf_repo": m["hf_repo"], "revision": m["revision"], "path": local,
            "bytes": size, "seconds": round(time.time() - t0, 1),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=1))
        print(f"done {m['id']} {size/1e9:.1f} GB in {time.time()-t0:.0f}s", flush=True)
