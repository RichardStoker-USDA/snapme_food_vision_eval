"""Add a model to the registry (configs/models/<id>.toml) from its Hugging Face card.

Pins the exact commit, records size, license and architecture, and proposes a GPU
plan for Atlas A100 (80 GB, NVLink) and L40S (48 GB, PCIe, native FP8).
Review the file before running it; the GPU plan is a starting point.

    python tools/add_model.py Qwen/Qwen3.8-27B --developer "Alibaba Qwen" --country China
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Usable memory per GPU type (GB) and GPUs per node, for the proposed plans. Add a type for another cluster.
GPU_GB = {"a100": 80, "l40s": 46, "h100": 80, "h200": 141, "b200": 180, "rtx-pro-6000": 96, "a6000": 48, "l4": 24}
UA = {"User-Agent": "snapme_eval-registry"}


def hf_json(url: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def weight_gb(repo: str, sha: str) -> float:
    tree = hf_json(f"https://huggingface.co/api/models/{repo}/tree/{sha}?recursive=true")
    st = [f for f in tree if f.get("path", "").endswith(".safetensors")]
    hf_format = [f for f in st if not f["path"].startswith(("consolidated", "original/"))]
    return sum(f.get("size", 0) for f in (hf_format or st)) / 1e9  # one weight set (Mistral ships two)


GPUS_PER_NODE = {"l40s": 4, "a6000": 4, "l4": 4}   # others: 8


def plan(gb: float, gpu: str) -> dict | None:
    """Smallest power-of-two tensor-parallel size where weights take at most 72% of GPU memory."""
    for tp in (1, 2, 4, 8):
        if gb / tp <= 0.72 * GPU_GB[gpu]:
            return {"gpus": tp, "tp": tp} if tp <= GPUS_PER_NODE.get(gpu, 8) else None  # one node
    return None


def weight_precision(cfg: dict, dtypes: list[str]) -> str:
    """Storage precision of the main weights, from the quantization config when there is one."""
    q = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config") or {}
    m = str(q.get("quant_method", "")).lower()
    if m in ("awq", "gptq", "autoround"):
        return f"int{q.get('bits', 4)}"
    if m == "modelopt":
        algo = str(q.get("quant_algo", "")).lower()
        return "nvfp4 mixed" if algo in ("mixed_precision", "") else algo
    if m.startswith("modelopt"):    # modelopt, modelopt_fp4, modelopt_mixed
        algo = str(q.get("quant_algo", "")).lower()
        return "nvfp4 mixed" if algo in ("mixed_precision", "") else algo
    if m in ("mxfp8", "mxfp4", "fp8", "bitsandbytes"):
        return m
    if m == "compressed-tensors":
        g = next(iter((q.get("config_groups") or {}).values()), {}).get("weights", {})
        return f"{g.get('type', 'int')}{g.get('num_bits', '')}"
    return "fp8" if "F8_E4M3" in dtypes else "bf16"


def slug(repo: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", repo.split("/")[1].lower()).strip("-")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--developer", required=True)
    ap.add_argument("--country", required=True)
    ap.add_argument("--id", help="registry id (default from repo name)")
    ap.add_argument("--architecture", default="", help="e.g. 'dense transformer' or 'mixture of experts, 3B active'")
    ap.add_argument("--active-b", type=float, default=None, help="active parameters for MoE, billions")
    ap.add_argument("--engine", default="vllm_0_32_0")
    ap.add_argument("--reasoning", choices=["none", "optional", "always"], default=None,
                    help="default: the --precision-of model's setting, else none")
    ap.add_argument("--tokenizer-mode", default="")
    ap.add_argument("--role", choices=["generalist", "decision", "perception"], default="generalist",
                    help="generalist writes a food record; decision answers typed questions; perception locates")
    ap.add_argument("--precision-of", default="", help="registry id of the reference-precision build of this model")
    ap.add_argument("--gpu-types", default="a100,l40s", help="GPU types to propose plans for (see GPU_GB)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=None, help="folder for the model file (default configs/models)")
    a = ap.parse_args()
    ref = {}
    if a.precision_of:   # a quantized build is run exactly like its reference: same thinking settings
        sys.path.insert(0, str(ROOT / "src"))
        from snapme_eval import registry
        ref = registry.model(a.precision_of)
    a.reasoning = a.reasoning or ref.get("reasoning", "none")

    info = hf_json(f"https://huggingface.co/api/models/{a.repo}?expand[]=sha&expand[]=createdAt"
                   f"&expand[]=cardData&expand[]=safetensors&expand[]=gated")
    sha = info["sha"]
    cfg = hf_json(f"https://huggingface.co/{a.repo}/resolve/{sha}/config.json")
    arch = (cfg.get("architectures") or ["?"])[0]
    params = (info.get("safetensors") or {}).get("total", 0) / 1e9
    dtypes = list((info.get("safetensors") or {}).get("parameters", {}))
    gb = weight_gb(a.repo, sha)
    mid = a.id or slug(a.repo)
    out = (pathlib.Path(a.out) if a.out else ROOT / "configs" / "models") / f"{mid}.toml"
    if out.exists() and not a.force:
        raise SystemExit(f"{out} exists (use --force)")

    plans = {g: plan(gb, g) for g in a.gpu_types.split(",")}
    moe = bool(re.search(r"expert|moe", a.architecture + json.dumps(cfg), re.I))
    lines = [
        f'id = "{mid}"',
        f'hf_repo = "{a.repo}"',
        f'revision = "{sha}"',
        f'developer = "{a.developer}"',
        f'country = "{a.country}"',
        f'released = "{info.get("createdAt", "")[:10]}"',
        f'license = "{(info.get("cardData") or {}).get("license", "unknown")}"',
        f'hf_architecture = "{arch}"',
        f'architecture = "{a.architecture}"',
        f"params_total_b = {params:.1f}",
        f"params_active_b = {a.active_b}" if a.active_b is not None else
        "# params_active_b: count from downloaded weights with tools/param_count.py" if moe else
        f"params_active_b = {params:.1f}",
        f'weights_dtype = "{weight_precision(cfg, dtypes)}"',
        f"weights_gb = {gb:.1f}",
        f'engine = "{a.engine}"',
        f'reasoning = "{a.reasoning}"',
        "max_model_len = 16384",
        f'role = "{a.role}"',
        *([f'precision_of = "{a.precision_of}"'] if a.precision_of else []),
        "",
        "# GPU plans per partition. Remove a block to keep the model off that partition.",
    ]
    for name, p in plans.items():
        if p:
            lines += [f"[serve.{name}]", f"gpus = {p['gpus']}", f"tp = {p['tp']}", ""]
    lines += ["[engine_args]", "# extra flags passed to the engine, e.g. tokenizer_mode = \"mistral\""]
    if a.tokenizer_mode:
        lines.append(f'tokenizer_mode = "{a.tokenizer_mode}"')
    if ref.get("chat_template_kwargs"):
        kw = ref["chat_template_kwargs"]
        lines += ["", "[chat_template_kwargs]", *(f"{k} = {json.dumps(v)}" for k, v in kw.items())]
    lines += ["", "[sampling]", "temperature = 0.0", "top_p = 1.0", "max_tokens = 4096", "seed = 20261108", ""]
    out.write_text("\n".join(lines))
    print(f"wrote {out}  weights {gb:.1f} GB  plans {plans}")


if __name__ == "__main__":
    main()
