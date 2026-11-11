"""Count total and active parameters from tensor shapes in downloaded safetensors headers.

Active parameters are counted per text token. Each language-model tensor is included, with each bank of routed
experts scaled by experts selected per token / experts in that bank. Routing settings come from config.json or
params.json. Shared experts count in full.

Vision and audio encoders count toward the total but not the active count, as in developers' "A3B"-style names.
The same exclusion applies to multi-token prediction layers, used only for speculative decoding, and per-layer
lookup tables such as n-gram embeddings, where each token reads a few rows. Quantization scales are excluded
from both counts. Two 4-bit weights packed into one byte count as two parameters.

The active count is unavailable when the files cannot determine it: the experts-per-token value is missing,
expert banks are shared across layers, or expert counts differ by layer.

    python tools/param_count.py MODEL_ID [MODEL_ID ...]      needs SNAPME_ROOT and the downloaded weights
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import struct
import sys
from collections import defaultdict

from snapme_eval import registry

ENCODER = re.compile(r"(^|\.)(vision|visual|vision_tower|vision_encoder|vision_model|audio|audio_tower|mm_projector|"
                     r"multi_modal_projector|vision_language_adapter|patch_merger|embed_vision|embed_audio|"
                     r"pre_mm_projector_norm|mm_proj|merger)(\.|_|$)")
LOOKUP = re.compile(r"ngram_embedding|ple_embedding|embed_tokens_per_layer|per_layer_embed")
SCALE = re.compile(r"(_scale(_inv|_2)?|\.scales?|_global_scale|\.zeros|\.qzeros|\.g_idx|\.absmax|_amax)$")
PER_EXPERT = re.compile(r"^(.*?)\.experts\.(\d+)\.")
FUSED_EXPERT = re.compile(r"^(.*?)\.experts\.(?!\d)")
LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")
K_KEYS = ("num_experts_per_tok", "top_k_experts", "moe_top_k", "moe_topk", "moe_k", "num_experts_per_token",
          "experts_per_token", "n_activated_experts", "num_selected_experts")
E_KEYS = ("num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts", "num_routed_experts")
L_KEYS = ("num_hidden_layers", "n_layers", "num_layers")


def shards(snap: pathlib.Path) -> list[pathlib.Path]:
    """The weight files of one format: the files a Hugging Face index lists, else Mistral's consolidated index."""
    for index in ("model.safetensors.index.json", "consolidated.safetensors.index.json"):
        if (snap / index).exists():
            return sorted({snap / f for f in json.loads((snap / index).read_text())["weight_map"].values()})
    single = [snap / f for f in ("model.safetensors", "consolidated.safetensors") if (snap / f).exists()]
    return single[:1] or sorted(snap.glob("*.safetensors"))


def headers(snap: pathlib.Path) -> dict[str, dict]:
    out = {}
    for f in shards(snap):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        h.pop("__metadata__", None)
        out.update(h)
    return out


def find(cfg, keys):
    """First integer value under any of keys, searching nested dicts."""
    if isinstance(cfg, dict):
        for k in keys:
            v = cfg.get(k)
            if isinstance(v, int) and not isinstance(v, bool):
                return v
        for v in cfg.values():
            r = find(v, keys)
            if r is not None:
                return r
    return None


def text_config(cfg: dict) -> dict:
    for c in cfg.values():
        for k in ("text_config", "llm_config", "language_config"):
            if isinstance(c.get(k), dict):
                return c[k]
    return next(iter(cfg.values()), {})


def count(snap: pathlib.Path) -> dict:
    cfg = {n: json.loads((snap / n).read_text()) for n in ("config.json", "params.json") if (snap / n).exists()}
    text = text_config(cfg)
    k, e_cfg = find(text, K_KEYS) or find(cfg, K_KEYS), find(text, E_KEYS) or find(cfg, E_KEYS)
    n_layers = find(text, L_KEYS)
    h = headers(snap)
    scaled = {SCALE.sub("", n) for n in h if SCALE.search(n)}   # tensor names associated with quantization scales
    total = encoder = skipped = dense = 0
    banks: dict[str, dict] = defaultdict(lambda: {"params": 0, "ids": set(), "lead": None})
    for name, t in h.items():
        if SCALE.search(name):
            continue
        n = 1
        for d in t["shape"]:
            n *= d
        if t["dtype"] == "U8" and name in scaled:
            n *= 2   # two 4-bit values per byte
        total += n
        layer = LAYER.search(name)
        if ENCODER.search(name):
            encoder += n
        elif name.startswith("mtp.") or ".mtp." in name or LOOKUP.search(name) or (
                layer and n_layers and int(layer.group(1)) >= n_layers and not name.startswith(("vision", "audio"))):
            skipped += n
        elif re.search(r"share[d]?_expert", name):
            dense += n
        elif m := PER_EXPERT.match(name):
            banks[m.group(1)]["params"] += n
            banks[m.group(1)]["ids"].add(m.group(2))
        elif (m := FUSED_EXPERT.match(name)) or (e_cfg and len(t["shape"]) == 3 and t["shape"][0] == e_cfg):
            b = banks[m.group(1) if m else name.rsplit(".", 2)[0]]
            b["params"] += n
            b["lead"] = t["shape"][0]
        else:
            dense += n
    routed = sum(b["params"] for b in banks.values())
    sizes = {len(b["ids"]) or b["lead"] for b in banks.values()}
    shared_banks = bool(banks) and not all(LAYER.search(p + ".") for p in banks)
    note = ("no experts-per-token value in the config" if banks and not k else
            "expert banks shared across layers" if shared_banks else
            "expert count differs by layer" if len(sizes) > 1 else "")
    active = None if note else dense + sum(b["params"] * k / (len(b["ids"]) or b["lead"]) for b in banks.values())
    return {"total_b": round(total / 1e9, 2), "active_b": round(active / 1e9, 2) if active is not None else None,
            "note": note, "encoder_b": round(encoder / 1e9, 2), "mtp_and_lookup_b": round(skipped / 1e9, 2),
            "routed_b": round(routed / 1e9, 2), "experts_per_token": k, "experts": sorted(sizes) if banks else None,
            "expert_banks": len(banks), "text_layers": n_layers}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    a = ap.parse_args()
    cache = pathlib.Path(os.environ.get("HF_HUB_CACHE", pathlib.Path(os.environ["SNAPME_ROOT"]) / "models/hf/hub"))
    for mid in a.models:
        m = registry.model(mid)
        snap = registry.snapshot_path(m, cache)
        if not shards(snap) or not shards(snap)[0].exists():
            print(f"{mid}: downloaded safetensors files not found at {snap}", file=sys.stderr)
            continue
        print(json.dumps({"model": mid, "config_total_b": m.get("params_total_b"),
                          "config_active_b": m.get("params_active_b"), **count(snap)}))


if __name__ == "__main__":
    main()
