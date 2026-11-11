"""Estimate a food list from one photo.

The request uses a prompt selected by photo type and returns names, preparation, ingredients and grams.
Other methods can use these lists as inputs.
"""
from __future__ import annotations

import hashlib
import pathlib

from snapme_eval import parse
from snapme_eval.client import Chat, image_data_url, user_message

PROMPTS = pathlib.Path(__file__).resolve().parents[1] / "prompts"
NAME, VERSION = "direct", "v2"


def prompt(photo_type: str) -> str:
    return (PROMPTS / f"direct_v2_{photo_type}.txt").read_text()


def prompt_hash() -> str:
    h = hashlib.sha256()
    for t in ("before", "front"):
        h.update(prompt(t).encode())
    return h.hexdigest()[:16]


async def predict(row: dict, chat: Chat, release_dir: pathlib.Path) -> dict:
    url, meta = image_data_url(release_dir / row["path"])
    reply = await chat([user_message(prompt(row["photo_type"]), [url])])
    out = {"image": meta, **{k: v for k, v in reply.items() if k != "content"}, "raw": reply.get("content", "")}
    if "error" in reply:
        out["items"], out["parse_error"] = None, "request failed"
        return out
    items, err = parse.items(reply["content"])
    out["items"], out["parse_error"] = items, err
    return out
