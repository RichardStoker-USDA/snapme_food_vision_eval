"""Estimate intake from before-and-after photos or a package front and ingredient label.

If the companion photo is absent, use the corresponding direct prompt with the first photo alone.
"""
from __future__ import annotations

import hashlib
import pathlib

from snapme_eval import parse
from snapme_eval.client import Chat, image_data_url, user_message
from snapme_eval.systems.direct import prompt as single_prompt

PROMPTS = pathlib.Path(__file__).resolve().parents[1] / "prompts"
NAME, VERSION = "consumed", "v2"


def prompt(photo_type: str) -> str:
    return (PROMPTS / f"consumed_v2_{photo_type}.txt").read_text()


def prompt_hash() -> str:
    h = hashlib.sha256()
    for t in ("before", "front"):
        h.update(prompt(t).encode())
    return h.hexdigest()[:16]


async def predict(row: dict, chat: Chat, release_dir: pathlib.Path) -> dict:
    url, meta = image_data_url(release_dir / row["path"])
    comp = release_dir / row["companion_path"]
    if row.get("companion_exists", True) and comp.exists():
        url2, _ = image_data_url(comp)
        msg, used = user_message(prompt(row["photo_type"]), [url, url2]), "pair"
    else:
        msg, used = user_message(single_prompt(row["photo_type"]), [url]), "single"
    reply = await chat([msg])
    out = {"image": meta, "photos_used": used, **{k: v for k, v in reply.items() if k != "content"},
           "raw": reply.get("content", "")}
    if "error" in reply:
        out["items"], out["parse_error"] = None, "request failed"
        return out
    out["items"], out["parse_error"] = parse.items(reply["content"])
    return out
