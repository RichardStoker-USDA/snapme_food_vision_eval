"""Combine earlier food lists using a model that also receives the photo.

Inputs are direct_v2 runs named by SNAPME_COMBINE_MEMBERS (comma-separated model IDs).
A missing or unreadable member list is omitted for that photo.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
from functools import cache

from snapme_eval import parse
from snapme_eval.client import Chat, image_data_url, user_message
from snapme_eval.paths import runs_dir

PROMPTS = pathlib.Path(__file__).resolve().parents[1] / "prompts"
NAME, VERSION = "combine", "v2"


def members() -> list[str]:
    return [m for m in os.environ.get("SNAPME_COMBINE_MEMBERS", "").split(",") if m]


def prompt_hash() -> str:
    t = (PROMPTS / "combine_v2.txt").read_text() + "|" + ",".join(members())
    return hashlib.sha256(t.encode()).hexdigest()[:16]


@cache
def member_predictions(model: str) -> dict:
    p = runs_dir() / os.environ.get("SNAPME_GT", "snapme_v1") / "direct_v2" / model / "predictions.jsonl"
    out = {}
    for line in p.open():
        r = json.loads(line)
        out[r["image_id"]] = r.get("items")
    return out


def render(items: list[dict]) -> str:
    rows = []
    for it in items:
        g, prep, ing = it.get("grams"), it.get("preparation"), it.get("ingredients") or []
        rows.append(f"  - {it['name']}" + (f" ({prep})" if prep and prep != "not applicable" else "")
                    + (f", {g:.0f} g" if isinstance(g, (int, float)) else "")
                    + (f"; ingredients: {', '.join(map(str, ing))}" if ing else ""))
    return "\n".join(rows) or "  (nothing listed)"


async def predict(row: dict, chat: Chat, release_dir: pathlib.Path) -> dict:
    blocks = []
    for i, m in enumerate(members()):
        items = member_predictions(m).get(row["image_id"])
        if items is not None:
            blocks.append(f"Reader {chr(65 + i)}:\n{render(items)}")
    text = (PROMPTS / "combine_v2.txt").read_text().format(records="\n\n".join(blocks))
    url, meta = image_data_url(release_dir / row["path"])
    reply = await chat([user_message(text, [url])])
    out = {"image": meta, "members": members(), "n_member_records": len(blocks),
           **{k: v for k, v in reply.items() if k != "content"}, "raw": reply.get("content", "")}
    if "error" in reply:
        out["items"], out["parse_error"] = None, "request failed"
        return out
    out["items"], out["parse_error"] = parse.items(reply["content"])
    return out
