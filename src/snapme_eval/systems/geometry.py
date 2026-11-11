"""Geometry: the model finds foods with boxes, a MoGe-2 point map turns each box into a volume, and the
model then estimates preparation, ingredients and grams with those measurements in view. Needs the point maps from
hpc/atlas/depth_moge.sbatch; a photo without one falls back to the direct prompt. Packaged foods (front
photos) also use the direct prompt, since the label carries the amount.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pathlib

from snapme_eval import geometry, parse
from snapme_eval.client import Chat, image_data_url, user_message
from snapme_eval.paths import root
from snapme_eval.systems.direct import prompt as direct_prompt

PROMPTS = pathlib.Path(__file__).resolve().parents[1] / "prompts"
NAME, VERSION = "geometry", "v2"
GEO_DIR = os.environ.get("SNAPME_GEOMETRY_DIR", "derived/moge2_vitl_normal")


def prompt_hash() -> str:
    t = "".join((PROMPTS / f"geometry_v2_{k}.txt").read_text() for k in ("locate", "amounts"))
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def x_first(box: list, order: str) -> list:
    """Convert the model's box order ("xy" or "yx") to [x_min, y_min, x_max, y_max]."""
    return [box[1], box[0], box[3], box[2]] if order == "yx" else list(box)


def _boxes(text: str, order: str = "xy"):
    body = parse.strip_reasoning(text or "")
    for cand in parse._candidates(body):
        obj = parse._loads(cand)
        if isinstance(obj, dict) and isinstance(obj.get("items"), list):
            items = [{**it, "box": x_first(it["box"], order)} for it in obj["items"]
                     if isinstance(it, dict) and it.get("name")
                     and isinstance(it.get("box"), list) and len(it["box"]) == 4
                     and all(isinstance(v, (int, float)) for v in it["box"])]
            card = obj.get("card")
            ok_card = isinstance(card, list) and len(card) == 4 and all(isinstance(v, (int, float)) for v in card)
            return items, x_first(card, order) if ok_card else None
    return None, None


async def _direct(row, chat, url, meta, why):
    reply = await chat([user_message(direct_prompt(row["photo_type"]), [url])])
    out = {"image": meta, "route": f"direct ({why})", **{k: v for k, v in reply.items() if k != "content"},
           "raw": reply.get("content", "")}
    if "error" in reply:
        out["items"], out["parse_error"] = None, "request failed"
    else:
        out["items"], out["parse_error"] = parse.items(reply["content"])
    return out


async def predict(row: dict, chat: Chat, release_dir: pathlib.Path) -> dict:
    url, meta = image_data_url(release_dir / row["path"])
    if row["photo_type"] != "before":
        return await _direct(row, chat, url, meta, "package")
    geo = await asyncio.to_thread(geometry.load, root() / GEO_DIR / f"{row['image_id']}.npz")
    if geo is None:
        return await _direct(row, chat, url, meta, "no point map")

    first = await chat([user_message((PROMPTS / "geometry_v2_locate.txt").read_text(), [url])])
    if "error" in first:
        return {"image": meta, "items": None, "parse_error": "request failed", "error": first["error"]}
    order = chat.info.get("box_order", "xy")
    found, card = _boxes(first.get("content", ""), order)
    if not found:
        return await _direct(row, chat, url, meta, "no boxes")
    m = await asyncio.to_thread(geometry.measure, geo, [it["box"] for it in found], card)
    if not m.get("ok"):
        return await _direct(row, chat, url, meta, m.get("reason", "geometry failed"))
    lines = [f"- {it['name']}: about {g['volume_ml']:.0f} mL, covering {g['footprint_cm2']:.0f} cm2 of the table, "
             f"up to {g['max_height_cm']:.1f} cm high" for it, g in zip(found, m["items"], strict=True)]
    text = (PROMPTS / "geometry_v2_amounts.txt").read_text().format(measurements="\n".join(lines))
    second = await chat([user_message(text, [url])])
    out = {"image": meta, "route": "geometry", "locate_raw": first.get("content", ""), "box_order": order,
           "boxes": found, "card": card, "geometry": m, **{k: v for k, v in second.items() if k != "content"},
           "raw": second.get("content", "")}
    out["usage_locate"] = first.get("usage")
    if "error" in second:
        out["items"], out["parse_error"] = None, "request failed"
        return out
    out["items"], out["parse_error"] = parse.items(second["content"])
    json.dumps(out)  # fail here, not in the writer, if anything is not serializable
    return out
