"""Compare scale-card conditions on meal photos with a located checkerboard card.
Vary whether the card is painted over and whether the prompt names it:

    marker:masked        card painted over, card instruction removed
    marker:masked_told   card painted over, standard prompt
    marker:untold        photo unchanged, card instruction removed
    marker:control       same-size patch painted elsewhere, card visible, standard prompt

Compare each condition with the standard one-photo run (direct): unchanged photo and standard prompt.
Select photos from tables/marker_v1.csv (marker.py), which excludes package fronts.
"""
from __future__ import annotations

import base64
import hashlib
import io
import pathlib
from functools import cache
from types import SimpleNamespace

from snapme_eval import marker, parse
from snapme_eval.client import MAX_SIDE, Chat, user_message
from snapme_eval.systems.direct import prompt as direct_prompt

NAME, VERSION = "marker", "v1"
ARMS = {"masked": ("card", False), "masked_told": ("card", True), "untold": (None, False),
        "control": ("control", True)}
CARD_SENTENCE = (" A black-and-white checkerboard card, 1.5 x 1.5 inches (3.8 x 3.8 cm) with four squares, is usually"
                 " placed at the lower left for scale.")
CARD_PHRASE = ", packaging or the scale card"


def prompt(told: bool) -> str:
    text = direct_prompt("before")
    if told:
        return text
    assert CARD_SENTENCE in text and CARD_PHRASE in text
    return text.replace(CARD_SENTENCE, "").replace(CARD_PHRASE, " or packaging")


@cache
def _table():
    return marker.table()


def select(images):
    return images[images["image_id"].isin(_table().index)]


def data_url(im) -> tuple[str, dict]:
    w, h = im.size
    scale = min(1.0, MAX_SIDE / max(w, h))
    if scale < 1.0:
        im = im.resize((round(w * scale), round(h * scale)), marker.Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92)
    return ("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
            {"orig_w": w, "orig_h": h, "sent_w": im.size[0], "sent_h": im.size[1]})


def variant(arm: str) -> SimpleNamespace:
    hidden, told = ARMS[arm]

    def prompt_hash() -> str:
        return hashlib.sha256(f"{arm}|{marker.MARGIN}|{marker.RING}|{prompt(told)}".encode()).hexdigest()[:16]

    async def predict(row: dict, chat: Chat, release_dir: pathlib.Path) -> dict:
        t = _table().loc[row["image_id"]]
        im = marker.open_photo(release_dir / row["path"])
        if hidden == "card":
            im = marker.hide(im, [t.x0, t.y0, t.x1, t.y1])
        elif hidden == "control":
            im = marker.hide(im, [t.cx0, t.cy0, t.cx1, t.cy1])
        url, meta = data_url(im)
        reply = await chat([user_message(prompt(told), [url])])
        out = {"image": meta, "arm": arm, **{k: v for k, v in reply.items() if k != "content"},
               "raw": reply.get("content", "")}
        if "error" in reply:
            out["items"], out["parse_error"] = None, "request failed"
        else:
            out["items"], out["parse_error"] = parse.items(reply["content"])
        return out

    return SimpleNamespace(NAME=NAME, VERSION=f"{VERSION}_{arm}", prompt_hash=prompt_hash, predict=predict,
                           select=select)
