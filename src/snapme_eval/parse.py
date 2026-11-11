"""Extract an item list from a model's text reply.

The parser accepts JSON with limited formatting repairs and preserves names, preparation, ingredients
and supported numeric fields. An unreadable list returns an error and is scored as an empty prediction.
"""
from __future__ import annotations

import json
import re

FIELDS = {"grams": "grams", "code": "code", "portion": "portion"}
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def strip_reasoning(text: str) -> str:
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return re.sub(r"<think>.*", "", text, flags=re.S) if "<think>" in text else text


def _candidates(text: str):
    for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.S):
        yield m.group(1)
    # balanced-brace scan for top-level objects, last one first
    depth, start, objs = 0, None, []
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                objs.append(text[start:i + 1])
    yield from reversed(objs)


def _loads(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        s2 = re.sub(r",\s*([}\]])", r"\1", s)            # trailing commas
        s2 = re.sub(r"(?<=\d)\s*(?:g|kcal|mg)\b", "", s2)  # units glued to numbers
        s2 = re.sub(r'(:\s*-?\d+(?:\.\d+)?)"(?=\s*[,}\]])', r"\1", s2)  # stray quote after a number: 0.3"}
        try:
            return json.loads(s2)
        except json.JSONDecodeError:
            return None


def _num(v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = _NUM.search(v.replace(",", ""))
        return float(m.group()) if m else None
    return None


def items(text: str) -> tuple[list[dict] | None, str | None]:
    """Return (items, error). Items carry name plus any numeric fields present."""
    body = strip_reasoning(text or "")
    for cand in _candidates(body):
        obj = _loads(cand)
        if isinstance(obj, dict):
            arr = obj["items"] if "items" in obj else obj.get("foods")  # an empty list is a valid answer
        elif isinstance(obj, list):
            arr = obj
        else:
            continue
        if not isinstance(arr, list):
            continue
        out = []
        for it in arr:
            if not isinstance(it, dict):
                continue
            name = it.get("name") or it.get("food") or it.get("description")
            if not name:
                continue
            row: dict = {"name": str(name).strip()}
            for k in FIELDS:
                if k in it:
                    row[k] = _num(it[k]) if k not in ("code", "portion") else it[k]
            # v2 text fields: preparation (string) and ingredients of a mixed dish (list of strings)
            if isinstance(it.get("preparation"), str) and it["preparation"].strip():
                row["preparation"] = it["preparation"].strip()
            if isinstance(it.get("ingredients"), list):
                row["ingredients"] = [str(x).strip() for x in it["ingredients"] if str(x).strip()][:20]
            out.append(row)
        return out, None
    return None, "no JSON object with an items list"
