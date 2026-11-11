"""Pair a photo's food list with its ASA24 record lines using grader align_v1.

Each request gives gpt-oss-120b the numbered ASA24 descriptions and lettered food list. Its developer has no
model in the benchmark. The grader sees no photo or food code and returns pairs graded as the same food,
the same kind of food, or a component of a mixed dish. It ignores details a photo cannot show, such as added
cooking fat, fresh or frozen origin, milk fat level, brand and "NS"/"NFS" wording. Cooking method is scored
separately and does not affect a "same" grade unless it makes a different food.

Saved grades are keyed by content in $SNAPME_ROOT/data/align/align_v1/. Each batch has its own file so grading
jobs can write concurrently. The key includes the grader version, model, record lines and list; a previously
graded list is reused. score/same_food.py calculates the measures from these pairs.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pathlib
import re

import httpx
import pandas as pd

from snapme_eval.paths import data_dir, gt_dir

VERSION = "align_v1"
# Requested reasoning effort; non-default settings use separate cache keys and folders.
EFFORT = os.environ.get("SNAPME_ALIGN_EFFORT", "medium")
SEED = int(os.environ.get("SNAPME_ALIGN_SEED", "20261110"))
TAG = VERSION + ("" if EFFORT == "medium" else f"_{EFFORT}") + ("" if SEED == 20261110 else f"_s{SEED}")
GRADES = ("same", "type", "part")
PROMPT = """A participant photographed a meal and separately recorded what they ate in a dietary recall (ASA24). \
A model then wrote a food list from the photo alone. Match the model's list against the record.

Record lines (numbered):
{lines}

Food list written from the photo (lettered):
{items}

Give a pair for every list item and record line that refer to the same thing, with one grade:
- "same": the same food. Two names for one food are the same (chicken leg and drumstick, chickpeas and garbanzo \
beans, zucchini and green summer squash). Ignore details a photo cannot show: fat or oil added in cooking or not, \
fresh, frozen or canned origin, milk fat level, salted or unsalted, brand, sweetened or unsweetened unless it \
would be visible, and "NS"/"NFS" wording. Ignore the cut or form (sliced, mashed, shredded, baby, whole). Also \
ignore raw versus cooked and the cooking method, unless they make a different food (French fries, potato chips \
and a baked potato are different foods). When the record line is general and the list item is a food it covers \
(beer recorded, sour ale listed; "grapes, NS as to type" recorded, red grapes listed), grade "same".
- "type": a different food of the same kind, such as two kinds of cheese, two breads, two leafy greens, chicken \
and turkey, or two kinds of soft drink. Also grade "type" when the list item is a general name and the record \
line is a specific food within it (cheese listed, provolone recorded; fish listed, tilapia recorded; tea listed, \
green tea recorded).
- "part": the list item is one component of a recorded mixed dish (rice listed, "chicken and rice" recorded), or \
the record line is one component of a mixed dish in the list ("ham sandwich" listed, bread and ham recorded). \
Give a "part" pair for each component, including the oils, sauces and seasonings recorded for that dish.

Leave out pairs of different foods (water and coffee, carrots and sweet potato). Each list item and each record \
line can be in at most one "same" or "type" pair; "part" pairs may share items or lines. A beverage and its \
additions (coffee and creamer) are separate foods.

Reply with JSON only, in this form: {{"pairs": [{{"item": "A", "line": 1, "grade": "same"}}]}}"""


def letters(n: int) -> list[str]:
    """Label list items A, B, ..., Z, AA, AB, ... ."""
    out = []
    for i in range(n):
        s, k = "", i
        while True:
            s = chr(65 + k % 26) + s
            k = k // 26 - 1
            if k < 0:
                break
        out.append(s)
    return out


def item_text(it: dict) -> str:
    name = str(it.get("name", "")).strip()
    ing = [str(x) for x in it.get("ingredients") or [] if str(x).strip()]
    return f"{name} (ingredients: {', '.join(ing)})" if ing else name


def prompt_for(lines: list[str], items: list[dict]) -> str:
    return PROMPT.format(lines="\n".join(f"{i + 1}. {d}" for i, d in enumerate(lines)),
                         items="\n".join(f"{a}. {item_text(it)}" for a, it in zip(letters(len(items)), items,
                                                                                     strict=True)))


def key_for(model: str, lines: list[str], items: list[dict]) -> str:
    return hashlib.sha256(f"{TAG}\n{model}\n{prompt_for(lines, items)}".encode()).hexdigest()[:24]


def parse(txt: str, n_items: int, n_lines: int) -> list[tuple[int, int, str]] | None:
    """Read (item index, line index, grade) pairs; return None if no JSON object can be read."""
    m = re.search(r"\{.*\}", txt or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    idx = {a: i for i, a in enumerate(letters(n_items))}
    out = []
    for p in obj.get("pairs") or []:
        try:
            i, j, g = idx.get(str(p["item"]).strip().upper()), int(p["line"]) - 1, str(p["grade"]).lower()
        except (KeyError, TypeError, ValueError):
            continue
        if i is not None and 0 <= j < n_lines and g in GRADES:
            out.append((i, j, g))
    return out


def cache_dir() -> pathlib.Path:
    return data_dir() / "align" / TAG


def load_cache() -> dict[str, list]:
    out: dict[str, list] = {}
    for p in sorted(cache_dir().glob("*.parquet")):
        d = pd.read_parquet(p, columns=["key", "pairs"])
        out.update({k: json.loads(v) for k, v in zip(d["key"], d["pairs"], strict=True) if isinstance(v, str)})
    return out


def record_lines(gt_version: str = "snapme_v1") -> dict[str, list[str]]:
    items = pd.read_parquet(gt_dir(gt_version) / "items.parquet").sort_values(["image_id", "line"])
    return {i: g["description"].tolist() for i, g in items.groupby("image_id")}


def collect(runs: list[pathlib.Path], model: str, gt_version: str = "snapme_v1") -> list[dict]:
    """Collect uncached requests from the given runs, removing duplicates."""
    from snapme_eval.score.score_run import load_predictions
    lines = record_lines(gt_version)
    have = set(load_cache())
    todo: dict[str, dict] = {}
    for run in runs:
        for image_id, p in load_predictions(run).items():
            its = p.get("items")
            if not its or image_id not in lines:
                continue
            k = key_for(model, lines[image_id], its)
            if k not in have and k not in todo:
                todo[k] = {"key": k, "image_id": image_id, "prompt": prompt_for(lines[image_id], its),
                           "n_items": len(its), "n_lines": len(lines[image_id])}
    return list(todo.values())


async def grade(todo: list[dict], urls: list[str], model: str, concurrency: int = 96,
                effort: str = EFFORT) -> list[dict]:
    """Distribute requests across grader endpoints in turn; save raw replies and parsed pairs."""
    sem = asyncio.Semaphore(concurrency * len(urls))
    clients = [httpx.AsyncClient(base_url=u, timeout=900, trust_env=False) for u in urls]

    async def one(i: int, t: dict) -> dict:
        body = {"model": model, "temperature": 0.0, "seed": SEED,
                "max_tokens": 6000 if effort == "medium" else 12000,
                "reasoning_effort": effort, "messages": [{"role": "user", "content": t["prompt"]}]}
        txt = ""
        async with sem:
            for attempt in range(4):
                try:
                    r = await clients[i % len(clients)].post("/chat/completions", json=body)
                    r.raise_for_status()
                    txt = r.json()["choices"][0]["message"]["content"] or ""
                    break
                except (httpx.HTTPError, KeyError, IndexError):
                    await asyncio.sleep(2 ** attempt)
        pairs = parse(txt, t["n_items"], t["n_lines"])
        return {"key": t["key"], "image_id": t["image_id"], "model": model, "grader": TAG,
                "pairs": json.dumps(pairs) if pairs is not None else None, "raw": txt[:2000]}

    try:
        return await asyncio.gather(*(one(i, t) for i, t in enumerate(todo)))
    finally:
        for c in clients:
            await c.aclose()


def in_shard(key: str, shard: tuple[int, int]) -> bool:
    return int(key, 16) % shard[1] == shard[0]


def update(runs: list[pathlib.Path], urls: list[str], model: str = "gpt-oss-120b", batch: int = 1000,
           limit: int | None = None, shard: tuple[int, int] | None = None) -> dict:
    """Grade uncached lists and save each batch so completed work survives a stopped job.
    shard (i, n) selects share i from n non-overlapping shares for concurrent jobs."""
    todo = collect(runs, model)
    if shard:
        todo = [t for t in todo if in_shard(t["key"], shard)]
    if limit:   # Select a fixed random pilot sample across the supplied runs.
        import random
        random.Random(20261110).shuffle(todo)
        todo = todo[:limit]
    d = cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    job = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    failed = 0
    for s in range(0, len(todo), batch):
        rows = pd.DataFrame(asyncio.run(grade(todo[s:s + batch], urls, model)))
        failed += int(rows["pairs"].isna().sum())
        # Leave failed replies uncached so a later run retries them.
        new = rows[rows["pairs"].notna()]
        if len(new):   # Rename the completed file so concurrent readers never see a partial write.
            part = d / f"part_{job}_{s // batch:04d}.parquet"
            new.to_parquet(part.with_suffix(".tmp"), index=False)
            part.with_suffix(".tmp").rename(part)
        print(f"graded {min(s + batch, len(todo))}/{len(todo)}, unreadable so far {failed}", flush=True)
    return {"requests": len(todo), "unreadable": failed}
