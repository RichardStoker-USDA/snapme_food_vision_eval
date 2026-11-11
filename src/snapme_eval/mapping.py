"""Assign FNDDS 2017-2018 codes to free-text food names.

Qwen3-Embedding-8B represents names and FNDDS descriptions as normalized vectors. The 20 candidates
with greatest cosine similarity are passed to a text model, which selects one or returns no match.
The CLI defaults to Qwen3.8-27B for selection and accepts explicit model IDs.
Neither photos nor recorded reference codes are supplied to this step.

Results are cached by normalized name in $SNAPME_ROOT/data/mapping/name_to_fndds_v1.parquet.
Methods that supply their own FNDDS code bypass the cached name code during scoring.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import pathlib
import re

import httpx
import numpy as np
import pandas as pd

from snapme_eval.paths import data_dir

VERSION = "mapper_v1"
K = 20
TASK = "Given a food or drink name written from a meal photo, retrieve the matching USDA food description"
JUDGE_PROMPT = """A food or drink seen in a meal photo was named: "{name}"

Pick the USDA FNDDS food that best matches it. Prefer the option that matches the food itself; \
preparation details (cooked, raw, with or without fat) matter less than the food.

{options}

Reply with only the number of the best option, or 0 if none of them is the same food."""


def norm(name: str) -> str:
    return " ".join(str(name).lower().split())


def corpus(foods: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    texts = []
    for r in foods.itertuples():
        t = r.description
        if isinstance(r.additional, str) and r.additional:
            t += "; also: " + r.additional[:300]
        texts.append(t)
    return foods["food_code"].to_numpy(), texts


async def embed(http: httpx.AsyncClient, model: str, texts: list[str], batch: int = 128) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch):
        r = await http.post("/embeddings", json={"model": model, "input": texts[i:i + batch]})
        r.raise_for_status()
        out += [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]
    v = np.asarray(out, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


async def build(names: list[str], embed_url: str, embed_model: str, judge_url: str, judge_model: str,
                concurrency: int = 64) -> pd.DataFrame:
    foods = pd.read_parquet(data_dir() / "fndds" / "fndds_2017_2018_foods.parquet")
    codes, texts = corpus(foods)
    desc = dict(zip(foods["food_code"], foods["description"], strict=True))
    cache_dir = data_dir() / "mapping"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256((embed_model + "\n".join(texts)).encode()).hexdigest()[:16]
    emb_path = cache_dir / f"fndds_corpus_{key}.npy"

    async with httpx.AsyncClient(base_url=embed_url, timeout=600, trust_env=False) as eh, \
               httpx.AsyncClient(base_url=judge_url, timeout=600, trust_env=False) as jh:
        if emb_path.exists():
            C = np.load(emb_path)
        else:
            C = await embed(eh, embed_model, texts)
            np.save(emb_path, C)
        Q = await embed(eh, embed_model, [f"Instruct: {TASK}\nQuery: {n}" for n in names])
        top = np.argsort(-(Q @ C.T), axis=1)[:, :K]

        sem = asyncio.Semaphore(concurrency)

        async def judge(i: int, name: str) -> dict:
            cand = [int(codes[j]) for j in top[i]]
            opts = "\n".join(f"{k + 1}. {desc[c]}" for k, c in enumerate(cand))
            body = {"model": judge_model, "temperature": 0.0, "max_tokens": 8, "seed": 20261108,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content": JUDGE_PROMPT.format(name=name, options=opts)}]}
            txt = ""
            async with sem:
                for attempt in range(4):
                    try:
                        r = await jh.post("/chat/completions", json=body)
                        r.raise_for_status()
                        txt = r.json()["choices"][0]["message"]["content"] or ""
                        break
                    except (httpx.HTTPError, KeyError):
                        txt = ""
                        await asyncio.sleep(2 ** attempt)
            m = re.search(r"\d+", txt.rsplit("</think>", 1)[-1])
            pick = int(m.group()) if m else -1
            code = cand[pick - 1] if 1 <= pick <= K else None
            return {"name_norm": norm(name), "name_example": name, "food_code": code,
                    "choice": pick, "embed_top1": cand[0], "candidates": json.dumps(cand),
                    "mapper": VERSION, "judge_raw": txt[:40]}

        rows = await asyncio.gather(*(judge(i, n) for i, n in enumerate(names)))
    return pd.DataFrame(rows)


def collect_names(runs_root: pathlib.Path) -> list[str]:
    seen: dict[str, str] = {}
    for p in runs_root.rglob("predictions.jsonl"):
        for line in p.open():
            try:
                for it in json.loads(line).get("items") or []:
                    for n in [it.get("name"), *(it.get("ingredients") or [])]:
                        if n:
                            seen.setdefault(norm(n), str(n))
            except (json.JSONDecodeError, AttributeError):
                continue
    # Reference names of other datasets scored by code (Nutrition5k ingredients), and the SNAPMe ASA24
    # descriptions themselves, which check the mapper against known codes (score/mapcheck.py). A cached code for a
    # description is the mapper's answer; recorded reference codes are not passed to the mapper.
    from snapme_eval.paths import gt_dir
    for v, col in (("nutrition5k_v1", "name"), ("snapme_v1", "description")):
        f = gt_dir(v) / "items.parquet"
        if f.exists():
            for n in pd.read_parquet(f)[col].unique():
                seen.setdefault(norm(n), str(n))
    return list(seen.values())


def update_cache(runs_root: pathlib.Path, embed_url: str, embed_model: str, judge_url: str,
                 judge_model: str) -> dict:
    path = data_dir() / "mapping" / "name_to_fndds_v1.parquet"
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["name_norm"])
    have = set(old["name_norm"])
    todo = [n for n in collect_names(runs_root) if norm(n) not in have]
    if todo:
        new = asyncio.run(build(todo, embed_url, embed_model, judge_url, judge_model))
        merged = pd.concat([old, new], ignore_index=True) if len(old) else new
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(path, index=False)
    else:
        merged = old
    return {"cached": int(len(merged)), "added": len(todo),
            "no_match": int(merged["food_code"].isna().sum()) if len(merged) else 0}
