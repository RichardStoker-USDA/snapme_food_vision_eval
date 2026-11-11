"""Check foods from an earlier run against the photo and FNDDS choices.

SNAPME_CHECK_SOURCE identifies the input run; the default is combine_v2_top3/gemma-4-31b-it.
For each item, the name-search cache supplies up to 15 FNDDS candidates. The decision model returns
probabilities for presence and food identity, then for portion options near the earlier gram estimate.
A presence probability below 0.5 excludes an item from the final list. A "none" food choice retains the
nearest candidate. Grams are the probability-weighted mean of the available household-portion options,
or the earlier estimate when there are fewer than two options.

Original names, grams, omitted items and answer probabilities are retained for analysis.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pathlib
from functools import cache

import pandas as pd

from snapme_eval import decision
from snapme_eval.client import Chat, image_data_url
from snapme_eval.mapping import norm
from snapme_eval.paths import data_dir, runs_dir

NAME, VERSION = "check", "v2"
K = 15                     # up to 15 foods plus "none", within JEV's 16-option limit
PRESENT_MIN = 0.5
COUNTS = [0.25, 0.33, 0.5, 0.67, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40]
STATE = "A photo of a meal or snack from a dietary study. A food-photo reader listed this item: {item}."


def source() -> str:
    return os.environ.get("SNAPME_CHECK_SOURCE") or "combine_v2_top3/gemma-4-31b-it"   # unset or empty


def prompt_hash() -> str:
    t = "|".join([pathlib.Path(__file__).read_text(), source()])
    return hashlib.sha256(t.encode()).hexdigest()[:16]


@cache
def reader_items() -> dict:
    p = runs_dir() / os.environ.get("SNAPME_GT", "snapme_v1") / source() / "predictions.jsonl"
    return {(r := json.loads(line))["image_id"]: r.get("items") for line in p.open()}


@cache
def fndds() -> tuple[dict, pd.DataFrame, dict]:
    d = data_dir()
    cache_ = pd.read_parquet(d / "mapping" / "name_to_fndds_v1.parquet")
    cands = {r.name_norm: [int(c) for c in (json.loads(r.candidates) if isinstance(r.candidates, str)
                                            else r.candidates)][:K] for r in cache_.itertuples()}
    foods = pd.read_parquet(d / "fndds" / "fndds_2017_2018_foods.parquet").set_index("food_code")
    por = pd.read_parquet(d / "fndds" / "fndds_2017_2018_portions.parquet")
    por = por[~por["portion"].str.startswith(("Guideline amount", "Quantity not specified"))]
    portions = {c: list(zip(g["portion"], g["grams"], strict=True)) for c, g in por.groupby("food_code")}
    return cands, foods, portions


def portion_options(code: int, grams_reader: float | None) -> dict[str, float]:
    """Return FNDDS portion multiples near the earlier gram estimate as {label: grams}."""
    units = fndds()[2].get(code, [])[:5]
    out: dict[str, float] = {}
    for unit, g in units:
        if not g or g <= 0:
            continue
        target = (grams_reader or g) / g
        i = min(range(len(COUNTS)), key=lambda j: abs(COUNTS[j] - target))
        for c in COUNTS[max(0, i - 1):i + 2]:
            label = f"{c:g} x {unit} ({c * g:.0f} g)"
            out.setdefault(label, c * g)
    return dict(list(out.items())[:16])


@cache
def decider(model: str, base_url: str):
    return decision.make(model, base_url)


async def one_item(it: dict, url: str, dec) -> dict:
    cands, foods, _ = fndds()
    name, g_reader = str(it.get("name", "")), it.get("grams")
    g_reader = float(g_reader) if isinstance(g_reader, (int, float)) and g_reader > 0 else None
    rec = {"name_reader": name, "grams_reader": g_reader, "preparation": it.get("preparation"),
           "ingredients": it.get("ingredients") or []}
    codes = [c for c in cands.get(norm(name), []) if c in foods.index]
    item = name + (f", about {g_reader:.0f} g" if g_reader else "")
    state = STATE.format(item=item)
    questions: dict = {"present": {"type": "noul",
                                   "instructions": f"Is this scenario one where: the photo shows {name}?"}}
    if codes:
        crit = {str(c): foods.at[c, "description"] for c in codes}
        crit["none"] = "None of these foods is the item"
        questions["food"] = {"type": "choice", "criteria": crit,
                             "instructions": f"Which FNDDS food is the item '{name}' in the photo?"}
    ans = await dec.ask(state, url, questions)
    rec["p_present"] = ans["present"]["true"]
    if not codes:
        return {**rec, "name": name, "grams": g_reader, "fndds_code": None, "code_source": "no candidates"}
    food = ans["food"]
    best = max(food, key=food.get)
    code, src = (codes[0], "embedding") if best == "none" else (int(best), "decision")
    rec.update(fndds_code=code, code_source=src, p_food=food[best], p_none=food.get("none"),
               candidates=codes, name=foods.at[code, "description"])
    opts = portion_options(code, g_reader)
    grams = g_reader
    if len(opts) >= 2:
        pa = await dec.ask(STATE.format(item=f"{foods.at[code, 'description']}"), url, {"portion": {
            "type": "choice", "criteria": {k: k for k in opts},
            "instructions": "How much of this food is in the photo?"}})
        pp = pa["portion"]
        grams = sum(pp[k] * opts[k] for k in opts)
        rec.update(portion_probs={k: round(v, 4) for k, v in pp.items()}, grams_portion=grams)
    rec["grams"] = grams
    return rec


async def predict(row: dict, chat: Chat, release_dir: pathlib.Path) -> dict:
    items = reader_items().get(row["image_id"])
    if items is None:
        return {"items": None, "parse_error": "reader failed on this photo", "source": source()}
    url, meta = image_data_url(release_dir / row["path"])
    dec = decider(chat.model, str(chat.http.base_url))
    recs = await asyncio.gather(*(one_item(it, url, dec) for it in items))
    kept = [r for r in recs if r["p_present"] >= PRESENT_MIN]
    return {"image": meta, "source": source(), "items": kept,
            "dropped": [r for r in recs if r["p_present"] < PRESENT_MIN], "parse_error": None}


ABLATIONS = {
    "keep_all": "presence check off: dropped items kept",
    "reader_grams": "reader's grams kept; the decision model still picks the FNDDS food",
}


def ablate(run: pathlib.Path, variant: str, out: pathlib.Path) -> None:
    """Write a variant that omits one step's effect, using existing predictions without new model requests."""
    if variant not in ABLATIONS:
        raise SystemExit(f"variant must be one of {', '.join(ABLATIONS)}")
    out.mkdir(parents=True, exist_ok=True)
    with (run / "predictions.jsonl").open() as fin, (out / "predictions.jsonl").open("w") as fout:
        for line in fin:
            r = json.loads(line)
            if r.get("items") is not None:
                items = r["items"] + (r.get("dropped") or []) if variant == "keep_all" else r["items"]
                if variant == "reader_grams":
                    for it in items:
                        it["grams"] = it.get("grams_reader")
                r["items"] = items
            fout.write(json.dumps(r) + "\n")
    for f in run.glob("run_*.json"):
        meta = json.loads(f.read_text())
        meta.update(ablation=variant, ablation_note=ABLATIONS[variant], ablation_of=str(run.name))
        (out / f.name).write_text(json.dumps(meta, indent=1))
