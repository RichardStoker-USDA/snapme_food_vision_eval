"""Extract preparation words and main ingredients from FNDDS descriptions and recipes.

Preparation terms come from a fixed list. Recipe ingredients exclude water and must each contribute at
least 5% of the remaining weight. Up to eight distinct descriptions are retained, largest first.
Descriptions use their first two parts after removal of uppercase brand components. A mixed dish requires
at least two different first description parts.
"""
from __future__ import annotations

import re

import pandas as pd

PREP_TERMS = ("raw", "baked", "fried", "grilled", "boiled", "roasted", "steamed", "broiled", "sauteed", "stewed",
              "scrambled", "poached", "toasted", "braised", "smoked", "canned", "dried", "mashed", "microwaved",
              "blanched", "breaded", "coated", "frozen", "cooked")
MIN_SHARE, MAX_INGREDIENTS = 0.05, 8


def words(text) -> list[str]:
    return re.findall(r"[a-z]+", str(text or "").lower())


def prep_terms(*texts) -> set[str]:
    out: set[str] = set()
    for t in texts:
        out |= set(words(t))
    return out & set(PREP_TERMS)


def preparation(description: str) -> str:
    """Preparation words of a description in their order, or "not applicable"."""
    found = [w for w in dict.fromkeys(words(description)) if w in PREP_TERMS]
    return ", ".join(found) or "not applicable"


def short_ingredient(desc: str) -> str:
    """First two parts of an SR Legacy description, lower case, brand parts dropped:
    "Cheese, cheddar, sharp" -> "cheese, cheddar"; "Cereals ready-to-eat, QUAKER, ..." -> "cereals ready-to-eat"."""
    parts = [p.strip() for p in str(desc).split(",") if p.strip()]
    parts = [p for i, p in enumerate(parts) if i == 0 or not (p.isupper() and len(p) > 2)]
    return ", ".join(parts[:2]).lower()


def recipe_index(ingredients: pd.DataFrame) -> dict[int, list[str]]:
    """Food code -> main ingredients of its standard recipe. Codes with fewer than two such ingredients get none,
    so single foods ("Apple, raw") carry an empty list as the prompt asks."""
    out: dict[int, list[str]] = {}
    for code, g in ingredients.groupby("food_code"):
        g = g[["water" not in words(short_ingredient(x)) for x in g["ingredient"]]]   # "Beverages, water, tap"
        total = g["grams"].sum()
        if total <= 0:
            continue
        main = g[g["grams"] / total >= MIN_SHARE].sort_values("grams", ascending=False)
        names = list(dict.fromkeys(short_ingredient(x) for x in main["ingredient"]))[:MAX_INGREDIENTS]
        if len({n.split(",")[0] for n in names}) >= 2:
            out[int(code)] = names
    return out
