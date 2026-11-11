"""USDA Food and Nutrient Database for Dietary Studies (FNDDS) 2017-2018.

ASA24-2020, which produced the SNAPMe food records, codes foods with FNDDS 2017-2018,
so predictions are mapped to and scored against this release. The tables kept are the food list (code, main and
additional descriptions, WWEIA category), household portions with their weights, and standard recipes. The food
list is read from the "FNDDS Nutrient Values" workbook, which is where FNDDS publishes it with WWEIA categories;
its nutrient columns are not read.
Source: USDA ARS Food Surveys Research Group, "FNDDS At A Glance" workbooks.
"""
from __future__ import annotations

import pathlib

import pandas as pd

FILES = {
    "food_list": "2017-2018_FNDDS_At_A_Glance_-_FNDDS_Nutrient_Values.xlsx",
    "foods": "2017-2018_FNDDS_At_A_Glance_-_Foods_and_Beverages.xlsx",
    "portions": "2017-2018_FNDDS_At_A_Glance_-_Portions_and_Weights.xlsx",
    "ingredients": "2017-2018_FNDDS_At_A_Glance_-_FNDDS_Ingredients.xlsx",
}


def _read(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0, header=1)
    df.columns = [str(c).strip() if "\n" not in str(c) else str(c) for c in df.columns]
    return df


def load(src: pathlib.Path) -> dict[str, pd.DataFrame]:
    """Read the workbooks into tidy tables keyed by 8-digit food code."""
    cols = ["Food code", "Main food description", "WWEIA Category number", "WWEIA Category description"]
    foods = pd.read_excel(src / FILES["food_list"], sheet_name=0, header=1, usecols=cols)[cols]
    foods.columns = ["food_code", "description", "wweia_code", "wweia_desc"]

    extra = _read(src / FILES["foods"])
    extra = extra.rename(columns={"Food code": "food_code", "Additional food description": "additional"})
    add = (extra.dropna(subset=["additional"]).groupby("food_code")["additional"]
           .apply(lambda s: "; ".join(dict.fromkeys(str(x).strip() for x in s))).reset_index())
    foods = foods.merge(add, on="food_code", how="left")

    por = _read(src / FILES["portions"])
    por = por.rename(columns={"Food code": "food_code", "Portion description": "portion",
                              "Portion weight (g)": "grams", "Seq num": "seq"})
    por = por[por["grams"] > 0][["food_code", "seq", "portion", "grams"]]

    # Standard recipe of each food code: ingredients (SR Legacy or other FNDDS codes) and their weights.
    ing = _read(src / FILES["ingredients"])
    ing = ing.rename(columns={"Food code": "food_code", "Seq num": "seq", "Ingredient code": "ingredient_code",
                              "Ingredient description": "ingredient", "Ingredient weight (g)": "grams"})
    ing = ing[["food_code", "seq", "ingredient_code", "ingredient", "grams"]]

    for df in (foods, por, ing):
        df["food_code"] = df["food_code"].astype("int64")
    foods["group1"] = foods["food_code"] // 10_000_000      # major group (1 milk ... 9 sugars/beverages)
    foods["group2"] = foods["food_code"] // 1_000_000       # subgroup
    return {"foods": foods, "portions": por, "ingredients": ing}


def write_parquet(tables: dict[str, pd.DataFrame], out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(out / f"fndds_2017_2018_{name}.parquet", index=False)


def read_parquet(dirpath: pathlib.Path) -> dict[str, pd.DataFrame]:
    names = ("foods", "portions", "ingredients")
    return {n: pd.read_parquet(dirpath / f"fndds_2017_2018_{n}.parquet") for n in names
            if (dirpath / f"fndds_2017_2018_{n}.parquet").exists()}
