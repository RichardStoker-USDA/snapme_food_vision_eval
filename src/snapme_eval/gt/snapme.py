"""Build the SNAPMe ground truth from the public Ag Data Commons release.

Source: SNAPMe DB v1.0 (December 9, 2022), doi:10.15482/USDA.ADC/1528346, CC BY-SA 4.0.
ASA24-2020 food record lines with photo filenames are linked to those photos. One evaluation
unit is a "before" photo of a non-packaged meal or a "front" photo of a packaged food.
Companion photos (after eating, ingredient label) are recorded for pipelines that use them.

Outputs in <out>/:
  images.parquet   one row per evaluable photo with summed grams
  items.parquet    one row per ASA24 food line, with FNDDS category fields
  unlinked.parquet ASA24 lines with no photo (excluded, kept for day-level analyses)
  build.json       source checksums, counts, exclusions and rules
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time

import pandas as pd

RELEASE = {
    "name": "SNAPMe DB v1.0",
    "file": "snapme_db_09Dec2022.tar.gz",
    "doi": "10.15482/USDA.ADC/1528346",
    "figshare_file_id": 44532971,
    "md5": "95383fd42b78eb45adbad03c7673e8f7",
    "license": "CC BY-SA 4.0",
}
# ASA24 eating occasion codes (Occ_Name). Checked against Table 2 of Larke et al. 2023.
OCCASIONS = {1: "breakfast", 2: "brunch", 3: "lunch", 4: "dinner", 5: "supper", 6: "snack",
             7: "drink only", 8: "supplement only"}
# The project's working name when the split was fixed; the prompts were developed on the photos it selects, so the
# string must stay as it is.
DEV_FRACTION_SALT = "fpbench-snapme-v1"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def split_for(subject: str) -> str:
    """Participant-level split: participants are assigned to dev or test by a fixed hash.
    The dev rule selects approximately 10% of participants."""
    d = hashlib.sha256(f"{DEV_FRACTION_SALT}:{subject}".encode()).digest()
    return "dev" if d[0] % 10 == 0 else "test"


def assign_categories(codes: pd.DataFrame, fndds_foods: pd.DataFrame) -> pd.DataFrame:
    """WWEIA category and FNDDS groups for every ASA24 code, with the method used.

    - code in FNDDS 2017-2018: taken as is ("fndds").
    - 8-digit code absent from FNDDS 2017-2018 (ASA24-specific variants such as
      "made with oil"): most common category among FNDDS codes sharing the longest code
      prefix, needing 1 code at 6+ digits or 3 codes at 4-5 digits ("prefixN").
    - SR Legacy ingredient number from the ASA24 recipe builder (fewer than 8 digits,
      e.g. salt, flour, baking soda): no WWEIA category; the FNDDS major group comes from
      the SR food group ("sr_group"). These lines remain in the reference, with no exact-code
      or WWEIA-category match available.
    """
    f = fndds_foods.set_index("food_code")
    fstr = f.index.astype(str)
    out = []
    for code, _desc in codes[["food_code", "description"]].drop_duplicates("food_code").itertuples(index=False):
        row: dict = {"food_code": code}
        if code in f.index:
            row.update(category_source="fndds", wweia_code=f.at[code, "wweia_code"],
                       wweia_desc=f.at[code, "wweia_desc"], group1=code // 10_000_000,
                       group2=code // 1_000_000)
        elif code >= 10_000_000:
            s = str(code)
            row["category_source"] = "none"
            for k in range(7, 3, -1):
                hits = f[fstr.str.startswith(s[:k])]
                if len(hits) >= (1 if k >= 6 else 3):
                    w = hits["wweia_code"].mode().iloc[0]
                    row.update(category_source=f"prefix{k}", wweia_code=w,
                               wweia_desc=hits.loc[hits["wweia_code"] == w, "wweia_desc"].iloc[0],
                               group1=code // 10_000_000, group2=code // 1_000_000)
                    break
        else:
            sr_group = code // 1000
            name, g1 = SR_GROUPS.get(sr_group, ("Other", None))
            row.update(category_source="sr_group", wweia_code=None,
                       wweia_desc=f"Recipe ingredient: {name}", group1=g1, group2=None)
        out.append(row)
    return pd.DataFrame(out)


# SR Legacy food groups (first digits of the NDB number) -> (name, FNDDS major group or None)
SR_GROUPS = {
    1: ("dairy and egg products", 1), 2: ("spices and herbs", None), 4: ("fats and oils", 8),
    6: ("soups, sauces and gravies", None), 9: ("fruits", 6), 11: ("vegetables", 7),
    12: ("nuts and seeds", 4), 16: ("legumes", 4), 18: ("baked products and leavening", 5),
    19: ("sweets", 9), 20: ("cereal grains and pasta", 5),
}


def build(release_dir: pathlib.Path, fndds_foods: pd.DataFrame, out: pathlib.Path,
          check_files: bool = True) -> dict:
    cs = release_dir / "snapme_cs_db"
    link_path = cs / "master_SNAPME_linkfile.csv"
    raw = pd.read_csv(link_path, na_values=["NA", "NA ", ""], keep_default_na=True)
    raw["filename"] = raw["filename"].astype("string").str.strip()
    raw.loc[raw["filename"].isin(["", "NA", "nan"]), "filename"] = pd.NA
    raw = raw.assign(line=range(len(raw)))

    unlinked = raw[raw["filename"].isna()].copy()
    rows = raw[raw["filename"].notna()].copy()

    rows["image_id"] = rows["filename"].str.replace(r"\.(jpeg|jpg)$", "", regex=True)
    rows["food_code"] = rows["FoodCode"].astype("int64")
    cats = assign_categories(rows.rename(columns={"Food_Description": "description"}), fndds_foods)
    rows = rows.merge(cats, on="food_code", how="left")
    items = pd.DataFrame({
        "image_id": rows["image_id"],
        "subject_id": rows["subject_id"],
        "study_day": rows["snapme_study_day"].astype("int64"),
        "line": rows["line"],
        "food_num": rows["FoodNum1"].astype("int64"),
        "dish_id": rows["image_id"] + "#" + rows["FoodNum1"].astype("int64").astype(str),
        "food_type": rows["FoodType"].map({1: "primary", 2: "addition"}),
        "food_code": rows["food_code"],
        "code_kind": (rows["food_code"] >= 10_000_000).map({True: "fndds", False: "sr_ingredient"}),
        "description": rows["Food_Description"],
        "grams": rows["FoodAmt"].astype(float),
        "wweia_code": rows["wweia_code"],
        "wweia_desc": rows["wweia_desc"],
        "group1": rows["group1"],
        "group2": rows["group2"],
        "category_source": rows["category_source"],
    })

    # One row per photo. Occasion and packaging come from the photo's food lines.
    g = rows.groupby("image_id", sort=True)
    img = pd.DataFrame({
        "image_id": g.size().index,
        "filename": g["filename"].first().values,
        "subject_id": g["subject_id"].first().values,
        "study_day": g["snapme_study_day"].first().astype("int64").values,
        "packaged": g["packaged_food"].first().astype("Int64").values,
        "occasion": g["Occ_Name"].agg(lambda s: OCCASIONS.get(int(s.mode().iloc[0]), "unknown")
                                      if s.notna().any() else "unknown").values,
        "n_items": g.size().values,
        "n_codes": g["food_code"].nunique().values,
        "n_dishes": g["FoodNum1"].nunique().values,
    })
    sums = items.groupby("image_id")[["grams"]].sum()
    img = img.merge(sums.rename(columns={"grams": "grams_total"}), left_on="image_id", right_index=True)
    img["photo_type"] = img["packaged"].map({0: "before", 1: "front"})
    img["path"] = img.apply(lambda r: f"snapme_cs_db/{'before' if r.packaged == 0 else 'front'}_photos/"
                                      f"{r.filename}", axis=1)
    img["companion_path"] = img.apply(
        lambda r: f"snapme_cs_db/after_photos/{r.image_id}_after.jpeg" if r.packaged == 0
        else f"snapme_cs_db/ingredients_photos/{r.image_id}_ingredients.jpeg", axis=1)
    img["split"] = img["subject_id"].map(split_for)
    img["food_code_bin"] = pd.cut(img["n_codes"], [0, 1, 3, 1000], labels=["1", "2-3", "4+"]).astype(str)

    excluded = []
    if check_files:
        img["image_exists"] = img["path"].map(lambda p: (release_dir / p).exists())
        img["companion_exists"] = img["companion_path"].map(lambda p: (release_dir / p).exists())
        for r in img[~img["image_exists"]].itertuples():
            excluded.append({"image_id": r.image_id, "reason": "photo file missing in release"})
        img = img[img["image_exists"]].drop(columns="image_exists")
        items = items[items["image_id"].isin(img["image_id"])]

    out.mkdir(parents=True, exist_ok=True)
    img.to_parquet(out / "images.parquet", index=False)
    items.to_parquet(out / "items.parquet", index=False)
    unl = pd.DataFrame({"subject_id": unlinked["subject_id"], "study_day": unlinked["snapme_study_day"],
                        "food_code": unlinked["FoodCode"], "description": unlinked["Food_Description"],
                        "grams": unlinked["FoodAmt"]})
    unl.to_parquet(out / "unlinked.parquet", index=False)

    info = {
        "benchmark": "snapme",
        "gt_version": out.name,
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {**RELEASE, "linkfile_sha256": sha256(link_path)},
        "fndds": "FNDDS 2017-2018 (used by ASA24-2020)",
        "counts": {
            "linkfile_rows": int(len(raw)),
            "linked_rows": int(len(rows)),
            "unlinked_rows": int(len(unlinked)),
            "images": int(len(img)),
            "images_by_type": img["photo_type"].value_counts().to_dict(),
            "images_by_split": img["split"].value_counts().to_dict(),
            "participants": int(img["subject_id"].nunique()),
            "participant_days": int(img[["subject_id", "study_day"]].drop_duplicates().shape[0]),
            "dishes": int(items["dish_id"].nunique()),
            "unique_food_codes": int(items["food_code"].nunique()),
            "companions_present": int(img.get("companion_exists", pd.Series(dtype=bool)).sum()),
            "category_source": items["category_source"].str.replace(r"\d", "", regex=True)
                                .value_counts().to_dict(),
            "items_without_category": int(items["wweia_code"].isna().sum()),
        },
        "excluded": excluded,
        "rules": [
            "Evaluation unit: one 'before' photo (non-packaged) or 'front' photo (packaged).",
            "Ground truth per photo: every ASA24 line linked to that photo, with its FNDDS code and grams eaten.",
            "A dish is the set of ASA24 lines sharing a FoodNum in one photo (a sandwich, a recipe); "
            "food identification is scored per dish.",
            "Category for codes outside FNDDS 2017-2018: 8-digit ASA24 variants take the category of the "
            "nearest FNDDS code by prefix; SR Legacy recipe ingredients get no WWEIA category or FNDDS "
            "subgroup, only their SR Legacy food group.",
            "ASA24 lines without a photo (NAlist) are kept in unlinked.parquet and excluded from photo scores.",
            "ASA24 records what was eaten; a 'before' photo shows what was served. Leftovers in the "
            "'after' photo explain part of any overestimate.",
            "Split: participants hashed into dev (about 10%) and test; prompts are tuned on dev only.",
        ],
    }
    (out / "build.json").write_text(json.dumps(info, indent=1, default=str))
    return info
