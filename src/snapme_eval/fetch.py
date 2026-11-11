"""Download the public inputs into $SNAPME_ROOT/data and check them against pinned hashes.

    snapme-eval fetch

- SNAPMe DB v1.0 from Ag Data Commons (figshare file 44532971), md5 from the release page, unpacked to
  data/snapme/.
- WordNet for the token F1 lemmatizer (NLTK), into $NLTK_DATA or $SNAPME_ROOT/env/nltk_data.
- FNDDS 2017-2018 "At A Glance" workbooks (food list, nutrient workbook, portions, standard recipes),
  from the USDA ARS Food Surveys Research Group; SHA-256 recorded on 2026-11-08.

A file that is already present with the right hash is not downloaded again. A hash mismatch stops the command:
the downloaded bytes differ from the recorded input.
"""
from __future__ import annotations

import hashlib
import pathlib
import tarfile
import urllib.parse
import urllib.request

from snapme_eval.gt.snapme import RELEASE
from snapme_eval.paths import data_dir, root

FSRG = "https://www.ars.usda.gov/ARSUserFiles/80400530/apps/"
FNDDS = {
    "2017-2018_FNDDS_At_A_Glance_-_Foods_and_Beverages.xlsx":
        ("2017-2018 FNDDS At A Glance - Foods and Beverages.xlsx",
         "c1c5c25ad0f58c1823990e09ca09d2c690aa70beb8f6d51d7e2fac8a010c3250"),
    "2017-2018_FNDDS_At_A_Glance_-_FNDDS_Nutrient_Values.xlsx":
        ("2017-2018 FNDDS At A Glance - FNDDS Nutrient Values.xlsx",
         "6b3eb74892069d9d78b83532b407a107536d66f6da8af0867809bb4f997ff166"),
    "2017-2018_FNDDS_At_A_Glance_-_Portions_and_Weights.xlsx":
        ("2017-2018 FNDDS At A Glance - Portions and Weights.xlsx",
         "03b809944cdcddcb753d1f774248b54c903e72ff94a567aef8a6f442d8c066fd"),
    "2017-2018_FNDDS_At_A_Glance_-_FNDDS_Ingredients.xlsx":
        ("2017-2018 FNDDS At A Glance - FNDDS Ingredients.xlsx",
         "e1a55b8a2d431f4bb52ce6e41bb9730337fa55d0cf25773fe8a6fb35e6d2b91a"),
}

# Ground truth and baseline predictions used in the SNAPMe paper (Larke et al. 2023, section 3.4), from the
# authors' repository github.com/JulesLarke-USDA/SNAPMe (CC BY-SA 4.0) at a fixed commit: per-photo token
# sets of ASA24 descriptions (manually converted to ingredients for mixed dishes) and predictions from
# Inverse Cooking and Im2Recipe, with their F1 scores.
LARKE_2023 = "https://raw.githubusercontent.com/JulesLarke-USDA/SNAPMe/ce09cee4a1e0c5b05f35abe5a8a058e686fefa74/"
LARKE_FILES = {
    "input/snapme_ingredientized_manual_fixes__tokens_cleaned.csv":
        "84a559eb21a847c2917ce25b55abdc326ecad91d547a4d6c9eb28eab4f7946f7",
    "output/snapme_inv_cook_f1_score.csv": "9ee4f748b10712cce6f3e0cfe7646ea05e42af82bafee235a513bbb022860f17",
    "output/snapme_im2r_f1_score.csv": "435da0e47db7b1bbaf8f38d4e81b6cb0d38c37d8bc527142e8925f200d524b1e",
}

def _digest(path: pathlib.Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _get(url: str, dst: pathlib.Path, algo: str, want: str) -> None:
    if dst.exists() and _digest(dst, algo) == want:
        print(f"ok      {dst.name}")
        return
    print(f"fetch   {url}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "snapme-eval"})
    with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:
        while b := r.read(1 << 20):
            f.write(b)
    got = _digest(tmp, algo)
    if got != want:
        raise SystemExit(f"{dst.name}: {algo} {got} does not match the pinned {want}; the source has changed")
    tmp.rename(dst)
    print(f"ok      {dst.name}")


def main() -> None:
    d = data_dir()
    (d / "fndds").mkdir(parents=True, exist_ok=True)
    (d / "snapme").mkdir(parents=True, exist_ok=True)
    for local, (remote, sha) in FNDDS.items():
        _get(FSRG + urllib.parse.quote(remote), d / "fndds" / local, "sha256", sha)
    (d / "larke_2023").mkdir(parents=True, exist_ok=True)
    for path, sha in LARKE_FILES.items():
        _get(LARKE_2023 + path, d / "larke_2023" / pathlib.Path(path).name, "sha256", sha)
    tar = d / "snapme" / RELEASE["file"]
    _get(f"https://ndownloader.figshare.com/files/{RELEASE['figshare_file_id']}", tar, "md5", RELEASE["md5"])
    if not any(p.is_dir() for p in (d / "snapme").iterdir()):
        with tarfile.open(tar) as t:
            t.extractall(d / "snapme", filter="data")
        print(f"unpacked to {d / 'snapme'}")
    import os

    import nltk
    nltk.download("wordnet", download_dir=os.environ.get("NLTK_DATA", str(root() / "env" / "nltk_data")),
                  quiet=True)
    print("ok      wordnet")
    print("next: snapme-eval fndds --src $SNAPME_ROOT/data/fndds && snapme-eval gt --version snapme_v1")
