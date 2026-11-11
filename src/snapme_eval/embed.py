"""EmbeddingGemma 2 vectors for FNDDS foods and SNAPMe photos, for comparing photos with food descriptions.

Runs inside an engine image with sentence-transformers on PYTHONPATH (hpc/atlas/embed.sbatch). Writes to
$SNAPME_ROOT/data/embeddings/:
  embeddinggemma2_fndds.npz    codes, 768-d unit vectors of "title: none | text: <description>; <additional>"
  embeddinggemma2_photos_<gt>.npz  image ids, unit vectors of each photo alone (no text prefix for images)

    python3 embed.py --model-dir SNAPSHOT --root $SNAPME_ROOT [--gt snapme_v1]
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--gt", default="snapme_v1")
    ap.add_argument("--images", default="data/snapme/snapme_db_09Dec2022")
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    out = root / "data" / "embeddings"
    out.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    m = SentenceTransformer(a.model_dir, model_kwargs={"torch_dtype": dtype}, config_kwargs={"audio_config": None},
                            device="cuda" if torch.cuda.is_available() else "cpu")

    foods = pd.read_parquet(root / "data" / "fndds" / "fndds_2017_2018_foods.parquet")
    texts = [f"title: none | text: {r.description}" + (f"; {r.additional}" if isinstance(r.additional, str) else "")
             for r in foods.itertuples()]
    v = m.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    np.savez(out / "embeddinggemma2_fndds.npz", codes=foods["food_code"].to_numpy(), vectors=v.astype(np.float32))

    from PIL import Image, ImageOps
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass
    imgs = pd.read_parquet(root / "gt" / a.gt / "images.parquet")
    vecs = []
    for i in range(0, len(imgs), 16):
        batch = []
        for p in imgs["path"].iloc[i:i + 16]:
            with Image.open(root / a.images / p) as im:
                batch.append({"image": ImageOps.exif_transpose(im).convert("RGB")})
        vecs.append(m.encode(batch, normalize_embeddings=True, show_progress_bar=False))
    np.savez(out / f"embeddinggemma2_photos_{a.gt}.npz", image_ids=imgs["image_id"].to_numpy(),
             vectors=np.concatenate(vecs).astype(np.float32))
    print("fndds", v.shape, "photos", sum(len(x) for x in vecs))


if __name__ == "__main__":
    main()
