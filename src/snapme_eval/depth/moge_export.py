"""Export MoGe-2 point maps for every photo in a list. Runs inside the engine container's Python
(torch present), so it imports nothing from snapme_eval.

    python moge_export.py <checkpoint model.pt> <list.tsv: image_id<TAB>path> <out dir>

Each photo becomes <out>/<image_id>.npz: metric point map (meters, OpenCV camera axes) at 512 px on the
long side, valid mask, normalized intrinsics and the original size. Existing files are skipped.
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageOps

try:
    import pillow_heif  # some SNAPMe photos are HEIC files named .jpeg
    pillow_heif.register_heif_opener()
except ImportError:
    pass

from moge.model.v2 import MoGeModel

INFER_SIDE, STORE_SIDE = 1024, 512


def load(path: pathlib.Path, side: int) -> tuple[np.ndarray, tuple[int, int]]:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        orig = im.size
        s = side / max(im.size)
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)), Image.Resampling.LANCZOS)
        return np.asarray(im), orig


def main(ckpt: str, listing: str, out_dir: str) -> None:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    model = MoGeModel.from_pretrained(ckpt).to(dev).eval()
    rows = [line.rstrip("\n").split("\t") for line in open(listing) if line.strip()]
    t0, n = time.time(), 0
    for image_id, path in rows:
        dst = out / f"{image_id}.npz"
        if dst.exists():
            continue
        try:
            rgb, orig = load(pathlib.Path(path), INFER_SIDE)
        except Exception as e:  # noqa: BLE001 - record and continue
            print(f"skip {image_id}: {type(e).__name__}: {e}", flush=True)
            continue
        x = torch.tensor(rgb / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1)
        with torch.inference_mode():
            o = model.infer(x)
        pts, mask = o["points"], o["mask"]
        h, w = pts.shape[:2]
        s = STORE_SIDE / max(h, w)
        size = (max(1, round(h * s)), max(1, round(w * s)))
        pts = torch.nn.functional.interpolate(pts.permute(2, 0, 1)[None], size=size, mode="nearest")[0].permute(1, 2, 0)
        mask = torch.nn.functional.interpolate(mask[None, None].float(), size=size, mode="nearest")[0, 0] > 0.5
        np.savez_compressed(dst, points=pts.cpu().numpy().astype(np.float32), mask=mask.cpu().numpy(),
                            intrinsics=o["intrinsics"].cpu().numpy(), orig_wh=np.array(orig))
        n += 1
        if n % 100 == 0:
            print(f"{n} done, {(time.time() - t0) / n:.2f} s/photo", flush=True)
    print(f"finished {n} photos in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
