"""Estimate food volumes from MoGe-2 point maps and model-drawn rectangles.

A fitted table plane defines the projection. Height is measured above a local base estimated from the
rectangle's border, then integrated over the projected food area. A usable checkerboard rectangle rescales
lengths by sqrt(known area / estimated area) and volumes by the cube of that ratio. Otherwise the MoGe-2
scale is retained. The returned scale_source identifies which scale was used.
"""
from __future__ import annotations

import pathlib

import numpy as np

CARD_CM2 = 3.81 ** 2          # SNAPMe checkerboard card, 1.5 x 1.5 in
MIN_H, MAX_H = 0.003, 0.15    # meters; retain estimated heights within this band
RING = 0.12                   # border band of the box used for the local base, as a fraction of box size


def load(path: pathlib.Path) -> dict | None:
    if not path.exists():
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def plane_ransac(p: np.ndarray, iters: int = 300, thresh: float = 0.004, seed: int = 0, sample: int = 20000):
    """Fit n.x + d = 0 to an (N, 3) cloud; n points toward the camera (d > 0).
    Hypotheses are scored on a random sample of points; the winner is refined on all of its inliers."""
    rng = np.random.default_rng(seed)
    best, best_n = None, -1
    if len(p) < 50:
        return None
    full = p
    if len(p) > sample:
        p = p[rng.choice(len(p), sample, replace=False)]
    for _ in range(iters):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -n @ a
        k = int((np.abs(p @ n + d) < thresh).sum())
        if k > best_n:
            best, best_n = (n, d), k
    if best is None:
        return None
    n, d = best
    inl = full[np.abs(full @ n + d) < thresh]
    c = inl.mean(0)                                   # refine with least squares on the inliers:
    n = np.linalg.eigh(np.cov((inl - c).T))[1][:, 0]  # normal = smallest-eigenvalue direction of the 3x3 covariance
    d = -n @ c
    return (n, d) if d > 0 else (-n, -d)


def footprint(points: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Area (m^2) of each pixel's surface patch projected onto the plane with normal n."""
    du = np.gradient(points, axis=1)
    dv = np.gradient(points, axis=0)
    return np.abs(np.cross(du, dv) @ n)


def box_slices(box, h: int, w: int):
    x0, y0, x1, y1 = (float(v) / 1000 for v in box)
    x0, x1 = sorted((min(max(x0, 0), 1), min(max(x1, 0), 1)))
    y0, y1 = sorted((min(max(y0, 0), 1), min(max(y1, 0), 1)))
    return slice(int(y0 * h), max(int(y1 * h), int(y0 * h) + 1)), slice(int(x0 * w), max(int(x1 * w), int(x0 * w) + 1))


def measure(geo: dict, boxes: list, card_box=None) -> dict:
    pts, mask = geo["points"].astype(np.float64), geo["mask"].astype(bool)
    h, w = mask.shape
    outside = mask.copy()
    for b in boxes:
        ys, xs = box_slices(b, h, w)
        outside[ys, xs] = False
    plane = plane_ransac(pts[outside]) or plane_ransac(pts[mask])
    if plane is None:
        return {"ok": False, "reason": "no table plane"}
    n, d = plane
    height = pts @ n + d
    area = footprint(pts, n)
    items = []
    for b in boxes:
        ys, xs = box_slices(b, h, w)
        hb, ab, mb = height[ys, xs], area[ys, xs], mask[ys, xs]
        ry, rx = max(1, int(hb.shape[0] * RING)), max(1, int(hb.shape[1] * RING))
        ring = np.ones_like(mb)
        ring[ry:-ry, rx:-rx] = False
        base_vals = hb[ring & mb]
        base = float(np.percentile(base_vals, 20)) if base_vals.size else 0.0
        rel = hb - max(base, 0.0)
        food = mb & (rel > MIN_H) & (rel < MAX_H)
        items.append({
            "volume_ml": float((rel[food] * ab[food]).sum() * 1e6),
            "footprint_cm2": float(ab[mb].sum() * 1e4),
            "max_height_cm": float(np.percentile(rel[food], 95) * 100) if food.any() else 0.0,
        })
    out = {"ok": True, "items": items, "camera_to_table_cm": float(d * 100)}
    if card_box is not None:
        ys, xs = box_slices(card_box, h, w)
        card_area = float(area[ys, xs][mask[ys, xs]].sum() * 1e4)
        out["card_area_cm2"] = card_area
        out["card_scale_ratio"] = float(np.sqrt(CARD_CM2 / card_area)) if card_area > 0 else None
    r = out.get("card_scale_ratio")
    out["scale_source"] = "moge"
    if r and 0.05 <= r <= 2.0:
        out["scale_source"] = "card"
        for it in out["items"]:
            it["volume_ml"] *= r ** 3
            it["footprint_cm2"] *= r ** 2
            it["max_height_cm"] *= r
        out["camera_to_table_cm"] *= r
    return out
