"""Locate and paint over the checkerboard scale card for comparisons on SNAPMe meal photos.

Geometry runs supply boxes around the card. checker_score tests the pixels inside each box for a two-by-two
checker pattern. When two models' boxes agree, with intersection over union at least IOU_MIN, also test their
mean box. Use the highest-scoring box and keep the photo if its score reaches CHECKER_MIN. The table includes
a control patch of the same size, placed away from the card and the foods boxed by those models.

Boxes are [x_min, y_min, x_max, y_max] on a 0 to 1000 scale from the left and top edges.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

IOU_MIN = 0.5
CHECKER_MIN = 0.65
MARGIN = 0.15      # Extend the painted area on each side by this share of the card box's size.
RING = 0.25        # Take the median fill color from a surrounding band this wide, as a share of patch size.
TABLE = pathlib.Path(__file__).resolve().parent / "tables" / "marker_v1.csv"


def iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(w, 0) * max(h, 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def tidy(box) -> list[float]:
    x0, y0, x1, y1 = (min(max(float(v), 0.0), 1000.0) for v in box)
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def checker_score(gray: np.ndarray, n: int = 48) -> float:
    """Score a central two-by-two checker pattern across rotations: 1 for a perfect card, 0 for plain surfaces,
    single edges, single corners and unstructured texture. Both quadrants on one diagonal must be brighter
    than both on the other. The score is half the gap between the dimmer bright quadrant and the brighter
    dark quadrant, measured in standard deviations of the patch."""
    im = Image.fromarray((np.clip(gray, 0, 1) * 255).astype(np.uint8)).resize((n, n), Image.Resampling.BILINEAR)
    g = np.asarray(im, float) / 255
    ys, xs = np.mgrid[0:n, 0:n]
    best = 0.0
    for cy in np.linspace(0.38, 0.62, 5) * n:
        for cx in np.linspace(0.38, 0.62, 5) * n:
            dx, dy = xs - cx, ys - cy
            disc = dx * dx + dy * dy <= (0.33 * n) ** 2
            v = g[disc]
            sd = v.std()
            if sd < 0.10:
                continue
            z = (g - v.mean()) / sd
            for th in np.deg2rad(np.arange(0, 90, 10)):
                u, w = dx * np.cos(th) + dy * np.sin(th), -dx * np.sin(th) + dy * np.cos(th)
                q = [z[disc & (u * a > 1) & (w * b > 1)] for a, b in ((1, 1), (-1, -1), (1, -1), (-1, 1))]
                if min(len(x) for x in q) < 8:
                    continue
                m = [float(x.mean()) for x in q]
                best = max(best, (min(m[0], m[1]) - max(m[2], m[3])) / 2, (min(m[2], m[3]) - max(m[0], m[1])) / 2)
    return best


def open_photo(path: pathlib.Path) -> Image.Image:
    import pillow_heif
    pillow_heif.register_heif_opener()
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def pixels(box, w: int, h: int, margin: float = 0.0) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return (max(int((x0 - mx) / 1000 * w), 0), max(int((y0 - my) / 1000 * h), 0),
            min(int(np.ceil((x1 + mx) / 1000 * w)), w), min(int(np.ceil((y1 + my) / 1000 * h)), h))


def hide(im: Image.Image, box, margin: float = MARGIN) -> Image.Image:
    """Copy the photo and fill the box plus margin with the surrounding band's median color."""
    a = np.asarray(im).copy()
    h, w = a.shape[:2]
    x0, y0, x1, y1 = pixels(box, w, h, margin)
    rx, ry = max(int((x1 - x0) * RING), 2), max(int((y1 - y0) * RING), 2)
    X0, Y0, X1, Y1 = max(x0 - rx, 0), max(y0 - ry, 0), min(x1 + rx, w), min(y1 + ry, h)
    band = np.ones((Y1 - Y0, X1 - X0), bool)
    band[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
    ring = a[Y0:Y1, X0:X1][band]
    a[y0:y1, x0:x1] = np.median(ring, axis=0).astype(np.uint8) if len(ring) else 128
    return Image.fromarray(a)


def control_box(card, foods: list, step: float = 25.0) -> tuple[list[float], float]:
    """Place a patch the size of the card plus margin without overlapping it and with minimum overlap of food
    boxes. Break ties by distance to the card's mirror position across the photo's vertical midline."""
    cw, ch = card[2] - card[0], card[3] - card[1]
    w, h = cw * (1 + 2 * MARGIN), ch * (1 + 2 * MARGIN)
    grown = [card[0] - cw * MARGIN, card[1] - ch * MARGIN, card[2] + cw * MARGIN, card[3] + ch * MARGIN]
    mx, my = 1000 - (card[0] + card[2]) / 2, (card[1] + card[3]) / 2
    best = None
    for y0 in np.arange(0, 1000 - h + 1e-6, step):
        for x0 in np.arange(0, 1000 - w + 1e-6, step):
            b = [x0, y0, x0 + w, y0 + h]
            if iou(b, grown) > 0:
                continue
            over = sum(max(min(b[2], f[2]) - max(b[0], f[0]), 0) * max(min(b[3], f[3]) - max(b[1], f[1]), 0)
                       for f in foods) / (w * h)
            key = (round(over, 2), (x0 + w / 2 - mx) ** 2 + (y0 + h / 2 - my) ** 2)
            if best is None or key < best[0]:
                best = (key, b, over)
    if best is None:
        return [0.0, 0.0, w, h], 1.0
    m = MARGIN / (1 + 2 * MARGIN)     # Remove the margin for storage; hide() adds it back as for a card box.
    b = best[1]
    return [b[0] + w * m, b[1] + h * m, b[2] - w * m, b[3] - h * m], float(best[2])


def located(run: pathlib.Path, order: str) -> dict[str, dict]:
    """Read card and food boxes from geometry replies, grouped by photo with x coordinates first."""
    from snapme_eval.systems.geometry import _boxes
    out = {}
    for line in (run / "predictions.jsonl").open():
        r = json.loads(line)
        if not r.get("locate_raw"):
            continue
        items, card = _boxes(r["locate_raw"], order)
        out[r["image_id"]] = {"card": tidy(card) if card else None,
                              "foods": [tidy(it["box"]) for it in items or []]}
    return out


def build(runs: list[pathlib.Path], images: pd.DataFrame, photo_root: pathlib.Path) -> pd.DataFrame:
    """Build a row for each meal photo with a located card, including box agreement and pixel checks."""
    from snapme_eval import registry
    loc = {r.name: located(r, registry.model(r.name).get("box_order", "xy")) for r in runs}
    rows = []
    for img in images[images["photo_type"] == "before"].sort_values("image_id").itertuples():
        cards = [(name, d[img.image_id]["card"]) for name, d in loc.items()
                 if d.get(img.image_id, {}).get("card")]
        if not cards:
            continue
        pair = max(((iou(a[1], b[1]), a[1], b[1]) for i, a in enumerate(cards) for b in cards[i + 1:]),
                   default=None, key=lambda t: t[0])
        gray = np.asarray(open_photo(photo_root / img.path).convert("L"), float) / 255
        h, w = gray.shape

        def scored(b, source, gray=gray, w=w, h=h):
            x0, y0, x1, y1 = pixels(b, w, h)
            return (checker_score(gray[y0:y1, x0:x1]) if x1 - x0 > 3 and y1 - y0 > 3 else 0.0, source, b)
        cands = [scored(b, name) for name, b in cards]
        if pair and pair[0] >= IOU_MIN:   # Put the mean box first so it wins a tie between checker scores.
            cands.insert(0, scored([float(np.mean(v)) for v in zip(pair[1], pair[2], strict=True)], "mean"))
        score, source, box = max(cands, key=lambda t: t[0])
        foods = [f for d in loc.values() for f in d.get(img.image_id, {}).get("foods", [])]
        ctrl, over = control_box(box, foods)
        rows.append({"image_id": img.image_id, "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3],
                     "locators": len(cards), "iou": pair[0] if pair else np.nan, "checker": score, "box_from": source,
                     "kept": bool(score >= CHECKER_MIN),
                     "cx0": ctrl[0], "cy0": ctrl[1], "cx1": ctrl[2], "cy1": ctrl[3], "control_on_food": over})
    return pd.DataFrame(rows).round(3)


def table() -> pd.DataFrame:
    """Read the committed table and return kept photos indexed by photo ID."""
    d = pd.read_csv(TABLE)
    return d[d["kept"]].set_index("image_id")
