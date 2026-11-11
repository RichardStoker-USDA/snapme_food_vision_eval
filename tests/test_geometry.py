import numpy as np

from snapme_eval import geometry


def scene(tilt: float = 0.0):
    """Construct a table scene with a 6 x 6 x 3 cm block (108 mL), viewed from 50 cm above."""
    h = w = 200
    ys, xs = np.mgrid[0:h, 0:w]
    x, y = (xs - w / 2) * 0.002, (ys - h / 2) * 0.002
    z = np.full((h, w), 0.5) + tilt * y
    z[(abs(x) < 0.03) & (abs(y) < 0.03)] -= 0.03
    return {"points": np.stack([x, y, z], -1), "mask": np.ones((h, w), bool)}


BOX = [300, 300, 700, 700]


def test_block_volume_within_ten_percent():
    m = geometry.measure(scene(), [BOX])
    assert m["ok"] and abs(m["items"][0]["volume_ml"] - 108) / 108 < 0.10
    assert abs(m["items"][0]["max_height_cm"] - 3.0) < 0.2


def test_tilted_table():
    m = geometry.measure(scene(tilt=0.3), [BOX])
    assert m["ok"] and abs(m["items"][0]["volume_ml"] - 108) / 108 < 0.15


def test_empty_box_has_no_volume():
    m = geometry.measure(scene(), [[0, 0, 100, 100]])
    assert m["items"][0]["volume_ml"] < 1


def test_boxes_in_y_first_order_are_turned_to_x_first():
    from snapme_eval.systems.geometry import _boxes
    reply = '{"card": [800, 30, 950, 170], "items": [{"name": "rice", "box": [250, 500, 690, 930]}]}'
    items, card = _boxes(reply, "yx")
    assert card == [30, 800, 170, 950] and items[0]["box"] == [500, 250, 930, 690]
    items, card = _boxes(reply)
    assert card == [800, 30, 950, 170] and items[0]["box"] == [250, 500, 690, 930]
