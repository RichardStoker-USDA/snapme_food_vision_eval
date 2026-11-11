import numpy as np
from PIL import Image

from snapme_eval import marker


def card(angle: float, n: int = 120, fill: float = 0.5) -> np.ndarray:
    """Draw a rotated two-by-two checker card on a mid-gray table."""
    ys, xs = np.mgrid[0:n, 0:n] - n / 2
    t = np.deg2rad(angle)
    u, v = xs * np.cos(t) + ys * np.sin(t), -xs * np.sin(t) + ys * np.cos(t)
    half = n * 0.5 / (abs(np.cos(t)) + abs(np.sin(t)))
    g = np.full((n, n), fill)
    on = (abs(u) < half) & (abs(v) < half)
    g[on] = np.where(u[on] * v[on] > 0, 0.92, 0.08)
    return g


def test_checker_score_finds_the_card_at_any_rotation():
    for a in (0, 20, 45, 70):
        assert marker.checker_score(card(a)) > 0.8


def test_checker_score_rejects_plain_edges_and_texture():
    rng = np.random.default_rng(0)
    edge = np.zeros((120, 120))
    edge[:, 60:] = 1.0
    corner = np.zeros((120, 120))
    corner[60:, 60:] = 1.0
    assert marker.checker_score(edge) < 0.2 and marker.checker_score(corner) < 0.2
    assert marker.checker_score(rng.random((120, 120))) < 0.2
    assert marker.checker_score(np.full((120, 120), 0.5)) == 0.0


def test_hide_paints_the_box_and_margin_only():
    a = np.full((200, 200, 3), 90, np.uint8)
    a[100:140, 20:60] = 255
    out = np.asarray(marker.hide(Image.fromarray(a), [100, 500, 300, 700]))
    assert (out[100:140, 20:60] == 90).all()          # Fill the card area with the surrounding color.
    assert (out == 90).all()


def test_control_box_avoids_the_card_and_foods():
    c = [100.0, 750.0, 250.0, 900.0]
    foods = [[300.0, 200.0, 800.0, 700.0]]
    b, over = marker.control_box(c, foods)
    assert over == 0 and marker.iou(b, c) == 0
    assert abs((b[2] - b[0]) - 150) < 1e-6 and abs((b[3] - b[1]) - 150) < 1e-6
    assert b[0] > 500                                  # Place the control on the opposite side of the photo.
