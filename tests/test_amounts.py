import math

from snapme_eval.score.score_run import amount, photo_grams


def test_zero_and_missing_grams_are_no_amount():
    assert amount({"grams": 0}) is None
    assert amount({"grams": None}) is None
    assert amount({}) is None
    assert amount({"grams": True}) is None
    assert amount({"grams": "120"}) is None
    assert amount({"grams": 120}) == 120.0


def test_photo_total_uses_only_given_amounts():
    assert photo_grams([{"grams": 100}, {"grams": 0}, {"grams": 50.5}]) == 150.5
    assert math.isnan(photo_grams([{"grams": 0}, {}]))
    assert math.isnan(photo_grams(None))
