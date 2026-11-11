import json

import pytest

from snapme_eval.score import report


def _scores(folder, **extra):
    f1 = {"mean": 0.5, "ci": [0.4, 0.6]}
    folder.mkdir(parents=True)
    (folder / "scores.json").write_text(json.dumps({
        "n_images": 2, "format_compliance": 1.0, "wweia_f1": f1, "code_f1": f1, "token_f1": f1,
        "dish_grams": {"median_ape": 0.3, "within_20pct": 0.4, "given": 1.0}, **extra}))


def test_nutrition5k_row_needs_every_name_coded(tmp_path):
    run = tmp_path / "runs" / "nutrition5k_v1" / "direct_v2" / "qwen3-5-9b"
    _scores(run, names_not_in_cache=0)
    rows = report.collect_nutrition5k(tmp_path / "runs", tmp_path / "results")
    assert [r["model"] for r in rows] == ["qwen3-5-9b"]


@pytest.mark.parametrize("extra", [{"names_not_in_cache": 3}, {}])
def test_nutrition5k_row_refused_when_names_lack_codes(tmp_path, extra):
    _scores(tmp_path / "runs" / "nutrition5k_v1" / "direct_v2" / "qwen3-5-9b", **extra)
    with pytest.raises(SystemExit, match="names_not_in_cache"):
        report.collect_nutrition5k(tmp_path / "runs", tmp_path / "results")
