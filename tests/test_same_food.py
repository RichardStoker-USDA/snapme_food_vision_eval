"""Check grader reply parsing and food-list measures in align.py and score/same_food.py."""
import pandas as pd
import pytest

from snapme_eval import align
from snapme_eval.score import same_food


def test_letters_continue_past_z():
    assert align.letters(28)[-3:] == ["Z", "AA", "AB"]


def test_parse_keeps_valid_pairs_only():
    txt = 'Here: ```json\n{"pairs": [{"item": "a", "line": 1, "grade": "SAME"}, {"item": "C", "line": 1, ' \
          '"grade": "same"}, {"item": "B", "line": 9, "grade": "type"}, ' \
          '{"item": "B", "line": 2, "grade": "maybe"}]}\n```'
    assert align.parse(txt, n_items=2, n_lines=2) == [(0, 0, "same")]
    assert align.parse("no json", 2, 2) is None


def test_key_changes_with_list_and_model():
    lines, items = ["Coffee, brewed"], [{"name": "coffee"}]
    k = align.key_for("m", lines, items)
    assert k == align.key_for("m", lines, [{"name": "coffee"}])
    assert k != align.key_for("m", lines, [{"name": "tea"}])
    assert k != align.key_for("other", lines, items)


@pytest.mark.parametrize("text,want", [
    ("Kale, cooked, from fresh, fat not added in cooking", "cooked"),
    ("Tomatoes, raw", "raw"),
    ("Coffee, brewed", None),
    ("fresh", "raw"),
])
def test_raw_or_cooked(text, want):
    assert same_food.raw_or_cooked(text) == want


def gt(n_lines, dishes):
    return pd.DataFrame({"line": range(1, n_lines + 1), "dish_id": dishes,
                         "description": [f"food {i}" for i in range(n_lines)]})


def test_photo_levels_and_parts():
    # Compare three record lines from two dishes with four listed foods.
    g = gt(3, ["d1", "d1", "d2"])
    items = [{"name": "a"}, {"name": "b"}, {"name": "c"}, {"name": "d"}]
    pairs = [(0, 0, "same"), (1, 2, "type"), (2, 1, "part"), (0, 2, "same")]   # The final pair reuses item 0.
    m = same_food.photo(g, items, pairs)
    # same: items 0 (same) and 2 (part) match lines 0 and 1, both lines of d1; d2 is not found.
    assert m["same_correct"] == pytest.approx(2 / 4)
    assert m["same_found_lines"] == pytest.approx(2 / 3)
    assert m["same_found_dishes"] == pytest.approx(1 / 2)
    # The type level also pairs item 1 with line 2.
    assert m["type_correct"] == pytest.approx(3 / 4)
    assert m["type_found_lines"] == pytest.approx(1.0)
    assert m["type_found_dishes"] == pytest.approx(1.0)
    assert m["pairs_same"] == [(0, 0)]
    # Remove unpaired item 3, add nothing because all lines are paired, and rename the one type pair.
    assert (m["edits_add"], m["edits_remove"], m["edits_rename"], m["edits"], m["no_edits"]) == (0, 1, 1, 2, 0)


def test_photo_needing_no_corrections():
    m = same_food.photo(gt(2, [None, None]), [{"name": "a"}, {"name": "b"}], [(0, 1, "same"), (1, 0, "same")])
    assert m["edits"] == 0 and m["no_edits"] == 1 and m["same_f1"] == 1


def test_photo_empty_list_finds_nothing():
    m = same_food.photo(gt(2, [None, None]), [], [])
    assert m["same_f1"] == 0 and m["same_found_lines"] == 0 and m["same_found_dishes"] == 0
    assert m["edits_add"] == 2 and m["edits"] == 2


def test_cache_parts_and_shares(tmp_path, monkeypatch):
    import json

    import pandas as pd
    monkeypatch.setattr(align, "cache_dir", lambda: tmp_path)
    assert align.load_cache() == {}
    first = pd.DataFrame({"key": ["a1", "b2"], "pairs": [json.dumps([[0, 0, "same"]]), None]})
    first.to_parquet(tmp_path / "part_1_0000.parquet")
    pd.DataFrame({"key": ["c3"], "pairs": [json.dumps([])]}).to_parquet(tmp_path / "part_2_0000.parquet")
    assert align.load_cache() == {"a1": [[0, 0, "same"]], "c3": []}
    keys = [align.key_for("m", ["Coffee"], [{"name": f"food {i}"}]) for i in range(200)]
    shares = [[k for k in keys if align.in_shard(k, (i, 8))] for i in range(8)]
    assert sorted(sum(shares, [])) == sorted(keys) and all(shares)


def test_grader_sheet_draw_and_agreement(tmp_path):
    import json

    import pandas as pd

    from snapme_eval.score import grader_check as gc
    lists = [{"run": f"direct_v2/m{k % 3}", "image_id": f"p{k}",
              "lines": ["Coffee, brewed", "Milk, whole", "Bread, white"],
              "items": ["coffee", "almond milk", "soup", "toast"],
              "pairs": [[0, 0, "same"], [1, 1, "type"], [3, 2, "part"]]} for k in range(40)]
    rows = gc.draw(lists, 40)
    assert len(rows) == 40 and {r["grader"] for r in rows} == set(gc.OUTCOMES)
    assert max(pd.Series([r["image_id"] for r in rows]).value_counts()) <= 2
    assert all((r["grader"] == "different") == ([r["item"], r["line"]] not in [[0, 0], [1, 1], [3, 2]]) for r in rows)
    key = {r["id"]: {"grader": r["grader"]} for r in rows}
    (tmp_path / "key.json").write_text(json.dumps(key))
    filled = pd.DataFrame([{"id": r["id"], "image_id": r["image_id"], "grade": r["grader"]} for r in rows])
    filled.loc[0, "grade"] = "same" if rows[0]["grader"] != "same" else "type"     # Introduce one disagreement.
    filled.to_csv(tmp_path / "sheet.csv", index=False)
    a = gc.agreement(tmp_path / "sheet.csv", tmp_path / "key.json", reps=200)
    assert a["decisions"] == 40 and abs(a["agree"]["value"] - 39 / 40) < 1e-9 and 0.9 < a["kappa"]["value"] < 1
    assert "Give a pair for every" in gc.rules() and "Reply with JSON" not in gc.rules()
