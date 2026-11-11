from snapme_eval import parse


def test_fenced_json_after_reasoning():
    text = '<think>plate has eggs</think>\n```json\n{"items": [{"name": "Egg, fried", "grams": 46}]}\n```'
    items, err = parse.items(text)
    assert err is None
    assert items[0]["name"] == "Egg, fried" and items[0]["grams"] == 46.0


def test_bare_object_with_trailing_text():
    items, err = parse.items('Here you go: {"items": [{"name": "Coffee", "grams": "240 g"}]} done')
    assert err is None and items[0]["grams"] == 240.0


def test_empty_list_is_a_valid_answer():
    items, err = parse.items('{"items": []}')
    assert items == [] and err is None


def test_unreadable_reply():
    items, err = parse.items("I cannot tell what this is.")
    assert items is None and err


def test_stray_quote_after_number():
    items, err = parse.items('{"items": [{"name": "Raspberry, fresh", "preparation": "raw", "grams": 30.5"}]}')
    assert err is None and items[0]["grams"] == 30.5 and items[0]["preparation"] == "raw"
