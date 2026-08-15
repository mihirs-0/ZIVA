import json

from ziva.parsing import parse_response


GOOD = {
    "visible_probability": 40,
    "binary_prediction": "not_visible",
    "confidence": 60,
    "evidence_sufficiency": "sufficient",
    "would_recommend_attempt": False,
    "short_explanation": "test",
}


def test_clean_json():
    r = parse_response(json.dumps(GOOD))
    assert r["status"] == "ok"
    assert r["parsed"]["visible_probability"] == 40


def test_fenced_json():
    r = parse_response("Here you go:\n```json\n" + json.dumps(GOOD) + "\n```\nHope that helps!")
    assert r["status"] == "ok"


def test_json_embedded_in_prose():
    r = parse_response("Sure! " + json.dumps(GOOD) + " Let me know if you need more.")
    assert r["status"] == "ok"


def test_fraction_probability_normalized():
    payload = {**GOOD, "visible_probability": 0.4, "confidence": 0.9}
    r = parse_response(json.dumps(payload))
    assert r["status"] == "ok"
    assert r["parsed"]["visible_probability"] == 40
    assert r["parsed"]["confidence"] == 90


def test_binary_synonyms():
    for raw, want in (("Visible", "visible"), ("NOT VISIBLE", "not_visible"), ("invisible", "not_visible")):
        r = parse_response(json.dumps({**GOOD, "binary_prediction": raw}))
        assert r["status"] == "ok"
        assert r["parsed"]["binary_prediction"] == want


def test_malformed_json_not_silently_dropped():
    r = parse_response("{ this is not json")
    assert r["status"] in ("json_error", "validation_error")
    assert r["parsed"] is None
    assert r["errors"]


def test_out_of_range_probability_rejected():
    r = parse_response(json.dumps({**GOOD, "visible_probability": 250}))
    assert r["status"] == "validation_error"


def test_empty_response():
    r = parse_response("   ")
    assert r["status"] == "empty"


def test_missing_field_rejected():
    bad = dict(GOOD)
    del bad["confidence"]
    r = parse_response(json.dumps(bad))
    assert r["status"] == "validation_error"
