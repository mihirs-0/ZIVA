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


def test_probability_policy_explicit_and_frozen():
    """Frozen policy: integers 0-100 only; integer-valued floats accepted as
    their integer; fractional floats REJECTED (never heuristically rescaled --
    1.0 is ambiguous between 1% and a 100% fraction)."""
    # integer-valued floats accepted
    r = parse_response(json.dumps({**GOOD, "visible_probability": 40.0, "confidence": 90.0}))
    assert r["status"] == "ok"
    assert r["parsed"]["visible_probability"] == 40
    assert r["parsed"]["confidence"] == 90
    # endpoints: 0.0 -> 0, 1.0 -> 1 (an integer-valued float on the 0-100 scale)
    r = parse_response(json.dumps({**GOOD, "visible_probability": 0.0}))
    assert r["status"] == "ok" and r["parsed"]["visible_probability"] == 0
    r = parse_response(json.dumps({**GOOD, "visible_probability": 1.0}))
    assert r["status"] == "ok" and r["parsed"]["visible_probability"] == 1
    # fractional values are malformed, not rescaled
    for bad in (0.4, 0.9, 62.5):
        r = parse_response(json.dumps({**GOOD, "visible_probability": bad}))
        assert r["status"] == "validation_error", f"{bad} must be rejected"
    r = parse_response(json.dumps({**GOOD, "confidence": 0.9}))
    assert r["status"] == "validation_error"


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
