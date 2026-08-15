from pathlib import Path

import pytest

from ziva.prompts import EVIDENCE_MODES, audit_pairing, compile_prompt, diff_prompts
from ziva.scenarios import case_000_davis_crescent, generate_scenarios
from ziva.stimuli import render_stimulus
from ziva.treatments import FAMILIES, TREATMENTS, select_treatments


def test_generation_deterministic_under_seed():
    a = [s.to_dict() for s in generate_scenarios(count=10, seed=7)]
    b = [s.to_dict() for s in generate_scenarios(count=10, seed=7)]
    assert a == b


def test_generation_differs_across_seeds():
    a = [s.scenario_id for s in generate_scenarios(count=10, seed=7, include_case_000=False)]
    b = [s.to_dict() for s in generate_scenarios(count=10, seed=8, include_case_000=False)]
    assert a != [s["scenario_id"] for s in b] or a == []  # ids share category names; compare payloads
    a_full = [s.to_dict() for s in generate_scenarios(count=10, seed=7, include_case_000=False)]
    assert a_full != b


def test_stratification_quotas(scenarios):
    cats = [s["classification"]["category"] for s in scenarios if not s["scenario_id"].startswith("case_000")]
    # the default mix guarantees daylight ambiguity is represented
    assert any(c.startswith("daylight") for c in cats)
    assert any(c == "trivial_invisible" for c in cats)


def test_case_000_matches_anecdote():
    cases = case_000_davis_crescent()
    assert len(cases) == 2
    for c in cases:
        ps = c.physical_state
        assert c.location["name"].startswith("Davis")
        assert ps["sky_regime"] == "daylight"
        assert 0.03 < ps["moon_illumination_fraction"] < 0.10
        assert "anecdotal" in c.notes


def test_paired_prompts_identical_facts(scenarios):
    treatments = select_treatments(FAMILIES, 2)
    for mode in EVIDENCE_MODES:
        record = audit_pairing(scenarios[0], treatments, mode)
        assert record["ok"]
        # every treatment yields a distinct full prompt but identical invariant
        assert len(set(record["prompt_hashes"].values())) == len(treatments)


def test_audit_catches_fact_differences(scenarios):
    """A treatment whose scenario gets altered must fail the audit."""
    from ziva.treatments import Treatment

    t1 = Treatment(id="x1", family="neutral", text="A", preferred_outcome=None)
    a = compile_prompt(scenarios[0], t1, "text")
    tampered = {**scenarios[0], "local_time": "1999-01-01 00:00 XXX (UTC+0000)"}
    b = compile_prompt(tampered, t1, "text")
    d = diff_prompts(a, b)
    assert not d["same_invariant"]


def test_treatment_wording_has_no_factual_sky_claims():
    banned = ["bright", "dark", "faint", "cloud", "clear sky", "horizon", "degree",
              "twilight", "glare", "daylight", "crescent", "full moon"]
    for t in TREATMENTS:
        low = t.text.lower()
        hits = [w for w in banned if w in low]
        assert not hits, f"{t.id} leaks factual sky terms: {hits}"


def test_treatment_families_have_paraphrases():
    for fam in ("neutral", "excited_positive"):
        assert len([t for t in TREATMENTS if t.family == fam]) >= 2


def test_directionality_labels():
    from ziva.treatments import get_treatment

    assert get_treatment("excited_positive_v1").preferred_outcome == "visible"
    assert get_treatment("negative_preference_v1").preferred_outcome == "not_visible"
    assert get_treatment("skeptical_negative_v1").preferred_outcome is None
    assert get_treatment("skeptical_negative_v1").expected_outcome == "not_visible"
    assert get_treatment("anti_sycophancy_v1").anti_sycophancy_instruction
    assert get_treatment("explicit_request_v1").explicit_answer_request


def test_primary_treatments_are_preference_only():
    """P0 regression: no primary valence treatment may request, confirm, or
    steer toward a particular answer. Explicit answer pressure lives only in
    the secondary explicit_request family."""
    from ziva.treatments import ANSWER_STEERING_MARKERS as steering
    from ziva.treatments import PRIMARY_ELIGIBLE_FAMILIES

    # the marker list itself must catch the known-bad wording
    assert any(m in "please tell me i'll be able to see it!" for m in steering)
    assert any(m in "please just say the moon is visible right now" for m in steering)
    for t in TREATMENTS:
        if t.family not in PRIMARY_ELIGIBLE_FAMILIES:
            continue
        low = t.text.lower()
        hits = [w for w in steering if w in low]
        assert not hits, f"primary treatment {t.id} steers toward an answer: {hits}"
        assert not t.explicit_answer_request, f"{t.id} flagged as answer request in a primary family"
    assert "explicit_request" not in PRIMARY_ELIGIBLE_FAMILIES


def test_negative_preference_symmetric_no_action_stakes():
    """P2 regression: negative preference is a pure preference statement --
    no promises, obligations, photos, or effort/action stakes."""
    stakes = ["promise", "owe", "friend", "photo", "bother", "go out", "check"]
    for t in TREATMENTS:
        if t.family != "negative_preference":
            continue
        low = t.text.lower()
        hits = [w for w in stakes if w in low]
        assert not hits, f"{t.id} carries action/utility stakes: {hits}"


def test_primary_endpoint_is_locatability():
    """P0 regression: the elicited affordance is locating the Moon (visual
    search included), not detection conditional on exact fixation."""
    from ziva.prompts import ELICITATION_BLOCKS, RESPONSE_JSON_SCHEMA
    from ziva.scenarios import TARGET_PROPOSITION

    assert "locate" in TARGET_PROPOSITION
    assert "two minutes" in TARGET_PROPOSITION
    assert "looking in the right direction" not in TARGET_PROPOSITION
    desc = RESPONSE_JSON_SCHEMA["properties"]["visible_probability"]["description"]
    assert "locate" in desc and "two minutes" in desc
    for block in ELICITATION_BLOCKS.values():
        assert "locate the Moon in the sky within two minutes" in block


def test_elicitation_regimes_differ_only_in_preamble():
    from ziva.prompts import ELICITATION_BLOCKS, _JSON_FIELD_SPEC

    nat, sep = ELICITATION_BLOCKS["naturalistic"], ELICITATION_BLOCKS["separated"]
    assert nat != sep
    assert nat.endswith(_JSON_FIELD_SPEC) and sep.endswith(_JSON_FIELD_SPEC)
    assert "separately from any recommendation" in sep
    assert "separately from any recommendation" not in nat


def test_structured_and_image_share_canonical_facts(scenarios):
    """P1 regression: structured JSON and the image card are deterministic
    encodings of exactly the same canonical fact dictionary."""
    import json as _json

    from ziva.prompts import canonical_facts, render_structured_evidence
    from ziva.stimuli import fact_rows

    sc = scenarios[0]
    facts = canonical_facts(sc)
    # structured mode embeds the dictionary verbatim
    rendered = render_structured_evidence(sc)
    embedded = _json.loads(rendered.split("```json\n")[1].split("\n```")[0])
    assert embedded == facts
    # the image renders exactly one row per canonical fact (no extra derived
    # cues, nothing missing), including the fields the old card leaked only
    # visually (regime label, above-horizon flags)
    rows = fact_rows(sc)
    assert len(rows) == len(facts)
    labels = [label for label, _ in rows]
    assert "Sky brightness" in labels
    assert "Moon above horizon" in labels and "Sun above horizon" in labels
    # ...and the same fields now appear in the structured block too
    assert "sky_brightness" in embedded and "moon_above_horizon" in embedded


def test_image_rendering_deterministic(scenarios, tmp_path):
    h1 = render_stimulus(scenarios[0], tmp_path / "a.png")
    h2 = render_stimulus(scenarios[0], tmp_path / "b.png")
    assert h1 == h2
    assert (tmp_path / "a.png").read_bytes() == (tmp_path / "b.png").read_bytes()


def test_image_differs_across_scenarios(scenarios, tmp_path):
    h1 = render_stimulus(scenarios[0], tmp_path / "a.png")
    h2 = render_stimulus(scenarios[1], tmp_path / "b.png")
    assert h1 != h2


def test_image_contains_no_visibility_verdict():
    """The renderer must not leak a verdict string; inspect the source contract."""
    import inspect

    from ziva import stimuli

    src = inspect.getsource(stimuli.render_stimulus).lower()
    assert "visible:" not in src
    assert "verdict" not in src


def test_unknown_evidence_mode_rejected(scenarios):
    t = select_treatments(["neutral"], 1)[0]
    with pytest.raises(ValueError):
        compile_prompt(scenarios[0], t, "hologram")


def test_scenario_generator_rejects_unknown_category():
    with pytest.raises(ValueError):
        generate_scenarios(count=5, seed=1, category_mix={"made_up_category": 1.0})


def test_stimulus_path_layout(tmp_path):
    from ziva.stimuli import stimulus_path

    assert stimulus_path(tmp_path, "scn_1") == Path(tmp_path) / "scn_1.png"
