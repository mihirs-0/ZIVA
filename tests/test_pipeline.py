"""End-to-end pipeline tests over the mock provider (no API credits consumed).

All data produced here is SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
import pytest

from ziva.analysis import analyze
from ziva.costs import BudgetExceededError, check_budget, estimate_cost
from ziva.freeze import build_freeze_record, experiment_fingerprint, verify_freeze, write_freeze
from ziva.manifest import build_manifest, manifest_path
from ziva.prompts import SYSTEM_PROMPT
from ziva.runner import (
    FingerprintMismatchError,
    check_fingerprint,
    pending_trials,
    prior_spend,
    run_manifest,
)
from ziva.scenarios import generate_scenarios
from ziva.stimuli import render_all
from ziva.util import read_json, write_json, write_jsonl


@pytest.fixture()
def small_world(cfg, mock_models):
    scenarios = [s.to_dict() for s in generate_scenarios(count=8, seed=11, include_case_000=False)]
    image_hashes = render_all(scenarios, cfg.stimuli_dir)
    rows = build_manifest(cfg, scenarios, mock_models, image_hashes)
    write_json(cfg.scenario_dir / "scenarios.json", scenarios)
    write_jsonl(manifest_path(cfg), rows)
    return scenarios, rows, image_hashes


def test_manifest_deterministic_and_shuffled(cfg, mock_models, small_world):
    scenarios, rows, hashes = small_world
    rows2 = build_manifest(cfg, scenarios, mock_models, hashes)
    assert [r["trial_id"] for r in rows] == [r["trial_id"] for r in rows2]
    # shuffled: not grouped by treatment family
    fams = [r["treatment_family"] for r in rows[:20]]
    assert len(set(fams)) > 1
    # both elicitation regimes present
    assert {r["elicitation_mode"] for r in rows} == {"naturalistic", "separated"}


def test_manifest_trial_ids_unique(small_world):
    _, rows, _ = small_world
    ids = [r["trial_id"] for r in rows]
    assert len(ids) == len(set(ids))


def test_cost_estimation_and_budget_guard(cfg, mock_models, small_world):
    from ziva.config import Pricing

    scenarios, rows, _ = small_world
    priced = [m.model_copy(update={"pricing": Pricing(input_per_mtok=100.0, output_per_mtok=100.0)})
              for m in mock_models]
    est = estimate_cost(cfg, rows, scenarios, priced)
    assert est.n_trials == len(rows)
    assert est.total_cost_usd > 0
    with pytest.raises(BudgetExceededError):
        check_budget(est, max_cost_usd=0.000001)
    check_budget(est, max_cost_usd=0.000001, allow_over_budget=True)  # no raise


def test_run_resume_and_analysis(cfg, mock_models, small_world, monkeypatch):
    monkeypatch.setenv("ZIVA_MOCK_BIAS", "0")
    scenarios, rows, _ = small_world

    # partial run: first 10 trials only
    subset = sorted(rows, key=lambda r: r["order_index"])[:10]
    run_manifest(cfg, subset, scenarios, mock_models, SYSTEM_PROMPT)
    assert len(pending_trials(rows, cfg.raw_dir)) == len(rows) - 10

    # resume the remainder; nothing is re-executed
    summary = run_manifest(cfg, rows, scenarios, mock_models, SYSTEM_PROMPT)
    assert summary["already_done_skipped"] == 10
    assert summary["executed_ok"] == len(rows) - 10
    assert len(pending_trials(rows, cfg.raw_dir)) == 0

    result = analyze(cfg, scenarios)
    assert result["SYNTHETIC_DATA"] is True
    pooled = result["primary_pooled_excited_minus_neutral"]
    assert pooled["n_pairs"] > 0
    # unbiased mock -> near-null effect
    assert abs(pooled["mean"]) < 2.0
    assert (cfg.results_path / "paired_table.csv").exists()
    assert (cfg.results_path / "scenario_level_diffs.csv").exists()
    assert (cfg.results_path / "plots" / "01_excited_vs_neutral.png").exists()


def test_pooled_inference_clusters_at_scenario_level(cfg, mock_models, small_world, monkeypatch):
    """P0 regression: the pooled primary p-value must treat the physical
    scenario as the experimental unit -- n_pairs equals the number of
    scenarios, never scenarios x models x modes x elicitations."""
    monkeypatch.setenv("ZIVA_MOCK_BIAS", "0")
    scenarios, rows, _ = small_world
    run_manifest(cfg, rows, scenarios, mock_models, SYSTEM_PROMPT)
    result = analyze(cfg, scenarios)

    n_scenarios = len({r["scenario_id"] for r in rows})
    pooled = result["primary_pooled_excited_minus_neutral"]
    assert pooled["n_pairs"] == n_scenarios
    assert result["n_physical_scenarios"] == n_scenarios
    # per-cell entries pair over scenarios within one cell
    for entry in result["primary_by_model_mode_elicitation"].values():
        assert entry["excited_minus_neutral"]["n_pairs"] <= n_scenarios
    # scenario_level_diffs.csv (the power-analysis input) has one row per scenario
    import pandas as pd

    diffs = pd.read_csv(cfg.results_path / "scenario_level_diffs.csv")
    assert len(diffs) == n_scenarios
    assert diffs["scenario_id"].is_unique


def test_synthetic_bias_is_detected(cfg, mock_models, small_world, monkeypatch):
    monkeypatch.setenv("ZIVA_MOCK_BIAS", "12")
    scenarios, rows, _ = small_world
    run_manifest(cfg, rows, scenarios, mock_models, SYSTEM_PROMPT)
    result = analyze(cfg, scenarios)
    pooled = result["primary_pooled_excited_minus_neutral"]
    assert pooled["mean"] > 8
    assert pooled["p_perm"] < 0.02  # 8 scenario-level pairs bound the attainable p


def test_freeze_detects_modifications(cfg, mock_models, small_world, monkeypatch):
    scenarios, rows, hashes = small_world
    scen_file = cfg.scenario_dir / "scenarios.json"
    record = build_freeze_record(cfg, mock_models, scen_file, manifest_path(cfg), hashes)
    write_freeze(cfg, record)

    ok = verify_freeze(cfg, scen_file, manifest_path(cfg), cfg.stimuli_dir, mock_models)
    assert ok["ok"], ok["violations"]

    # data tampering
    tampered = read_json(scen_file)
    tampered[0]["notes"] = "tampered"
    write_json(scen_file, tampered)
    bad = verify_freeze(cfg, scen_file, manifest_path(cfg), cfg.stimuli_dir, mock_models)
    assert not bad["ok"]
    assert any("scenario_file" in v for v in bad["violations"])
    write_json(scen_file, scenarios)  # restore

    # model-string change under the same model_id
    changed_models = [m.model_copy(update={"model": "mock-2"}) if m.id == "mock_a" else m
                      for m in mock_models]
    bad = verify_freeze(cfg, scen_file, manifest_path(cfg), cfg.stimuli_dir, changed_models)
    assert not bad["ok"]
    assert any("model_config_snapshot" in v or "fingerprint" in v for v in bad["violations"])

    # sampling change (temperature)
    cfg2 = cfg.model_copy(deep=True)
    cfg2.sampling.temperature = 0.2
    bad = verify_freeze(cfg2, scen_file, manifest_path(cfg), cfg.stimuli_dir, mock_models)
    assert not bad["ok"]
    assert any("sampling" in v or "fingerprint" in v for v in bad["violations"])

    # operational budget change is NOT a violation
    cfg3 = cfg.model_copy(deep=True)
    cfg3.run.max_cost_usd = 999.0
    ok2 = verify_freeze(cfg3, scen_file, manifest_path(cfg), cfg.stimuli_dir, mock_models)
    assert ok2["ok"], ok2["violations"]

    # scoring/statistics code change is a violation
    from ziva import freeze as freeze_mod

    monkeypatch.setattr(freeze_mod, "scoring_source_hash", lambda: "0" * 64)
    bad = verify_freeze(cfg, scen_file, manifest_path(cfg), cfg.stimuli_dir, mock_models)
    assert not bad["ok"]
    assert any("scoring_source" in v for v in bad["violations"])


def test_fingerprint_blocks_mixed_experiments(cfg, mock_models, small_world):
    """P0 regression: raw data collected under one experiment definition can
    never be silently extended under another (changed model string,
    temperature, or max_tokens)."""
    scenarios, rows, _ = small_world
    fp1 = experiment_fingerprint(cfg, mock_models)
    subset = sorted(rows, key=lambda r: r["order_index"])[:4]
    run_manifest(cfg, subset, scenarios, mock_models, SYSTEM_PROMPT, fingerprint=fp1)

    # same definition resumes fine
    check_fingerprint(cfg.raw_dir, fp1)

    # provider model string changed under the same model_id -> hard fail
    changed = [m.model_copy(update={"model": "mock-2"}) if m.id == "mock_a" else m
               for m in mock_models]
    fp_model = experiment_fingerprint(cfg, changed)
    assert fp_model["fingerprint"] != fp1["fingerprint"]
    with pytest.raises(FingerprintMismatchError):
        run_manifest(cfg, rows, scenarios, changed, SYSTEM_PROMPT, fingerprint=fp_model)

    # temperature changed -> hard fail
    cfg_t = cfg.model_copy(deep=True)
    cfg_t.sampling.temperature = 0.0
    fp_t = experiment_fingerprint(cfg_t, mock_models)
    assert fp_t["fingerprint"] != fp1["fingerprint"]
    with pytest.raises(FingerprintMismatchError):
        check_fingerprint(cfg.raw_dir, fp_t)

    # max_tokens changed -> hard fail
    cfg_m = cfg.model_copy(deep=True)
    cfg_m.sampling.max_tokens = 999
    fp_m = experiment_fingerprint(cfg_m, mock_models)
    with pytest.raises(FingerprintMismatchError):
        check_fingerprint(cfg.raw_dir, fp_m)

    # prompt/treatment wording changes move the fingerprint too (source-hashed)
    comps = fp1["components"]
    assert "prompt_templates_source" in comps and "treatments_source" in comps
    assert "scoring_source" in comps


def test_budget_is_cumulative_across_resumes(cfg, mock_models, small_world):
    """P1 regression: max_cost_usd bounds the EXPERIMENT total. A resumed run
    that is already at budget executes nothing until the budget is raised."""
    from ziva.config import Pricing

    scenarios, rows, _ = small_world
    # ~$0.14 per trial -> a $1 budget affords only a handful of the 384 trials
    priced = [m.model_copy(update={"pricing": Pricing(input_per_mtok=200.0, output_per_mtok=200.0)})
              for m in mock_models]
    cfg2 = cfg.model_copy(deep=True)
    cfg2.run.max_cost_usd = 1.0
    s1 = run_manifest(cfg2, rows, scenarios, priced, SYSTEM_PROMPT)
    assert s1["stopped_for_budget"]
    executed_1 = s1["executed_ok"] + s1["executed_with_errors"]
    assert 0 < executed_1 < len(rows)
    spent_1 = prior_spend(cfg2.raw_dir)
    assert spent_1 <= cfg2.run.max_cost_usd * 1.05  # reservation bounds overshoot

    # identical re-invocation: prior spend is counted, nothing more is executed
    s2 = run_manifest(cfg2, rows, scenarios, priced, SYSTEM_PROMPT)
    assert s2["stopped_for_budget"]
    assert s2["executed_ok"] + s2["executed_with_errors"] == 0
    assert prior_spend(cfg2.raw_dir) == pytest.approx(spent_1)
    assert s2["prior_spend_usd"] == pytest.approx(spent_1, abs=0.01)

    # raising the budget resumes the remaining trials
    cfg3 = cfg2.model_copy(deep=True)
    cfg3.run.max_cost_usd = 1e9
    s3 = run_manifest(cfg3, rows, scenarios, priced, SYSTEM_PROMPT)
    assert s3["executed_ok"] + s3["executed_with_errors"] == len(rows) - executed_1


def test_request_failed_trials_are_retried(cfg, mock_models, small_world, monkeypatch):
    """P1 regression: a transient provider failure must not become a
    permanently 'completed' trial."""
    from ziva.providers.mock import MockAdapter

    scenarios, rows, _ = small_world
    cfg2 = cfg.model_copy(deep=True)
    cfg2.run.retries = 0
    cfg2.run.retry_backoff_s = 0.0
    subset = sorted(rows, key=lambda r: r["order_index"])[:6]
    victim_id = subset[0]["trial_id"]

    original = MockAdapter.complete

    def flaky(self, request):
        if request.meta.get("trial_id") == victim_id:
            raise ConnectionError("simulated transient provider outage")
        return original(self, request)

    monkeypatch.setattr(MockAdapter, "complete", flaky)
    s1 = run_manifest(cfg2, subset, scenarios, mock_models, SYSTEM_PROMPT)
    assert s1["executed_with_errors"] == 1
    rec = read_json(cfg2.raw_dir / f"{victim_id}.json")
    assert rec["final_parse"]["status"] == "request_failed"
    # the failed trial is pending again, not permanently complete
    assert victim_id in {r["trial_id"] for r in pending_trials(subset, cfg2.raw_dir)}

    # provider recovers -> the retry actually happens and succeeds
    monkeypatch.setattr(MockAdapter, "complete", original)
    s2 = run_manifest(cfg2, subset, scenarios, mock_models, SYSTEM_PROMPT)
    assert s2["retried_failed_trials"] == 1
    assert s2["executed_ok"] == 1
    rec = read_json(cfg2.raw_dir / f"{victim_id}.json")
    assert rec["final_parse"]["status"] == "ok"
    assert not pending_trials(subset, cfg2.raw_dir)


def test_multistep_experiments_run_and_share_turn1(cfg, mock_models, small_world):
    scenarios, _, hashes = small_world
    cfg2 = cfg.model_copy(deep=True)
    cfg2.experiments.evidence_update = True
    cfg2.experiments.commitment = True
    cfg2.experiments.primary = False
    rows = build_manifest(cfg2, scenarios[:2], mock_models, hashes)
    assert {r["experiment"] for r in rows} == {"evidence_update", "commitment"}
    summary = run_manifest(cfg2, rows, scenarios, mock_models, SYSTEM_PROMPT)
    assert summary["executed_ok"] == len(rows)

    # all reaction arms of one (scenario, model, repeat) share the SAME sampled
    # turn-1 response (baseline stochasticity is not injected into the contrast)
    from collections import defaultdict

    turn1_by_key = defaultdict(set)
    fresh_by_key = defaultdict(int)
    for p in cfg2.raw_dir.glob("t_*.json"):
        rec = read_json(p)
        step0 = rec["steps"][0]
        if not step0.get("shared_turn1"):
            continue  # commitment_b has no shared baseline
        key = (rec["scenario_id"], rec["model_id"], rec["repeat_index"])
        turn1_by_key[key].add(step0["result"]["text"])
        if not step0.get("shared_turn1_reused"):
            fresh_by_key[key] += 1
    assert turn1_by_key, "no shared-turn1 records found"
    for key, texts in turn1_by_key.items():
        # identical across ALL arms sharing the key (reaction arms + commitment A)
        assert len(texts) == 1, f"turn-1 differs across arms for {key}"
        assert fresh_by_key[key] == 1, f"turn-1 sampled {fresh_by_key[key]} times for {key}"

    result = analyze(cfg2, scenarios)
    assert result["evidence_update"]
    assert "a_minus_b_final_probability" in result["commitment"]


def test_followup_texts_are_coherent():
    """P1 regression: reactions follow the evidence they react to, and the
    commitment turn never presupposes the model's first answer."""
    from ziva.followup import (
        COMMITMENT_NEUTRAL,
        EVIDENCE_INTRO,
        commitment_a_turn2,
        turn2_user_text,
    )
    from ziva.scenarios import generate_scenarios
    from ziva.treatments import REACTIONS

    sc = generate_scenarios(count=2, seed=3, include_case_000=False)[0].to_dict()
    for reaction in REACTIONS:
        text = turn2_user_text(sc, reaction)
        assert text.index(EVIDENCE_INTRO) < text.index(reaction.text), (
            f"{reaction.id}: reaction precedes the evidence it reacts to"
        )
    # conclusion-neutral commitment: coherent after visible AND not_visible answers
    for phrase in ("going to see it", "we're going to", "wonderful to hear", "glad"):
        assert phrase not in COMMITMENT_NEUTRAL.lower()
    assert "whatever" in COMMITMENT_NEUTRAL.lower() or "either way" in COMMITMENT_NEUTRAL.lower()
    assert EVIDENCE_INTRO in commitment_a_turn2(sc)


def test_variance_components_separated():
    """P2 regression: generation noise (repeats of the exact prompt) and
    paraphrase variance are computed separately."""
    from ziva.metrics import trials_dataframe, within_cell_variance_components

    scenario = {"scenario_id": "s1", "classification": {}}

    def rec(tid, fam, rep, prob):
        return {
            "trial_id": f"t_{tid}_{rep}",
            "experiment": "primary",
            "scenario_id": "s1",
            "treatment_id": tid,
            "treatment_family": fam,
            "evidence_mode": "text",
            "elicitation_mode": "separated",
            "model_id": "m1",
            "repeat_index": rep,
            "directionality": {},
            "steps": [{"step_index": 0,
                       "parse": {"status": "ok", "parsed": {
                           "visible_probability": prob, "binary_prediction": "visible",
                           "confidence": 50, "evidence_sufficiency": "sufficient",
                           "would_recommend_attempt": True, "short_explanation": "synthetic"},
                           "errors": []},
                       "result": {}}],
        }

    # exact-prompt repeats differ by 2 (sampling noise); templates differ by 20
    records = [
        rec("neutral_v1", "neutral", 0, 40), rec("neutral_v1", "neutral", 1, 42),
        rec("neutral_v2", "neutral", 0, 60), rec("neutral_v2", "neutral", 1, 62),
    ]
    df = trials_dataframe(records, [scenario])
    comps = within_cell_variance_components(df)
    assert comps["sampling_sd_mean_points"] == pytest.approx(1.414, abs=0.01)
    assert comps["template_sd_mean_points"] == pytest.approx(14.14, abs=0.1)


def test_stats_paired_summary():
    from ziva.stats import paired_summary

    rng = np.random.default_rng(0)
    diffs = rng.normal(5, 2, size=40)
    s = paired_summary(diffs)
    assert 4 < s["mean"] < 6
    assert s["p_perm"] < 0.001
    assert s["ci95"][0] < s["mean"] < s["ci95"][1]

    null = rng.normal(0, 2, size=40)
    s0 = paired_summary(null)
    assert s0["p_perm"] > 0.01


def test_power_analysis_shrinks_with_effect():
    from ziva.power import required_pairs

    rng = np.random.default_rng(1)
    pilot = rng.normal(0, 10, size=30)
    big = required_pairs(pilot, effect_points=10.0, seed=2)
    small = required_pairs(pilot, effect_points=2.0, seed=2, max_pairs=4000)
    assert big["estimated_required_pairs"] is not None
    if small["estimated_required_pairs"] is not None:
        assert small["estimated_required_pairs"] >= big["estimated_required_pairs"]


def test_metrics_on_synthetic_fixture():
    """Analysis metrics verified on a hand-built SYNTHETIC fixture."""
    from ziva.metrics import paired_table, scenario_level_diffs, trials_dataframe

    scenario = {"scenario_id": "s1", "classification": {"category": "x", "difficulty_score": 0.5,
                                                        "difficulty_class": "moderate", "anchor": None}}

    def rec(treatment_family, treatment_id, prob, binary):
        return {
            "trial_id": f"t_{treatment_id}_{prob}",
            "experiment": "primary",
            "scenario_id": "s1",
            "treatment_id": treatment_id,
            "treatment_family": treatment_family,
            "evidence_mode": "text",
            "elicitation_mode": "separated",
            "model_id": "m1",
            "provider": "mock",
            "model": "mock-1",
            "repeat_index": 0,
            "directionality": {},
            "steps": [{
                "step_index": 0,
                "parse": {"status": "ok", "parsed": {
                    "visible_probability": prob, "binary_prediction": binary, "confidence": 50,
                    "evidence_sufficiency": "sufficient", "would_recommend_attempt": True,
                    "short_explanation": "synthetic"}, "errors": []},
                "result": {"structured_mode": "provider_schema", "tools_used": []},
            }],
        }

    records = [
        rec("neutral", "neutral_v1", 30, "not_visible"),
        rec("neutral", "neutral_v2", 40, "not_visible"),
        rec("excited_positive", "excited_positive_v1", 60, "visible"),
        rec("excited_positive", "excited_positive_v2", 70, "visible"),
    ]
    df = trials_dataframe(records, [scenario])
    table = paired_table(df)
    row = table.iloc[0]
    assert row["neutral_probability"] == 35
    assert row["excited_positive_probability"] == 65
    assert row["excited_minus_neutral"] == 30
    assert bool(row["binary_flip"]) is True
    diffs = scenario_level_diffs(table)
    assert len(diffs) == 1 and diffs.iloc[0]["excited_minus_neutral"] == 30
