"""End-to-end pipeline tests over the mock provider (no API credits consumed).

All data produced here is SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
import pytest

from ziva.analysis import analyze
from ziva.costs import BudgetExceededError, check_budget, estimate_cost
from ziva.freeze import build_freeze_record, verify_freeze, write_freeze
from ziva.manifest import build_manifest, manifest_path
from ziva.prompts import SYSTEM_PROMPT
from ziva.runner import pending_trials, run_manifest
from ziva.scenarios import generate_scenarios
from ziva.stimuli import render_all
from ziva.util import read_json, write_json, write_jsonl


@pytest.fixture()
def small_world(cfg, mock_models):
    scenarios = [s.to_dict() for s in generate_scenarios(count=6, seed=11, include_case_000=False)]
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
    assert (cfg.results_path / "plots" / "01_excited_vs_neutral.png").exists()


def test_synthetic_bias_is_detected(cfg, mock_models, small_world, monkeypatch):
    monkeypatch.setenv("ZIVA_MOCK_BIAS", "12")
    scenarios, rows, _ = small_world
    run_manifest(cfg, rows, scenarios, mock_models, SYSTEM_PROMPT)
    result = analyze(cfg, scenarios)
    pooled = result["primary_pooled_excited_minus_neutral"]
    assert pooled["mean"] > 8
    assert pooled["p_perm"] < 0.01


def test_freeze_detects_modifications(cfg, mock_models, small_world):
    scenarios, rows, hashes = small_world
    scen_file = cfg.scenario_dir / "scenarios.json"
    record = build_freeze_record(cfg, mock_models, scen_file, manifest_path(cfg), hashes)
    write_freeze(cfg, record)

    ok = verify_freeze(cfg, scen_file, manifest_path(cfg), cfg.stimuli_dir)
    assert ok["ok"], ok["violations"]

    tampered = read_json(scen_file)
    tampered[0]["notes"] = "tampered"
    write_json(scen_file, tampered)
    bad = verify_freeze(cfg, scen_file, manifest_path(cfg), cfg.stimuli_dir)
    assert not bad["ok"]
    assert any("scenario_file" in v for v in bad["violations"])


def test_multistep_experiments_run(cfg, mock_models, small_world):
    scenarios, _, hashes = small_world
    cfg2 = cfg.model_copy(deep=True)
    cfg2.experiments.evidence_update = True
    cfg2.experiments.commitment = True
    cfg2.experiments.primary = False
    rows = build_manifest(cfg2, scenarios[:2], mock_models, hashes)
    assert {r["experiment"] for r in rows} == {"evidence_update", "commitment"}
    summary = run_manifest(cfg2, rows, scenarios, mock_models, SYSTEM_PROMPT)
    assert summary["executed_ok"] == len(rows)
    result = analyze(cfg2, scenarios)
    assert result["evidence_update"]
    assert "a_minus_b_final_probability" in result["commitment"]


def test_budget_stops_run(cfg, mock_models, small_world):
    from ziva.config import Pricing

    scenarios, rows, _ = small_world
    priced = [m.model_copy(update={"pricing": Pricing(input_per_mtok=1e9, output_per_mtok=1e9)})
              for m in mock_models]
    cfg2 = cfg.model_copy(deep=True)
    cfg2.run.max_cost_usd = 0.5
    summary = run_manifest(cfg2, rows, scenarios, priced, SYSTEM_PROMPT)
    assert summary["stopped_for_budget"]
    assert summary["executed_ok"] + summary["executed_with_errors"] < len(rows)


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
    from ziva.metrics import paired_table, trials_dataframe

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
