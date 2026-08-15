from __future__ import annotations

import pytest

from ziva.config import ExperimentConfig, ModelConfig
from ziva.scenarios import generate_scenarios


@pytest.fixture(scope="session")
def scenarios():
    """Small deterministic scenario set shared across tests (SYNTHETIC fixture)."""
    return [s.to_dict() for s in generate_scenarios(count=12, seed=99, include_case_000=True)]


@pytest.fixture()
def mock_models():
    return [
        ModelConfig(id="mock_a", provider="mock", model="mock-1"),
        ModelConfig(id="mock_b", provider="mock", model="mock-1"),
    ]


@pytest.fixture()
def cfg(tmp_path):
    return ExperimentConfig(
        experiment_name="testexp",
        scenarios={"count": 8, "seed": 5, "include_case_000": False},
        treatments={"families": ["neutral", "excited_positive", "skeptical_negative"],
                    "variants_per_family": 2},
        evidence_modes=["text", "structured"],
        sampling={"repeats": 2, "temperature": 1.0, "max_tokens": 400},
        run={"max_cost_usd": 5.0, "concurrency": 2, "retries": 1, "shuffle_seed": 3},
        data_dir=str(tmp_path / "data"),
        results_dir=str(tmp_path / "results"),
        models_config=str(tmp_path / "models.yaml"),
    )
