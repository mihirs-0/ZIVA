"""Experiment and model configuration.

Experiment configs are human-editable YAML (configs/pilot.yaml etc.).
Model configs live in a separate YAML file so provider/model choices can be
swapped without touching the frozen experiment design. API keys come ONLY from
the environment (.env), never from config files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from .prompts import ELICITATION_MODES, EVIDENCE_MODES, SECONDARY_EVIDENCE_MODES
from .treatments import FAMILIES

PROVIDERS = ["openai", "anthropic", "google", "openai_compatible", "mock"]

DEFAULT_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openai_compatible": "OPENAI_API_KEY",
    "mock": None,
}


class Pricing(BaseModel):
    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0


class ModelSupports(BaseModel):
    json_mode: bool = True          # provider-enforced JSON/schema output
    images: bool = True             # multimodal input
    web_search: bool = False        # provider server-side web search tool
    temperature: bool = True        # accepts a temperature parameter


class ModelConfig(BaseModel):
    id: str
    provider: Literal["openai", "anthropic", "google", "openai_compatible", "mock"]
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    pricing: Pricing = Field(default_factory=Pricing)
    supports: ModelSupports = Field(default_factory=ModelSupports)
    max_concurrent: int = 4
    min_interval_s: float = 0.0
    max_tokens: int = 600
    temperature: float | None = None   # None -> omit the parameter entirely
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    # OpenAI-compatible servers such as vLLM accept these through extra_body.
    # Keeping them out of prompt text is essential for stimulus fidelity.
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    # Reproducibility metadata for locally served/open-weight checkpoints.
    # This is frozen and fingerprinted but is not sent to the model.
    runtime_metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def key_env(self) -> str | None:
        return self.api_key_env or DEFAULT_KEY_ENV[self.provider]

    def api_key(self) -> str | None:
        env = self.key_env
        return os.environ.get(env) if env else None

    def enabled(self) -> bool:
        """A model is runnable if it's the mock provider or its key is present."""
        return self.provider == "mock" or bool(self.api_key())


class ModelsFile(BaseModel):
    models: list[ModelConfig]


class ScenarioConfig(BaseModel):
    count: int = 30
    year: int = 2026
    seed: int = 20260815
    include_case_000: bool = True
    category_mix: dict[str, float] | None = None


class TreatmentConfig(BaseModel):
    families: list[str] = Field(default_factory=lambda: ["neutral", "excited_positive", "skeptical_negative"])
    variants_per_family: int = 2

    @field_validator("families")
    @classmethod
    def _known(cls, v: list[str]) -> list[str]:
        unknown = set(v) - set(FAMILIES)
        if unknown:
            raise ValueError(f"unknown treatment families: {sorted(unknown)}")
        return v


class SamplingConfig(BaseModel):
    repeats: int = 2
    temperature: float | None = 1.0     # applied only where the model config allows it
    max_tokens: int = 600


class RunConfig(BaseModel):
    max_cost_usd: float = 25.0
    concurrency: int = 4
    retries: int = 3
    retry_backoff_s: float = 2.0
    shuffle_seed: int = 7


class ExperimentsConfig(BaseModel):
    primary: bool = True
    evidence_update: bool = False
    commitment: bool = False
    web: bool = False


class ExperimentConfig(BaseModel):
    experiment_name: str = "pilot"
    scenarios: ScenarioConfig = Field(default_factory=ScenarioConfig)
    treatments: TreatmentConfig = Field(default_factory=TreatmentConfig)
    evidence_modes: list[str] = Field(default_factory=lambda: ["text", "structured"])
    elicitation_modes: list[str] = Field(default_factory=lambda: ["naturalistic", "separated"])
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    run: RunConfig = Field(default_factory=RunConfig)
    experiments: ExperimentsConfig = Field(default_factory=ExperimentsConfig)
    models_config: str = "configs/models.yaml"
    data_dir: str = "data"
    results_dir: str = "results"

    @field_validator("evidence_modes")
    @classmethod
    def _modes(cls, v: list[str]) -> list[str]:
        allowed = set(EVIDENCE_MODES) | set(SECONDARY_EVIDENCE_MODES)
        unknown = set(v) - allowed
        if unknown:
            raise ValueError(f"unknown evidence modes: {sorted(unknown)} (allowed: {sorted(allowed)})")
        return v

    @field_validator("elicitation_modes")
    @classmethod
    def _elicitations(cls, v: list[str]) -> list[str]:
        unknown = set(v) - set(ELICITATION_MODES)
        if unknown:
            raise ValueError(f"unknown elicitation modes: {sorted(unknown)} (allowed: {ELICITATION_MODES})")
        if not v:
            raise ValueError("at least one elicitation mode is required")
        return v

    # -- derived paths -----------------------------------------------------
    @property
    def scenario_dir(self) -> Path:
        return Path(self.data_dir) / "scenarios" / self.experiment_name

    @property
    def stimuli_dir(self) -> Path:
        return Path(self.data_dir) / "stimuli" / self.experiment_name

    @property
    def manifest_dir(self) -> Path:
        return Path(self.data_dir) / "manifests" / self.experiment_name

    @property
    def raw_dir(self) -> Path:
        return Path(self.data_dir) / "raw" / self.experiment_name

    @property
    def parsed_dir(self) -> Path:
        return Path(self.data_dir) / "parsed" / self.experiment_name

    @property
    def results_path(self) -> Path:
        return Path(self.results_dir) / self.experiment_name

    @property
    def freeze_path(self) -> Path:
        return self.manifest_dir / "freeze.json"


def load_env() -> None:
    load_dotenv(override=False)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    load_env()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return ExperimentConfig.model_validate(raw)


def load_models_config(path: str | Path) -> list[ModelConfig]:
    load_env()
    p = Path(path)
    if not p.exists():
        example = p.with_name("models.example.yaml")
        if example.exists():
            raise FileNotFoundError(
                f"{p} not found. Copy {example} to {p} and edit the model names/pricing."
            )
        raise FileNotFoundError(f"models config not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return ModelsFile.model_validate(raw).models


def enabled_models(models: list[ModelConfig]) -> list[ModelConfig]:
    return [m for m in models if m.enabled()]
