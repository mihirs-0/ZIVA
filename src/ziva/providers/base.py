"""Provider-agnostic adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ..config import ModelConfig


@dataclass
class CompletionRequest:
    model: ModelConfig
    system: str
    messages: list[dict]                 # [{"role": "user"|"assistant", "content": str}, ...]
    max_tokens: int = 600
    temperature: float | None = None     # None -> parameter omitted entirely
    json_schema: dict | None = None      # provider-enforced schema where supported
    image_path: Path | None = None       # attached to the first user message
    web_search: bool = False             # secondary web-enabled condition only
    meta: dict = field(default_factory=dict)   # trial metadata; ignored by real providers


@dataclass
class CompletionResult:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_s: float | None = None
    model_version: str | None = None
    stop_reason: str | None = None
    structured_mode: str = "prompt"      # "provider_schema" | "provider_json" | "prompt"
    tools_available: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)   # provider payload (JSON-serializable subset)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_s": self.latency_s,
            "model_version": self.model_version,
            "stop_reason": self.stop_reason,
            "structured_mode": self.structured_mode,
            "tools_available": self.tools_available,
            "tools_used": self.tools_used,
            "raw": self.raw,
        }


class ProviderAdapter(ABC):
    """One instance per configured model."""

    def __init__(self, model: ModelConfig):
        self.model = model

    @abstractmethod
    def complete(self, request: CompletionRequest) -> CompletionResult:
        ...


class ProviderNotConfiguredError(RuntimeError):
    pass
