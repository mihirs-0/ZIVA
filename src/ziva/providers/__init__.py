"""Provider adapter registry."""

from __future__ import annotations

from ..config import ModelConfig
from .base import CompletionRequest, CompletionResult, ProviderAdapter, ProviderNotConfiguredError

__all__ = [
    "CompletionRequest",
    "CompletionResult",
    "ProviderAdapter",
    "ProviderNotConfiguredError",
    "get_adapter",
]

_ADAPTER_CACHE: dict[str, ProviderAdapter] = {}


def get_adapter(model: ModelConfig) -> ProviderAdapter:
    """Return (and cache) the adapter instance for a configured model."""
    if model.id in _ADAPTER_CACHE:
        return _ADAPTER_CACHE[model.id]
    if model.provider == "mock":
        from .mock import MockAdapter

        adapter: ProviderAdapter = MockAdapter(model)
    elif model.provider in ("openai", "openai_compatible"):
        from .openai_adapter import OpenAIAdapter

        adapter = OpenAIAdapter(model)
    elif model.provider == "anthropic":
        from .anthropic_adapter import AnthropicAdapter

        adapter = AnthropicAdapter(model)
    elif model.provider == "google":
        from .google_adapter import GoogleAdapter

        adapter = GoogleAdapter(model)
    else:  # pragma: no cover - config validation prevents this
        raise ValueError(f"unknown provider: {model.provider}")
    _ADAPTER_CACHE[model.id] = adapter
    return adapter
