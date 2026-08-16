"""Provider adapter contract tests (mocked SDK clients; no credits consumed)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ziva.config import ModelConfig
from ziva.prompts import RESPONSE_JSON_SCHEMA
from ziva.providers.base import CompletionRequest
from ziva.providers.mock import MockAdapter


def _request(model, **over):
    base = dict(
        model=model,
        system="sys",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=300,
        temperature=1.0,
        json_schema=RESPONSE_JSON_SCHEMA,
        meta={"invariant_hash": "abc", "directionality": {}, "repeat_index": 0},
    )
    base.update(over)
    return CompletionRequest(**base)


def test_mock_adapter_valence_invariant_by_default(monkeypatch):
    monkeypatch.delenv("ZIVA_MOCK_BIAS", raising=False)
    m = ModelConfig(id="m", provider="mock", model="mock-1")
    a = MockAdapter(m)
    r1 = a.complete(_request(m, meta={"invariant_hash": "same", "directionality": {"preferred_outcome": "visible"}}))
    r2 = a.complete(_request(m, meta={"invariant_hash": "same", "directionality": {"preferred_outcome": None}}))
    p1 = json.loads(r1.text)["visible_probability"]
    p2 = json.loads(r2.text)["visible_probability"]
    assert abs(p1 - p2) <= 4  # only deterministic jitter differs


def test_openai_adapter_request_shape(monkeypatch):
    from ziva.providers.openai_adapter import OpenAIAdapter

    m = ModelConfig(id="oai", provider="openai", model="gpt-test",
                    supports={"json_mode": True, "images": True, "temperature": True})
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    adapter = OpenAIAdapter(m)

    fake_resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"x":1}'), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        model="gpt-test-2026",
        id="resp1",
        system_fingerprint="fp",
    )
    client = MagicMock()
    client.chat.completions.create.return_value = fake_resp
    adapter.__dict__["_client"] = client

    result = adapter.complete(_request(m))
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-test"
    assert kwargs["messages"][0] == {"role": "system", "content": "sys"}
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["schema"] == RESPONSE_JSON_SCHEMA
    assert kwargs["temperature"] == 1.0
    assert result.structured_mode == "provider_schema"
    assert result.input_tokens == 10


def test_openai_compatible_sampling_and_nonthinking_are_transport_metadata(monkeypatch):
    """OSS inference controls must be sent out-of-band, never appended to stimuli."""
    from ziva.providers.openai_adapter import OpenAIAdapter

    m = ModelConfig(
        id="qwen",
        provider="openai_compatible",
        model="Qwen/Qwen3-14B",
        api_key_env="VLLM_API_KEY",
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        chat_template_kwargs={"enable_thinking": False},
    )
    monkeypatch.setenv("VLLM_API_KEY", "EMPTY")
    adapter = OpenAIAdapter(m)
    fake_resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"x":1}'), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        model="Qwen/Qwen3-14B", id="qwen1", system_fingerprint=None,
    )
    client = MagicMock()
    client.chat.completions.create.return_value = fake_resp
    adapter.__dict__["_client"] = client

    adapter.complete(_request(m, temperature=0.7))
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
    assert "/no_think" not in json.dumps(kwargs["messages"])
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.8
    assert kwargs["presence_penalty"] == 0.0
    assert kwargs["extra_body"] == {
        "top_k": 20,
        "min_p": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_anthropic_adapter_omits_temperature_when_unsupported(monkeypatch):
    from ziva.providers.anthropic_adapter import AnthropicAdapter

    m = ModelConfig(id="ant", provider="anthropic", model="claude-test",
                    supports={"json_mode": True, "images": True, "temperature": False})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    adapter = AnthropicAdapter(m)

    fake_resp = SimpleNamespace(
        content=[SimpleNamespace(type="text", text='{"x":1}')],
        usage=SimpleNamespace(input_tokens=11, output_tokens=6),
        model="claude-test-1",
        stop_reason="end_turn",
        id="msg1",
    )
    client = MagicMock()
    client.messages.create.return_value = fake_resp
    adapter.__dict__["_client"] = client

    result = adapter.complete(_request(m))
    kwargs = client.messages.create.call_args.kwargs
    assert "temperature" not in kwargs, "must omit temperature for models that reject it"
    assert kwargs["system"] == "sys"
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    assert result.stop_reason == "end_turn"


def test_anthropic_adapter_records_refusal(monkeypatch):
    from ziva.providers.anthropic_adapter import AnthropicAdapter

    m = ModelConfig(id="ant", provider="anthropic", model="claude-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    adapter = AnthropicAdapter(m)
    fake_resp = SimpleNamespace(
        content=[], usage=SimpleNamespace(input_tokens=5, output_tokens=0),
        model="claude-test-1", stop_reason="refusal", id="msg2",
    )
    client = MagicMock()
    client.messages.create.return_value = fake_resp
    adapter.__dict__["_client"] = client
    result = adapter.complete(_request(m))
    assert result.stop_reason == "refusal"
    assert result.text == ""


def test_google_adapter_schema_mode(monkeypatch):
    from ziva.providers.google_adapter import GoogleAdapter

    m = ModelConfig(id="goo", provider="google", model="gemini-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    adapter = GoogleAdapter(m)

    fake_resp = SimpleNamespace(
        text='{"x":1}',
        candidates=[SimpleNamespace(finish_reason="STOP", grounding_metadata=None)],
        usage_metadata=SimpleNamespace(prompt_token_count=9, candidates_token_count=4),
        model_version="gemini-test-001",
        response_id="g1",
    )
    client = MagicMock()
    client.models.generate_content.return_value = fake_resp
    adapter.__dict__["_client"] = client

    result = adapter.complete(_request(m))
    kwargs = client.models.generate_content.call_args.kwargs
    assert kwargs["model"] == "gemini-test"
    assert kwargs["config"].response_mime_type == "application/json"
    assert result.structured_mode == "provider_schema"
    assert result.model_version == "gemini-test-001"


def test_missing_key_raises(monkeypatch):
    from ziva.providers.base import ProviderNotConfiguredError
    from ziva.providers.openai_adapter import OpenAIAdapter

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    m = ModelConfig(id="oai", provider="openai", model="gpt-test")
    adapter = OpenAIAdapter(m)
    with pytest.raises(ProviderNotConfiguredError):
        adapter.complete(_request(m))


def test_image_attachment_openai(monkeypatch, tmp_path):
    from ziva.providers.openai_adapter import OpenAIAdapter

    m = ModelConfig(id="oai", provider="openai", model="gpt-test")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    adapter = OpenAIAdapter(m)
    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    fake_resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}"), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1), model="m", id="i",
        system_fingerprint=None,
    )
    client = MagicMock()
    client.chat.completions.create.return_value = fake_resp
    adapter.__dict__["_client"] = client

    adapter.complete(_request(m, image_path=img))
    content = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
