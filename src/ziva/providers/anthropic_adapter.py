"""Anthropic adapter, using the official `anthropic` SDK.

Structured output: `output_config.format` with a JSON schema on models that
support structured outputs (`supports.json_mode`); prompt-based JSON with
fallback parsing otherwise.

Sampling: recent Anthropic models (Opus 4.7+/Sonnet 5+) reject non-default
`temperature`; the parameter is only sent when the model config explicitly
enables `supports.temperature` AND a temperature is set. Otherwise it is
omitted and the omission is recorded in the trial's request parameters.

Refusals: a `stop_reason == "refusal"` is recorded verbatim; the empty/partial
content is stored and counted as a malformed response downstream.

Web-enabled condition: server-side `web_search` tool (secondary experiment
only; incompatible with schema-enforced output, so it uses fallback parsing).
"""

from __future__ import annotations

import base64
import time
from functools import cached_property

from .base import CompletionRequest, CompletionResult, ProviderAdapter, ProviderNotConfiguredError

WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search"}


class AnthropicAdapter(ProviderAdapter):
    @cached_property
    def _client(self):
        import anthropic

        key = self.model.api_key()
        if not key:
            raise ProviderNotConfiguredError(
                f"missing API key env var {self.model.key_env} for model {self.model.id}"
            )
        return anthropic.Anthropic(api_key=key)

    def _messages(self, request: CompletionRequest) -> list[dict]:
        out = []
        first_user = True
        for m in request.messages:
            if m["role"] == "user" and first_user and request.image_path is not None:
                data = base64.standard_b64encode(request.image_path.read_bytes()).decode()
                out.append({
                    "role": "user",
                    "content": [
                        {"type": "image",
                         "source": {"type": "base64", "media_type": "image/png", "data": data}},
                        {"type": "text", "text": m["content"]},
                    ],
                })
                first_user = False
            else:
                out.append({"role": m["role"], "content": m["content"]})
                if m["role"] == "user":
                    first_user = False
        return out

    def complete(self, request: CompletionRequest) -> CompletionResult:
        kwargs: dict = {
            "model": self.model.model,
            "max_tokens": request.max_tokens,
            "system": request.system,
            "messages": self._messages(request),
        }
        if request.temperature is not None and self.model.supports.temperature:
            kwargs["temperature"] = request.temperature

        structured_mode = "prompt"
        tools_available: list[str] = []
        if request.web_search:
            kwargs["tools"] = [WEB_SEARCH_TOOL]
            kwargs["max_tokens"] = max(request.max_tokens, 1500)
            tools_available = ["web_search"]
        elif request.json_schema and self.model.supports.json_mode:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": request.json_schema}}
            structured_mode = "provider_schema"

        t0 = time.monotonic()
        resp = self._client.messages.create(**kwargs)
        latency = time.monotonic() - t0

        text_parts = []
        tools_used = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "server_tool_use":
                tools_used.append(getattr(block, "name", "server_tool"))
        return CompletionResult(
            text="".join(text_parts),
            input_tokens=getattr(resp.usage, "input_tokens", None),
            output_tokens=getattr(resp.usage, "output_tokens", None),
            latency_s=round(latency, 3),
            model_version=resp.model,
            stop_reason=resp.stop_reason,
            structured_mode=structured_mode,
            tools_available=tools_available,
            tools_used=tools_used,
            raw={"id": resp.id},
        )
