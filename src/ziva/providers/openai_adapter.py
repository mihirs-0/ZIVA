"""OpenAI (and OpenAI-compatible) adapter.

Uses the official `openai` SDK. `openai_compatible` provider entries (e.g.
xAI/Grok at https://api.x.ai/v1) reuse this adapter with a `base_url` and
`api_key_env` override.

Structured output: Chat Completions `response_format` with a strict JSON
schema where `supports.json_mode` is set; otherwise prompt-based JSON with
fallback parsing.

Web-enabled condition: uses the Responses API `web_search` tool. This path is
best-effort across OpenAI-compatible vendors and is only exercised by the
separate secondary web experiment.
"""

from __future__ import annotations

import base64
import time
from functools import cached_property

from .base import CompletionRequest, CompletionResult, ProviderAdapter, ProviderNotConfiguredError


class OpenAIAdapter(ProviderAdapter):
    @cached_property
    def _client(self):
        from openai import OpenAI

        key = self.model.api_key()
        if not key:
            raise ProviderNotConfiguredError(
                f"missing API key env var {self.model.key_env} for model {self.model.id}"
            )
        kwargs = {"api_key": key}
        if self.model.base_url:
            kwargs["base_url"] = self.model.base_url
        return OpenAI(**kwargs)

    def _content_for_user(self, text: str, request: CompletionRequest, first_user: bool):
        if first_user and request.image_path is not None:
            data = base64.standard_b64encode(request.image_path.read_bytes()).decode()
            return [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
                {"type": "text", "text": text},
            ]
        return text

    def complete(self, request: CompletionRequest) -> CompletionResult:
        if request.web_search:
            return self._complete_web(request)

        messages = [{"role": "system", "content": request.system}]
        first_user = True
        for m in request.messages:
            if m["role"] == "user":
                messages.append({"role": "user", "content": self._content_for_user(m["content"], request, first_user)})
                first_user = False
            else:
                messages.append({"role": "assistant", "content": m["content"]})

        kwargs: dict = {
            "model": self.model.model,
            "messages": messages,
            "max_completion_tokens": request.max_tokens,
        }
        if request.temperature is not None and self.model.supports.temperature:
            kwargs["temperature"] = request.temperature
        if self.model.top_p is not None:
            kwargs["top_p"] = self.model.top_p
        if self.model.presence_penalty is not None:
            kwargs["presence_penalty"] = self.model.presence_penalty
        extra_body: dict = {}
        if self.model.top_k is not None:
            extra_body["top_k"] = self.model.top_k
        if self.model.min_p is not None:
            extra_body["min_p"] = self.model.min_p
        if self.model.chat_template_kwargs:
            extra_body["chat_template_kwargs"] = self.model.chat_template_kwargs
        if extra_body:
            kwargs["extra_body"] = extra_body
        structured_mode = "prompt"
        if request.json_schema and self.model.supports.json_mode:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "ziva_response", "strict": True, "schema": request.json_schema},
            }
            structured_mode = "provider_schema"

        t0 = time.monotonic()
        resp = self._client.chat.completions.create(**kwargs)
        latency = time.monotonic() - t0
        choice = resp.choices[0]
        usage = resp.usage
        return CompletionResult(
            text=choice.message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            latency_s=round(latency, 3),
            model_version=resp.model,
            stop_reason=choice.finish_reason,
            structured_mode=structured_mode,
            raw={
                "id": resp.id,
                "system_fingerprint": getattr(resp, "system_fingerprint", None),
                "generation_settings": {
                    "temperature": request.temperature if self.model.supports.temperature else None,
                    "top_p": self.model.top_p,
                    "top_k": self.model.top_k,
                    "min_p": self.model.min_p,
                    "presence_penalty": self.model.presence_penalty,
                    "chat_template_kwargs": self.model.chat_template_kwargs,
                },
            },
        )

    def _complete_web(self, request: CompletionRequest) -> CompletionResult:
        input_items = [{"role": "system", "content": request.system}] + [
            {"role": m["role"], "content": m["content"]} for m in request.messages
        ]
        t0 = time.monotonic()
        resp = self._client.responses.create(
            model=self.model.model,
            tools=[{"type": "web_search"}],
            input=input_items,
            max_output_tokens=max(request.max_tokens, 1200),
        )
        latency = time.monotonic() - t0
        tools_used = []
        try:
            for item in resp.output:
                if getattr(item, "type", "") == "web_search_call":
                    tools_used.append("web_search")
        except Exception:
            pass
        usage = getattr(resp, "usage", None)
        return CompletionResult(
            text=getattr(resp, "output_text", "") or "",
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
            latency_s=round(latency, 3),
            model_version=getattr(resp, "model", None),
            stop_reason=getattr(resp, "status", None),
            structured_mode="prompt",
            tools_available=["web_search"],
            tools_used=tools_used,
            raw={"id": getattr(resp, "id", None)},
        )
