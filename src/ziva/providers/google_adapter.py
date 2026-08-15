"""Google Gemini adapter, using the official `google-genai` SDK.

Structured output: `response_mime_type="application/json"` +
`response_json_schema` where `supports.json_mode` is set.

Web-enabled condition: `google_search` grounding tool (secondary experiment;
incompatible with enforced JSON schema, so fallback parsing applies).
"""

from __future__ import annotations

import time
from functools import cached_property

from .base import CompletionRequest, CompletionResult, ProviderAdapter, ProviderNotConfiguredError


class GoogleAdapter(ProviderAdapter):
    @cached_property
    def _client(self):
        from google import genai

        key = self.model.api_key()
        if not key:
            raise ProviderNotConfiguredError(
                f"missing API key env var {self.model.key_env} for model {self.model.id}"
            )
        return genai.Client(api_key=key)

    def complete(self, request: CompletionRequest) -> CompletionResult:
        from google.genai import types

        contents = []
        first_user = True
        for m in request.messages:
            role = "user" if m["role"] == "user" else "model"
            parts = []
            if role == "user" and first_user and request.image_path is not None:
                parts.append(types.Part.from_bytes(data=request.image_path.read_bytes(), mime_type="image/png"))
                first_user = False
            elif role == "user":
                first_user = False
            parts.append(types.Part.from_text(text=m["content"]))
            contents.append(types.Content(role=role, parts=parts))

        cfg_kwargs: dict = {
            "system_instruction": request.system,
            "max_output_tokens": request.max_tokens,
        }
        if request.temperature is not None and self.model.supports.temperature:
            cfg_kwargs["temperature"] = request.temperature

        structured_mode = "prompt"
        tools_available: list[str] = []
        if request.web_search:
            cfg_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
            cfg_kwargs["max_output_tokens"] = max(request.max_tokens, 1500)
            tools_available = ["google_search"]
        elif request.json_schema and self.model.supports.json_mode:
            cfg_kwargs["response_mime_type"] = "application/json"
            cfg_kwargs["response_json_schema"] = request.json_schema
            structured_mode = "provider_schema"

        t0 = time.monotonic()
        resp = self._client.models.generate_content(
            model=self.model.model,
            contents=contents,
            config=types.GenerateContentConfig(**cfg_kwargs),
        )
        latency = time.monotonic() - t0

        tools_used = []
        try:
            cand = resp.candidates[0]
            if getattr(cand, "grounding_metadata", None) and getattr(
                cand.grounding_metadata, "web_search_queries", None
            ):
                tools_used.append("google_search")
        except Exception:
            pass
        usage = getattr(resp, "usage_metadata", None)
        return CompletionResult(
            text=resp.text or "",
            input_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
            output_tokens=getattr(usage, "candidates_token_count", None) if usage else None,
            latency_s=round(latency, 3),
            model_version=getattr(resp, "model_version", None) or self.model.model,
            stop_reason=str(getattr(resp.candidates[0], "finish_reason", None)) if resp.candidates else None,
            structured_mode=structured_mode,
            tools_available=tools_available,
            tools_used=tools_used,
            raw={"response_id": getattr(resp, "response_id", None)},
        )
