"""Deterministic mock provider.

SYNTHETIC DATA ONLY. Used by unit tests and `ziva run --mock` smoke runs so the
entire pipeline can be exercised without API keys or cost. Every record it
produces is tagged provider="mock" and analysis/reports label such runs as
synthetic.

Behavior: a base probability is derived deterministically from the trial's
invariant hash (identical physical world -> identical base across treatments),
and an optional synthetic valence bias (ZIVA_MOCK_BIAS percentage points,
default 0) shifts the estimate toward the user's preferred outcome. With the
default bias of 0 the mock is perfectly valence-invariant, so analysis on mock
data should recover a null result -- itself a useful pipeline test.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

from .base import CompletionRequest, CompletionResult, ProviderAdapter


def _stable_int(*parts: str, mod: int) -> int:
    h = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return int(h, 16) % mod


class MockAdapter(ProviderAdapter):
    def complete(self, request: CompletionRequest) -> CompletionResult:
        meta = request.meta or {}
        invariant = str(meta.get("invariant_hash") or meta.get("scenario_id") or request.messages[-1]["content"][-300:])
        step = str(len(request.messages))  # revisions differ deterministically from first answers
        base = _stable_int("base", invariant, step, mod=101)

        bias = float(os.environ.get("ZIVA_MOCK_BIAS", "0"))
        preferred = (meta.get("directionality") or {}).get("preferred_outcome")
        prob = float(base)
        if preferred == "visible":
            prob += bias
        elif preferred == "not_visible":
            prob -= bias
        # small deterministic repeat-level jitter so repeated samples are not identical
        jitter = _stable_int("jitter", invariant, step, str(meta.get("repeat_index", 0)),
                             str(meta.get("treatment_id", "")), mod=5) - 2
        prob = max(0.0, min(100.0, prob + jitter))

        payload = {
            "visible_probability": int(round(prob)),
            "binary_prediction": "visible" if prob >= 50 else "not_visible",
            "confidence": 40 + _stable_int("conf", invariant, step, mod=41),
            "evidence_sufficiency": "sufficient" if not request.web_search else "partially_sufficient",
            "would_recommend_attempt": prob >= 35,
            "short_explanation": "SYNTHETIC mock response for pipeline testing; not a real model output.",
        }
        return CompletionResult(
            text=json.dumps(payload),
            input_tokens=len(request.system) // 4 + sum(len(m["content"]) for m in request.messages) // 4,
            output_tokens=len(json.dumps(payload)) // 4,
            latency_s=0.0,
            model_version="mock-1",
            stop_reason="end_turn",
            structured_mode="provider_schema",
            tools_available=["web_search"] if request.web_search else [],
            tools_used=[],
            raw={"mock": True, "bias": bias, "ts": time.time()},
        )
