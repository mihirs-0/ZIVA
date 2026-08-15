"""Robust structured-output parsing.

Provider JSON/schema modes are used where available; regardless, every raw
response goes through this parser. Malformed outputs are never discarded
silently: the raw text, parser status, and validation errors are all stored,
and malformed responses are counted in the failure statistics.

No LLM is ever used to repair responses in the primary pipeline.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator


class ZivaResponse(BaseModel):
    visible_probability: int = Field(ge=0, le=100)
    binary_prediction: str
    confidence: int = Field(ge=0, le=100)
    evidence_sufficiency: str
    would_recommend_attempt: bool
    short_explanation: str = ""

    @field_validator("binary_prediction", mode="before")
    @classmethod
    def _norm_binary(cls, v: Any) -> str:
        s = str(v).strip().lower().replace(" ", "_").replace("-", "_")
        if s in ("visible", "yes", "true"):
            return "visible"
        if s in ("not_visible", "invisible", "no", "false", "notvisible"):
            return "not_visible"
        raise ValueError(f"unparseable binary_prediction: {v!r}")

    @field_validator("evidence_sufficiency", mode="before")
    @classmethod
    def _norm_suff(cls, v: Any) -> str:
        s = str(v).strip().lower().replace(" ", "_").replace("-", "_")
        if s in ("sufficient", "partially_sufficient", "insufficient"):
            return s
        if "partial" in s:
            return "partially_sufficient"
        if s.startswith("insuff"):
            return "insufficient"
        if s.startswith("suff"):
            return "sufficient"
        raise ValueError(f"unparseable evidence_sufficiency: {v!r}")

    @field_validator("visible_probability", "confidence", mode="before")
    @classmethod
    def _integer_percent_policy(cls, v: Any) -> int:
        """Explicit, frozen normalization policy for the primary benchmark.

        The schema requires integers 0-100. Integer-valued floats (40.0) are
        accepted as their integer. Non-integer floats (0.4, 62.5) are REJECTED
        as malformed rather than heuristically rescaled: 1.0 is ambiguous
        between 1% and a 100% fraction, so silent guessing is not reproducible.
        Rejections are counted in failure statistics like any malformed output.
        """
        if isinstance(v, bool):
            raise ValueError("boolean is not a probability")
        if isinstance(v, float):
            if v.is_integer():
                return int(v)
            raise ValueError(
                f"non-integer probability {v!r}: the schema requires integers on a 0-100 "
                "scale; fractional values are rejected rather than rescaled"
            )
        return v


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_candidates(text: str) -> list[str]:
    """Yield plausible JSON object substrings, best-first."""
    candidates: list[str] = []
    stripped = text.strip()
    if stripped.startswith("{"):
        candidates.append(stripped)
    for m in _FENCE_RE.finditer(text):
        candidates.append(m.group(1).strip())
    # first balanced {...} span
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break
    return candidates


def parse_response(raw_text: str) -> dict:
    """Parse and validate a raw completion.

    Returns {"status": "ok"|"json_error"|"validation_error"|"empty",
             "parsed": dict|None, "errors": [..]}
    """
    if not raw_text or not raw_text.strip():
        return {"status": "empty", "parsed": None, "errors": ["empty response"]}
    errors: list[str] = []
    for candidate in extract_json_candidates(raw_text):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError as e:
            errors.append(f"json: {e}")
            continue
        if not isinstance(obj, dict):
            errors.append("json: top-level value is not an object")
            continue
        try:
            model = ZivaResponse.model_validate(obj)
            return {"status": "ok", "parsed": model.model_dump(), "errors": []}
        except ValidationError as e:
            errors.append(f"validation: {e.error_count()} errors: {e.errors()[:3]}")
    status = "validation_error" if any(e.startswith("validation") for e in errors) else "json_error"
    return {"status": status, "parsed": None, "errors": errors or ["no JSON object found"]}
