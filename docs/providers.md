# Provider adapters

Configuration lives in `configs/models.yaml` (copy from `configs/models.example.yaml`).
Model names are plain strings — nothing in the harness is tied to specific models, so
swap them freely as providers ship new ones. API keys come only from the environment
(`.env`); a model whose key is missing is skipped gracefully and reported, so the
benchmark degrades cleanly when only one or two providers are configured.

| provider | SDK | structured output | images | web tool | notes |
|---|---|---|---|---|---|
| `openai` | `openai` | Chat Completions `response_format: json_schema` (strict) | base64 data-URI parts | Responses API `web_search` | |
| `openai_compatible` | `openai` | same, if the vendor supports it | same | usually no | set `base_url` + `api_key_env` (e.g. xAI/Grok at `https://api.x.ai/v1`) |
| `anthropic` | `anthropic` | `output_config.format` JSON schema | base64 image blocks | `web_search` server tool | recent models reject non-default `temperature` — keep `supports.temperature: false` so the parameter is omitted; `stop_reason: refusal` is recorded and counted as a failure downstream |
| `google` | `google-genai` | `response_mime_type: application/json` + `response_json_schema` | inline `Part.from_bytes` | `google_search` grounding | |
| `mock` | none | deterministic synthetic JSON | n/a | n/a | for tests and `--mock` smoke runs; outputs are labelled SYNTHETIC everywhere |

Behavior common to all adapters:

* `supports.json_mode: false` disables provider-enforced schemas; the harness then
  relies on prompt instructions plus the robust fallback parser (`ziva.parsing`).
  Either way the parse status and raw text are stored; nothing is repaired by an LLM.
* The web-enabled condition disables schema enforcement (provider tools and enforced
  schemas conflict) and records which tools were available and observably used.
* Per-model `max_concurrent` and `min_interval_s` implement provider-specific rate
  limiting; the runner adds global concurrency, retries with exponential backoff, and
  a live budget guard on actual reported usage.
* Every trial record stores the request parameters, structured-output mode actually
  used, reported token usage, latency, provider response IDs, and the model version
  string the provider returned.

## Live validation status

Adapters are written against the official SDK contracts and covered by mocked
contract tests (`tests/test_providers.py`); they have **not been exercised against
live APIs in this repository** because no credentials were available at build time.
The first live pilot run will validate them; the mock provider exercises every other
part of the pipeline end-to-end.
