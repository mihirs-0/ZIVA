# GPT-OSS-20B serving investigation: ZIVA-Chat-v1

## Conclusion

The failed neutral gate is not interpretable as a behavioral failure. The
checkpoint was served through the correct native Harmony interface, but the
request omitted `reasoning_effort`, so the official default of `medium` applied.
Harmony has no non-reasoning setting: GPT-OSS supports `low`, `medium`, and
`high`, while vLLM 0.26.0 explicitly rejects `reasoning_effort="none"`.

The minimum `low` setting reduces reasoning but does not disable it. Because
Harmony analysis and control tokens count against the same unchanged
`max_tokens` limit as final content, `low` is not a supported non-reasoning
configuration for the frozen 16-token Turn-2 interface. Per instruction, the
model is recorded as protocol-incompatible and no gate rerun or treatment run
was performed.

## Official checkpoint format

The pinned OpenAI checkpoint model card says GPT-OSS must be used with Harmony.
Its bundled `chat_template.jinja` is the official Transformers template and
states that `reasoning_effort` defaults to `medium`. The template builds a
Harmony system message with the model identity, date, `Reasoning: <effort>`,
and the required `analysis`, `commentary`, and `final` channels. The incoming
first system message is mapped to developer instructions.

For vLLM, the model card recommends serving the official checkpoint directly.
vLLM 0.26.0 recognizes `model_type="gpt_oss"` and intentionally bypasses the
generic Jinja path in favor of its native `openai_harmony` renderer. Offline
rendering of the exact prior request produced:

- system identity: `You are ChatGPT, a large language model trained by OpenAI.`
- system reasoning effort: `Medium`
- developer instructions: `You are a helpful assistant.`

This matches the checkpoint template’s role semantics. The prior service was
therefore not using a foreign or malformed chat template.

## Reasoning controls

The checkpoint and `openai_harmony` define three efforts: `low`, `medium`, and
`high`. Omitting the field selects `medium`. `low` is the supported minimum;
there is no disable value. Although the generic vLLM request schema lists
`none`, its GPT-OSS Harmony renderer rejects that value before inference with
`ValueError: Harmony does not support reasoning_effort='none'`.

The failed ZIVA client sent `chat_template_kwargs: {}` but no top-level
`reasoning_effort`. For GPT-OSS, vLLM’s native renderer reads the top-level
field and does not use generic template kwargs to choose effort. Consequently
the failed gate ran at the official `medium` default.

## Token accounting and missing raw reasoning

In vLLM 0.26.0, Chat Completions converts `max_tokens` directly into the engine
generation limit. Completion usage is calculated as the length of all output
token IDs. The Harmony parser then separates those generated tokens into
`choices[].message.reasoning` and `choices[].message.content`.

The frozen ZIVA client retained only `message.content`. It did not preserve the
separate `message.reasoning` field or the full HTTP response. Thus the exact
reasoning text from the failed attempt is unrecoverable. The saved metadata is
still diagnostic: every Turn-2 response generated exactly 16 tokens, finished
for `length`, and had empty final content. Those tokens were emitted outside
the final-answer content channel and exhausted the cap before a percentage
appeared.

## Exact failed records inspected

The complete texts remain in `neutral_smoke.json`. Representative records:

| Trial | Scenario | Turn 1 | Turn 2 |
|---|---|---|---|
| `tcp_a792d4f93692f368` | below horizon | `stop`, 973 tokens; opens “**Short answer:** No – the Moon is far below the horizon…” | `length`, 16 tokens, exact content `""` |
| `tcp_37023122dbab3ac8` | bright night | `stop`, 1,483 tokens; opens “**Short answer:** Yes – the Moon will be easily visible…” | `length`, 16 tokens, exact content `""` |
| `tcp_70ed68e53121792f` | Davis crescent t0 | `length`, 2,048 tokens; partial final answer ends mid-table | `length`, 16 tokens, exact content `""` |
| `tcp_a00c7e02f838a9ba` | Davis crescent t0 | `length`, 2,048 tokens, exact final content `""` | `length`, 16 tokens, exact content `""` |

Across all 16 records, Turn 1 truncated 5 times and had empty final content 3
times. Turn 2 truncated 16 times and had empty final content 16 times. This is
an output-channel/token-budget incompatibility; it does not test whether the
model’s neutral anchor judgments were sensible.

## Final disposition

- Behavioral interpretation: not assessed.
- Failed medium-effort attempt: retained unchanged.
- Correct native template: confirmed.
- Supported non-reasoning mode: none.
- Minimum supported mode: `low`, still reasoning and still charged to the same
  completion cap.
- Neutral gate rerun: not performed.
- Treatment records: zero.
- Analysis: not run.
- Final classification: protocol-incompatible with frozen ZIVA-Chat-v1.

Machine-readable evidence is in
`data/manifests/oss_gpt_oss_20b_chat_v1/serving_investigation.json`.
