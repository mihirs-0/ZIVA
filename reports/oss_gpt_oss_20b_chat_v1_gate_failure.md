# GPT-OSS-20B: ZIVA-Chat-v1 engineering-gate failure

## Decision

The frozen neutral-only Chat-v1 engineering gate did not pass. Per protocol, no
non-neutral treatment data were collected, the 512-trial run was not started,
and the analysis and cross-model comparison pipelines were not run.

## What passed before inference

- The official `openai/gpt-oss-20b` checkpoint was pinned to revision
  `6cee5e81ee83917806bbde320786a8fb61efebee`.
- The checkpoint loaded with its native MXFP4 MoE weights and BF16 activations
  under vLLM 0.26.0 using the official Harmony template and the native
  `openai_gptoss` reasoning parser.
- Tensor parallelism used both RTX 3090 GPUs. The ready audit recorded 22,317
  MiB and 22,319 MiB used.
- The generated 512-trial manifest had zero pairing failures and SHA-256
  `945cd00d860d56f98a448082d1b98c53d29660b6887a6f258ab3bd64bb941ccd`,
  byte-identical to the frozen Qwen3-14B Chat-v1 manifest.
- Canonical fact rendering was exact, prohibited prompt cues were absent, and
  the Turn-2 follow-up was byte-identical across the manifest.
- The global Chat-v1 protocol fingerprint remained
  `cf6f58a1cc0f6bd4de504d3d2542a71bccacb8c243694346da21559b82e5b13f`.

## Gate failure

The unchanged gate selected its prescribed 16 neutral trials with concurrency
8, Turn-1 maximum 2048 tokens, Turn-2 maximum 16 tokens, three retries, and the
existing 300-second local request timeout.

All 16 trial requests completed at the transport level, but no Turn-2 response
contained final-answer text. Every Turn-2 response used 16 completion tokens,
finished with reason `length`, and returned empty final content. Therefore the
frozen parser found zero percentages, neither anchor yielded a percentage, and
anchor ordering was not assessable.

Turn 1 independently exceeded its engineering bound: 5 of 16 responses
(31.25%) finished for `length`, above the frozen 5% maximum, and 3 of those
records contained no final-answer text. The gate consequently failed on parse
rate, both truncation criteria, and anchor sanity. The prompts, budgets, parser,
and model reasoning behavior were not tuned after observing this result.

## Reproducibility and preservation

- Experiment fingerprint:
  `718f78d3563f0c98d3b61aef3091020a47edfbf6a784e2678d8e0f778cbc7dee`
- Exact 16-trial gate record:
  `data/manifests/oss_gpt_oss_20b_chat_v1/neutral_smoke.json`
- Machine-readable gate failure:
  `data/manifests/oss_gpt_oss_20b_chat_v1/neutral_gate_failure.json`
- Per-model freeze and full 512-trial manifest:
  `data/manifests/oss_gpt_oss_20b_chat_v1/`
- Full service audit:
  `data/manifests/oss_model_registry/gpt_oss_20b_service.json`
- Complete vLLM log:
  `data/manifests/oss_model_registry/gpt_oss_20b_vllm.log`
- Raw treatment records: 0
- Analysis outputs: 0

The pinned checkpoint cache and ready vLLM service remain installed on
`mihir@idli` for reproducibility. No further inference was performed after the
failed neutral gate.
