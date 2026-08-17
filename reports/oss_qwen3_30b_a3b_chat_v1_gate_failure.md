# Qwen3-30B-A3B: ZIVA-Chat-v1 engineering-gate failure

## Decision

The official `Qwen/Qwen3-30B-A3B` checkpoint completed the frozen Structured
benchmark successfully, but its final frozen neutral-only Chat-v1 engineering
gate did not pass. Per protocol, the final 512-trial Chat-v1 treatment run was
not started, and Chat-v1, within-model, and Chat cross-model analyses were not
run.

## What passed

- The checkpoint was pinned to revision
  `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` and all 16 BF16 safetensor
  shards were verified in the local cache.
- vLLM 0.26.0 loaded `Qwen3MoeForCausalLM` as unquantized BF16 with tensor
  parallelism across both RTX 3090 GPUs.
- The checkpoint's native tokenizer template was used with the official
  `enable_thinking=false` template argument. No model-facing reasoning text or
  `/no_think` marker was added.
- The unchanged Structured neutral gate passed, and the complete 768-trial
  Structured experiment finished with 768 valid parses, zero malformed
  records, zero thinking markers, and zero execution errors. Its standard
  report and runtime audit are banked.
- The generated Chat-v1 manifest contained 512 trials with zero pairing
  failures and retained global protocol fingerprint
  `cf6f58a1cc0f6bd4de504d3d2542a71bccacb8c243694346da21559b82e5b13f`.

## First Chat runtime attempt

The first Chat runtime used 16 GiB of UVA CPU weight offload per GPU. Its
neutral gate passed with 16/16 parsed percentages, no Turn-1 truncations, one
Turn-2 truncation, sane 0%/95% anchors, exact prompt fidelity, and no thinking
content.

The treatment run then exposed an engineering throughput failure at the
unchanged 300-second local request boundary. Of the first 26 banked records, 20
completed and 6 exhausted all three retries with `TimeoutError('timed out')`.
This partial dataset is excluded from every analysis and preserved under
`data/engineering_failures/oss_qwen3_30b_a3b_chat_v1_timeout_attempt1/`.

## Final neutral-gate failure

A second runtime retained the identical checkpoint, BF16 weights, TP2, native
template, prompts, sampling, token caps, parser, anchors, and gate thresholds.
Only memory placement changed: CPU offload was reduced to 10 GiB per GPU,
`max_num_seqs` was set to the protocol's actual concurrency of 8, and the KV
cache was explicitly set to 1.5 GiB per GPU. The ready audit recorded 21,143
MiB on each card.

The unchanged 16-trial neutral gate produced:

- percentage parses: 16/16 (100%)
- Turn-1 truncations: 0/16 (0%)
- Turn-2 truncations: 2/16 (12.5%), above the frozen 10% maximum
- below-horizon anchor: 0% in all four cells
- bright-night anchor: 95% in all four cells
- thinking markers: 0
- canonical fact rendering, prompt-cue absence, pairing, and byte-identical
  Turn-2 follow-up checks: passed

Because the Turn-2 truncation rate exceeded the frozen threshold, the gate
failed and collection stopped. No treatment record was collected under this
final runtime.

## Reproducibility and preservation

- Final experiment fingerprint:
  `261d2680d89442f72a834d64c78aa5617a4ef106d7e1ffd4d71987cdd1eb3203`
- Exact failed gate: `data/manifests/oss_qwen3_30b_a3b_chat_v1/neutral_smoke.json`
- Machine-readable failure record:
  `data/manifests/oss_qwen3_30b_a3b_chat_v1/neutral_gate_failure.json`
- Final service audit and log:
  `data/manifests/oss_model_registry/qwen3_30b_a3b_service_attempt2.json` and
  `qwen3_30b_a3b_vllm_attempt2.log`
- First service audit and log:
  `data/manifests/oss_model_registry/qwen3_30b_a3b_service.json` and
  `qwen3_30b_a3b_vllm.log`
- Structured report: `reports/oss_qwen3_30b_a3b_structured_report.md`

The final vLLM service was stopped after the failed gate. No further inference
or downstream Chat analysis was performed.
