# DeepSeek-R1-Distill-Qwen-32B: ZIVA-Chat-v1 engineering-gate failure

## Decision

The frozen neutral-only Chat-v1 engineering gate did not pass. Per protocol, no
non-neutral treatment data were collected, the 512-trial run was not started,
and the analysis and cross-model comparison pipelines were not run.

## What passed before inference

- The official `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` checkpoint was pinned
  to revision `711ad2ea6aa40cfca18895e8aca02ab92df1a746`.
- The 32.764B-parameter checkpoint loaded as unquantized BF16 with vLLM 0.26.0.
- Tensor parallelism used both RTX 3090 GPUs. The ready audit recorded 22,243
  MiB on each card.
- vLLM used 16 GiB of UVA CPU weight offload per GPU. Its V1 model runner was
  required because the auto-selected V2 runner did not initialize the offloader.
- The generated 512-trial manifest had zero pairing failures and SHA-256
  `945cd00d860d56f98a448082d1b98c53d29660b6887a6f258ab3bd64bb941ccd`,
  byte-identical to the frozen Qwen3-14B Chat-v1 manifest.
- The global Chat-v1 protocol fingerprint remained
  `cf6f58a1cc0f6bd4de504d3d2542a71bccacb8c243694346da21559b82e5b13f`.

## Gate failure

The unchanged gate selected its prescribed 16 neutral trials with concurrency
8, Turn-1 maximum 2048 tokens, Turn-2 maximum 16 tokens, three retries, and the
existing 300-second local request timeout. Stable aggregate decoding throughput
was approximately 4.8 tokens/second because substantial BF16 weights were read
over PCIe through UVA offload.

No Turn-1 request completed within the frozen request boundary. All 16 neutral
trials exhausted all three attempts before reaching Turn 2. Consequently there
were zero percentage responses, and anchor sanity, percentage parsing, and
finish-reason truncation could not be assessed.

After the execution-error records returned, the unchanged gate reporter raised
`KeyError: 'turn_1'` while computing its truncation rate. This reporter exception
is downstream of the substantive failure: every record lacked a completed
Turn-1 object.

## Earlier model-loading attempts

Two pre-gate startup failures are preserved:

1. With 12 GiB offload per GPU, vLLM's auto-selected V2 model runner attempted
   to construct the full shard on each GPU and OOMed at 23.44 GiB used while
   requesting another 136 MiB.
2. Raising offload to 16 GiB produced the identical OOM, confirming that V2 had
   not initialized its offloader. The installed vLLM source showed that the V1
   runner initializes `UVAOffloader`; setting `VLLM_USE_V2_MODEL_RUNNER=0`
   resolved loading without changing weights or benchmark behavior.

These failures and the successful service configuration are retained in the
model registry namespace.

## Reproducibility and preservation

- Experiment fingerprint:
  `c7352806d378796c5fb9e2d6eb6b5bb73a4f0afe92ba7bb349647821cfb5310b`
- Full service audit:
  `data/manifests/oss_model_registry/deepseek_r1_distill_qwen_32b_service.json`
- Complete vLLM log:
  `data/manifests/oss_model_registry/deepseek_r1_distill_qwen_32b_vllm.log`
- Machine-readable gate failure:
  `data/manifests/oss_deepseek_r1_distill_qwen_32b_chat_v1/neutral_gate_failure.json`
- Raw treatment records: 0
- Preservation audit: passed; all 1,917 previously banked files remained
  unchanged.

The checkpoint cache and ready vLLM service remain installed on `mihir@idli`.
