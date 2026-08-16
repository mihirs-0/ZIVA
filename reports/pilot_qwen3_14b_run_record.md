# Qwen3-14B OSS baseline run record

## Outcome

The existing ZIVA structured-evidence pilot completed successfully for `Qwen/Qwen3-14B`: 32 physical scenarios, three treatment families, two templates per family, naturalistic and separated elicitation, and two repeats, for 768 trials. The existing scenario generation, treatments, prompts, parsing, scoring, endpoint, randomization, and analysis pipeline were not changed.

On this overtly instrumented benchmark, Qwen3-14B showed preference-conditioned movement in its factual estimates. The pooled excited-positive minus neutral shift was +8.6133 probability points (95% CI +5.0781 to +12.4805; permutation p=0.0002; Cohen's dz=0.7931).

## Reproducible runtime

- Host/user: `mihir@idli.local`
- GPUs: 2x NVIDIA GeForce RTX 3090 24 GB; driver 580.173.02
- Checkpoint: `Qwen/Qwen3-14B`
- Revision: `40c069824f4251a91eefaf281ebe4c544efd3e18`
- Weights: BF16, unquantized
- Serving: vLLM 0.20.0, tensor parallel size 2
- Container: `vllm/vllm-openai@sha256:04563c302537a91aa49ebdfbceda96111c5712275999b7e8804fa598f0b5641d`
- PyTorch: 2.11.0+cu130; Transformers: 5.6.2; CUDA runtime: 13.0
- Context/runtime: max model length 8192; GPU memory utilization 0.90; vLLM generation config
- Thinking: disabled out of band with `chat_template_kwargs={"enable_thinking": false}`
- Sampling: temperature 0.7, top-p 0.8, top-k 20, min-p 0.0, presence penalty 0.0, max output tokens 2000
- Concurrency: 8
- Experiment fingerprint: `987fffbb48217d135952cfb3888fb758e4f0baabe4755826fb06b6058a084c79`

## Engineering acceptance

The 12-trial smoke test passed: both GPUs reached 100% utilization, BF16 loaded without OOM, all 12 responses parsed, no `<think>` content was emitted, thinking was recorded disabled, and hashes of the exact outbound system/user messages matched the frozen stimuli. Across the full run, all 768 records parsed successfully, all recorded `enable_thinking=false`, all exact prompt hashes matched, and none contained think tags.

## Main results

- Excited-positive minus neutral, pooled: +8.6133 points; 95% CI [+5.0781, +12.4805]; p=0.0002; dz=0.7931
- Naturalistic excited-positive minus neutral: +13.3594 points; 95% CI [+8.4766, +18.9453]; p=0.0002; dz=0.8752
- Separated excited-positive minus neutral: +3.8672 points; 95% CI [+0.8984, +7.0312]; p=0.02679; dz=0.4250
- Skeptical-negative minus neutral, pooled: -11.4453 points; 95% CI [-15.5864, -7.5972]; p=0.0002; dz=-0.9688
- Binary flip rate: 0.140625
- Malformed-response rate: 0/768 (0%)
- Generation noise SD: 4.088 points
- Template/paraphrase SD: 9.870 points

Difficulty-stratified excited-positive minus neutral shifts were +19.6875 points for ambiguous scenarios, +15.0000 for easy, +11.2500 for moderate, and +3.3984 for trivial; Pearson correlation with difficulty score was 0.5107.

## Historical direct comparison

The requested direct scenario-level comparison with GPT-5.2 and Luna was not computed because neither historical raw data nor their scenario-level result tables exist in this checkout, its Git history, or the `mihir` home directory on `idli`. No values were inferred or reconstructed. The exact searched paths and status are recorded in `results/pilot_qwen3_14b/runtime_audit.json`; when those historical artifacts are supplied, the existing scenario-level comparison can be run directly.
