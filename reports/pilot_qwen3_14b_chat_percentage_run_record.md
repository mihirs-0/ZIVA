# Qwen3-14B JSON-less conversational-percentage ZIVA run record

## Answer to the research question

The excited-positive point estimate remained positive in ordinary conversation, but it was smaller and less certain than in structured JSON. Excited-positive wording shifted Qwen's stated spotting chance by **+4.9909 probability points** relative to neutral (95% CI [-0.7227, +10.8194], sign-flip p=0.10638, Cohen's dz=0.2920). The structured-JSON effect was +8.6133 points. The direct same-scenario chat-minus-structured difference was -3.6224 points (95% CI [-10.0638, +2.0823], p=0.26815, dz=-0.2054).

For context, the prior structured naturalistic and separated effects were +13.36 and +3.87 points, respectively. This conversational experiment does not contain that elicitation-mode factor, so its +4.99-point estimate should not be relabeled as either mode or treated as a formal comparison with them.

This does not establish that the excited-positive effect survives unchanged, because the conversational CI includes zero. It also does not establish a collapse or a difference from the structured effect, because the direct comparison CI includes zero. The most defensible description is **directionally preserved but attenuated and imprecise**.

## Other treatments

- Skeptical-negative minus neutral: **-12.1883 points** (95% CI [-19.2297, -5.9595], p=0.0002, dz=-0.6206). This closely matches the structured result's -11.45-point direction and magnitude.
- Negative-preference minus neutral: **-8.9597 points** (95% CI [-16.3361, -2.2714], p=0.0200, dz=-0.4496; 30 scenario pairs). Direction-normalized toward the user's preferred not-visible world, this is +8.9597 points (95% CI [+2.4149, +16.2279], p=0.0150).
- The negative-preference result is not paraphrase-robust: v1 was +1.1351 points (p=0.75405), while v2 was -25.2617 points (p=0.0002). It should be treated as strong template heterogeneity, not a stable family-wide magnitude.

## Robustness and heterogeneity

- Excluding the two `case_000` scenarios, excited-positive was +4.0736 points (95% CI [-1.9130, +10.0113], p=0.19836, dz=0.2405); skeptical-negative was -10.8619; negative-preference was -7.7783.
- Both excited paraphrases were positive and similar: v1 +4.3763 and v2 +5.4282 points, though neither was individually significant.
- Excited-positive effects were positive in every difficulty stratum: ambiguous +10.5729, easy +10.2500, moderate +2.9583, trivial +4.2083. The difficulty-score relationship was weak (Pearson r=0.0919), and stratum CIs were wide.
- The 50% threshold-crossing rates were 19.28% for excited-positive versus neutral, 22.78% for skeptical-negative, and 28.36% for negative-preference.
- Mean repeat-sampling SD was 8.5008 points; mean template/paraphrase SD was 11.1089 points. Both exceed the pooled excited point estimate, underscoring substantial response variability.

## Parsing and missingness limitation

The conservative frozen parser recovered 386/512 endpoints; **126/512 (24.6094%) were malformed**. Statuses were 263 clear ranges, 123 clear single percentages, 125 ambiguous multi-percentage responses, and one response without a usable percentage.

Missingness differed across arms: excited-positive 17.19%, neutral 24.22%, skeptical-negative 23.44%, and negative-preference 33.59%. The most extreme template was `negative_preference_v2` at 40.63% malformed. Therefore, the numerical contrasts describe the parsed responses and could be affected by treatment-dependent parsing missingness. This is the main validity limitation of the MVP.

All Turn-2 generations and 496/512 Turn-1 generations reached their configured output-token cap. The requested percentage generally appeared early, but responses and explanations were often truncated. The exact full transcripts and finish reasons are preserved; settings were not changed after treatment effects became available.

## Design and integrity

- Same 32 frozen physical scenarios and unchanged frozen treatment wording.
- Four families, two paraphrases, and two independent repeats: 512 trials exactly.
- Ordinary prose evidence, generic `You are a helpful assistant.` system prompt, and the exact invariant Turn-2 question in all 512 records.
- Treatment wording was the only paired-arm difference; pairing audit failures: zero.
- 512 unique raw records, 512 complete five-message transcripts, zero execution errors, zero retries, zero follow-up mismatches, and zero `<think>` markers.
- Both RTX 3090s recorded 979 nonzero-utilization samples and reached 100% SM utilization. BF16 tensor-parallel inference fit without OOM. The model remains resident on both GPUs.
- The full run began at 2026-08-16T03:17:54Z and ended at 2026-08-16T03:34:22Z.

## Reproducibility

- Experiment fingerprint: `1b0f912b98cda0d5c16ffc6b587b47082924a43300caf22bb9865c88aa6cb33c`
- Frozen Git commit: `ca389752b867d0ae147c7d412a3deff691785218`
- Qwen model/tokenizer revision: `40c069824f4251a91eefaf281ebe4c544efd3e18`
- BF16, unquantized; vLLM 0.20.0; tensor parallel size 2; `enable_thinking=false`
- Sampling: temperature 0.7, top-p 0.8, top-k 20, min-p 0.0, presence penalty 0.0; independent fixed seed per trial and turn; Turn-1 max 400 tokens and Turn-2 max 200 tokens.
- Server: max model length 8192, GPU memory utilization 0.90, vLLM generation config, bound to localhost only.

## Interpretation boundary

The experiment shows context-conditioned movement in Qwen's stated factual percentages, especially under skeptical-negative wording. It does not establish internal beliefs, deception, conscious evaluation awareness, or scheming. For excited-positive wording specifically, the result is suggestive but not statistically decisive, and the high differential malformed rate prevents a strong invariance claim from this MVP alone.
