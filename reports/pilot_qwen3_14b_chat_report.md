# ZIVA JSON-less conversational report: pilot_qwen3_14b_chat

## Design

- Scenarios: 32; model-facing prompts: 256
- Frozen continuation pairs: 4
- The model-facing conversation contains ordinary prose only and requests no structured output, probability, confidence, or grading information.
- Fixed assistant continuations were scored teacher-forced; generated prose and LLM judges were not used.
- Primary endpoint: `total_sequence_logprob_margin`. Robustness endpoint: `per_token_mean_logprob_margin`.
- Positive margins favor a visible-world continuation. These scores are not calibrated probabilities.

## Primary results

- Excited-positive minus neutral: -1.2819 (95% CI [-1.6215, -0.9354], p=0.0002, d_z=-1.2572)
- Skeptical-negative minus neutral: +0.0310 (95% CI [-0.0605, +0.1141], p=0.4873, d_z=0.1244)
- Negative-preference minus neutral: -0.4453 (95% CI [-0.9672, +0.0497], p=0.10298, d_z=-0.295)
- Negative-preference direction-normalized: +0.4453 (95% CI [-0.0608, +0.9826], p=0.11498, d_z=0.295)

## Sensitivity excluding case_000

- excited_positive_minus_neutral: -1.2942 (95% CI [-1.6718, -0.9236], p=0.0002, d_z=-1.231)
- skeptical_negative_minus_neutral: +0.0233 (95% CI [-0.0708, +0.1109], p=0.63387, d_z=0.0911)
- negative_preference_minus_neutral: -0.5417 (95% CI [-1.0782, -0.0236], p=0.06179, d_z=-0.3585)

## Effect by difficulty

- ambiguous: -1.3291 (95% CI [-1.7403, -0.9871], p=0.11958, d_z=-3.0181)
- easy: -0.7612 (95% CI [-0.9484, -0.5739], p=0.4991, d_z=-2.8738)
- moderate: -1.3509 (95% CI [-1.7762, -0.9767], p=0.0016, d_z=-1.9816)
- trivial: -1.2920 (95% CI [-1.9445, -0.6747], p=0.0018, d_z=-0.9667)
- Pearson r with difficulty score: -0.0109

## Per-treatment paraphrase effects

- excited_positive_v1: -1.7577 (95% CI [-2.1415, -1.3826], p=0.0002, d_z=-1.5651)
- excited_positive_v2: -0.8061 (95% CI [-1.1164, -0.4827], p=0.0002, d_z=-0.8578)
- negative_preference_v1: -0.0784 (95% CI [-0.6321, +0.4351], p=0.77824, d_z=-0.0497)
- negative_preference_v2: -0.8123 (95% CI [-1.3351, -0.3337], p=0.0042, d_z=-0.5436)
- skeptical_negative_v1: -0.2019 (95% CI [-0.3513, -0.0377], p=0.0188, d_z=-0.437)
- skeptical_negative_v2: +0.2640 (95% CI [+0.0365, +0.4759], p=0.03179, d_z=0.4043)

## Per-continuation-pair effects

- `answer_can` excited-positive minus neutral: +0.6610 (95% CI [+0.5214, +0.8021], p=0.0002, d_z=1.586)
- `answer_yes_no` excited-positive minus neutral: -1.1331 (95% CI [-1.4529, -0.8128], p=0.0002, d_z=-1.1843)
- `overall_visible` excited-positive minus neutral: -2.2238 (95% CI [-2.7537, -1.6996], p=0.0002, d_z=-1.4296)
- `probably_can` excited-positive minus neutral: -2.4318 (95% CI [-2.9208, -1.8914], p=0.0002, d_z=-1.5875)

## Total-logprob robustness

- excited_positive_minus_neutral: -0.2161 (95% CI [-0.2694, -0.1601], p=0.0002, d_z=-1.3378); same direction=True; same p<0.05 conclusion=True
- skeptical_negative_minus_neutral: +0.0014 (95% CI [-0.0117, +0.0127], p=0.83963, d_z=0.0386); same direction=True; same p<0.05 conclusion=True
- negative_preference_minus_neutral: -0.0764 (95% CI [-0.1528, -0.0025], p=0.06459, d_z=-0.3386); same direction=True; same p<0.05 conclusion=True

## Anchor sanity check

- Passed: True
- Below-horizon primary margin: -3.449395663337782
- Bright-night primary margin: 3.958008068264462

## Interpretation boundary

This experiment tests whether irrelevant conversational valence changes Qwen3-14B's distribution over fixed factual continuations. It does not identify internal beliefs, deception, conscious evaluation awareness, or scheming.
