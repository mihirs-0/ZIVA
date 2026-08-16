# ZIVA conversational-percentage report: pilot_qwen3_14b_chat_percentage_concise

## Design and completeness

- Trials: 512; scenarios: 32
- Valid percentages: 512; malformed: 0 (0.00%)
- Primary endpoint: the explicit rough percentage in the second ordinary conversational turn.

## Results

- Excited-positive minus neutral: +3.96 points (95% CI [-0.43, +8.62], p=0.09858, d_z=0.2997)
- Skeptical-negative minus neutral: -11.75 points (95% CI [-18.05, -6.45], p=0.0004, d_z=-0.6868)
- Negative-preference minus neutral: -12.14 points (95% CI [-18.14, -6.63], p=0.0004, d_z=-0.7164)
- Negative-preference direction-normalized: +12.14 points (95% CI [+6.70, +18.00], p=0.0004, d_z=0.7164)

## Sensitivity excluding case_000

- excited_positive_minus_neutral: +3.84 points (95% CI [-0.85, +8.71], p=0.13457, d_z=0.2823)
- skeptical_negative_minus_neutral: -10.71 points (95% CI [-17.58, -5.27], p=0.0004, d_z=-0.6276)
- negative_preference_minus_neutral: -11.58 points (95% CI [-17.56, -5.82], p=0.0006, d_z=-0.6907)

## Effect by difficulty

- ambiguous: +8.75 points (95% CI [+1.56, +22.19], p=0.12757, d_z=0.6556)
- easy: +4.38 points (95% CI [+1.00, +7.75], p=0.4943, d_z=0.9166)
- moderate: +2.64 points (95% CI [-6.73, +10.56], p=0.60988, d_z=0.1769)
- trivial: +3.53 points (95% CI [-1.92, +10.89], p=0.37752, d_z=0.2615)
- Pearson r with difficulty: 0.1111

## Per-template effects

- excited_positive_v1: +7.16 points (95% CI [+2.59, +12.13], p=0.0044, d_z=0.5122)
- excited_positive_v2: +0.75 points (95% CI [-6.09, +7.30], p=0.82823, d_z=0.0392)
- negative_preference_v1: -4.00 points (95% CI [-9.61, +1.64], p=0.19436, d_z=-0.2327)
- negative_preference_v2: -20.29 points (95% CI [-29.30, -12.12], p=0.0002, d_z=-0.8125)
- skeptical_negative_v1: -12.78 points (95% CI [-20.16, -6.30], p=0.0002, d_z=-0.6222)
- skeptical_negative_v2: -10.73 points (95% CI [-17.09, -5.58], p=0.0002, d_z=-0.633)

## Variance and binary crossings

- Repeat-sampling SD: 9.01 points
- Template/paraphrase SD: 9.30 points
- excited_positive_vs_neutral 50% crossing rate: 18.75% (n=128)
- skeptical_negative_vs_neutral 50% crossing rate: 15.62% (n=128)
- negative_preference_vs_neutral 50% crossing rate: 20.31% (n=128)

## Direct comparison with structured Qwen

- Chat excited effect: +3.96 points
- Structured excited effect: +8.61 points
- Chat minus structured scenario-level effect: -4.66 points (95% CI [-9.23, -0.18], p=0.06899, d_z=-0.3482)

## Reliability rerun

- Turn-2 truncation: 0/512 (0.00%)
- Malformed percentage rate: 0.00%; previous run: 24.61%
- Malformed-rate change: -24.61 percentage points
- Malformed by family: excited_positive=0.00%, negative_preference=0.00%, neutral=0.00%, skeptical_negative=0.00%
- Malformed by template: excited_positive_v1=0.00%, excited_positive_v2=0.00%, negative_preference_v1=0.00%, negative_preference_v2=0.00%, neutral_v1=0.00%, neutral_v2=0.00%, skeptical_negative_v1=0.00%, skeptical_negative_v2=0.00%

## Direct comparison with the previous chat run

- Concise excited effect: +3.96 points
- Previous chat excited effect: +4.99 points
- Concise minus previous: -1.03 points (95% CI [-6.50, +4.79], p=0.73065, d_z=-0.0618)

## Interpretation boundary

This is a behavioral comparison of stated percentages under different conversational contexts. It does not establish internal beliefs, deception, conscious evaluation awareness, or scheming.
