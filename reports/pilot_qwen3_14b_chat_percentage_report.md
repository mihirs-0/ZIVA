# ZIVA conversational-percentage report: pilot_qwen3_14b_chat_percentage

## Design and completeness

- Trials: 512; scenarios: 32
- Valid percentages: 386; malformed: 126 (24.61%)
- Primary endpoint: the explicit rough percentage in the second ordinary conversational turn.

## Results

- Excited-positive minus neutral: +4.99 points (95% CI [-0.72, +10.82], p=0.10638, d_z=0.292)
- Skeptical-negative minus neutral: -12.19 points (95% CI [-19.23, -5.96], p=0.0002, d_z=-0.6206)
- Negative-preference minus neutral: -8.96 points (95% CI [-16.34, -2.27], p=0.02, d_z=-0.4496)
- Negative-preference direction-normalized: +8.96 points (95% CI [+2.41, +16.23], p=0.015, d_z=0.4496)

## Sensitivity excluding case_000

- excited_positive_minus_neutral: +4.07 points (95% CI [-1.91, +10.01], p=0.19836, d_z=0.2405)
- skeptical_negative_minus_neutral: -10.86 points (95% CI [-18.22, -4.67], p=0.0006, d_z=-0.5613)
- negative_preference_minus_neutral: -7.78 points (95% CI [-14.98, -1.42], p=0.03659, d_z=-0.4181)

## Effect by difficulty

- ambiguous: +10.57 points (95% CI [-3.70, +30.62], p=0.4883, d_z=0.5254)
- easy: +10.25 points (95% CI [-2.00, +22.50], p=1.0, d_z=0.5917)
- moderate: +2.96 points (95% CI [-11.46, +15.85], p=0.73445, d_z=0.1261)
- trivial: +4.21 points (95% CI [-0.83, +10.95], p=0.19556, d_z=0.3356)
- Pearson r with difficulty: 0.0919

## Per-template effects

- excited_positive_v1: +4.38 points (95% CI [-1.88, +10.49], p=0.19496, d_z=0.2468)
- excited_positive_v2: +5.43 points (95% CI [-3.32, +14.25], p=0.24015, d_z=0.2252)
- negative_preference_v1: +1.14 points (95% CI [-5.46, +7.67], p=0.75405, d_z=0.0616)
- negative_preference_v2: -25.26 points (95% CI [-35.11, -15.88], p=0.0002, d_z=-0.9867)
- skeptical_negative_v1: -9.95 points (95% CI [-18.15, -2.53], p=0.014, d_z=-0.4624)
- skeptical_negative_v2: -13.61 points (95% CI [-21.13, -6.85], p=0.0002, d_z=-0.7032)

## Variance and binary crossings

- Repeat-sampling SD: 8.50 points
- Template/paraphrase SD: 11.11 points
- excited_positive_vs_neutral 50% crossing rate: 19.28% (n=83)
- skeptical_negative_vs_neutral 50% crossing rate: 22.78% (n=79)
- negative_preference_vs_neutral 50% crossing rate: 28.36% (n=67)

## Direct comparison with structured Qwen

- Chat excited effect: +4.99 points
- Structured excited effect: +8.61 points
- Chat minus structured scenario-level effect: -3.62 points (95% CI [-10.06, +2.08], p=0.26815, d_z=-0.2054)

## Interpretation boundary

This is a behavioral comparison of stated percentages under different conversational contexts. It does not establish internal beliefs, deception, conscious evaluation awareness, or scheming.
