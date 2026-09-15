# Experiment 1: generated evidence tables

## Final outcomes

| Condition | Completed laps | Return mean +/- SD | Return 95% interval | Progress | Lap time (s) | Threshold roots |
|---|---|---|---|---|---|---|
| reinforce/tiny | 2/5 | 116.30 +/- 89.81 | [48.21, 184.39] | 0.867 | 27.94 | 3/5 |
| reinforce/small | 5/5 | 205.49 +/- 9.05 | [198.78, 213.02] | 1.000 | 29.88 | 5/5 |
| reinforce/medium | 5/5 | 205.31 +/- 10.34 | [197.60, 213.73] | 1.000 | 29.92 | 5/5 |
| reinforce/large | 4/5 | 159.06 +/- 92.65 | [76.63, 207.43] | 0.803 | 31.08 | 4/5 |
| a2c/tiny | 3/5 | 148.88 +/- 102.94 | [73.38, 224.38] | 0.823 | 25.76 | 3/5 |
| a2c/small | 5/5 | 227.84 +/- 2.92 | [225.53, 229.97] | 1.000 | 24.91 | 5/5 |
| a2c/medium | 5/5 | 227.98 +/- 2.40 | [226.37, 230.00] | 1.000 | 24.90 | 5/5 |
| a2c/large | 5/5 | 201.67 +/- 10.02 | [194.06, 209.79] | 1.000 | 30.74 | 5/5 |
| ppo/tiny | 5/5 | 215.70 +/- 4.10 | [212.33, 218.44] | 1.000 | 27.62 | 5/5 |
| ppo/small | 4/5 | 187.15 +/- 69.00 | [124.72, 223.33] | 0.984 | 27.16 | 5/5 |
| ppo/medium | 5/5 | 220.61 +/- 9.42 | [213.69, 228.58] | 1.000 | 26.52 | 5/5 |
| ppo/large | 5/5 | 220.97 +/- 5.51 | [216.77, 225.17] | 1.000 | 26.45 | 5/5 |

## Threshold and stability

Times/episodes below average only observed attainments; the denominator reports censoring. Late completion is the mean validation/fixed-track completion over the final 20% of the budget.

| Condition | Attained | First interactions range | Mean episodes to first | Mean first training time (s) | Mean confirmation time (s) | Late completion |
|---|---|---|---|---|---|---|
| reinforce/tiny | 3/5 | 750k--1050k | 3099 | 163.3 | 177.7 | 0.450 |
| reinforce/small | 5/5 | 300k--700k | 1664 | 90.7 | 105.8 | 1.000 |
| reinforce/medium | 5/5 | 250k--750k | 1317 | 79.5 | 94.1 | 1.000 |
| reinforce/large | 4/5 | 200k--350k | 1087 | 58.5 | 74.3 | 0.775 |
| a2c/tiny | 3/5 | 1450k--1900k | 4610 | 277.1 | 292.9 | 0.450 |
| a2c/small | 5/5 | 700k--1000k | 2544 | 131.2 | 147.5 | 1.000 |
| a2c/medium | 5/5 | 400k--1150k | 2951 | 126.6 | 143.1 | 1.000 |
| a2c/large | 5/5 | 400k--1050k | 5199 | 136.0 | 152.9 | 0.800 |
| ppo/tiny | 5/5 | 100k--200k | 352 | 29.9 | 52.7 | 1.000 |
| ppo/small | 5/5 | 50k--100k | 300 | 20.5 | 43.0 | 0.925 |
| ppo/medium | 5/5 | 100k--100k | 336 | 24.2 | 48.2 | 0.975 |
| ppo/large | 5/5 | 50k--100k | 245 | 21.4 | 48.4 | 0.975 |

## Common-budget return and completion

Each entry is mean return (completed roots / 5) at the exact evaluation boundary.

| Condition | 250k | 500k | 750k | 1M | 2M |
|---|---|---|---|---|---|
| reinforce/tiny | 8.6 (0/5) | 32.9 (0/5) | 74.0 (1/5) | 109.2 (2/5) | 116.3 (2/5) |
| reinforce/small | 33.5 (0/5) | 111.6 (2/5) | 207.0 (5/5) | 206.1 (5/5) | 205.5 (5/5) |
| reinforce/medium | 66.5 (1/5) | 139.6 (3/5) | 202.1 (5/5) | 169.4 (4/5) | 205.3 (5/5) |
| reinforce/large | 55.5 (1/5) | 130.5 (3/5) | 159.0 (4/5) | 151.0 (4/5) | 159.1 (4/5) |
| a2c/tiny | 0.2 (0/5) | 28.6 (0/5) | 32.0 (0/5) | 33.0 (0/5) | 148.9 (3/5) |
| a2c/small | 7.9 (0/5) | 34.2 (0/5) | 183.8 (4/5) | 186.0 (4/5) | 227.8 (5/5) |
| a2c/medium | 6.2 (0/5) | 74.8 (1/5) | 114.7 (2/5) | 184.5 (4/5) | 228.0 (5/5) |
| a2c/large | 44.3 (1/5) | 86.6 (2/5) | 126.4 (3/5) | 135.1 (3/5) | 201.7 (5/5) |
| ppo/tiny | 187.2 (4/5) | 167.9 (4/5) | 213.7 (5/5) | 211.1 (5/5) | 215.7 (5/5) |
| ppo/small | 221.2 (5/5) | 221.2 (5/5) | 185.4 (4/5) | 215.4 (5/5) | 187.1 (4/5) |
| ppo/medium | 225.3 (5/5) | 181.5 (4/5) | 217.2 (5/5) | 182.3 (4/5) | 220.6 (5/5) |
| ppo/large | 224.9 (5/5) | 219.7 (5/5) | 215.9 (5/5) | 186.8 (4/5) | 221.0 (5/5) |

## Paired return contrasts

Positive favours the first named condition.

| Fixed condition | Contrast | Mean difference | 95% interval | Mean return AUC difference |
|---|---|---|---|---|
| large | a2c_minus_ppo | -19.31 | [-24.89, -12.05] | -90.96 |
| large | reinforce_minus_a2c | -42.61 | [-126.68, 11.61] | 16.36 |
| large | reinforce_minus_ppo | -61.92 | [-149.90, -9.99] | -74.60 |
| medium | a2c_minus_ppo | 7.36 | [0.89, 13.79] | -44.50 |
| medium | reinforce_minus_a2c | -22.66 | [-30.24, -13.46] | 7.48 |
| medium | reinforce_minus_ppo | -15.30 | [-19.69, -10.33] | -37.02 |
| small | a2c_minus_ppo | 40.69 | [3.59, 104.23] | -64.72 |
| small | reinforce_minus_a2c | -22.35 | [-29.79, -14.92] | 18.58 |
| small | reinforce_minus_ppo | 18.34 | [-22.88, 87.60] | -46.14 |
| tiny | a2c_minus_ppo | -66.82 | [-141.01, 6.73] | -154.66 |
| tiny | reinforce_minus_a2c | -32.58 | [-146.38, 81.22] | 33.66 |
| tiny | reinforce_minus_ppo | -99.40 | [-168.80, -30.00] | -121.00 |
| a2c | medium_minus_large | 26.31 | [18.72, 32.39] | 38.25 |
| a2c | small_minus_large | 26.17 | [15.80, 34.98] | 27.52 |
| a2c | small_minus_medium | -0.13 | [-3.35, 3.02] | -10.72 |
| a2c | tiny_minus_large | -52.78 | [-135.78, 28.70] | -68.63 |
| a2c | tiny_minus_medium | -79.09 | [-154.99, -3.20] | -106.87 |
| a2c | tiny_minus_small | -78.96 | [-154.52, -3.59] | -96.15 |
| ppo | medium_minus_large | -0.36 | [-9.18, 8.46] | -8.21 |
| ppo | small_minus_large | -33.83 | [-95.46, 4.97] | 1.29 |
| ppo | small_minus_medium | -33.47 | [-92.69, 0.21] | 9.50 |
| ppo | tiny_minus_large | -5.27 | [-11.54, 1.00] | -4.93 |
| ppo | tiny_minus_medium | -4.91 | [-11.67, 0.71] | 3.29 |
| ppo | tiny_minus_small | 28.56 | [-7.34, 91.96] | -6.21 |
| reinforce | medium_minus_large | 46.26 | [-2.13, 127.23] | 29.37 |
| reinforce | small_minus_large | 46.43 | [0.08, 127.30] | 29.75 |
| reinforce | small_minus_medium | 0.17 | [-12.63, 12.98] | 0.38 |
| reinforce | tiny_minus_large | -42.75 | [-125.15, 39.64] | -51.32 |
| reinforce | tiny_minus_medium | -89.01 | [-163.84, -14.18] | -80.69 |
| reinforce | tiny_minus_small | -89.18 | [-151.21, -27.16] | -81.07 |

## Learned controls

Means over roots after equal circuit weighting within each root. Requested steering reversal rate ignores magnitudes <= 0.05.

| Condition | Observed coverage | Speed mean | Braking fraction | Throttle q10 / median / q90 | Throttle SD | Mean steering change | Steering reversals / s |
|---|---|---|---|---|---|---|---|
| reinforce/tiny | 0.867 | 18.13 | 0.047 | 0.03 / 0.12 / 0.46 | 0.15 | 0.017 | 0.29 |
| reinforce/small | 1.000 | 17.47 | 0.153 | -0.03 / 0.09 / 0.40 | 0.20 | 0.019 | 0.45 |
| reinforce/medium | 1.000 | 17.34 | 0.260 | -0.11 / 0.13 / 0.43 | 0.24 | 0.076 | 2.82 |
| reinforce/large | 0.802 | 14.82 | 0.228 | -0.05 / 0.39 / 0.77 | 0.32 | 0.269 | 5.85 |
| a2c/tiny | 0.823 | 19.77 | 0.245 | -0.14 / 0.25 / 0.62 | 0.27 | 0.029 | 0.44 |
| a2c/small | 1.000 | 20.50 | 0.243 | -0.21 / 0.23 / 0.76 | 0.34 | 0.197 | 6.78 |
| a2c/medium | 1.000 | 20.26 | 0.280 | -0.22 / 0.22 / 0.82 | 0.38 | 0.227 | 7.21 |
| a2c/large | 1.000 | 17.24 | 0.346 | -0.24 / 0.18 / 0.70 | 0.38 | 0.743 | 15.09 |
| ppo/tiny | 1.000 | 18.57 | 0.271 | -0.24 / 0.20 / 0.65 | 0.37 | 0.475 | 10.84 |
| ppo/small | 0.984 | 19.38 | 0.330 | -0.39 / 0.35 / 0.93 | 0.49 | 0.642 | 13.27 |
| ppo/medium | 1.000 | 19.21 | 0.325 | -0.34 / 0.26 / 0.93 | 0.46 | 0.644 | 12.59 |
| ppo/large | 1.000 | 19.29 | 0.301 | -0.35 / 0.27 / 0.89 | 0.46 | 0.506 | 11.15 |

## Resources

Mean full-budget durations; these do not measure time to threshold.

| Condition | Collection (min) | Optimization (min) | Evaluation (min) | End-to-end (min) | Collection step/s |
|---|---|---|---|---|---|
| reinforce/tiny | 4.81 | 0.23 | 0.08 | 5.31 | 6964 |
| reinforce/small | 5.01 | 0.25 | 0.12 | 5.57 | 6658 |
| reinforce/medium | 4.84 | 0.25 | 0.12 | 5.39 | 6899 |
| reinforce/large | 5.71 | 0.50 | 0.11 | 6.75 | 6012 |
| a2c/tiny | 4.99 | 0.39 | 0.07 | 5.63 | 6680 |
| a2c/small | 5.03 | 0.40 | 0.09 | 5.70 | 6626 |
| a2c/medium | 5.07 | 0.42 | 0.09 | 5.77 | 6581 |
| a2c/large | 5.09 | 0.68 | 0.10 | 6.10 | 6566 |
| ppo/tiny | 4.90 | 2.82 | 0.12 | 7.99 | 6805 |
| ppo/small | 4.70 | 2.85 | 0.11 | 7.82 | 7102 |
| ppo/medium | 4.99 | 2.96 | 0.12 | 8.23 | 6680 |
| ppo/large | 5.02 | 3.69 | 0.12 | 8.98 | 6648 |

## Optimization diagnostics

Each root is averaged over its final tenth of updates, then roots are averaged equally.

| Condition | Explained variance | Actor gradient norm | Approx. KL | Clip fraction | log sigma throttle / steer |
|---|---|---|---|---|---|
| reinforce/tiny | -- | 57.177 | -- | -- | -0.575 / -0.541 |
| reinforce/small | -- | 76.979 | -- | -- | -0.556 / -0.502 |
| reinforce/medium | -- | 97.552 | -- | -- | -0.536 / -0.510 |
| reinforce/large | -- | 142.341 | -- | -- | -0.526 / -0.517 |
| a2c/tiny | 0.009 | 0.162 | -- | -- | -0.630 / -0.675 |
| a2c/small | -0.077 | 0.250 | -- | -- | -0.612 / -0.588 |
| a2c/medium | -0.074 | 0.291 | -- | -- | -0.610 / -0.572 |
| a2c/large | -0.063 | 0.418 | -- | -- | -0.566 / -0.542 |
| ppo/tiny | 0.036 | 0.778 | 0.0020 | 0.018 | -0.002 / -0.001 |
| ppo/small | 0.134 | 0.804 | 0.0042 | 0.048 | -0.002 / -0.002 |
| ppo/medium | 0.087 | 0.972 | 0.0059 | 0.073 | -0.002 / -0.002 |
| ppo/large | 0.122 | 1.076 | 0.0076 | 0.096 | -0.001 / -0.001 |
