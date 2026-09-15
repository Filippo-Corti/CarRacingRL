# Experiment 2: generated evidence tables

## Final outcomes

| Condition | Completed laps | Return mean +/- SD | Return 95% interval | Progress | Lap time (s) | Threshold roots |
|---|---|---|---|---|---|---|
| frenet | 315/320 | 221.36 +/- 7.48 | [217.24, 226.03] | 0.990 | 25.64 | 10/10 |
| lidar | 318/320 | 233.21 +/- 7.76 | [229.11, 238.05] | 0.998 | 23.45 | 10/10 |

## Threshold and stability

Times/episodes below average only observed attainments; the denominator reports censoring. Late completion is the mean validation/fixed-track completion over the final 20% of the budget.

| Condition | Attained | First interactions range | Mean episodes to first | Mean first training time (s) | Mean confirmation time (s) | Late completion |
|---|---|---|---|---|---|---|
| frenet | 10/10 | 100k--650k | 480 | 57.7 | 86.5 | 0.920 |
| lidar | 10/10 | 50k--200k | 281 | 37.9 | 69.7 | 0.975 |

## Paired return contrasts

Positive favours the first named condition.

| Fixed condition | Contrast | Mean difference | 95% interval | Mean return AUC difference |
|---|---|---|---|---|
| ppo | frenet_minus_lidar | -11.85 | [-18.78, -4.59] | -19.59 |

## Learned controls

Means over roots after equal circuit weighting within each root. Requested steering reversal rate ignores magnitudes <= 0.05.

| Condition | Observed coverage | Speed mean | Braking fraction | Throttle q10 / median / q90 | Throttle SD | Mean steering change | Steering reversals / s |
|---|---|---|---|---|---|---|---|
| frenet | 0.990 | 17.92 | 0.317 | -0.44 / 0.40 / 0.85 | 0.49 | 0.589 | 13.61 |
| lidar | 0.998 | 19.66 | 0.307 | -0.44 / 0.38 / 0.94 | 0.52 | 0.345 | 8.13 |

## Resources

Mean full-budget durations; these do not measure time to threshold.

| Condition | Collection (min) | Optimization (min) | Evaluation (min) | End-to-end (min) | Collection step/s |
|---|---|---|---|---|---|
| frenet | 3.35 | 1.46 | 0.90 | 5.88 | 4980 |
| lidar | 3.83 | 1.43 | 1.33 | 6.79 | 4359 |

## Optimization diagnostics

Each root is averaged over its final tenth of updates, then roots are averaged equally.

| Condition | Explained variance | Actor gradient norm | Approx. KL | Clip fraction | log sigma throttle / steer |
|---|---|---|---|---|---|
| frenet | 0.164 | 0.788 | 0.0053 | 0.064 | -0.014 / -0.003 |
| lidar | 0.091 | 0.914 | 0.0057 | 0.070 | -0.001 / -0.002 |

## Paired test completion

| Contrast | Difference | 95% interval |
|---|---|---|
| frenet_minus_lidar | -0.009 | [-0.028, 0.013] |

| Root | Frenet completion | LiDAR completion | Difference | Return difference |
|---|---|---|---|---|
| 0 | 1.000 | 1.000 | 0.000 | -7.38 |
| 1 | 0.969 | 1.000 | -0.031 | -5.14 |
| 2 | 1.000 | 0.938 | 0.062 | -11.22 |
| 3 | 0.969 | 1.000 | -0.031 | -15.84 |
| 4 | 0.969 | 1.000 | -0.031 | -27.91 |
| 5 | 0.938 | 1.000 | -0.062 | -19.84 |
| 6 | 1.000 | 1.000 | 0.000 | -4.96 |
| 7 | 1.000 | 1.000 | 0.000 | 12.59 |
| 8 | 1.000 | 1.000 | 0.000 | -11.26 |
| 9 | 1.000 | 1.000 | 0.000 | -27.51 |

## Generalization splits

| Observation | Split | Mean completion | Mean progress | Mean return |
|---|---|---|---|---|
| frenet | training_reference | 0.938 | 0.968 | 212.53 |
| frenet | validation | 0.912 | 0.977 | 209.24 |
| frenet | test | 0.984 | 0.990 | 221.36 |
| lidar | training_reference | 1.000 | 1.000 | 234.98 |
| lidar | validation | 0.981 | 0.993 | 230.95 |
| lidar | test | 0.994 | 0.998 | 233.21 |
