# Report-ready experiment data

## Included studies

| Study | Raw inputs | Runs | Budget per run | Compact data | Manifest |
|---|---|---:|---:|---|---|
| Experiment 1, combined | `reported_experiments/experiment_1/` + `reported_experiments/experiment_1_extension/` | 60 | 2,000,000 | [data summary](analysis/reported_experiments/experiment_1_combined/DATA_SUMMARY.md) | [manifest](analysis/reported_experiments/experiment_1_combined/analysis_manifest.json) |
| Experiment 2, revised | `reported_experiments/experiment_2_revised/` | 20 | 1,000,000 | [data summary](analysis/reported_experiments/experiment_2_revised/DATA_SUMMARY.md) | [manifest](analysis/reported_experiments/experiment_2_revised/analysis_manifest.json) |

`reported_experiments/experiment_2/` is historical and is excluded from these
outputs. Reduced-budget validation and pre-experiment configuration records are
also excluded.

## Compact machine-readable tables

Every table is available as CSV and JSON in each study directory.

| Table stem | Unit / contents | Experiment 1 | Experiment 2 |
|---|---|:---:|:---:|
| `cell_summaries` | one condition; final outcomes, intervals, convergence and costs | yes | yes |
| `run_summaries` | one training root; final, convergence, stability and resources | yes | yes |
| `paired_summaries` | one paired contrast; root-bootstrap summaries | yes | yes |
| `paired_differences` | one paired contrast per root | yes | yes |
| `learning_curves` | one root and evaluation boundary | yes | yes |
| `common_budget_outcomes` | declared common interaction boundaries | yes | yes |
| `root_controls` | final controls aggregated equally across circuits within root | yes | yes |
| `circuit_controls` | final control distribution for one root/circuit | yes | yes |
| `curvature_controls` | final controls by straight/curved geometry group | yes | yes |
| `optimization_diagnostics` | one optimizer update | yes | yes |
| `evaluation_outcomes` | one deterministic evaluation circuit | yes | yes |
| `training_episodes` | one completed training episode | yes | yes |
| `run_inventory` | full configuration, provenance and checksums | yes | yes |
| `final_split_summaries` | one root and circuit split |  | yes |
| `generalization_gaps` | validation/test and training-reference/test per root |  | yes |
| `paired_circuit_differences` | Frenet minus LiDAR per root/test circuit |  | yes |
| `geometry_strata` | test outcomes by frozen length/curvature strata |  | yes |

## Plots

### Experiment 1

- [Final outcomes](analysis/reported_experiments/experiment_1_combined/task_outcomes.png)
- [Learning curves](analysis/reported_experiments/experiment_1_combined/learning_curves.png)
- [Sizes within each algorithm](analysis/reported_experiments/experiment_1_combined/sizes_within_algorithms.png)
- [Algorithms within each size](analysis/reported_experiments/experiment_1_combined/algorithms_within_sizes.png)
- [Threshold and resource measurements](analysis/reported_experiments/experiment_1_combined/convergence_resources.png)
- [Root-level control distributions](analysis/reported_experiments/experiment_1_combined/control_summaries.png)
- [Fixed-root signed control and trajectory traces](analysis/reported_experiments/experiment_1_combined/control_traces.png)
- [Controls grouped by curvature](analysis/reported_experiments/experiment_1_combined/curvature_controls.png)
- [Optimization diagnostics](analysis/reported_experiments/experiment_1_combined/optimization_diagnostics.png)

### Experiment 2

- [Final test outcomes](analysis/reported_experiments/experiment_2_revised/task_outcomes.png)
- [Validation learning curves](analysis/reported_experiments/experiment_2_revised/learning_curves.png)
- [Threshold and resource measurements](analysis/reported_experiments/experiment_2_revised/convergence_resources.png)
- [Root-level control distributions](analysis/reported_experiments/experiment_2_revised/control_summaries.png)
- [Fixed-root, fixed-test-circuit signed control and trajectory traces](analysis/reported_experiments/experiment_2_revised/control_traces.png)
- [Controls grouped by curvature](analysis/reported_experiments/experiment_2_revised/curvature_controls.png)
- [Test outcomes by circuit geometry](analysis/reported_experiments/experiment_2_revised/circuit_geometry.png)
- [Optimization diagnostics](analysis/reported_experiments/experiment_2_revised/optimization_diagnostics.png)

## Statistical and aggregation metadata

The manifests record input checksums and analysis conventions. Experiment 1
uses exhaustive whole-root bootstrap enumeration over five roots. Experiment 2
uses 10,000 whole-root resamples with seed 0. Paired intervals resample paired
root differences. Final Experiment 2 controls include all 32 test circuits for
each root and observation; circuits are averaged within roots before root-level
comparisons.
