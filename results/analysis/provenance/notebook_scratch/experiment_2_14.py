from analyze_results import analyze_results

MANIFEST = analyze_results(
    results_root=RESULTS_ROOT,
    output_directory=ANALYSIS_ROOT,
    experiment=2,
    category=RUN_CATEGORY,
    geometry_specification=SPLITS_PATH,
)
print(f"analyzed {len(MANIFEST['inputs'])} runs -> {ANALYSIS_ROOT}")

INVENTORY = read_table(ANALYSIS_ROOT, "run_inventory")
SUMMARIES = read_table(ANALYSIS_ROOT, "run_summaries")
SPLIT_SUMMARIES = read_table(ANALYSIS_ROOT, "final_split_summaries")
GAPS = read_table(ANALYSIS_ROOT, "generalization_gaps")
PAIRED_CIRCUITS = read_table(ANALYSIS_ROOT, "paired_circuit_differences")
PAIRED = read_table(ANALYSIS_ROOT, "paired_summaries")
GEOMETRY = read_table(ANALYSIS_ROOT, "geometry_strata")
