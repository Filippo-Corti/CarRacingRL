from analyze_results import analyze_results

MANIFEST = analyze_results(
    results_root=RESULTS_ROOT,
    output_directory=ANALYSIS_ROOT,
    experiment=1,
    category=RUN_CATEGORY,
)
print(f"analyzed {len(MANIFEST['inputs'])} runs -> {ANALYSIS_ROOT}")

INVENTORY = read_table(ANALYSIS_ROOT, "run_inventory")
SUMMARIES = read_table(ANALYSIS_ROOT, "run_summaries")
CELLS = read_table(ANALYSIS_ROOT, "cell_summaries")
CURVES = read_table(ANALYSIS_ROOT, "learning_curves")
PAIRED = read_table(ANALYSIS_ROOT, "paired_summaries")
UPDATES = read_table(ANALYSIS_ROOT, "optimization_diagnostics")
