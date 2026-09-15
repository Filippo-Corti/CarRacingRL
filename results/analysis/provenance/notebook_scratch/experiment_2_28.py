show_table(
    SUMMARIES,
    columns=[
        "observation_type", "root_identity", "collection_throughput",
        "collection_duration", "optimization_duration", "evaluation_duration",
        "end_to_end_duration", "peak_process_memory",
    ],
    sort_by=["observation_type", "root_identity"],
    title="Computational cost",
)
show_figure(ANALYSIS_ROOT, "optimization_diagnostics")
