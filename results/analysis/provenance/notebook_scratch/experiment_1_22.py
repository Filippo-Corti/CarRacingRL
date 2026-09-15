show_table(
    SUMMARIES,
    columns=[
        "algorithm", "actor_name", "root_identity",
        "collection_throughput", "collection_duration", "optimization_duration",
        "evaluation_duration", "end_to_end_duration", "peak_process_memory",
    ],
    sort_by=["algorithm", "actor_name", "root_identity"],
    title="Computational cost",
)
