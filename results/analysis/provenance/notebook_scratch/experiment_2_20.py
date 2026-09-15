show_table(
    SPLIT_SUMMARIES,
    columns=[
        "observation_type", "root_identity", "circuit_split", "circuit_count",
        "completion_rate", "mean_progress", "mean_return",
    ],
    sort_by=["observation_type", "root_identity", "circuit_split"],
    title="Every split, every root",
)
show_table(GAPS, title="Generalization gaps")
