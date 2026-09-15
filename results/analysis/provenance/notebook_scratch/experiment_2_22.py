show_table(PAIRED, title="Paired Frenet-minus-LiDAR root-level differences")
show_table(
    PAIRED_CIRCUITS,
    columns=[
        "root_identity", "circuit_identity", "circuit_split",
        "return", "maximum_progress",
    ],
    sort_by=["root_identity", "circuit_identity"],
    limit=16,
    title="Per-circuit paired differences (first rows; full table on disk)",
)
