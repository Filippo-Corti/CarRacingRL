show_table(
    SPLIT_SUMMARIES,
    columns=[
        "observation_type", "root_identity", "circuit_split", "circuit_count",
        "completion_count", "completion_rate", "crash_rate",
        "mean_return", "mean_progress",
    ],
    sort_by=["circuit_split", "observation_type", "root_identity"],
    where=lambda row: row["circuit_split"] == "test",
    title="Final performance on the 32 unseen test circuits",
)

for observation in ("frenet", "lidar"):
    values = [
        row["completion_rate"]
        for row in SPLIT_SUMMARIES
        if row["circuit_split"] == "test" and row["observation_type"] == observation
    ]
    print(f"{observation:>7} test completion rate: {describe(values)}")
