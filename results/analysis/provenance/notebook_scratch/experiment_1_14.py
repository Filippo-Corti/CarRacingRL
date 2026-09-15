show_table(
    CELLS,
    columns=[
        "algorithm", "actor_name", "run_count",
        "final_completion_rate_mean", "final_mean_return_mean",
        "final_mean_return_sample_standard_deviation",
        "final_mean_progress_mean", "final_crash_rate_mean",
        "converged_mean",
    ],
    sort_by=["algorithm", "actor_name"],
    title="Cell summaries (mean over roots)",
)

show_table(
    SUMMARIES,
    columns=[
        "algorithm", "actor_name", "root_identity",
        "final_mean_return", "final_mean_progress",
        "final_completion_count", "final_crash_count",
        "completed_lap_time_mean", "completed_lap_denominator",
        "converged", "censored",
    ],
    sort_by=["algorithm", "actor_name", "root_identity"],
    title="Every root, raw",
)
