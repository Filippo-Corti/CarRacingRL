show_table(
    SUMMARIES,
    columns=[
        "algorithm", "actor_name", "root_identity", "converged", "censored",
        "convergence_interactions", "convergence_duration", "episodes_to_convergence",
    ],
    sort_by=["algorithm", "actor_name", "root_identity"],
    title="Convergence with censoring",
)
show_figure(ANALYSIS_ROOT, "convergence_resources")
