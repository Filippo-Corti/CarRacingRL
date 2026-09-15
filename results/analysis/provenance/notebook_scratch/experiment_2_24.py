show_figure(ANALYSIS_ROOT, "learning_curves")
show_table(
    SUMMARIES,
    columns=[
        "observation_type", "root_identity", "converged", "censored",
        "convergence_interactions", "convergence_duration",
        "return_auc", "progress_auc",
    ],
    sort_by=["observation_type", "root_identity"],
    title="Convergence and curve area",
)
show_figure(ANALYSIS_ROOT, "convergence_resources")
