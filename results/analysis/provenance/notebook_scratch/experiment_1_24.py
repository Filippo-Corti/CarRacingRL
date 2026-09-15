show_figure(ANALYSIS_ROOT, "optimization_diagnostics")
show_table(
    UPDATES,
    columns=[
        "run_id", "training_interactions", "actor_loss", "actor_gradient_norm",
        "entropy_proxy", "log_standard_deviation_0", "explained_variance",
    ],
    sort_by=["run_id", "training_interactions"],
    limit=15,
    title="Optimization diagnostics (first rows; the full table is on disk)",
)
