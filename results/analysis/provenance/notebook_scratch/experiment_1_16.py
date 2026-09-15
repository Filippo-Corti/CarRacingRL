show_figure(ANALYSIS_ROOT, "learning_curves")
show_table(
    CELLS,
    columns=[
        "algorithm", "actor_name",
        "return_auc_mean", "return_auc_sample_standard_deviation",
        "progress_auc_mean", "progress_auc_sample_standard_deviation",
    ],
    sort_by=["algorithm", "actor_name"],
    title="Normalized curve area",
)
