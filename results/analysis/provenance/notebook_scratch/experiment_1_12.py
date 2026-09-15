show_table(
    INVENTORY,
    columns=[
        "run_id", "algorithm", "actor_name", "root_identity",
        "actor_parameters", "critic_parameters", "total_parameters",
        "training_interactions",
    ],
    sort_by=["algorithm", "actor_parameters", "root_identity"],
    title="Run inventory and parameter counts",
)
