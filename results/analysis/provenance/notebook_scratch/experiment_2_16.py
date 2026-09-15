show_table(
    INVENTORY,
    columns=[
        "run_id", "observation_type", "root_identity",
        "actor_parameters", "critic_parameters", "total_parameters",
        "training_interactions",
    ],
    sort_by=["observation_type", "root_identity"],
    title="Run inventory",
)
