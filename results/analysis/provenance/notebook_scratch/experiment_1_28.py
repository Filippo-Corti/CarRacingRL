import json

SELECTION = ppo_actor_selection_rows(SUMMARIES)
SELECTED_ACTOR = selected_ppo_actor(SELECTION)

show_table(
    SELECTION,
    columns=[
        "actor_name", "actor_parameters", "paired_root_count",
        "mean_final_return", "completion_count", "is_best_mean_return",
        "mean_paired_deficit", "paired_deficit_standard_error",
        "within_one_standard_error", "completion_within_one_root",
        "admitted", "selected",
    ],
    title="PPO actor-size selection",
)

selection_path = ANALYSIS_ROOT / "ppo_actor_selection.json"
selection_path.write_text(
    json.dumps(
        {"selected_actor": SELECTED_ACTOR, "candidates": SELECTION},
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
print(f"Experiment 2 will use the {SELECTED_ACTOR!r} actor.")
print(f"Recorded at {selection_path}")
