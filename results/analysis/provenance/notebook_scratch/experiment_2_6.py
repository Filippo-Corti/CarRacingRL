ACTOR_OVERRIDE = None  # set to "small" / "medium" / "large" only to deviate deliberately

selection_path = EXPERIMENT_1_ANALYSIS / "ppo_actor_selection.json"
if ACTOR_OVERRIDE is not None:
    SELECTED_ACTOR = ACTOR_OVERRIDE
    print(f"OVERRIDDEN: using the {SELECTED_ACTOR!r} actor, not the recorded selection.")
elif selection_path.is_file():
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    SELECTED_ACTOR = selection["selected_actor"]
    show_table(selection["candidates"], title="The recorded Experiment 1 selection")
    print(f"Experiment 1 selected the {SELECTED_ACTOR!r} actor.")
else:
    raise FileNotFoundError(
        f"no recorded actor selection at {selection_path}. "
        "Run experiment_1.ipynb first, at the same run category."
    )

ACTOR_CONFIG = ACTORS[SELECTED_ACTOR]
print(f"hidden widths {ACTOR_CONFIG.hidden_sizes}")
