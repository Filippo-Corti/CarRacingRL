from dataclasses import replace

CIRCUITS = {
    name: {
        split.value: load_split_circuits(
            SPLITS_PATH,
            split,
            environment_config=replace(ENVIRONMENT_CONFIG, observation_type=observation),
        )
        for split in (CircuitSplit.VALIDATION, CircuitSplit.TEST)
    }
    for name, observation in OBSERVATIONS.items()
}
SPLIT_MANIFEST = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
STRATA = SPLIT_MANIFEST["geometry_strata"]

for name, splits in CIRCUITS.items():
    counts = ", ".join(f"{split} {len(circuits)}" for split, circuits in splits.items())
    print(f"{name:>7}: {counts}")
print(f"geometry bin edges:  {STRATA}")

show_table(
    SPLIT_MANIFEST["splits"]["validation"]["circuits"],
    columns=["identity", "track_seed", "track_length", "straight_fraction"],
    title="Validation split (rebuilt and geometry-verified)",
    limit=6,
)
