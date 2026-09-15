def launch_for(observation_name: str, root: int, path: Path):
    """
    Return a no-argument callable that trains one condition of one root.
    """
    return partial(
        run_ppo_training,
        seed=root,
        run_path=path,
        actor_config=ACTOR_CONFIG,
        actor_learning_rate=ACTOR_LEARNING_RATE,
        critic_learning_rate=CRITIC_LEARNING_RATE,
        training_interaction_budget=TRAINING_INTERACTION_BUDGET,
        environment_config=ENVIRONMENT_CONFIG,
        evaluation_interval=EVALUATION_INTERVAL,
        execution_config=EXECUTION_CONFIG,
        near_saturated_steering_threshold=STEERING_THRESHOLD,
        # An unbounded schedule of generated circuits: this is what makes the
        # experiment about circuits rather than about one circuit.
        training_circuit_schedule=TrainingCircuitSchedule(),
        evaluation_circuits=CIRCUITS[observation_name]["validation"],
        final_evaluation_circuits=CIRCUITS[observation_name]["test"],
        training_reference_circuits=TRAINING_REFERENCE_CIRCUITS,
        observation=OBSERVATIONS[observation_name],
        run_category=RUN_CATEGORY,
    )


SPECIFICATIONS = [
    RunSpecification(
        run_id := f"ppo-{SELECTED_ACTOR}-{observation_name}-seed-{root}",
        RESULTS_ROOT / run_id,
        launch_for(observation_name, root, RESULTS_ROOT / run_id),
    )
    for observation_name in OBSERVATIONS
    for root in ROOTS
]

show_table(
    [
        {"run_id": s.run_id, "already complete": (s.path / "completion.json").is_file()}
        for s in SPECIFICATIONS
    ],
    title=f"{len(SPECIFICATIONS)} runs",
)
