def launch_for(algorithm: str, actor_name: str, root: int, path: Path):
    """
    Return a no-argument callable that trains one cell of the matrix.

    The learning rate travels with the algorithm rather than with the size:
    holding it fixed across the size ladder is what makes the comparison about
    capacity instead of about tuning.
    """
    common = dict(
        seed=root,
        track_path=TRACK_PATH,
        run_path=path,
        actor_config=ACTORS[actor_name],
        actor_learning_rate=ACTOR_LEARNING_RATE[algorithm],
        training_interaction_budget=TRAINING_INTERACTION_BUDGET,
        evaluation_interval=EVALUATION_INTERVAL,
        execution_config=EXECUTION_CONFIG,
        near_saturated_steering_threshold=STEERING_THRESHOLD,
        run_category=RUN_CATEGORY,
    )
    if algorithm == "reinforce":
        return partial(run_reinforce_training, **common)
    runner = run_a2c_training if algorithm == "a2c" else run_ppo_training
    return partial(runner, critic_learning_rate=CRITIC_LEARNING_RATE[algorithm], **common)


SPECIFICATIONS = [
    RunSpecification(
        run_id := f"{algorithm}-{actor_name}-frenet-seed-{root}",
        RESULTS_ROOT / run_id,
        launch_for(algorithm, actor_name, root, RESULTS_ROOT / run_id),
    )
    for algorithm in ("reinforce", "a2c", "ppo")
    for actor_name in ACTORS
    for root in ROOTS
]

show_table(
    [
        {"run_id": s.run_id, "already complete": (s.path / "completion.json").is_file()}
        for s in SPECIFICATIONS
    ],
    title=f"{len(SPECIFICATIONS)} runs",
)
