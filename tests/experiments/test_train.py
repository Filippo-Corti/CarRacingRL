from __future__ import annotations

import json
from pathlib import Path

import torch

from circuits import CircuitSplit, EvaluationCircuit
from configs import (
    SMALL_ACTOR_CONFIG,
    A2CConfig,
    EnvironmentConfig,
    ExecutionConfig,
    PPOConfig,
    ReinforceConfig,
    SimulationConfig,
    TrackGenerationConfig,
)
from envs.racing import RacingEnv
from envs.tracks import Track, TrackWithGeometry
from experiments.train import (
    parse_arguments,
    run_a2c_training,
    run_ppo_training,
    run_reinforce_training,
)
from recording import RunCategory, RunDirectory
from training.checkpointing import load_checkpoint


def test_train_entry_point_runs_reinforce_and_writes_shared_records(tmp_path) -> None:
    root = Path(__file__).parents[1]
    engine = run_reinforce_training(
        seed=3,
        track_path=root / "fixtures" / "tracks" / "valid_circle.json",
        run_path=tmp_path / "run",
        actor_config=SMALL_ACTOR_CONFIG,
        actor_learning_rate=0.01,
        training_interaction_budget=8,
        environment_config=_fixture_environment_config(),
        execution_config=_reinforce_execution_config(),
    )
    run = RunDirectory.open(
        tmp_path / "run",
        expected_category=RunCategory.REDUCED_VALIDATION,
        require_complete=True,
    )

    assert engine.state().counters.optimizer_updates == 1
    assert len(run.records("episodes")) == 8
    assert len(run.records("updates")) == 1
    assert run.records("updates")[0]["actor_loss"] is not None
    assert run.records("evaluations") == []
    completion = run.require_complete()
    assert completion["timing"]["persistence"] > 0.0
    assert completion["resources"]["training_interactions"] == 8


def test_train_cli_requires_learning_rate_and_accepts_output_alias() -> None:
    parsed = parse_arguments(
        [
            "--seed",
            "3",
            "--track",
            "track.json",
            "--output",
            "run",
            "--actor-learning-rate",
            "0.001",
            "--interaction-budget",
            "8",
        ]
    )

    assert parsed.run_path == "run"
    assert parsed.actor_learning_rate == 0.001


def test_reported_training_records_the_frozen_steering_saturation_threshold(
    tmp_path,
) -> None:
    """
    The threshold is frozen in configuration, so a reported run inherits it.

    It used to be absent by default and a reported run refused to start without
    it. The guard in the training path remains for a configuration that clears
    the value deliberately, but the value itself is now fixed in one place.
    """
    root = Path(__file__).parents[1]
    run_reinforce_training(
        seed=3,
        track_path=root / "fixtures" / "tracks" / "valid_circle.json",
        run_path=tmp_path / "run",
        actor_config=SMALL_ACTOR_CONFIG,
        actor_learning_rate=0.01,
        training_interaction_budget=8,
        environment_config=_fixture_environment_config(),
        execution_config=_reinforce_execution_config(),
        run_category=RunCategory.REPORTED,
    )
    run = RunDirectory.open(
        tmp_path / "run",
        expected_category=RunCategory.REPORTED,
        require_complete=True,
    )

    stored = json.loads((run.path / "config.json").read_text(encoding="utf-8"))
    logging_config = stored["training"]["logging"]
    assert logging_config["near_saturated_steering_threshold"] == 0.9
    assert logging_config["gradient_dispersion_subbatch"] == 256


def test_train_entry_point_runs_a2c_and_writes_shared_records(tmp_path) -> None:
    root = Path(__file__).parents[1]
    engine = run_a2c_training(
        seed=3,
        track_path=root / "fixtures" / "tracks" / "valid_circle.json",
        run_path=tmp_path / "run",
        actor_config=SMALL_ACTOR_CONFIG,
        actor_learning_rate=0.01,
        critic_learning_rate=0.01,
        training_interaction_budget=8,
        a2c_config=A2CConfig(transitions_per_rollout=8),
        environment_config=_fixture_environment_config(),
        execution_config=_reinforce_execution_config(),
    )
    run = RunDirectory.open(
        tmp_path / "run",
        expected_category=RunCategory.REDUCED_VALIDATION,
        require_complete=True,
    )

    assert engine.state().counters.optimizer_updates == 1
    assert len(run.records("episodes")) == 8
    assert len(run.records("updates")) == 1
    assert run.records("updates")[0]["critic_loss"] is not None
    assert run.require_complete()["critic_parameters"] is not None


def test_train_entry_point_runs_ppo_and_writes_shared_records(tmp_path) -> None:
    root = Path(__file__).parents[1]
    engine = run_ppo_training(
        seed=3,
        track_path=root / "fixtures" / "tracks" / "valid_circle.json",
        run_path=tmp_path / "run",
        actor_config=SMALL_ACTOR_CONFIG,
        actor_learning_rate=0.01,
        critic_learning_rate=0.01,
        training_interaction_budget=8,
        evaluation_interval=8,
        near_saturated_steering_threshold=0.9,
        ppo_config=PPOConfig(
            transitions_per_rollout=8,
            optimization_epochs=2,
            minibatch_size=4,
        ),
        environment_config=_fixture_environment_config(),
        execution_config=_reinforce_execution_config(),
    )
    run = RunDirectory.open(
        tmp_path / "run",
        expected_category=RunCategory.REDUCED_VALIDATION,
        require_complete=True,
    )

    assert engine.state().counters.optimizer_updates == 1
    assert len(run.records("episodes")) == 8
    assert len(run.records("updates")) == 1
    assert run.records("updates")[0]["critic_loss"] is not None
    assert run.records("updates")[0]["approximate_kl"] is not None
    assert run.records("updates")[0]["clip_fraction"] is not None
    evaluation = run.records("evaluations")[0]
    assert evaluation["training_duration"] > 0.0
    assert evaluation["episode"]["circuit_geometry"]["track_length"] > 0.0
    assert evaluation["episode"]["near_saturated_steering_fraction"] is not None
    trajectory = next((run.path / "trajectories").glob("*.json")).read_text(
        encoding="utf-8"
    )
    assert '"current_curvature":' in trajectory
    assert '"lateral_acceleration_proxy":' in trajectory


def test_final_test_trajectories_have_a_separate_retention_quota(tmp_path) -> None:
    """
    Retain final test drives after validation has filled the boundary quota.

    Experiment 2 evaluates validation circuits at the final boundary before its
    held-out circuits. The held-out retention request must therefore not share
    the ordinary per-boundary quota.
    """
    validation_circuits = _evaluation_circuits(CircuitSplit.VALIDATION, 2)
    test_circuits = _evaluation_circuits(CircuitSplit.TEST, 3)
    root = Path(__file__).parents[1]
    run_ppo_training(
        seed=3,
        track_path=root / "fixtures" / "tracks" / "valid_circle.json",
        run_path=tmp_path / "run",
        actor_config=SMALL_ACTOR_CONFIG,
        actor_learning_rate=0.01,
        critic_learning_rate=0.01,
        training_interaction_budget=8,
        evaluation_interval=8,
        near_saturated_steering_threshold=0.9,
        ppo_config=PPOConfig(
            transitions_per_rollout=8,
            optimization_epochs=2,
            minibatch_size=4,
        ),
        environment_config=_fixture_environment_config(),
        execution_config=_reinforce_execution_config(),
        evaluation_circuits=validation_circuits,
        final_evaluation_circuits=test_circuits,
        final_evaluation_trajectory_circuits=3,
    )
    run = RunDirectory.open(
        tmp_path / "run",
        expected_category=RunCategory.REDUCED_VALIDATION,
        require_complete=True,
    )

    trajectories = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (run.path / "trajectories").glob("*.json")
    ]
    retained_test_ids = {
        trajectory["evaluation"]["episode"]["circuit_identity"]
        for trajectory in trajectories
        if trajectory["evaluation"]["episode"]["circuit_split"] == "test"
    }

    assert retained_test_ids == {circuit.identity for circuit in test_circuits}
    assert len(trajectories) == 5
    stored = json.loads((run.path / "config.json").read_text(encoding="utf-8"))
    assert stored["training"]["logging"]["final_evaluation_trajectory_circuits"] == 3


def test_final_test_retention_does_not_change_training_or_evaluation(tmp_path) -> None:
    """
    Keep retention outside the learning and deterministic-evaluation state.

    Recording happens after the final checkpoint and all evaluations. Changing
    only its quota must leave the learned models, normalizer, RNG state, and
    substantive episode/evaluation records identical.
    """
    validation_circuits = _evaluation_circuits(CircuitSplit.VALIDATION, 2)
    test_circuits = _evaluation_circuits(CircuitSplit.TEST, 3)
    common = {
        "seed": 3,
        "actor_config": SMALL_ACTOR_CONFIG,
        "actor_learning_rate": 0.01,
        "critic_learning_rate": 0.01,
        "training_interaction_budget": 8,
        "evaluation_interval": 8,
        "near_saturated_steering_threshold": 0.9,
        "ppo_config": PPOConfig(
            transitions_per_rollout=8,
            optimization_epochs=2,
            minibatch_size=4,
        ),
        "environment_config": _fixture_environment_config(),
        "execution_config": _reinforce_execution_config(),
        "evaluation_circuits": validation_circuits,
        "final_evaluation_circuits": test_circuits,
    }
    root = Path(__file__).parents[1]
    run_ppo_training(
        **common,
        track_path=root / "fixtures" / "tracks" / "valid_circle.json",
        run_path=tmp_path / "without-retention",
        final_evaluation_trajectory_circuits=0,
    )
    run_ppo_training(
        **common,
        track_path=root / "fixtures" / "tracks" / "valid_circle.json",
        run_path=tmp_path / "with-retention",
        final_evaluation_trajectory_circuits=32,
    )
    without_retention = RunDirectory.open(
        tmp_path / "without-retention",
        expected_category=RunCategory.REDUCED_VALIDATION,
        require_complete=True,
    )
    with_retention = RunDirectory.open(
        tmp_path / "with-retention",
        expected_category=RunCategory.REDUCED_VALIDATION,
        require_complete=True,
    )

    assert without_retention.records("episodes") == with_retention.records("episodes")
    assert _records_without_timing(
        without_retention.records("updates")
    ) == _records_without_timing(with_retention.records("updates"))
    assert _records_without_timing(
        without_retention.records("evaluations")
    ) == _records_without_timing(with_retention.records("evaluations"))

    without_checkpoint = load_checkpoint(
        without_retention.path / "checkpoints" / "final.pt"
    )
    with_checkpoint = load_checkpoint(with_retention.path / "checkpoints" / "final.pt")
    _assert_checkpoint_values_equal(
        without_checkpoint["agent"], with_checkpoint["agent"]
    )
    assert without_checkpoint["normalizer"] == with_checkpoint["normalizer"]
    without_vector_state = without_checkpoint["environments"]["vector_environment"]
    with_vector_state = with_checkpoint["environments"]["vector_environment"]
    assert without_vector_state.reset_generators == with_vector_state.reset_generators
    assert without_vector_state.track_generators == with_vector_state.track_generators


def _reinforce_execution_config() -> ExecutionConfig:
    """
    Pin one worker per REINFORCE trajectory instead of inheriting the host's cores.
    """
    return ExecutionConfig(
        device="cpu",
        environment_workers=ReinforceConfig().completed_episodes_per_update,
    )


def _fixture_environment_config() -> EnvironmentConfig:
    """
    Use the length range of the deliberately long legacy test fixture.
    """
    return EnvironmentConfig(
        simulation=SimulationConfig(max_episode_steps=1),
        track=TrackGenerationConfig(min_length=1_000.0, max_length=3_000.0),
    )


def _evaluation_circuits(
    split: CircuitSplit, count: int
) -> tuple[EvaluationCircuit, ...]:
    """
    Build fresh fixed-track evaluation environments for the retention test.
    """
    return tuple(
        EvaluationCircuit(
            identity=f"{split.value}-{index}",
            split=split,
            factory=_fixture_evaluation_environment,
        )
        for index in range(count)
    )


def _fixture_evaluation_environment() -> RacingEnv:
    """
    Return one isolated environment for a deterministic evaluation.
    """
    root = Path(__file__).parents[1]
    track = TrackWithGeometry(
        Track.load(root / "fixtures" / "tracks" / "valid_circle.json")
    )
    return RacingEnv(track, config=_fixture_environment_config())


def _records_without_timing(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    """
    Remove wall-clock measurements from otherwise deterministic records.
    """
    timing_fields = {
        "collection_duration",
        "optimization_duration",
        "training_duration",
    }
    return [
        {name: value for name, value in record.items() if name not in timing_fields}
        for record in records
    ]


def _assert_checkpoint_values_equal(first: object, second: object) -> None:
    """
    Compare nested checkpoint values, including exact tensor and RNG state.
    """
    if isinstance(first, torch.Tensor):
        assert isinstance(second, torch.Tensor)
        torch.testing.assert_close(first, second, rtol=0, atol=0, equal_nan=True)
        return
    if isinstance(first, dict):
        assert isinstance(second, dict)
        assert first.keys() == second.keys()
        for name in first:
            _assert_checkpoint_values_equal(first[name], second[name])
        return
    if isinstance(first, (list, tuple)):
        assert isinstance(second, type(first))
        assert len(first) == len(second)
        for first_item, second_item in zip(first, second, strict=True):
            _assert_checkpoint_values_equal(first_item, second_item)
        return
    assert first == second


def test_reported_evaluation_uses_the_canonical_start(tmp_path) -> None:
    """
    Training samples a start pose; a reported evaluation curve must not.

    Every checkpoint has to answer the same question, so the deterministic
    evaluation environment always launches from the canonical start line even
    though the training environments do not.
    """
    root = Path(__file__).parents[1]
    engine = run_ppo_training(
        seed=3,
        track_path=root / "fixtures" / "tracks" / "valid_circle.json",
        run_path=tmp_path / "run",
        actor_config=SMALL_ACTOR_CONFIG,
        actor_learning_rate=0.01,
        critic_learning_rate=0.01,
        training_interaction_budget=8,
        evaluation_interval=8,
        near_saturated_steering_threshold=0.9,
        ppo_config=PPOConfig(
            transitions_per_rollout=8,
            optimization_epochs=2,
            minibatch_size=4,
        ),
        environment_config=_fixture_environment_config(),
        execution_config=_reinforce_execution_config(),
    )

    assert engine.environment_config.start.randomized
    factory = engine.evaluation_environment_factory
    assert factory is not None
    evaluation_environment = factory()
    try:
        assert not evaluation_environment.config.start.randomized
    finally:
        evaluation_environment.close()
