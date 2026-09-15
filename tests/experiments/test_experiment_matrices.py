"""Reported experiment matrix definitions."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def experiment_modules(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """
    Load the executable experiment modules through their command-line import path.
    """
    project_root = Path(__file__).parents[2]
    monkeypatch.syspath_prepend(str(project_root / "experiments"))
    return importlib.import_module("experiment_1"), importlib.import_module(
        "experiment_2"
    )


def test_experiment_1_combines_the_original_matrix_and_tiny_extension(
    experiment_modules: tuple[Any, Any],
) -> None:
    """
    The full plan includes 60 runs while the tiny filter schedules only 15.
    """
    experiment_1, _ = experiment_modules

    all_runs = experiment_1.specifications(experiment_1.PROTOCOL)
    tiny_runs = experiment_1.specifications(experiment_1.PROTOCOL, ("tiny",))

    assert len(all_runs) == 60
    assert len(tiny_runs) == 15
    assert {run.run_id.split("-")[1] for run in tiny_runs} == {"tiny"}
    assert {
        run.launch.keywords["training_interaction_budget"] for run in tiny_runs
    } == {2_000_000}
    assert {
        run.launch.keywords["execution_config"].environment_workers for run in tiny_runs
    } == {8}
    assert {
        run.expected_contract["training.training_interaction_budget"]
        for run in tiny_runs
    } == {2_000_000}
    assert {
        run.expected_contract["training.evaluation.evaluation_interval"]
        for run in tiny_runs
    } == {50_000}
    assert {
        tuple(run.expected_contract["training.actor"]["hidden_sizes"])
        for run in tiny_runs
    } == {(8, 8)}
    assert experiment_1.PPO_SELECTION_ACTORS == ("small", "medium", "large")
    assert experiment_1.REHEARSAL.results_root.name == "experiment_1_extension"


def test_revised_experiment_2_has_twenty_medium_actor_runs(
    experiment_modules: tuple[Any, Any],
) -> None:
    """
    The rerun uses ten paired roots, one million interactions, and a new root.
    """
    _, experiment_2 = experiment_modules

    runs = experiment_2.specifications(experiment_2.PROTOCOL)

    assert len(runs) == 20
    assert experiment_2.PROTOCOL.results_root.name == "experiment_2_revised"
    assert experiment_2.REHEARSAL.results_root.name == "experiment_2_revised"
    assert {run.run_id.split("-")[1] for run in runs} == {"medium"}
    assert {run.launch.keywords["seed"] for run in runs} == set(range(10))
    assert {run.launch.keywords["training_interaction_budget"] for run in runs} == {
        1_000_000
    }
    assert {
        run.launch.keywords["final_evaluation_trajectory_circuits"] for run in runs
    } == {32}
    assert {
        run.launch.keywords["execution_config"].environment_workers for run in runs
    } == {8}
    assert {
        run.expected_contract["training.training_interaction_budget"] for run in runs
    } == {1_000_000}
    assert {
        run.expected_contract["training.logging.final_evaluation_trajectory_circuits"]
        for run in runs
    } == {32}


def test_existing_medium_ppo_selection_bytes_are_preserved(
    experiment_modules: tuple[Any, Any], tmp_path: Path
) -> None:
    """
    Combining tiny results cannot rewrite the original PPO architecture decision.
    """
    experiment_1, _ = experiment_modules
    selection = [{"actor_name": "medium", "selected": True}]

    experiment_1._record_ppo_selection(tmp_path, "medium", selection)
    path = tmp_path / "ppo_actor_selection.json"
    original = path.read_bytes()
    experiment_1._record_ppo_selection(tmp_path, "medium", selection)

    assert path.read_bytes() == original
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        experiment_1._record_ppo_selection(tmp_path, "small", selection)
    assert path.read_bytes() == original
