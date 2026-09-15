from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from configs import EnvironmentConfig, PPOConfig
from experiments.matrix import (
    RunSpecification,
    contract_mismatch,
    execute,
    is_complete,
    learning_contract,
    summarize,
)


def _finished_run(directory: Path, *, environment: EnvironmentConfig) -> Path:
    """
    Write the two documents a matrix reads: the config and the completion.
    """
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "environment": environment.to_dict(),
                "training": {"ppo": PPOConfig().to_dict()},
            }
        ),
        encoding="utf-8",
    )
    (directory / "completion.json").write_text("{}", encoding="utf-8")
    return directory


def test_a_finished_run_is_skipped(tmp_path: Path) -> None:
    path = _finished_run(tmp_path / "run", environment=EnvironmentConfig())
    started: list[str] = []

    outcomes = execute(
        [RunSpecification("run", path, lambda: started.append("run"))],
        report=lambda _: None,
    )

    assert is_complete(path)
    assert [outcome.status for outcome in outcomes] == ["skipped"]
    assert started == []


def test_a_run_recorded_under_other_constants_is_refused_and_preserved(
    tmp_path: Path,
) -> None:
    """
    A completed result from a superseded contract cannot be reused or deleted.
    """
    superseded = EnvironmentConfig()
    superseded = replace(
        superseded, reward=replace(superseded.reward, lap_time_bonus=100.0)
    )
    path = _finished_run(tmp_path / "run", environment=superseded)
    started: list[str] = []

    outcomes = execute(
        [RunSpecification("run", path, lambda: started.append("run"))],
        contract=learning_contract(EnvironmentConfig(), PPOConfig()),
        report=lambda _: None,
    )

    assert [outcome.status for outcome in outcomes] == ["failed"]
    assert started == []
    assert (path / "completion.json").is_file()
    assert (path / "config.json").is_file()


def test_a_matching_contract_still_skips(tmp_path: Path) -> None:
    path = _finished_run(tmp_path / "run", environment=EnvironmentConfig())
    started: list[str] = []

    outcomes = execute(
        [RunSpecification("run", path, lambda: started.append("run"))],
        contract=learning_contract(EnvironmentConfig(), PPOConfig()),
        report=lambda _: None,
    )

    assert [outcome.status for outcome in outcomes] == ["skipped"]
    assert started == []


@pytest.mark.parametrize(
    ("dotted", "recorded_value", "expected_value"),
    (
        ("training.training_interaction_budget", 2_000_000, 1_000_000),
        ("training.evaluation.evaluation_interval", 50_000, 25_000),
    ),
)
def test_a_completed_run_with_changed_per_run_settings_is_refused(
    tmp_path: Path,
    dotted: str,
    recorded_value: int,
    expected_value: int,
) -> None:
    """
    Same-identity runs cannot silently mix training budgets or evaluation cadence.
    """
    path = _finished_run(tmp_path / "run", environment=EnvironmentConfig())
    document = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if dotted.endswith("training_interaction_budget"):
        document["training"]["training_interaction_budget"] = recorded_value
    else:
        document["training"]["evaluation"] = {"evaluation_interval": recorded_value}
    config_path = path / "config.json"
    config_path.write_text(json.dumps(document), encoding="utf-8")
    original_config = config_path.read_bytes()
    original_completion = (path / "completion.json").read_bytes()
    started: list[str] = []

    outcomes = execute(
        [
            RunSpecification(
                "run",
                path,
                lambda: started.append("run"),
                {dotted: expected_value},
            )
        ],
        report=lambda _: None,
    )

    assert [outcome.status for outcome in outcomes] == ["failed"]
    assert started == []
    assert config_path.read_bytes() == original_config
    assert (path / "completion.json").read_bytes() == original_completion


def test_the_mismatch_names_the_field_that_changed(tmp_path: Path) -> None:
    superseded = EnvironmentConfig()
    superseded = replace(
        superseded, reward=replace(superseded.reward, lap_time_bonus=100.0)
    )
    path = _finished_run(tmp_path / "run", environment=superseded)

    reason = contract_mismatch(
        path, learning_contract(EnvironmentConfig(), PPOConfig())
    )

    assert reason is not None
    assert "lap_time_bonus" in reason
    assert "100.0" in reason and "140.0" in reason


def test_a_discount_change_is_detected(tmp_path: Path) -> None:
    path = _finished_run(tmp_path / "run", environment=EnvironmentConfig())

    reason = contract_mismatch(
        path,
        learning_contract(EnvironmentConfig(), replace(PPOConfig(), discount=0.9995)),
    )

    assert reason is not None
    assert "training.ppo.discount" in reason


def test_one_failure_does_not_stop_the_queue(tmp_path: Path) -> None:
    """
    An overnight matrix must finish the runs that can finish.
    """
    started: list[str] = []

    def fail() -> None:
        raise RuntimeError("deliberate")

    outcomes = execute(
        [
            RunSpecification("bad", tmp_path / "bad", fail),
            RunSpecification("good", tmp_path / "good", lambda: started.append("good")),
        ],
        report=lambda _: None,
    )

    assert [outcome.status for outcome in outcomes] == ["failed", "completed"]
    assert started == ["good"]
    assert summarize(outcomes, report=lambda _: None) == 1


def test_an_incomplete_directory_is_archived_before_a_rerun(tmp_path: Path) -> None:
    """
    The recorder refuses a non-empty directory, so debris is retained elsewhere.
    """
    path = tmp_path / "run"
    path.mkdir()
    (path / "episodes.jsonl").write_text("partial", encoding="utf-8")

    def launch() -> None:
        assert not path.exists(), "the interrupted run's debris was left behind"

    outcomes = execute([RunSpecification("run", path, launch)], report=lambda _: None)

    assert [outcome.status for outcome in outcomes] == ["completed"]
    archived = tmp_path / ".incomplete" / "run-1"
    assert (archived / "episodes.jsonl").read_text(encoding="utf-8") == "partial"
