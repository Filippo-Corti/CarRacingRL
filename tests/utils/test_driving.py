"""Tests for control alignment, tied geometry edges and root weighting."""

from __future__ import annotations

import pytest

from recording import EpisodeOutcome
from tests.fixtures.analysis_runs import write_analysis_run
from utils.analysis import descriptive_statistics, load_recorded_runs
from utils.driving import FinalDrive, root_control_rows


def test_ten_root_bootstrap_is_seeded_and_resamples_whole_pairs() -> None:
    values = [float(index) for index in range(10)]
    first = descriptive_statistics(values, seed=0, resamples=10_000)
    assert first == descriptive_statistics(values, seed=0, resamples=10_000)
    assert first.count == 10
    assert first.confidence_interval_low < first.mean < first.confidence_interval_high
    # A constant paired contrast must stay constant under every resampling.
    paired = descriptive_statistics([3.0] * 10, seed=0)
    assert paired.confidence_interval_low == paired.confidence_interval_high == 3


def test_pre_action_alignment_and_tied_curvature_edges(tmp_path) -> None:
    write_analysis_run(
        tmp_path / "run", root_identity=0, outcomes=(EpisodeOutcome.COMPLETED,) * 4
    )
    drive = FinalDrive.from_runs(load_recorded_runs(tmp_path), experiment=1)[0]
    rows = drive.aligned_rows()
    assert [row["distance"] for row in rows] == [0, 25, 50, 75]
    assert [row["time"] for row in rows] == [0, 1, 2, 3]
    assert [row["speed"] for row in rows] == [10, 11, 12, 13]
    geometry = drive.episode["circuit_geometry"]["absolute_curvature"]
    geometry["quantiles"].update(q25=0, q50=0, q75=0.02)
    for transition, curvature in zip(
        drive.transitions, (0, 1e-12, 0.015, 0.035), strict=True
    ):
        transition["current_curvature"] = curvature
    bins = drive.curvature_rows()
    assert {row["curvature_bin"]: row["sample_count"] for row in bins} == {
        "straight": 2,
        "curve_1": 1,
        "curve_2": 1,
    }
    assert FinalDrive.from_runs((drive.run,), experiment=2) == []


def test_controls_weight_circuits_equally_and_keep_failures(tmp_path) -> None:
    write_analysis_run(
        tmp_path / "run", root_identity=0, outcomes=(EpisodeOutcome.COMPLETED,) * 4
    )
    drive = FinalDrive.from_runs(load_recorded_runs(tmp_path), experiment=1)[0]
    first = drive.summary()
    second = {
        **first,
        "circuit_identity": "failed",
        "sample_count": 1000,
        "outcome": "crashed",
        "mean_speed": 0.0,
        "coverage": 0.1,
    }
    root = root_control_rows([first, second])[0]
    assert root["mean_speed"] == pytest.approx(first["mean_speed"] / 2)
    assert root["circuit_count"] == 2
    assert root["completed_circuit_count"] == 1
    assert root["coverage"] == pytest.approx(0.55)
