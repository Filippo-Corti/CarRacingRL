"""Final-policy control summaries and correctly aligned trajectory tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .analysis import RecordedRun, TableRow

# Analysis thresholds, fixed before new runs: curvature interpolation can leave
# roundoff on straight segments; steering reversals ignore commands near zero.
STRAIGHT_TOLERANCE = 1e-8
STEERING_DEADBAND = 0.05
NEAR_ZERO_THROTTLE = 0.05


@dataclass(frozen=True)
class FinalDrive:
    """
    Represent one retained final deterministic drive on a named circuit.

    Fields:
        * run: Source run, including its configuration and provenance.
        * episode: Recorded complete-episode outcome and circuit geometry.
        * transitions: Ordered pre-action signals with post-action progress.
    """

    run: RecordedRun
    episode: dict[str, Any]
    transitions: list[dict[str, Any]]

    def summary(self) -> TableRow:
        """
        Summarize one circuit with equal time-step weight, including failures.

        Braking means strictly negative requested throttle, as in the original
        recorder. Steering variation uses adjacent commands; sign reversals
        discard the declared deadband and count switches between remaining signs.
        These measure requested actions, not tire slip or physical oscillation.
        """
        actions = np.asarray([row["action"] for row in self.transitions])
        speed = np.asarray([row["speed"] for row in self.transitions])
        throttle, steering = actions.T
        signs = np.sign(steering[np.abs(steering) > STEERING_DEADBAND])
        reversals = int(np.count_nonzero(np.diff(signs)))
        variation = np.abs(np.diff(steering))
        duration = float(self.episode["simulated_time"])
        return {
            **self.identity,
            "sample_count": len(self.transitions),
            "coverage": min(1.0, float(self.episode["maximum_progress"])),
            "simulated_time": duration,
            "braking_fraction": float(np.mean(throttle < 0)),
            "positive_throttle_fraction": float(np.mean(throttle > 0)),
            "near_zero_throttle_fraction": float(
                np.mean(np.abs(throttle) <= NEAR_ZERO_THROTTLE)
            ),
            "mean_throttle": float(np.mean(throttle)),
            "throttle_standard_deviation": float(np.std(throttle)),
            "throttle_q10": float(np.quantile(throttle, 0.1)),
            "throttle_median": float(np.median(throttle)),
            "throttle_q90": float(np.quantile(throttle, 0.9)),
            "mean_speed": float(np.mean(speed)),
            "speed_standard_deviation": float(np.std(speed)),
            "speed_q10": float(np.quantile(speed, 0.1)),
            "speed_median": float(np.median(speed)),
            "speed_q90": float(np.quantile(speed, 0.9)),
            "mean_absolute_steering": float(np.mean(np.abs(steering))),
            "mean_steering_change": (
                float(np.mean(variation)) if len(variation) else 0.0
            ),
            "steering_reversals": reversals,
            "steering_reversals_per_time": reversals / duration,
            "near_saturated_steering_fraction": float(np.mean(np.abs(steering) >= 0.9)),
        }

    def aligned_rows(self) -> list[TableRow]:
        """
        Align pre-action signals with distance since the canonical start.

        The recorder's progress and elapsed time are post-action. At canonical
        reset both are zero; shifting them by one transition aligns them with
        the recorded speed, position and observation. Preserve temporal order
        even if the car briefly reverses its progress around the centerline.
        """
        distance = 0.0
        time = 0.0
        length = float(self.episode["circuit_geometry"]["track_length"])
        rows = []
        for index, transition in enumerate(self.transitions):
            observation = transition["observation"]
            wheel_index = 3 if self.run.observation_type == "frenet" else 1
            rows.append(
                {
                    **self.identity,
                    "step_index": index,
                    "distance": distance,
                    "time": time,
                    "speed": float(transition["speed"]),
                    "throttle": float(transition["action"][0]),
                    "requested_steering": float(transition["action"][1]),
                    "wheel_angle": float(observation[wheel_index]),
                    "curvature": float(transition["current_curvature"]),
                    "x": float(transition["position"][0]),
                    "y": float(transition["position"][1]),
                }
            )
            distance = float(transition["progress"]) * length
            time = float(transition["elapsed_time"])
        return rows

    def curvature_rows(self) -> list[TableRow]:
        """
        Summarize straights separately and split curves at unique positive edges.

        Edges come from the recorded circuit's geometric quartiles, not the
        policy's visited samples. Removing zero/duplicate/maximum edges avoids
        the empty pseudo-quartiles caused by predominantly straight circuits.
        Numbered curve groups are relative to each circuit, not equal curvature
        ranges across different circuits. Each row retains its numeric bounds.
        """
        geometry = self.episode["circuit_geometry"]["absolute_curvature"]
        quantiles = geometry["quantiles"]
        edges = sorted(
            {
                float(quantiles[name])
                for name in ("q25", "q50", "q75")
                if STRAIGHT_TOLERANCE
                < float(quantiles[name])
                < float(geometry["maximum"])
            }
        )
        curvature = np.abs([row["current_curvature"] for row in self.transitions])
        groups = np.where(
            curvature <= STRAIGHT_TOLERANCE,
            0,
            1 + np.searchsorted(edges, curvature, side="right"),
        )
        rows = []
        for index in sorted(set(groups)):
            selected = [
                row
                for row, group in zip(self.transitions, groups, strict=True)
                if group == index
            ]
            actions = np.asarray([row["action"] for row in selected])
            rows.append(
                {
                    **self.identity,
                    "curvature_bin": "straight" if index == 0 else f"curve_{index}",
                    "lower_curvature": 0.0 if index <= 1 else edges[index - 2],
                    "upper_curvature": (
                        STRAIGHT_TOLERANCE
                        if index == 0
                        else (
                            edges[index - 1]
                            if index <= len(edges)
                            else float(geometry["maximum"])
                        )
                    ),
                    "sample_count": len(selected),
                    "mean_speed": float(np.mean([row["speed"] for row in selected])),
                    "mean_throttle": float(np.mean(actions[:, 0])),
                    "braking_fraction": float(np.mean(actions[:, 0] < 0)),
                    "mean_absolute_steering": float(np.mean(np.abs(actions[:, 1]))),
                }
            )
        return rows

    @property
    def identity(self) -> TableRow:
        """
        Return the source checkpoint, root, circuit, split and outcome.
        """
        return {
            "run_id": self.run.directory.run_id,
            "algorithm": self.run.algorithm,
            "actor_name": self.run.actor_name,
            "observation_type": self.run.observation_type,
            "root_identity": self.run.root_identity,
            "training_interactions": self.run.completion["training_interactions"],
            "circuit_identity": self.episode["circuit_identity"],
            "circuit_seed": self.episode.get("circuit_seed"),
            "circuit_split": self.episode.get("circuit_split"),
            "outcome": self.episode["outcome"],
        }

    @classmethod
    def from_runs(
        cls, runs: tuple[RecordedRun, ...], *, experiment: int
    ) -> list[FinalDrive]:
        """
        Select final trajectories, restricting the multi-circuit study to test.
        """
        drives = []
        for run in runs:
            for trajectory in run.trajectories:
                evaluation = trajectory["evaluation"]
                if (
                    evaluation["training_interactions"]
                    != run.completion["training_interactions"]
                ):
                    continue
                episode = evaluation["episode"]
                if experiment == 2 and episode.get("circuit_split") != "test":
                    continue
                drives.append(cls(run, episode, trajectory["transitions"]))
        return drives


def root_control_rows(rows: list[TableRow]) -> list[TableRow]:
    """
    Give each retained circuit equal weight within its training root.

    A failure contributes the observed segment without extrapolation; report
    mean coverage and completion count so truncated coverage stays visible.
    """
    output = []
    identity_keys = (
        "run_id",
        "algorithm",
        "actor_name",
        "observation_type",
        "root_identity",
        "training_interactions",
    )
    excluded = {
        *identity_keys,
        "circuit_identity",
        "circuit_seed",
        "circuit_split",
        "outcome",
    }
    for run_id in sorted({row["run_id"] for row in rows}):
        circuits = [row for row in rows if row["run_id"] == run_id]
        output.append(
            {
                **{key: circuits[0][key] for key in identity_keys},
                "circuit_count": len(circuits),
                "completed_circuit_count": sum(
                    row["outcome"] == "completed" for row in circuits
                ),
                **{
                    key: float(np.mean([row[key] for row in circuits]))
                    for key in circuits[0]
                    if key not in excluded
                },
            }
        )
    return output
