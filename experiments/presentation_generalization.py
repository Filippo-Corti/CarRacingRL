"""Generate Experiment 2 presentation figures and verify its paired statistics."""

import argparse
import csv
import json
from itertools import pairwise

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from experiments.presentation_figures import (
    COLORS,
    OUTPUT,
    ROOT,
    configure_style,
    records,
    save,
)

STUDY = ROOT / "results/reported_experiments/experiment_2_revised"
OBSERVATIONS = ("frenet", "lidar")
LABELS = ("Frenet", "LiDAR")
PALETTE = (COLORS[0], COLORS[2])


def learning_figures():
    """
    Plot exploratory episode returns and held-out validation returns by root.
    """
    edges = np.arange(0, 1_000_001, 50_000)
    for kind in ("training", "evaluation"):
        positions = (edges[:-1] + edges[1:]) / 2 if kind == "training" else edges[1:]
        figure, axis = plt.subplots(figsize=(9, 5.1), layout="constrained")
        for observation, label, color in zip(
            OBSERVATIONS, LABELS, PALETTE, strict=True
        ):
            curves = []
            for root in range(10):
                path = STUDY / f"ppo-medium-{observation}-seed-{root}"
                if kind == "training":
                    rows = records(path / "episodes.jsonl")
                    rows = [row for row in rows if row["scope"] == "training"]
                    x = np.array([row["training_interactions"] for row in rows])
                    y = np.array([row["undiscounted_return"] for row in rows])
                    curve = np.array(
                        [
                            y[(x > left) & (x <= right)].mean()
                            for left, right in pairwise(edges)
                        ]
                    )
                else:
                    rows = records(path / "evaluations.jsonl")
                    rows = [
                        row
                        for row in rows
                        if row["episode"]["circuit_split"] == "validation"
                    ]
                    curve = np.array(
                        [
                            np.mean(
                                [
                                    row["episode"]["undiscounted_return"]
                                    for row in rows
                                    if row["training_interactions"] == boundary
                                ]
                            )
                            for boundary in positions
                        ]
                    )
                    assert all(
                        sum(row["training_interactions"] == boundary for row in rows)
                        == 16
                        for boundary in positions
                    )
                curves.append(curve)
            curves = np.array(curves)
            assert np.isfinite(curves).all()
            mean = curves.mean(axis=0)
            deviation = curves.std(axis=0, ddof=1)
            axis.plot(positions / 1e6, mean, label=label, color=color, linewidth=2)
            axis.fill_between(
                positions / 1e6,
                mean - deviation,
                mean + deviation,
                color=color,
                alpha=0.14,
            )
        description = (
            "Exploratory training" if kind == "training" else "Deterministic validation"
        )
        axis.set(
            title=f"PPO · {description.lower()} returns",
            xlabel="Training interactions (millions)",
            ylabel="Undiscounted episode return",
            xlim=(0, 1),
        )
        axis.grid(alpha=0.2)
        axis.legend(title="Observation", loc="lower right")
        aggregation = (
            "50k-interaction episode bins"
            if kind == "training"
            else "16 circuits per root and checkpoint"
        )
        figure.supxlabel(
            f"Equal-weight mean of ten roots ± 1 sample SD\n{aggregation}", fontsize=11
        )
        save(figure, f"experiment_2_{kind}_returns")


def control_figures():
    """
    Compare pre-action speed and throttle on root 0, test circuit 0.
    """
    trajectories = []
    for observation in OBSERVATIONS:
        folder = STUDY / f"ppo-medium-{observation}-seed-0" / "trajectories"
        for path in folder.glob("*interaction_1000000.json"):
            trajectory = json.loads(path.read_text(encoding="utf-8"))
            episode = trajectory["evaluation"]["episode"]
            if (
                episode["circuit_split"] == "test"
                and episode["circuit_identity"] == "0"
            ):
                assert episode["outcome"] == "completed"
                trajectories.append(trajectory)
                break
    assert len(trajectories) == 2
    reference = trajectories[0]
    length = reference["evaluation"]["episode"]["circuit_geometry"]["track_length"]
    reference_rows = reference["transitions"]
    distances = np.r_[0, [row["progress"] for row in reference_rows]] * length
    curved = np.array([abs(row["current_curvature"]) > 1e-8 for row in reference_rows])
    starts = np.flatnonzero(curved & ~np.r_[False, curved[:-1]])
    ends = np.flatnonzero(curved & ~np.r_[curved[1:], False]) + 1
    for metric in ("speed", "throttle"):
        figure, axis = plt.subplots(figsize=(9, 5.1), layout="constrained")
        for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
            axis.axvspan(
                distances[start],
                distances[end],
                color="#D5D9DE",
                alpha=0.55,
                label="Curved track" if index == 0 else None,
            )
        for trajectory, label, color in zip(trajectories, LABELS, PALETTE, strict=True):
            rows = trajectory["transitions"]
            x = np.r_[0, [row["progress"] for row in rows[:-1]]] * length
            y = [
                row["speed"] if metric == "speed" else row["action"][0] for row in rows
            ]
            axis.plot(x, y, label=label, color=color, linewidth=1.5)
        if metric == "throttle":
            axis.axhline(0, color="#596574", linewidth=0.8)
            axis.set_ylim(-1.05, 1.05)
        axis.set(
            title=f"{'Speed schedule' if metric == 'speed' else 'Throttle and braking'} · root 0, test circuit 0",
            xlabel="Distance along lap (m)",
            ylabel=(
                "Speed (m/s)"
                if metric == "speed"
                else "Normalized throttle / brake request"
            ),
            xlim=(0, length),
        )
        axis.grid(alpha=0.2)
        handles, labels = axis.get_legend_handles_labels()
        axis.legend(
            [handles[1], handles[2], handles[0]],
            [labels[1], labels[2], labels[0]],
            loc="lower right" if metric == "speed" else "upper right",
        )
        figure.supxlabel(
            "Final deterministic policies · Frenet 26.92 s; LiDAR 24.20 s\n"
            "Pre-action values; curve shading follows the Frenet reference trace",
            fontsize=11,
        )
        save(figure, f"experiment_2_{metric}")


def verify_statistics(seed):
    """
    Recompute outcomes, validation attainment and the matched-lap root test.
    """
    all_test = {}
    summary = {}
    indices = np.random.default_rng(seed).integers(10, size=(10_000, 10))
    for observation in OBSERVATIONS:
        roots = []
        firsts = []
        late_completion = []
        durations = []
        for root in range(10):
            folder = STUDY / f"ppo-medium-{observation}-seed-{root}"
            rows = records(folder / "evaluations.jsonl")
            test = [
                row["episode"]
                for row in rows
                if row["episode"]["circuit_split"] == "test"
            ]
            assert len(test) == 32
            all_test[observation, root] = {e["circuit_identity"]: e for e in test}
            completed = [e["lap_time"] for e in test if e["outcome"] == "completed"]
            roots.append(
                {
                    "completed": len(completed),
                    "return": np.mean([e["undiscounted_return"] for e in test]),
                    "progress": np.mean([e["maximum_progress"] for e in test]),
                    "lap_time": np.mean(completed),
                }
            )
            qualifies = []
            boundaries = np.arange(50_000, 1_000_001, 50_000)
            for boundary in boundaries:
                validation = [
                    r["episode"]
                    for r in rows
                    if r["training_interactions"] == boundary
                    and r["episode"]["circuit_split"] == "validation"
                ]
                assert len(validation) == 16
                completion = np.mean([e["outcome"] == "completed" for e in validation])
                qualifies.append(
                    completion >= 0.75
                    and np.median([e["maximum_progress"] for e in validation]) >= 0.95
                )
                if boundary > 800_000:
                    late_completion.append(completion)
            first = next(
                i for i in range(len(qualifies) - 2) if all(qualifies[i : i + 3])
            )
            firsts.append(int(boundaries[first]))
            durations.append(
                json.loads((folder / "completion.json").read_text(encoding="utf-8"))[
                    "timing"
                ]["end_to_end"]
            )
        completion_rates = np.array([r["completed"] / 32 for r in roots])
        returns = np.array([r["return"] for r in roots])
        summary[observation] = {
            "completed": sum(r["completed"] for r in roots),
            "all_circuits_roots": sum(r["completed"] == 32 for r in roots),
            "mean_return": float(np.mean([r["return"] for r in roots])),
            "root_completion_rates": completion_rates.tolist(),
            "completion_95_interval": np.quantile(
                completion_rates[indices].mean(axis=1), [0.025, 0.975]
            ).tolist(),
            "return_95_interval": np.quantile(
                returns[indices].mean(axis=1), [0.025, 0.975]
            ).tolist(),
            "return_sd": float(np.std([r["return"] for r in roots], ddof=1)),
            "mean_progress": float(np.mean([r["progress"] for r in roots])),
            "mean_lap_time": float(np.mean([r["lap_time"] for r in roots])),
            "first_interactions": firsts,
            "mean_first_interactions": float(np.mean(firsts)),
            "late_validation_completion": float(np.mean(late_completion)),
            "total_end_to_end_minutes": float(sum(durations) / 60),
        }
    completion_difference = np.array(
        summary["lidar"]["root_completion_rates"]
    ) - np.array(summary["frenet"]["root_completion_rates"])
    first_difference = np.array(summary["frenet"]["first_interactions"]) - np.array(
        summary["lidar"]["first_interactions"]
    )
    summary["paired_intervals"] = {
        "root_order": list(range(10)),
        "bootstrap_seed": seed,
        "bootstrap_resamples": 10_000,
        "lidar_minus_frenet_completion_95_interval": np.quantile(
            completion_difference[indices].mean(axis=1), [0.025, 0.975]
        ).tolist(),
        "frenet_minus_lidar_first_interactions_95_interval": np.quantile(
            first_difference[indices].mean(axis=1), [0.025, 0.975]
        ).tolist(),
    }
    differences = []
    counts = []
    for root in range(10):
        frenet, lidar = all_test["frenet", root], all_test["lidar", root]
        common = [
            identity
            for identity in frenet
            if frenet[identity]["outcome"] == lidar[identity]["outcome"] == "completed"
        ]
        counts.append(len(common))
        differences.append(
            float(
                np.mean([frenet[i]["lap_time"] - lidar[i]["lap_time"] for i in common])
            )
        )
    with (ROOT / "docs/tables/experiment_2_matched_laps.csv").open(
        encoding="utf-8"
    ) as source:
        saved = list(csv.DictReader(source))
    assert np.allclose(
        differences, [float(r["frenet_minus_lidar_lap_time"]) for r in saved]
    )
    assert counts == [int(r["common_completed_circuit_count"]) for r in saved]
    test = wilcoxon(differences, alternative="two-sided", method="exact")
    array = np.array(differences)
    interval = np.quantile(array[indices].mean(axis=1), [0.025, 0.975])
    summary["matched_laps"] = {
        "paired_circuits": sum(counts),
        "paired_roots": 10,
        "root_differences": differences,
        "counts_per_root": counts,
        "mean_frenet_minus_lidar": float(array.mean()),
        "bootstrap_seed": seed,
        "bootstrap_resamples": 10_000,
        "bootstrap_95_interval": interval.tolist(),
        "wilcoxon_statistic": float(test.statistic),
        "wilcoxon_pvalue": float(test.pvalue),
        "wilcoxon_method": "exact",
        "alternative": "two-sided",
    }
    destination = ROOT / "docs/tables/presentation_experiment_2_statistics.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    """
    Generate Experiment 2 figures and statistics with an explicit bootstrap seed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    arguments = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configure_style()
    verify_statistics(arguments.seed)
    learning_figures()
    control_figures()


if __name__ == "__main__":
    main()
