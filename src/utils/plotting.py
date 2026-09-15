"""Deterministic plots generated from root-level analysis tables."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from .analysis import TableRow
from .driving import STRAIGHT_TOLERANCE, FinalDrive

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


def plot_learning_curves(rows: list[TableRow]) -> Figure:
    """
    Plot root-mean return and progress with sample-standard-deviation bands.
    """
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for label, condition_rows in _condition_groups(rows).items():
        by_boundary: dict[int, list[TableRow]] = defaultdict(list)
        for row in condition_rows:
            by_boundary[int(row["training_interactions"])].append(row)
        x = np.asarray(sorted(by_boundary), dtype=np.float64)
        for axis, metric, title in (
            (axes[0], "mean_return", "Evaluation return"),
            (axes[1], "mean_progress", "Normalized progress"),
        ):
            means = np.asarray(
                [
                    np.mean([float(row[metric]) for row in by_boundary[int(point)]])
                    for point in x
                ]
            )
            deviations = np.asarray(
                [
                    _sample_standard_deviation(
                        [float(row[metric]) for row in by_boundary[int(point)]]
                    )
                    for point in x
                ]
            )
            line = axis.plot(x, means, label=label)[0]
            axis.fill_between(
                x,
                means - deviations,
                means + deviations,
                color=line.get_color(),
                alpha=0.18,
            )
            axis.set_title(title)
            axis.set_xlabel("Training interactions")
    axes[1].set_ylim(top=max(1.0, axes[1].get_ylim()[1]))
    axes[0].set_ylabel("Root mean ± sample SD")
    axes[1].legend(fontsize="small")
    return figure


def plot_task_outcomes(
    rows: list[dict[str, Any]], root_rows: list[TableRow] | None = None
) -> Figure:
    """
    Plot final completion, progress, and return for each experiment cell.
    """
    ordered = sorted(rows, key=_cell_label)
    labels = [_cell_label(row) for row in ordered]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    metrics = (
        ("final_completion_rate", "Completion rate"),
        ("final_mean_progress", "Final progress"),
        ("final_mean_return", "Final return"),
    )
    x = np.arange(len(ordered))
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        means = [float(row[metric]["mean"]) for row in ordered]
        lower = [
            mean - float(row[metric]["confidence_interval_low"])
            for mean, row in zip(means, ordered, strict=True)
        ]
        upper = [
            float(row[metric]["confidence_interval_high"]) - mean
            for mean, row in zip(means, ordered, strict=True)
        ]
        axis.errorbar(x, means, yerr=(lower, upper), fmt="o", capsize=3)
        if root_rows is not None:
            cell_positions = {label: index for index, label in enumerate(labels)}
            for root in root_rows:
                position = next(
                    (
                        cell_positions[label]
                        for label in _cell_label_candidates(root)
                        if label in cell_positions
                    ),
                    None,
                )
                if position is not None:
                    axis.scatter(
                        position,
                        float(root[metric]),
                        facecolors="none",
                        edgecolors="black",
                        linewidths=0.7,
                        zorder=3,
                    )
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=45, ha="right")
    axes[0].set_ylim(0.0, 1.05)
    return figure


def plot_convergence_resources(rows: list[TableRow]) -> Figure:
    """
    Plot final performance and convergence cost against actor parameter count.
    """
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for label, condition_rows in _condition_groups(rows).items():
        parameters = np.asarray(
            [float(row["actor_parameters"]) for row in condition_rows]
        )
        progress = np.asarray(
            [float(row["final_mean_progress"]) for row in condition_rows]
        )
        convergence = np.asarray(
            [
                float(row["restricted_convergence_interactions"])
                for row in condition_rows
            ]
        )
        duration = np.asarray(
            [float(row["restricted_convergence_duration"]) for row in condition_rows]
        )
        axes[0].scatter(parameters, progress, label=label, alpha=0.8)
        for censored, marker in ((False, "o"), (True, "x")):
            mask = np.asarray(
                [bool(row["censored"]) == censored for row in condition_rows]
            )
            axes[1].scatter(
                parameters[mask],
                convergence[mask],
                label=label if not censored else None,
                marker=marker,
                alpha=0.8,
            )
            axes[2].scatter(
                parameters[mask],
                duration[mask],
                label=label if not censored else None,
                marker=marker,
                alpha=0.8,
            )
    axes[0].set_title("Performance versus capacity")
    axes[0].set_ylabel("Final normalized progress")
    axes[1].set_title("Convergence or censoring boundary")
    axes[1].set_ylabel("Training interactions")
    axes[2].set_title("Training time to threshold (× censored)")
    axes[2].set_ylabel("Recorded training seconds")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("Actor parameters")
    axes[2].legend(fontsize="small")
    return figure


def plot_optimization_diagnostics(rows: list[TableRow]) -> Figure:
    """
    Plot recorded loss, gradient dispersion, critic fit, and PPO clipping.

    The two dispersion panels are the ones that speak to estimator variance
    rather than to optimization progress. Prefer the cosine when comparing
    algorithms: the ratio is dominated by whichever sub-batch gradient was
    largest, and REINFORCE's magnitudes are heavy-tailed.
    """
    figure, axes = plt.subplots(2, 5, figsize=(19, 7), constrained_layout=True)
    panels = (
        ("actor_loss", "Actor loss"),
        ("critic_loss", "Critic loss"),
        ("actor_gradient_norm", "Actor gradient norm"),
        ("entropy_proxy", "Entropy proxy"),
        ("gradient_cosine_similarity", "Gradient agreement (cosine)"),
        ("gradient_signal_to_noise", "Gradient signal-to-noise"),
        ("explained_variance", "Explained variance"),
        ("approximate_kl", "Approximate KL"),
        ("clip_fraction", "PPO clip fraction"),
        ("ratio_mean", "PPO ratio mean"),
    )
    for label, condition_rows in _condition_groups(rows).items():
        by_boundary: dict[int, list[TableRow]] = defaultdict(list)
        for row in condition_rows:
            by_boundary[int(row["training_interactions"])].append(row)
        for axis, (metric, title) in zip(axes.flat, panels, strict=True):
            points: list[tuple[int, float]] = []
            for boundary, boundary_rows in sorted(by_boundary.items()):
                values = [
                    float(row[metric])
                    for row in boundary_rows
                    if row.get(metric) is not None
                ]
                if values:
                    points.append((boundary, float(np.mean(values))))
            if points:
                axis.plot(
                    [point[0] for point in points],
                    [float(point[1]) for point in points],
                    label=label,
                    alpha=0.75,
                )
            axis.set_title(title)
            axis.set_xlabel("Training interactions")
    axes[0, 0].legend(fontsize="small")
    return figure


def plot_curvature_controls(rows: list[TableRow]) -> Figure:
    """
    Plot circuit-then-root averages by straight/positive-curvature group.
    """
    figure, axes = plt.subplots(1, 3, figsize=(11, 4), constrained_layout=True)
    bins = sorted(
        {str(row["curvature_bin"]) for row in rows},
        key=lambda name: (name != "straight", name),
    )
    for label, condition_rows in _condition_groups(rows).items():
        for axis, metric, title in (
            (axes[0], "mean_speed", "Speed"),
            (axes[1], "mean_throttle", "Throttle / brake"),
            (axes[2], "mean_absolute_steering", "Absolute steering"),
        ):
            values: list[float] = []
            for curvature_bin in bins:
                selected = [
                    row
                    for row in condition_rows
                    if row["curvature_bin"] == curvature_bin
                ]
                roots: dict[str, list[float]] = defaultdict(list)
                for row in selected:
                    roots[str(row["run_id"])].append(float(row[metric]))
                values.append(
                    float(np.mean([np.mean(values) for values in roots.values()]))
                    if selected
                    else np.nan
                )
            axis.plot(bins, values, marker="o", label=label)
            axis.set_title(title)
            axis.set_xlabel("Circuit-relative curvature group")
    axes[0].legend(fontsize="small")
    return figure


def plot_circuit_geometry(rows: list[TableRow]) -> Figure:
    """
    Plot held-out progress against recorded circuit length and curvature.
    """
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    markers = {"completed": "o", "crashed": "x", "time_limit": "^"}
    for label, condition_rows in _condition_groups(rows).items():
        for outcome, marker in markers.items():
            selected = [row for row in condition_rows if row["outcome"] == outcome]
            if not selected:
                continue
            axes[0].scatter(
                [float(row["circuit_length"]) for row in selected],
                [float(row["maximum_progress"]) for row in selected],
                marker=marker,
                label=f"{label}/{outcome}",
                alpha=0.75,
            )
            axes[1].scatter(
                [float(row["curvature_q90"]) for row in selected],
                [float(row["maximum_progress"]) for row in selected],
                marker=marker,
                label=f"{label}/{outcome}",
                alpha=0.75,
            )
    axes[0].set_xlabel("Circuit length")
    axes[1].set_xlabel("90th-percentile absolute curvature")
    for axis in axes:
        axis.set_ylabel("Maximum normalized progress")
    axes[0].legend(fontsize="x-small")
    return figure


def plot_comparison_curves(
    rows: list[TableRow], *, fixed_key: str, varying_key: str
) -> Figure:
    """
    Compare one factor per panel with thin root traces and thick root means.
    """
    order = (
        ("tiny", "small", "medium", "large")
        if fixed_key == "actor_name"
        else ("reinforce", "a2c", "ppo")
    )
    panels = [name for name in order if any(row[fixed_key] == name for row in rows)]
    figure, axes = plt.subplots(
        len(panels),
        2,
        figsize=(11, 3 * len(panels)),
        squeeze=False,
        constrained_layout=True,
    )
    conditions = (
        ("tiny", "small", "medium", "large")
        if varying_key == "actor_name"
        else ("reinforce", "a2c", "ppo")
    )
    for panel, name in enumerate(panels):
        for color_index, condition in enumerate(conditions):
            selected = [
                row
                for row in rows
                if row[fixed_key] == name and row[varying_key] == condition
            ]
            if not selected:
                continue
            color = f"C{color_index}"
            for column, metric in enumerate(("mean_return", "completion_rate")):
                axis = axes[panel, column]
                for root in sorted({row["root_identity"] for row in selected}):
                    trace = sorted(
                        [row for row in selected if row["root_identity"] == root],
                        key=lambda row: row["training_interactions"],
                    )
                    axis.plot(
                        [row["training_interactions"] / 1e6 for row in trace],
                        [row[metric] for row in trace],
                        color=color,
                        alpha=0.22,
                        linewidth=0.7,
                    )
                boundaries = sorted({row["training_interactions"] for row in selected})
                axis.plot(
                    np.asarray(boundaries) / 1e6,
                    [
                        np.mean(
                            [
                                row[metric]
                                for row in selected
                                if row["training_interactions"] == boundary
                            ]
                        )
                        for boundary in boundaries
                    ],
                    color=color,
                    linewidth=2,
                    label=condition,
                )
                axis.set_title(f"{name}: {'return' if column == 0 else 'completion'}")
                axis.set_xlabel("Training interactions (millions)")
                axis.grid(alpha=0.15)
        axes[panel, 0].legend(fontsize="small")
        axes[panel, 1].set_ylim(-0.04, 1.04)
    return figure


def plot_control_summaries(rows: list[TableRow], *, experiment: int) -> Figure:
    """
    Show every root's control statistics with an equally weighted root mean.
    """
    groups = _condition_groups(rows)
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    metrics = (
        ("mean_speed", "Mean speed (m/s)"),
        ("braking_fraction", "Fraction braking"),
        ("near_zero_throttle_fraction", "Fraction |throttle| ≤ 0.05"),
        ("throttle_standard_deviation", "Throttle standard deviation"),
        ("mean_steering_change", "Mean adjacent steering change"),
        ("steering_reversals_per_time", "Steering sign reversals / s"),
    )
    for axis, (metric, title) in zip(axes.flat, metrics, strict=True):
        for index, (label, roots) in enumerate(groups.items()):
            values = [float(row[metric]) for row in roots]
            offsets = np.linspace(-0.14, 0.14, len(values))
            axis.scatter(
                index + offsets, values, color=f"C{index % 10}", alpha=0.65, s=20
            )
            axis.scatter(index, np.mean(values), color="black", marker="_", s=150)
        axis.set_xticks(
            range(len(groups)), list(groups), rotation=50, ha="right", fontsize=8
        )
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.15)
    figure.suptitle(
        "Final controls: each dot is a training root; black marks show means"
    )
    return figure


def plot_control_traces(drives: list[FinalDrive], *, experiment: int) -> Figure:
    """
    Illustrate signed controls and driving lines for fixed root/circuit choices.

    Distance is shifted to pre-action time by FinalDrive. Grey shading marks
    curved samples of the reference trace, which is the first listed drive;
    uncovered portions after a failure remain unshaded rather than invented.
    """
    panels = (
        sorted({drive.run.algorithm for drive in drives})
        if experiment == 1
        else ["ppo"]
    )
    figure, axes = plt.subplots(
        len(panels),
        4,
        figsize=(17, 3.4 * len(panels)),
        squeeze=False,
        constrained_layout=True,
    )
    for panel, algorithm in enumerate(panels):
        selected = [drive for drive in drives if drive.run.algorithm == algorithm]
        selected.sort(
            key=lambda drive: (drive.run.actor_name, drive.run.observation_type)
        )
        for index, drive in enumerate(selected):
            rows = drive.aligned_rows()
            distance = np.asarray([row["distance"] for row in rows])
            label = (
                drive.run.actor_name if experiment == 1 else drive.run.observation_type
            ) + f" ({drive.episode['outcome']})"
            for column, (metric, title) in enumerate(
                (
                    ("speed", "Speed (m/s)"),
                    ("throttle", "Throttle / brake"),
                    ("requested_steering", "Requested steering"),
                )
            ):
                axis = axes[panel, column]
                axis.plot(
                    distance,
                    [row[metric] for row in rows],
                    label=label,
                    linewidth=0.9,
                    alpha=0.85,
                )
                if index == 0:
                    axis.fill_between(
                        distance,
                        0,
                        1,
                        where=np.abs([row["curvature"] for row in rows])
                        > STRAIGHT_TOLERANCE,
                        transform=axis.get_xaxis_transform(),
                        color="grey",
                        alpha=0.12,
                        linewidth=0,
                    )
                axis.set_title(f"{algorithm}: {title}")
                axis.set_xlabel("Pre-action distance from start (m)")
                if column > 0:
                    axis.set_ylim(-1.05, 1.05)
                    axis.axhline(0, color="grey", linewidth=0.4)
            axes[panel, 3].plot(
                [row["x"] for row in rows],
                [row["y"] for row in rows],
                label=label,
                linewidth=1,
            )
        axes[panel, 0].legend(fontsize=7)
        axes[panel, 3].set_aspect("equal")
        axes[panel, 3].set_title("XY driving line")
        axes[panel, 3].set_xlabel("x (m)")
        axes[panel, 3].set_ylabel("y (m)")
    figure.suptitle(
        "Illustrative root 0 final policies"
        + (" — test circuit 0" if experiment == 2 else "")
    )
    return figure


def save_figure(figure: Figure, path: str | Path) -> None:
    """
    Save one plot with fixed rendering metadata and close its resources.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination,
        dpi=150,
        metadata={"Software": "reinforcement-learning-car-racing"},
    )
    plt.close(figure)


def _condition_groups(rows: list[TableRow]) -> dict[str, list[TableRow]]:
    groups: dict[str, list[TableRow]] = defaultdict(list)
    for row in rows:
        groups[_condition_label(row)].append(row)
    return dict(sorted(groups.items()))


def _condition_label(row: TableRow) -> str:
    observation = row.get("observation_type")
    algorithm = row.get("algorithm")
    actor = row.get("actor_name")
    return "/".join(str(value) for value in (algorithm, actor, observation) if value)


def _cell_label(row: dict[str, Any]) -> str:
    return "/".join(
        str(row[key])
        for key in ("algorithm", "actor_name", "observation_type")
        if row.get(key) is not None
    )


def _cell_label_candidates(row: dict[str, Any]) -> tuple[str, ...]:
    """
    Return labels for complete, actor-size, and observation-only cell designs.
    """
    return (
        _cell_label(row),
        "/".join(str(row[key]) for key in ("algorithm", "actor_name") if row.get(key)),
        str(row.get("observation_type", "")),
    )


def _sample_standard_deviation(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
