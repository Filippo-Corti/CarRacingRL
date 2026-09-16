"""Generate the static figures used by the project presentation notebooks."""

import argparse
import json
from itertools import pairwise, product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from agents.models import ScriptedFrenetPolicy
from configs import EnvironmentConfig, StartStateConfig
from envs import RacingEnv, TrackWithGeometry

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/figures/presentation"
ALGORITHMS = ("reinforce", "a2c", "ppo")
NAMES = {"reinforce": "REINFORCE", "a2c": "A2C+GAE", "ppo": "PPO"}
SIZES = ("tiny", "small", "medium", "large")
COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")


def save(figure, name):
    """
    Save a presentation figure at print resolution.
    """
    figure.savefig(OUTPUT / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def records(path):
    """
    Read the saved per-episode or per-evaluation records.
    """
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def track_figures(seed):
    """
    Draw three generated circuits and capture the broadcast racing renderer.
    """
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.7), layout="constrained")
    for offset, axis in enumerate(axes):
        geometry = TrackWithGeometry.generate(seed + offset)
        track = geometry.track
        for boundary in (geometry.left_boundary, geometry.right_boundary):
            closed = np.vstack((boundary, boundary[0]))
            axis.plot(*closed.T, color="#596574", linewidth=1.3)
        axis.plot(
            np.r_[track.x, track.x[0]],
            np.r_[track.y, track.y[0]],
            color=COLORS[0],
            linewidth=1.4,
            linestyle="--",
        )
        index = track.start_index
        gate = np.stack((geometry.left_boundary[index], geometry.right_boundary[index]))
        axis.plot(*gate.T, color=COLORS[2], linewidth=3)
        axis.set(
            title=f"Seed {seed + offset} · length {track.track_length:.0f} m",
            xlabel="x (m)",
            ylabel="y (m)",
            aspect="equal",
        )
        axis.margins(0.12)
        axis.grid(alpha=0.15)
    figure.suptitle("Procedurally generated racing circuits", fontsize=17)
    figure.legend(
        handles=[
            Line2D([], [], color="#596574", label="Track boundaries"),
            Line2D([], [], color=COLORS[0], linestyle="--", label="Centerline"),
            Line2D([], [], color=COLORS[2], linewidth=3, label="Start / finish"),
        ],
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    save(figure, "generated_tracks")
    environment = RacingEnv(
        TrackWithGeometry.load(ROOT / "tracks/experiment_1.json"),
        config=EnvironmentConfig(start=StartStateConfig(randomized=False)),
        render_mode="rgb_array",
    )
    observation, _ = environment.reset(seed=seed)
    controller = ScriptedFrenetPolicy()
    try:
        for _ in range(240):
            observation, _, terminated, truncated, _ = environment.step(
                controller.action(observation)
            )
            if terminated or truncated:
                break
        figure, axis = plt.subplots(figsize=(12, 7.5), layout="constrained")
        frame = environment.render()
        assert frame is not None
        axis.imshow(frame)
        axis.set_title("Racing simulation · reference controller", fontsize=17, pad=12)
        axis.axis("off")
        save(figure, "racing_simulation")
    finally:
        environment.close()


def result_figures():
    """
    Plot all sixty runs with equal root weights and explicit aggregation labels.
    """
    runs = {}
    for study in ("experiment_1", "experiment_1_extension"):
        for path in (ROOT / "results/reported_experiments" / study).iterdir():
            if not (path / "completion.json").exists():
                continue
            algorithm, size = path.name.split("-")[:2]
            runs.setdefault((algorithm, size), []).append(path)
    assert sum(map(len, runs.values())) == 60
    finals = []
    timing = []
    for algorithm in ALGORITHMS:
        for size in SIZES:
            paths = sorted(runs[algorithm, size])
            evaluations = [records(path / "evaluations.jsonl") for path in paths]
            finals.append(
                [rows[-1]["episode"]["undiscounted_return"] for rows in evaluations]
            )
            completion = [
                json.loads((path / "completion.json").read_text(encoding="utf-8"))
                for path in paths
            ]
            durations = np.array(
                [
                    [
                        row["timing"][key]
                        for key in (
                            "collection",
                            "optimization",
                            "evaluation",
                            "end_to_end",
                        )
                    ]
                    for row in completion
                ]
            )
            timing.append(durations.mean(axis=0))

    figure, axis = plt.subplots(figsize=(14, 5.5), layout="constrained")
    # Enumerate every root resample, matching the five-root reporting protocol.
    median_intervals = [
        np.quantile(
            [np.median(sample) for sample in product(values, repeat=len(values))],
            [0.025, 0.975],
        )
        for values in finals
    ]
    boxes = axis.boxplot(
        finals,
        patch_artist=True,
        notch=True,
        conf_intervals=median_intervals,
        showfliers=True,
        widths=0.6,
    )
    for index, box in enumerate(boxes["boxes"]):
        color = COLORS[index // 4]
        box.set(facecolor=color, alpha=0.45, edgecolor=color)
        boxes["fliers"][index].set(
            marker="o", markerfacecolor=color, markeredgecolor=color, markersize=4
        )
    for median in boxes["medians"]:
        median.set(color="#202A35", linewidth=1.7)
    labels = [f"{NAMES[a]}\n{s.title()}" for a in ALGORITHMS for s in SIZES]
    axis.set_xticks(range(1, 13), labels, fontsize=10)
    minimum = min(map(min, finals))
    maximum = max(map(max, finals))
    axis.set(
        title="Final deterministic driving performance · all 60 runs",
        ylabel="Undiscounted episode return",
        ylim=(
            minimum - 0.045 * (maximum - minimum),
            maximum + 0.075 * (maximum - minimum),
        ),
    )
    axis.grid(axis="y", alpha=0.2)
    for x in (4.5, 8.5):
        axis.axvline(x, color="#B8BFC7", linewidth=0.8)
    figure.supxlabel(
        "Boxes: quartiles and median; notches: 95% bootstrap median interval\n"
        "Whiskers: 1.5 × IQR; points: outliers only",
        fontsize=11,
    )
    save(figure, "final_returns")

    for algorithm in ALGORITHMS:
        for kind in ("training", "evaluation"):
            figure, axis = plt.subplots(figsize=(9, 5.1), layout="constrained")
            edges = np.arange(0, 2_000_001, 50_000)
            positions = (
                (edges[:-1] + edges[1:]) / 2 if kind == "training" else edges[1:]
            )
            for size, color in zip(SIZES, COLORS, strict=True):
                curves = []
                for path in sorted(runs[algorithm, size]):
                    if kind == "training":
                        rows = records(path / "episodes.jsonl")
                        rows = [row for row in rows if row["scope"] == "training"]
                        x = np.array([row["training_interactions"] for row in rows])
                        y = np.array([row["undiscounted_return"] for row in rows])
                        curve = np.array(
                            [
                                (
                                    y[(x > left) & (x <= right)].mean()
                                    if np.any((x > left) & (x <= right))
                                    else np.nan
                                )
                                for left, right in pairwise(edges)
                            ]
                        )
                    else:
                        rows = records(path / "evaluations.jsonl")
                        # Root averages require the same checkpoint grid.
                        assert np.array_equal(
                            [row["training_interactions"] for row in rows], positions
                        )
                        curve = np.array(
                            [row["episode"]["undiscounted_return"] for row in rows]
                        )
                    curves.append(curve)
                curves = np.array(curves)
                mean = np.nanmean(curves, axis=0)
                deviation = np.nanstd(curves, axis=0, ddof=1)
                axis.plot(
                    positions / 1e6, mean, label=size.title(), color=color, linewidth=2
                )
                axis.fill_between(
                    positions / 1e6,
                    mean - deviation,
                    mean + deviation,
                    color=color,
                    alpha=0.14,
                )
            description = (
                "Exploratory training"
                if kind == "training"
                else "Deterministic evaluation"
            )
            axis.set(
                title=f"{NAMES[algorithm]} · {description.lower()} returns",
                xlabel="Training interactions (millions)",
                ylabel="Undiscounted episode return",
                xlim=(0, 2),
            )
            axis.grid(alpha=0.2)
            axis.legend(title="Actor size", ncol=4, loc="lower right", frameon=True)
            aggregation = (
                "50k-interaction episode bins"
                if kind == "training"
                else "50k-interaction checkpoints"
            )
            figure.supxlabel(
                f"Equal-weight mean of five roots ± 1 sample SD · {aggregation}",
                fontsize=11,
            )
            save(figure, f"{algorithm}_{kind}_returns")

    timing = np.array(timing)
    overhead = timing[:, 3] - timing[:, :3].sum(axis=1)
    shares = np.column_stack((timing[:, :3], overhead)) / timing[:, 3, None] * 100
    figure, axis = plt.subplots(figsize=(12, 6.5), layout="constrained")
    left = np.zeros(12)
    for phase, color, values in zip(
        ("Collection", "Optimization", "Evaluation", "Other overhead"),
        (*COLORS[:3], "#939CA6"),
        shares.T,
        strict=True,
    ):
        axis.barh(
            np.arange(12), values, left=left, color=color, label=phase, height=0.7
        )
        for index, (start, value) in enumerate(zip(left, values, strict=True)):
            if value > 8:
                axis.text(
                    start + value / 2,
                    index,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if phase == "Collection" else "#16212B",
                    fontsize=10,
                )
        left += values
    axis.set_yticks(
        np.arange(12), [f"{NAMES[a]} / {s.title()}" for a in ALGORITHMS for s in SIZES]
    )
    axis.invert_yaxis()
    axis.set(
        title="Where the full-budget run time goes",
        xlabel="Share of mean end-to-end duration (%)",
        xlim=(0, 100),
    )
    figure.legend(loc="outside lower center", ncol=4, frameon=False)
    save(figure, "runtime_phases")


def main():
    """
    Generate all presentation images using an explicit track seed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)
    arguments = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    track_figures(arguments.seed)
    result_figures()


if __name__ == "__main__":
    main()
