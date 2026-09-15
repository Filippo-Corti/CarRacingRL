import matplotlib.pyplot as plt

figure, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
for algorithm in ("reinforce", "a2c", "ppo"):
    rows = sorted(
        (row for row in SUMMARIES if row["algorithm"] == algorithm),
        key=lambda row: row["actor_parameters"],
    )
    parameters = [row["actor_parameters"] for row in rows]
    axes[0].scatter(parameters, [row["final_mean_return"] for row in rows], label=algorithm)
    axes[1].scatter(parameters, [row["final_mean_progress"] for row in rows], label=algorithm)
    axes[2].scatter(parameters, [row["end_to_end_duration"] / 60 for row in rows], label=algorithm)
for axis, title, ylabel in zip(
    axes,
    ("Final return", "Final progress", "End-to-end cost"),
    ("deterministic return", "normalized progress", "minutes"),
):
    axis.set(title=title, xlabel="actor parameters", ylabel=ylabel, xscale="log")
    axis.grid(alpha=0.3)
    axis.legend()
plt.show()
