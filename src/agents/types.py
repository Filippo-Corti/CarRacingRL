"""The records an agent and an engine exchange, independent of either one."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from training.buffers import Trajectory
    from training.multienvs import VectorRollout


class CollectionMode(StrEnum):
    """
    Specify whether an agent learns from whole episodes or fixed rollouts.
    """

    COMPLETE_EPISODES = "complete_episodes"
    FIXED_ROLLOUT = "fixed_rollout"


@dataclass(frozen=True, slots=True)
class CollectedAction:
    """
    Record one training action and the detached quantities it produced.

    The collector sends `env_action` to the environment while retaining
    `raw_action` so a squashed Gaussian policy can later score that action
    exactly. Actor-critic collectors additionally retain their value estimate.

    Fields:
        * raw_action: Action sampled before policy-specific post-processing.
        * env_action: Bounded action sent to the environment.
        * behaviour_log_probability: Collection-policy action log probability.
        * current_value: Critic estimate at the acted-on observation, if present.
    """

    raw_action: NDArray[np.float32]
    env_action: NDArray[np.float32]
    behaviour_log_probability: float | None
    current_value: float | None

    def __post_init__(self) -> None:
        """
        Preserve private float32 action copies with one shared vector shape.
        """
        raw_action = np.asarray(self.raw_action, dtype=np.float32)
        env_action = np.asarray(self.env_action, dtype=np.float32)
        if (
            raw_action.ndim != 1
            or env_action.ndim != 1
            or raw_action.shape != env_action.shape
        ):
            raise ValueError("Collected raw and environment actions must match.")
        raw_action = raw_action.copy()
        env_action = env_action.copy()
        raw_action.setflags(write=False)
        env_action.setflags(write=False)
        object.__setattr__(self, "raw_action", raw_action)
        object.__setattr__(self, "env_action", env_action)
        if self.behaviour_log_probability is not None:
            object.__setattr__(
                self, "behaviour_log_probability", float(self.behaviour_log_probability)
            )
        if self.current_value is not None:
            object.__setattr__(self, "current_value", float(self.current_value))


@dataclass(frozen=True, slots=True)
class CollectedActionBatch:
    """
    Record one detached training action per vector-environment row.

    Each `NDArray[np.float32]` is a NumPy array of single-precision floating
    point values. Its leading dimension identifies the vector-environment row,
    so every row keeps the raw action needed to score the bounded action later.

    Fields:
        * raw_actions: Pre-squash actions with shape `(environments, actions)`.
        * env_actions: Bounded environment actions with the same shape.
        * behaviour_log_probabilities: One joint action log probability per row.
        * current_values: One critic estimate per row, or `None` for REINFORCE.
    """

    raw_actions: NDArray[np.float32]
    env_actions: NDArray[np.float32]
    behaviour_log_probabilities: NDArray[np.float32]
    current_values: NDArray[np.float32] | None

    def __post_init__(self) -> None:
        """
        Preserve private float32 copies and require one common batch dimension.
        """
        raw_actions = np.asarray(self.raw_actions, dtype=np.float32)
        env_actions = np.asarray(self.env_actions, dtype=np.float32)
        probabilities = np.asarray(self.behaviour_log_probabilities, dtype=np.float32)
        if raw_actions.ndim != 2 or env_actions.shape != raw_actions.shape:
            raise ValueError("Batched raw and environment actions must match.")
        if probabilities.shape != (raw_actions.shape[0],):
            raise ValueError("Batched probabilities require one value per action row.")
        values = self.current_values
        if values is not None:
            values = np.asarray(values, dtype=np.float32)
            if values.shape != (raw_actions.shape[0],):
                raise ValueError("Batched critic values require one value per row.")
            values = values.copy()
            values.setflags(write=False)
        raw_actions = raw_actions.copy()
        env_actions = env_actions.copy()
        probabilities = probabilities.copy()
        raw_actions.setflags(write=False)
        env_actions.setflags(write=False)
        probabilities.setflags(write=False)
        object.__setattr__(self, "raw_actions", raw_actions)
        object.__setattr__(self, "env_actions", env_actions)
        object.__setattr__(self, "behaviour_log_probabilities", probabilities)
        object.__setattr__(self, "current_values", values)


@dataclass(frozen=True, slots=True)
class AgentUpdateInput:
    """
    Carry whatever one optimizer step is allowed to learn from.

    There is no field here, on purpose. What an update needs depends entirely
    on which boundary released it, and the two boundaries have nothing in
    common: a Monte Carlo estimator needs finished episodes and an actor-critic
    needs a fixed rollout. A single class carrying both would be half empty
    whichever way it was built, and would have to be told by a mode flag which
    half to believe. The subclasses make that unrepresentable instead.
    """


@dataclass(frozen=True, slots=True)
class CompleteEpisodesInput(AgentUpdateInput):
    """
    Input data for a Monte Carlo Agent (e.g., running REINFORCE) to perform its update.
    The data is in the shape of a list of full trajectories, from episode start to finish.

    Fields:
        * episodes: Complete trajectories, one per finished episode.
    """

    episodes: tuple[Trajectory, ...]

    def __post_init__(self) -> None:
        """
        Require episodes that actually ended, since the returns depend on it.
        """
        if not self.episodes:
            raise ValueError("A complete-episode update requires episodes.")
        if not all(episode.is_complete for episode in self.episodes):
            raise ValueError("A complete-episode update requires ended episodes.")


@dataclass(frozen=True, slots=True)
class FixedRolloutInput(AgentUpdateInput):
    """
    Input data for an Actor-Critic Agent (e.g., running A2C or PPO) to perform its update.
    The data is in the shape of a list of transitions, not organized by episodes/trajectories.

    Fields:
        * rollout: Collected transitions in time-by-worker order.
    """

    rollout: VectorRollout

    def __post_init__(self) -> None:
        """
        Reject an empty rollout, whose targets would have no meaningful shape.
        """
        if not self.rollout.transition_count:
            raise ValueError("A fixed-rollout update requires a transition.")


@dataclass(frozen=True, slots=True)
class AgentUpdateOutput:
    """
    Return scalar diagnostics generated by one completed optimizer update.

    Fields:
        * diagnostics: JSON-compatible scalar diagnostics owned by the algorithm.
    """

    diagnostics: dict[str, float | int | None] = field(default_factory=dict)
