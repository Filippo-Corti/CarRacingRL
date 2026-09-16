"""Restore a saved actor and its frozen normalization for presentation demos."""

import json
from pathlib import Path

import torch

from agents.models import ActorNetwork
from configs import ActorConfig
from configs.training import ObservationNormalizationConfig
from normalization import RunningObservationNormalizer
from recording.records import ObservationNormalizerStateRecord
from training.checkpointing import load_checkpoint


class SavedActor:
    """
    Run deterministic actions with the observation statistics saved in a run.

    Fields:
        * actor_network: The restored actor used for deterministic evaluation.
        * observation_normalizer: Frozen training statistics for the actor inputs.
    """

    def __init__(self, actor_network, observation_normalizer):
        self.actor_network = actor_network
        self.observation_normalizer = observation_normalizer

    def action(self, observation):
        """
        Normalize an observation without updating statistics and choose an action.
        """
        return self.actor_network.action(
            self.observation_normalizer.normalize(observation)
        )

    @classmethod
    def from_run(cls, run, *, seed):
        """
        Restore a run's final actor and normalizer using its saved configuration.
        """
        run = Path(run)
        state = load_checkpoint(run / "checkpoints/final.pt")
        configuration = json.loads((run / "config.json").read_text(encoding="utf-8"))
        dimensions = len(state["normalizer"]["sums"])
        actor_network = ActorNetwork(
            dimensions,
            ActorConfig(**state["agent"]["actor_config"]),
            torch.Generator().manual_seed(seed),
        )
        actor_network.load_state_dict(state["agent"]["actor"])
        actor_network.eval()
        observation_normalizer = RunningObservationNormalizer(
            dimensions,
            ObservationNormalizationConfig(
                **configuration["training"]["normalization"]
            ),
        )
        observation_normalizer.restore(
            ObservationNormalizerStateRecord(**state["normalizer"])
        )
        return cls(actor_network, observation_normalizer)
