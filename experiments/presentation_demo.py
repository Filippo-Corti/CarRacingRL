"""Restore saved policies and display their racing views in presentation demos."""

import json
from base64 import b64encode
from html import escape
from io import BytesIO
from pathlib import Path

import torch
from IPython.display import HTML, display
from PIL import Image

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


class RacingPairViewer:
    """
    Update two labeled racing views together in one width-limited notebook row.

    Fields:
        * labels: Captions identifying the two policies.
        * width: Maximum combined display width in pixels.
        * display_handle: Notebook output updated in place during the demo.
    """

    def __init__(self, frames, *, labels, width=1200):
        self.labels = labels
        self.width = width
        display_handle = display(self._html(frames), display_id=True)
        # Updating in place requires the display handle provided by a notebook.
        assert display_handle is not None
        self.display_handle = display_handle

    def update(self, frames):
        """
        Replace both displayed frames while retaining the same output row.
        """
        self.display_handle.update(self._html(frames))

    def _html(self, frames):
        panels = []
        for label, frame in zip(self.labels, frames, strict=True):
            buffer = BytesIO()
            Image.fromarray(frame).save(buffer, format="JPEG")
            encoded = b64encode(buffer.getvalue()).decode("ascii")
            panels.append(
                '<div style="width:50%;min-width:0;text-align:center;">'
                f'<p style="font-family:serif;font-size:18px;">{escape(label)}</p>'
                f'<img src="data:image/jpeg;base64,{encoded}" '
                f'alt="{escape(label)} racing view" style="width:100%;">'
                "</div>"
            )
        return HTML(
            f'<div style="display:flex;max-width:{self.width}px;width:100%;">'
            + "".join(panels)
            + "</div>"
        )
