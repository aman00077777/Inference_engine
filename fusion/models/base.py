"""
fusion/models/base.py

Phase 5.2 — Base Model (Owner: Suyash)

Defines `BaseFusionModel`, the abstract base class every fusion model in the
FUSION framework subclasses. It fixes the model contract to a three-stage
pipeline — `encode -> fuse -> predict` — and provides shared, non-abstract
plumbing (schema validation, checkpointing, and small introspection helpers)
so that every subclass gets this behavior for free.

Depends on:
    Phase 1 — Config, ConfigDict, save_checkpoint, load_checkpoint, count_parameters
    Phase 2 — Modality, ModalTensor (encoder output types)
    Phase 4 — ModalitySchema, ModalityError (input validation)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
from torch import nn

from fusion.utils.config import Config
from fusion.core.types import ConfigDict
from fusion.constants import Modality
from fusion.core.modal_tensor import ModalTensor
from fusion.core.schema import ModalitySchema
from fusion.exceptions import ModalityError
from fusion.utils.helpers import count_parameters
from fusion.utils.io import save_checkpoint, load_checkpoint


class BaseFusionModel(nn.Module, ABC):
    """
    Abstract base class for every fusion model in the framework.

    Subclasses must implement the three pipeline stages:
        encode  : raw modality inputs -> per-modality encoded tensors
        fuse    : per-modality encoded tensors -> a single fused tensor
        predict : fused tensor -> task outputs (logits, embeddings, etc.)

    `forward()` wires these three stages together and, if a `ModalitySchema`
    has been attached, validates inputs before running the pipeline.

    Concrete subclasses are typically registered into `MODEL_REGISTRY`
    (see fusion/models/registry.py) via the `@register_model("name")`
    decorator so they can be built by name/config through `build_model()`.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        # Set by subclasses (or externally) when input validation against a
        # fixed modality schema is desired. Left unset (None) means forward()
        # skips validation entirely.
        self.schema: Optional[ModalitySchema] = None

    # ------------------------------------------------------------------ #
    # Abstract pipeline stages — every subclass must implement these.
    # ------------------------------------------------------------------ #
    @abstractmethod
    def encode(self, inputs: Dict[Modality, Any]) -> Dict[Modality, ModalTensor]:
        """
        Run each modality's encoder over its raw input.

        Args:
            inputs: mapping of Modality -> raw input for that modality
                (e.g. a batch of images, a batch of tokenized text, etc.)

        Returns:
            Mapping of Modality -> ModalTensor holding the encoded
            representation for that modality.
        """
        raise NotImplementedError

    @abstractmethod
    def fuse(self, encoded: Dict[Modality, ModalTensor]) -> torch.Tensor:
        """
        Combine the per-modality encoded representations into a single
        fused tensor using this model's fusion strategy.

        Args:
            encoded: output of `encode()`.

        Returns:
            A single fused torch.Tensor ready to be passed to `predict()`.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, fused: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply the task head(s) to the fused representation.

        Args:
            fused: output of `fuse()`.

        Returns:
            Mapping of output name -> torch.Tensor (e.g. {"logits": ...}).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Concrete, shared behavior.
    # ------------------------------------------------------------------ #
    def forward(self, inputs: Dict[Modality, Any]) -> Dict[str, torch.Tensor]:
        """
        Run the full encode -> fuse -> predict pipeline.

        If `self.schema` is set, `inputs` is validated against it first;
        a `ModalityError` is raised (not swallowed) if validation fails,
        so callers see a clear, early error instead of a confusing failure
        deeper in `encode()`.
        """
        if self.schema is not None:
            self.schema.validate(inputs)

        encoded = self.encode(inputs)
        fused = self.fuse(encoded)
        outputs = self.predict(fused)
        return outputs

    def save(self, path: str) -> None:
        """
        Save this model's weights and enough metadata to reconstruct it.

        Persists:
            - state_dict:  this module's parameters/buffers
            - config:      `self.config.to_dict()`, so `load()` can rebuild
                            the exact Config used to construct this model
            - class_name:  `type(self).__name__`, so `load()` can look the
                            class up (e.g. via MODEL_REGISTRY) and dispatch
                            to the right constructor
        """
        checkpoint = {
            "state_dict": self.state_dict(),
            "config": self.config.to_dict(),
            "class_name": type(self).__name__,
        }
        save_checkpoint(checkpoint, path)

    @classmethod
    def load(cls, path: str) -> "BaseFusionModel":
        """
        Reconstruct a model from a checkpoint written by `save()`.

        This is implemented on the base class so any subclass can call
        `SubclassName.load(path)`. It:
            1. Loads the raw checkpoint dict from disk.
            2. Rebuilds a `Config` from the stored `config` dict.
            3. Instantiates `cls` with that config.
            4. Loads the saved `state_dict` into the new instance.

        Note: `cls` here should be the concrete subclass matching the
        checkpoint's `class_name` (or a caller that already knows which
        subclass to instantiate, e.g. via `MODEL_REGISTRY.build(...)`).
        """
        checkpoint = load_checkpoint(path)

        config = Config.from_dict(checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])
        return model

    def get_num_parameters(self) -> int:
        """Return the total number of parameters in this model."""
        return count_parameters(self)

    def get_config(self) -> ConfigDict:
        """Return this model's config as a plain dict."""
        return self.config.to_dict()
