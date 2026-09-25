"""Input/Output utilities for the Fusion framework.

Provides helper functions for saving and loading checkpoints,
configuration files, and JSON data.

Expected config params: None — this module defines only I/O helpers.
"""

import json
import os
from typing import Any, Dict, Optional

import torch
import yaml

from fusion.exceptions import CheckpointError, ConfigError
from fusion.utils.config import Config


# ------------------------------------------------------------------
# Checkpoint I/O
# ------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a model checkpoint to disk.

    The checkpoint dict always contains ``"epoch"`` and
    ``"model_state_dict"``.  If an optimiser is supplied its state is
    stored under ``"optimizer_state_dict"``.  Any caller-supplied
    payload (loss, metric scores, config snapshots, …) is merged in
    via *extra*.

    Args:
        path (str): Destination file path (e.g. ``"runs/epoch5.pt"``).
        model (torch.nn.Module): The model whose ``state_dict`` will
            be serialised.
        epoch (int): Current training epoch (stored for resume logic).
        optimizer (Optional[torch.optim.Optimizer]): If provided, its
            ``state_dict`` is saved alongside the model weights.
        extra (Optional[Dict[str, Any]]): Arbitrary additional keys
            to include in the checkpoint (e.g. ``{"loss": 0.42}``).

    Raises:
        CheckpointError: If the directory cannot be created or
            ``torch.save`` fails for any reason.
    """
    try:
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        checkpoint: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if extra:
            checkpoint.update(extra)

        torch.save(checkpoint, path)

    except Exception as exc:
        raise CheckpointError(
            f"Failed to save checkpoint to '{path}': {exc}",
            details={"path": path, "epoch": epoch},
        ) from exc


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load a checkpoint from disk and restore model (and optionally optimiser) state.

    Args:
        path (str): Path to the ``.pt`` / ``.pth`` checkpoint file.
        model (torch.nn.Module): Model whose weights will be restored
            in-place via ``load_state_dict``.
        optimizer (Optional[torch.optim.Optimizer]): If provided,
            its state is restored from ``"optimizer_state_dict"`` when
            that key is present in the checkpoint.
        device (Optional[torch.device]): Device to map tensors to when
            loading (e.g. ``torch.device("cpu")`` to load a GPU
            checkpoint on a CPU-only machine).  Defaults to
            ``torch.device("cpu")``.

    Returns:
        Dict[str, Any]: The full checkpoint dictionary, so callers can
        access extra fields such as ``"epoch"`` or ``"loss"``.

    Raises:
        CheckpointError: If the file does not exist, is unreadable,
            or the ``state_dict`` is incompatible with *model*.
    """
    if not os.path.isfile(path):
        raise CheckpointError(
            f"Checkpoint file not found: '{path}'",
            details={"path": path},
        )

    map_location = device if device is not None else torch.device("cpu")

    try:
        checkpoint: Dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise CheckpointError(
            f"Failed to load checkpoint from '{path}': {exc}",
            details={"path": path},
        ) from exc

    if "model_state_dict" not in checkpoint:
        raise CheckpointError(
            f"Checkpoint at '{path}' is missing 'model_state_dict'. "
            f"Found keys: {list(checkpoint.keys())}",
            details={"path": path, "found_keys": list(checkpoint.keys())},
        )

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise CheckpointError(
            f"state_dict mismatch while loading '{path}': {exc}",
            details={"path": path},
        ) from exc

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


__all__ = [
    "save_checkpoint",
    "load_checkpoint",
]
