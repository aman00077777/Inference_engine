"""Type aliases for the Fusion framework.

Centralised type definitions used across all core modules.  These aliases
provide consistent typing for tensor dictionaries, modality mappings,
embedding containers, and component configuration dicts.

Expected config params: None — this module defines only type aliases.
"""

from typing import Any, Dict, List, Tuple

import torch

from fusion.constants import Modality


# Mapping of arbitrary string keys to tensors (e.g. named layer outputs).
TensorDict = Dict[str, torch.Tensor]

# Mapping of Modality enum members to their corresponding tensors.
ModalityDict = Dict[Modality, torch.Tensor]

# Mapping of string keys to embedding tensors.
EmbeddingDict = Dict[str, torch.Tensor]

# Generic configuration dictionary used across components.
ConfigDict = Dict[str, Any]

__all__ = ["TensorDict", "ModalityDict", "EmbeddingDict", "ConfigDict"]
