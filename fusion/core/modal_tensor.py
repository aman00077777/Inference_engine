"""Lightweight tensor wrapper that binds data to its modality metadata."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from fusion.constants import Modality


@dataclass
class ModalTensor:
    """A tensor annotated with modality information."""

    data: torch.Tensor
    modality: Modality
    _embedding: Optional[torch.Tensor] = field(default=None, repr=False)
    mask: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Property alias for Universal Encoder Contract
    # ------------------------------------------------------------------

    @property
    def embedding(self) -> torch.Tensor:
        """Alias for `data` (or explicit `_embedding` if provided)."""
        return self._embedding if self._embedding is not None else self.data

    # ------------------------------------------------------------------
    # Device / Gradient Helpers
    # ------------------------------------------------------------------

    def to(self, device: torch.device) -> ModalTensor:
        return ModalTensor(
            data=self.data.to(device),
            modality=self.modality,
            _embedding=self._embedding.to(device) if self._embedding is not None else None,
            mask=self.mask.to(device) if self.mask is not None else None,
            attention_mask=self.attention_mask.to(device) if self.attention_mask is not None else None,
            metadata=self.metadata,
        )

    def detach(self) -> ModalTensor:
        return ModalTensor(
            data=self.data.detach(),
            modality=self.modality,
            _embedding=self._embedding.detach() if self._embedding is not None else None,
            mask=self.mask.detach() if self.mask is not None else None,
            attention_mask=self.attention_mask.detach() if self.attention_mask is not None else None,
            metadata=self.metadata,
        )

    # ------------------------------------------------------------------
    # Convenience Properties & Methods
    # ------------------------------------------------------------------

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    @property
    def batch_size(self) -> int:
        return int(self.data.shape[0])

    def has_nan_or_inf(self) -> bool:
        return bool(torch.isnan(self.data).any() or torch.isinf(self.data).any())

    def __repr__(self) -> str:
        return (
            f"ModalTensor(modality={self.modality.value!r}, "
            f"shape={self.shape}, "
            f"has_embedding={self.embedding is not None}, "
            f"has_mask={self.mask is not None or self.attention_mask is not None})"
        )
