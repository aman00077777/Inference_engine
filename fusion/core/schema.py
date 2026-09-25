"""Schema-level validation for multimodal pipeline inputs.

Provides ``ModalitySchema``, a declarative dataclass that describes the
expected modality contract.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from fusion.constants import Modality, TaskType
from fusion.exceptions import ModalityError


@dataclass
class ModalitySchema:
    """Declarative schema describing a multimodal pipeline's I/O contract."""

    input_modalities: List[Modality]
    output_type: TaskType
    expected_shapes: Dict[Modality, Tuple]
    optional_modalities: List[Modality] = field(default_factory=list)

    def validate(self, inputs: Dict[Modality, Any]) -> bool:
        """Validate *inputs* against this schema.

        Raises:
            ModalityError: If a required modality is missing or shape mismatch.
        """
        required = [
            m for m in self.input_modalities
            if m not in self.optional_modalities
        ]
        for modality in required:
            if modality not in inputs:
                raise ModalityError(
                    f"Required modality '{modality.value}' is missing from "
                    f"inputs. Provided: {[m.value for m in inputs.keys()]}",
                    details={
                        "missing": modality.value,
                        "provided": [m.value for m in inputs.keys()],
                    },
                )

        for modality, tensor in inputs.items():
            if modality not in self.expected_shapes:
                continue

            expected = self.expected_shapes[modality]
            actual = tuple(tensor.shape[1:])

            if len(expected) != len(actual):
                raise ModalityError(
                    f"Shape mismatch for modality '{modality.value}': "
                    f"expected {len(expected)}D (shape {expected}) excluding "
                    f"batch, got {len(actual)}D (shape {actual})",
                    details={
                        "modality": modality.value,
                        "expected_shape": expected,
                        "actual_shape": actual,
                    },
                )

            for dim_idx, (exp_dim, act_dim) in enumerate(zip(expected, actual)):
                if exp_dim != -1 and exp_dim != act_dim:
                    raise ModalityError(
                        f"Shape mismatch for modality '{modality.value}' at "
                        f"dim {dim_idx + 1} (excluding batch): expected "
                        f"{expected}, got {actual}",
                        details={
                            "modality": modality.value,
                            "dim": dim_idx + 1,
                            "expected_shape": expected,
                            "actual_shape": actual,
                        },
                    )

        return True
