"""Custom exception hierarchy for the Fusion multimodal AI framework.

Every domain-specific error raised by Fusion modules is defined here so
that downstream consumers can catch fine-grained failures or the broad
``FusionError`` base with a single import.

Expected config params: None — this module defines only exception classes.
"""

from typing import Optional


class FusionError(Exception):
    """Root exception for all Fusion-related errors.

    Args:
        message (str): Human-readable error description.
        details (Optional[dict]): Structured diagnostic payload
            (e.g. tensor shapes, config keys) for programmatic inspection.
    """

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        self.message = message
        self.details = details
        super().__init__(message)

    def __str__(self) -> str:
        if self.details:
            formatted = ", ".join(f"{k}={v!r}" for k, v in self.details.items())
            return f"{self.message} [{formatted}]"
        return self.message


# ------------------------------------------------------------------
# Modality & encoding
# ------------------------------------------------------------------

class ModalityError(FusionError):
    """Wrong or unsupported modality encountered.

    Args:
        message (str): Description of the modality issue.
        details (Optional[dict]): Extra context (e.g. expected vs. received).
    """

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message, details)


class EncoderError(FusionError):
    """Encoder initialisation or forward-pass failure.

    Args:
        message (str): Description of the encoder failure.
        details (Optional[dict]): Extra context (e.g. layer name, input shape).
    """

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message, details)


# ------------------------------------------------------------------
# Fusion strategy
# ------------------------------------------------------------------

class FusionStrategyError(FusionError):
    """Incompatible shapes or wrong modality count in a fusion strategy.

    Args:
        message (str): Description of the strategy failure.
        details (Optional[dict]): Extra context (e.g. shapes, modality count).
    """

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message, details)


class FusionRegistryError(FusionStrategyError):
    """Errors in fusion strategy registry operations (duplicate key, missing key, etc.).

    Args:
        message (str): Description of the registry failure.
        details (Optional[dict]): Extra context (e.g. registry name, duplicate key).
    """

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message, details)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

class ConfigError(FusionError):
    """Malformed YAML, missing keys, type mismatches, or unknown registry names.

    Args:
        message (str): Description of the config issue.
        details (Optional[dict]): Extra context (e.g. key path, expected type).
    """

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message, details)


# ------------------------------------------------------------------
# Checkpoints
# ------------------------------------------------------------------

class CheckpointError(FusionError):
    """Missing/corrupted checkpoints or version mismatches.

    Args:
        message (str): Description of the checkpoint failure.
        details (Optional[dict]): Extra context (e.g. file path, version).
    """

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message, details)


# ------------------------------------------------------------------
# Data pipeline
# ------------------------------------------------------------------

class DataError(FusionError):
    """Dataset file, corruption, label, or collation issues.

    Args:
        message (str): Description of the data issue.
        details (Optional[dict]): Extra context (e.g. file path, row index).
    """

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message, details)


# ------------------------------------------------------------------
# Deployment / export
# ------------------------------------------------------------------

class DeploymentError(FusionError):
    """Export (ONNX / TorchScript) or FastAPI server failures.

    Args:
        message (str): Description of the deployment failure.
        details (Optional[dict]): Extra context (e.g. export format, endpoint).
    """

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message, details)


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------

class InferenceError(FusionError):
    """Prediction-time failures: bad input shape, missing modality, or
    forward-pass errors during inference.

    Args:
        message (str): Description of the inference failure.
        details (Optional[dict]): Extra context (e.g. input keys, shapes).
    """

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message, details)


# ------------------------------------------------------------------
# Vector search
# ------------------------------------------------------------------

class SearchError(FusionError):
    """Vector search index, dimension, or missing-dependency failures.

    Args:
        message (str): Description of the search failure.
        details (Optional[dict]): Extra context (e.g. index name, dimension).
    """

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message, details)


__all__ = [
    "FusionError",
    "ModalityError",
    "EncoderError",
    "FusionStrategyError",
    "FusionRegistryError",
    "ConfigError",
    "CheckpointError",
    "DataError",
    "DeploymentError",
    "InferenceError",
    "SearchError",
]
