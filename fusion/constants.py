"""Framework-wide constants and enumerations for the Fusion library.

Defines the canonical set of modality identifiers, fusion strategies,
task types, data splits, precision modes, and scalar defaults used
across all Fusion modules.

Expected config params: None — this module defines only constants.
"""

from enum import Enum


# ------------------------------------------------------------------
# Modality identifiers
# ------------------------------------------------------------------

class Modality(str, Enum):
    """Supported input/output modalities.

    Inherits from ``str`` so that ``.value`` is a plain string and
    instances can be used directly as dictionary keys, JSON values,
    and ``nn.ModuleDict`` keys without explicit conversion.
    """

    VISION = "vision"
    LANGUAGE = "language"
    AUDIO = "audio"


# ------------------------------------------------------------------
# Fusion strategies
# ------------------------------------------------------------------

class FusionType(str, Enum):
    """Available multimodal fusion strategies."""

    EARLY = "early"
    LATE = "late"
    CROSS_ATTENTION = "cross_attention"
    CO_ATTENTION = "co_attention"
    GATED = "gated"
    TENSOR = "tensor"
    PERCEIVER = "perceiver"
    TRANSFORMER = "transformer"
    BILINEAR = "bilinear"
    CUSTOM = "custom"


# ------------------------------------------------------------------
# Task types
# ------------------------------------------------------------------

class TaskType(str, Enum):
    """Downstream task categories the framework supports."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"
    QA = "qa"


# ------------------------------------------------------------------
# Data splits
# ------------------------------------------------------------------

class DataSplit(str, Enum):
    """Standard train / validation / test splits."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


# ------------------------------------------------------------------
# Precision modes
# ------------------------------------------------------------------

class PrecisionType(str, Enum):
    """Numeric precision modes for training and inference."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"


# ------------------------------------------------------------------
# Scalar defaults
# ------------------------------------------------------------------

DEFAULT_MAX_SEQ_LEN: int = 512
DEFAULT_IMAGE_SIZE: int = 224
DEFAULT_AUDIO_SAMPLE_RATE: int = 16000

# ------------------------------------------------------------------
# Derived constants
# ------------------------------------------------------------------

SUPPORTED_MODALITIES: tuple = (Modality.VISION, Modality.LANGUAGE, Modality.AUDIO)


__all__ = [
    # Enums
    "Modality",
    "FusionType",
    "TaskType",
    "DataSplit",
    "PrecisionType",
    # Scalars
    "DEFAULT_MAX_SEQ_LEN",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_AUDIO_SAMPLE_RATE",
    # Derived
    "SUPPORTED_MODALITIES",
]
