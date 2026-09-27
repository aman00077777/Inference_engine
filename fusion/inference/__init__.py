"""
Public re-exports for the inference sub-package (Phase 12).
"""

from fusion.inference.batch_inference import BatchInference, FusionDataset
from fusion.inference.predictor import FusionPredictor

__all__ = ["BatchInference", "FusionDataset", "FusionPredictor"]
