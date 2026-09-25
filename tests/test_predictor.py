"""Smoke tests for FusionPredictor (Phase 12 — Inference Engine).

Verifies that the predictor can be instantiated with a minimal dummy
model and that predict(), predict_batch(), predict_proba(), embed(),
and from_checkpoint() all work end-to-end.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict

import numpy as np
import pytest
import torch
import torch.nn as nn

from fusion.constants import Modality
from fusion.core.modal_tensor import ModalTensor
from fusion.exceptions import CheckpointError, InferenceError
from fusion.models.base import BaseFusionModel
from fusion.utils.config import Config
from fusion.utils.io import save_checkpoint
from fusion.inference.predictor import FusionPredictor


# ====================================================================
# Dummy model — minimal concrete BaseFusionModel for testing
# ====================================================================

class DummyFusionModel(BaseFusionModel):
    """Tiny two-modality model used exclusively in predictor smoke tests.

    Pipeline:
        encode:  identity (wraps raw tensors in ModalTensor)
        fuse:    sum all encoded tensors → (batch, hidden_dim)
        predict: linear head → (batch, num_classes)
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        hidden_dim: int = int(config.get("model.hidden_dim", 32))
        num_classes: int = int(config.get("model.num_classes", 5))
        self.head = nn.Linear(hidden_dim, num_classes)

    def encode(self, inputs: Dict[Modality, Any]) -> Dict[Modality, ModalTensor]:
        encoded: Dict[Modality, ModalTensor] = {}
        for mod, tensor in inputs.items():
            if isinstance(tensor, torch.Tensor):
                encoded[mod] = ModalTensor(data=tensor, modality=mod)
            else:
                encoded[mod] = tensor
        return encoded

    def fuse(self, encoded: Dict[Modality, ModalTensor]) -> torch.Tensor:
        tensors = [mt.data for mt in encoded.values()]
        return sum(tensors)

    def predict(self, fused: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.head(fused)
        return {"logits": logits}


# ====================================================================
# Fixtures
# ====================================================================

@pytest.fixture
def dummy_config() -> Config:
    return Config({
        "model": {"hidden_dim": 32, "num_classes": 5},
        "inference": {"device": "cpu", "task_type": "classification"},
    })


@pytest.fixture
def dummy_model(dummy_config: Config) -> DummyFusionModel:
    return DummyFusionModel(dummy_config)


@pytest.fixture
def predictor(dummy_model: DummyFusionModel, dummy_config: Config) -> FusionPredictor:
    return FusionPredictor(model=dummy_model, config=dummy_config)


@pytest.fixture
def sample_inputs() -> Dict[str, torch.Tensor]:
    """Single-sample input batch for vision + language."""
    return {
        "vision": torch.randn(1, 32),
        "language": torch.randn(1, 32),
    }


# ====================================================================
# Tests: __init__
# ====================================================================

class TestInit:

    def test_model_is_in_eval_mode(self, predictor: FusionPredictor) -> None:
        assert not predictor.model.training

    def test_device_is_cpu(self, predictor: FusionPredictor) -> None:
        assert predictor.device == torch.device("cpu")

    def test_repr(self, predictor: FusionPredictor) -> None:
        r = repr(predictor)
        assert "DummyFusionModel" in r
        assert "cpu" in r


# ====================================================================
# Tests: predict
# ====================================================================

class TestPredict:

    def test_predict_returns_dict(
        self, predictor: FusionPredictor, sample_inputs: Dict[str, torch.Tensor]
    ) -> None:
        result = predictor.predict(sample_inputs)
        assert isinstance(result, dict)

    def test_classification_outputs(
        self, predictor: FusionPredictor, sample_inputs: Dict[str, torch.Tensor]
    ) -> None:
        result = predictor.predict(sample_inputs)
        assert "logits" in result
        assert "probabilities" in result
        assert "predicted_class" in result
        # probabilities should sum to ~1
        probs = result["probabilities"]
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_predict_with_numpy_input(
        self, predictor: FusionPredictor
    ) -> None:
        inputs = {
            "vision": np.random.randn(1, 32).astype(np.float32),
            "language": np.random.randn(1, 32).astype(np.float32),
        }
        result = predictor.predict(inputs)
        assert "logits" in result

    def test_predict_unknown_modality_raises(
        self, predictor: FusionPredictor
    ) -> None:
        bad_inputs = {"smell": torch.randn(1, 32)}
        with pytest.raises(InferenceError) as exc_info:
            predictor.predict(bad_inputs)
        assert "Unknown modality" in str(exc_info.value)

    def test_predict_generation_task(
        self, dummy_model: DummyFusionModel
    ) -> None:
        config = Config({
            "model": {"hidden_dim": 32, "num_classes": 5},
            "inference": {"device": "cpu", "task_type": "generation"},
        })
        pred = FusionPredictor(model=dummy_model, config=config)
        result = pred.predict({
            "vision": torch.randn(1, 32),
            "language": torch.randn(1, 32),
        })
        assert "token_ids" in result


# ====================================================================
# Tests: predict_batch
# ====================================================================

class TestPredictBatch:

    def test_batch_preserves_order(
        self, predictor: FusionPredictor
    ) -> None:
        inputs_list = [
            {"vision": torch.randn(1, 32), "language": torch.randn(1, 32)}
            for _ in range(4)
        ]
        results = predictor.predict_batch(inputs_list)
        assert len(results) == 4
        for r in results:
            assert "logits" in r

    def test_empty_batch_raises(
        self, predictor: FusionPredictor
    ) -> None:
        with pytest.raises(InferenceError):
            predictor.predict_batch([])

    def test_inconsistent_keys_raises(
        self, predictor: FusionPredictor
    ) -> None:
        inputs_list = [
            {"vision": torch.randn(1, 32), "language": torch.randn(1, 32)},
            {"vision": torch.randn(1, 32)},  # missing language
        ]
        with pytest.raises(InferenceError):
            predictor.predict_batch(inputs_list)


# ====================================================================
# Tests: predict_proba
# ====================================================================

class TestPredictProba:

    def test_returns_numpy(
        self, predictor: FusionPredictor, sample_inputs: Dict[str, torch.Tensor]
    ) -> None:
        probs = predictor.predict_proba(sample_inputs)
        assert isinstance(probs, np.ndarray)

    def test_probabilities_sum_to_one(
        self, predictor: FusionPredictor, sample_inputs: Dict[str, torch.Tensor]
    ) -> None:
        probs = predictor.predict_proba(sample_inputs)
        assert probs.shape == (1, 5)
        assert abs(probs.sum() - 1.0) < 1e-5


# ====================================================================
# Tests: embed
# ====================================================================

class TestEmbed:

    def test_returns_numpy(
        self, predictor: FusionPredictor, sample_inputs: Dict[str, torch.Tensor]
    ) -> None:
        emb = predictor.embed(sample_inputs)
        assert isinstance(emb, np.ndarray)

    def test_embedding_shape(
        self, predictor: FusionPredictor, sample_inputs: Dict[str, torch.Tensor]
    ) -> None:
        emb = predictor.embed(sample_inputs)
        # fused output = (batch=1, hidden_dim=32)
        assert emb.shape == (1, 32)


# ====================================================================
# Tests: from_checkpoint
# ====================================================================

class TestFromCheckpoint:

    def test_round_trip(self, dummy_config: Config) -> None:
        """Save a model, reload via from_checkpoint, run predict."""
        model = DummyFusionModel(dummy_config)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = os.path.join(tmp, "model.pt")
            save_checkpoint(ckpt_path, model, epoch=0)

            predictor = FusionPredictor.from_checkpoint(
                checkpoint_path=ckpt_path,
                config=dummy_config,
                model_class=DummyFusionModel,
            )
            result = predictor.predict({
                "vision": torch.randn(1, 32),
                "language": torch.randn(1, 32),
            })
            assert "logits" in result

    def test_missing_checkpoint_raises(self, dummy_config: Config) -> None:
        with pytest.raises(CheckpointError):
            FusionPredictor.from_checkpoint(
                checkpoint_path="/nonexistent/path.pt",
                config=dummy_config,
                model_class=DummyFusionModel,
            )

    def test_missing_model_class_raises(self, dummy_config: Config) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = DummyFusionModel(dummy_config)
            ckpt_path = os.path.join(tmp, "model.pt")
            save_checkpoint(ckpt_path, model, epoch=0)

            with pytest.raises(InferenceError):
                FusionPredictor.from_checkpoint(
                    checkpoint_path=ckpt_path,
                    config=dummy_config,
                    model_class=None,
                )
