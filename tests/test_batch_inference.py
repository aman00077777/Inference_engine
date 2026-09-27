"""Smoke tests for BatchInference (Phase 12 — Inference Engine).

Verifies that the batch runner can be instantiated with a mock predictor
and that ``run()`` produces a valid JSONL file with the expected number
of predictions.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
import torch

from fusion.exceptions import DataError, InferenceError
from fusion.inference.batch_inference import BatchInference, FusionDataset


# ====================================================================
# Helpers — mock predictor
# ====================================================================

def _make_mock_predictor() -> MagicMock:
    """Return a MagicMock that mimics FusionPredictor.predict_batch.

    Each call returns one prediction dict per input sample with a dummy
    ``score`` tensor and a ``label`` string.
    """
    mock = MagicMock()

    def _predict_batch(inputs_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for sample in inputs_list:
            results.append({
                "score": torch.tensor([0.9, 0.1]),
                "label": "positive",
            })
        return results

    mock.predict_batch = MagicMock(side_effect=_predict_batch)
    return mock


# ====================================================================
# FusionDataset tests
# ====================================================================

class TestFusionDataset:

    def test_len(self) -> None:
        ds = FusionDataset([{"a": 1}, {"a": 2}, {"a": 3}])
        assert len(ds) == 3

    def test_getitem(self) -> None:
        ds = FusionDataset([{"a": 1}, {"a": 2}])
        assert ds[0] == {"a": 1}
        assert ds[1] == {"a": 2}

    def test_empty_raises_data_error(self) -> None:
        with pytest.raises(DataError):
            FusionDataset([])

    def test_non_list_raises_data_error(self) -> None:
        with pytest.raises(DataError):
            FusionDataset("not_a_list")  # type: ignore[arg-type]

    def test_index_out_of_range_raises(self) -> None:
        ds = FusionDataset([{"a": 1}])
        with pytest.raises(DataError):
            _ = ds[5]


# ====================================================================
# BatchInference.__init__ tests
# ====================================================================

class TestBatchInferenceInit:

    def test_stores_params(self) -> None:
        pred = _make_mock_predictor()
        runner = BatchInference(pred, batch_size=16, num_workers=2)
        assert runner.batch_size == 16
        assert runner.num_workers == 2
        assert runner.predictor is pred

    def test_none_predictor_raises(self) -> None:
        with pytest.raises(InferenceError):
            BatchInference(None, batch_size=8)  # type: ignore[arg-type]

    def test_invalid_batch_size_raises(self) -> None:
        with pytest.raises(InferenceError):
            BatchInference(_make_mock_predictor(), batch_size=0)


# ====================================================================
# BatchInference.run tests
# ====================================================================

class TestBatchInferenceRun:

    @pytest.fixture
    def small_dataset(self) -> FusionDataset:
        """A tiny 5-sample dataset with vision + language tensors."""
        samples = [
            {
                "vision": torch.randn(1, 32),
                "language": torch.randn(1, 32),
            }
            for _ in range(5)
        ]
        return FusionDataset(samples)

    def test_run_produces_valid_jsonl(self, small_dataset: FusionDataset) -> None:
        """The full happy-path smoke test."""
        pred = _make_mock_predictor()
        runner = BatchInference(pred, batch_size=2, num_workers=0)

        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "preds.jsonl")
            runner.run(small_dataset, out_path)

            # File should exist
            assert os.path.isfile(out_path)

            # Read and validate each line
            with open(out_path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()

            assert len(lines) == 5  # one per sample

            for line in lines:
                obj = json.loads(line)
                assert "score" in obj
                assert "label" in obj
                assert obj["label"] == "positive"
                # Tensor should have been serialised as a list
                assert isinstance(obj["score"], list)

    def test_run_creates_output_dir(self, small_dataset: FusionDataset) -> None:
        pred = _make_mock_predictor()
        runner = BatchInference(pred, batch_size=4, num_workers=0)

        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "sub", "dir", "preds.jsonl")
            runner.run(small_dataset, nested)
            assert os.path.isfile(nested)

    def test_run_rejects_non_dataset(self) -> None:
        pred = _make_mock_predictor()
        runner = BatchInference(pred, batch_size=2, num_workers=0)

        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "out.jsonl")
            with pytest.raises(DataError):
                runner.run([{"a": 1}], out_path)  # type: ignore[arg-type]

    def test_run_surfaces_predict_batch_failure(
        self, small_dataset: FusionDataset
    ) -> None:
        pred = _make_mock_predictor()
        pred.predict_batch.side_effect = RuntimeError("boom")
        runner = BatchInference(pred, batch_size=2, num_workers=0)

        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "out.jsonl")
            with pytest.raises(InferenceError, match="predict_batch failed"):
                runner.run(small_dataset, out_path)

    def test_repr(self) -> None:
        pred = _make_mock_predictor()
        runner = BatchInference(pred, batch_size=8, num_workers=1)
        r = repr(runner)
        assert "BatchInference" in r
        assert "batch_size=8" in r
