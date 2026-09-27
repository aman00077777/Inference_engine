"""
fusion/inference/batch_inference.py

Phase 12 — Inference Engine: BatchInference (Owner: Aman Sharma)

Drives batched, offline prediction over an entire dataset using
``FusionPredictor``.  Results are written to disk as a JSONL file
(one JSON object per prediction, one per line).

Depends on:
    Phase 1  — Config
    Phase 2  — Modality
    Phase 12 — FusionPredictor, InferenceError, DataError

Expected config params: None — all tuning knobs are constructor args.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from fusion.exceptions import DataError, InferenceError
from fusion.inference.predictor import FusionPredictor
from fusion.utils.logging import get_logger, log_event

logger = get_logger(__name__)


# ====================================================================
# FusionDataset — lightweight Dataset wrapper for batch inference
# ====================================================================

class FusionDataset(Dataset):
    """Minimal ``torch.utils.data.Dataset`` wrapping a list of sample dicts.

    Each sample is a ``Dict[str, Any]`` mapping modality names to their
    raw data (tensors, arrays, lists, etc.) — the same schema that
    ``FusionPredictor.predict()`` expects.

    This class is intentionally thin: it adds ``len`` / ``__getitem__``
    semantics on top of a plain Python list so the dataset can be fed
    to a standard ``torch.utils.data.DataLoader``.

    Args:
        samples (List[Dict[str, Any]]): List of per-sample input dicts.

    Raises:
        DataError: If *samples* is empty or not a list.

    Example::

        ds = FusionDataset([
            {"vision": img_tensor_1, "language": tok_ids_1},
            {"vision": img_tensor_2, "language": tok_ids_2},
        ])
        print(len(ds))   # 2
        print(ds[0])      # first sample dict
    """

    def __init__(self, samples: List[Dict[str, Any]]) -> None:
        if not isinstance(samples, list):
            raise DataError(
                "FusionDataset expects a list of sample dicts",
                details={"got_type": type(samples).__name__},
            )
        if len(samples) == 0:
            raise DataError(
                "FusionDataset received an empty sample list",
                details={"length": 0},
            )
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0 or index >= len(self._samples):
            raise DataError(
                f"FusionDataset index {index} out of range [0, {len(self._samples)})",
                details={"index": index, "dataset_size": len(self._samples)},
            )
        return self._samples[index]


# ====================================================================
# BatchInference
# ====================================================================

class BatchInference:
    """Offline batch-inference runner for the FUSION framework.

    Iterates over a :class:`FusionDataset` via a ``DataLoader``, calls
    :meth:`FusionPredictor.predict_batch` on each batch, and writes
    the per-sample prediction dicts as JSONL (one JSON object per line)
    to an output file.

    Args:
        predictor (FusionPredictor): A fully-initialised predictor
            instance (model loaded, device set).
        batch_size (int): Number of samples per DataLoader batch.
        num_workers (int): Number of DataLoader worker processes.

    Raises:
        InferenceError: If *predictor* is ``None`` or *batch_size* < 1.

    Example::

        predictor = FusionPredictor(model=my_model, config=my_config)
        runner = BatchInference(predictor, batch_size=64, num_workers=2)
        runner.run(dataset, "predictions.jsonl")
    """

    def __init__(
        self,
        predictor: FusionPredictor,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        if predictor is None:
            raise InferenceError(
                "BatchInference requires a non-None FusionPredictor",
                details={},
            )
        if batch_size < 1:
            raise InferenceError(
                "batch_size must be >= 1",
                details={"batch_size": batch_size},
            )

        self.predictor: FusionPredictor = predictor
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self, dataset: FusionDataset, output_path: str) -> None:
        """Execute batch inference over *dataset* and write JSONL results.

        **Workflow**:
            1. Validate the dataset.
            2. Build a ``torch.utils.data.DataLoader`` with ``batch_size``
               and ``num_workers`` (collation is a passthrough that keeps
               each sample as a plain dict).
            3. Iterate over batches, calling
               ``predictor.predict_batch(batch)`` on each.
            4. Serialise each per-sample result dict as a single JSON
               line and write it to *output_path*.
            5. Log progress via ``tqdm`` (if available) and structured
               log events for start/completion.

        Args:
            dataset (FusionDataset): The dataset to run inference on.
            output_path (str): Path to the output JSONL file.  Parent
                directories are created automatically.

        Raises:
            DataError: If *dataset* is not a ``FusionDataset`` or is
                empty.
            InferenceError: If any batch prediction fails or the results
                cannot be written to disk.
        """
        # --- validate dataset -----------------------------------------------
        if not isinstance(dataset, FusionDataset):
            raise DataError(
                "BatchInference.run() expects a FusionDataset instance",
                details={"got_type": type(dataset).__name__},
            )

        total_samples = len(dataset)
        total_batches = (total_samples + self.batch_size - 1) // self.batch_size

        log_event(
            logger, logging.INFO, "batch_run_start",
            total_samples=total_samples,
            batch_size=self.batch_size,
            total_batches=total_batches,
            output_path=output_path,
        )

        # --- build DataLoader -----------------------------------------------
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_passthrough_collate,
        )

        # --- ensure output directory exists ---------------------------------
        output_dir = os.path.dirname(output_path)
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as exc:
                raise InferenceError(
                    f"Cannot create output directory '{output_dir}'",
                    details={"output_dir": output_dir, "error": str(exc)},
                ) from exc

        # --- iterate & predict ----------------------------------------------
        t0 = time.monotonic()
        samples_written = 0

        # Import tqdm for progress bar (graceful fallback if missing)
        try:
            from tqdm import tqdm
            batch_iter = tqdm(
                loader,
                total=total_batches,
                desc="BatchInference",
                unit="batch",
            )
        except ImportError:
            logger.warning(
                "event=tqdm_unavailable msg='tqdm not installed; "
                "progress bar disabled'"
            )
            batch_iter = loader

        try:
            with open(output_path, "w", encoding="utf-8") as fh:
                for batch_idx, batch in enumerate(batch_iter):
                    # batch is List[Dict[str, Any]] thanks to
                    # _passthrough_collate
                    try:
                        predictions = self.predictor.predict_batch(batch)
                    except Exception as exc:
                        raise InferenceError(
                            f"predict_batch failed on batch {batch_idx}",
                            details={
                                "batch_idx": batch_idx,
                                "batch_size": len(batch),
                                "error": str(exc),
                            },
                        ) from exc

                    for pred in predictions:
                        line = json.dumps(
                            _serialise_prediction(pred),
                            ensure_ascii=False,
                        )
                        fh.write(line + "\n")
                        samples_written += 1

                    if batch_idx % max(1, total_batches // 10) == 0:
                        log_event(
                            logger, logging.DEBUG, "batch_progress",
                            batch_idx=batch_idx,
                            total_batches=total_batches,
                            samples_written=samples_written,
                        )

        except InferenceError:
            raise
        except OSError as exc:
            raise InferenceError(
                f"Failed to write results to '{output_path}'",
                details={"output_path": output_path, "error": str(exc)},
            ) from exc

        duration_s = round(time.monotonic() - t0, 3)
        log_event(
            logger, logging.INFO, "batch_run_complete",
            samples_written=samples_written,
            output_path=output_path,
            duration_s=duration_s,
        )

    # ------------------------------------------------------------------ #
    # Repr
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"BatchInference("
            f"predictor={self.predictor!r}, "
            f"batch_size={self.batch_size}, "
            f"num_workers={self.num_workers})"
        )


# ====================================================================
# Module-private helpers
# ====================================================================

def _passthrough_collate(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """DataLoader collate function that returns the batch unchanged.

    The default ``DataLoader`` collate would attempt to stack tensors
    across samples.  Since ``FusionPredictor.predict_batch`` expects a
    ``List[Dict]`` and handles stacking internally, we bypass that
    behaviour entirely.
    """
    return batch


def _serialise_prediction(pred: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single prediction dict to a JSON-safe form.

    ``torch.Tensor`` values are converted to nested Python lists;
    everything else is passed through.
    """
    out: Dict[str, Any] = {}
    for key, value in pred.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().cpu().tolist()
        else:
            out[key] = value
    return out


__all__ = ["FusionDataset", "BatchInference"]
