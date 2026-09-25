"""
fusion/inference/predictor.py

Phase 12 — Inference Engine: FusionPredictor (Owner: Phase 12 Team)

Provides ``FusionPredictor``, the single-entry-point class for running
inference with any trained ``BaseFusionModel``.  It handles device
placement, pre/post-processing, batched prediction, probability
extraction, and embedding generation.

Other Phase 12 modules (``batch_inference.py``, ``streaming.py``,
``optimization.py``) depend directly on this class's public interface —
do **not** rename or restructure methods without coordinating downstream.

Depends on:
    Phase 1  — Config, load_checkpoint
    Phase 2  — Modality, ModalTensor
    Phase 5  — BaseFusionModel
    Phase 12 — InferenceError (new, added to fusion.exceptions)

Expected config params (all optional, with sensible defaults):
    inference.device        : str   — ``"cpu"`` or ``"cuda"``
    inference.task_type     : str   — ``"classification"`` | ``"generation"``
    inference.softmax_dim   : int   — axis for softmax (default ``-1``)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch
import torch.nn.functional as F

from fusion.constants import Modality
from fusion.core.modal_tensor import ModalTensor
from fusion.exceptions import CheckpointError, InferenceError, ModalityError
from fusion.models.base import BaseFusionModel
from fusion.utils.config import Config
from fusion.utils.io import load_checkpoint
from fusion.utils.logging import get_logger, log_event

logger = get_logger(__name__)


# ====================================================================
# FusionPredictor
# ====================================================================

class FusionPredictor:
    """High-level inference wrapper around any ``BaseFusionModel``.

    Responsibilities:
        * Device placement (CPU / CUDA, respects ``config.inference.device``)
        * Input preprocessing (tensor conversion, device transfer)
        * Forward pass under ``torch.no_grad()``
        * Output postprocessing (softmax for classification, argmax for
          generation heads)
        * Batched prediction with automatic stacking / splitting
        * Probability extraction (``predict_proba``)
        * Embedding extraction via ``embed()`` (encode + fuse only,
          skipping the prediction head)

    Attributes:
        model (BaseFusionModel): The wrapped model, set to eval mode.
        config (Config): Runtime configuration.
        device (torch.device): Resolved compute device.
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(self, model: BaseFusionModel, config: Config) -> None:
        """Initialise the predictor and prepare the model for inference.

        Args:
            model (BaseFusionModel): A trained fusion model instance.
            config (Config): Runtime configuration.  The predictor reads
                ``config.get("inference.device", "cpu")`` to resolve the
                target device and ``config.get("inference.task_type",
                "classification")`` to choose postprocessing behaviour.

        Side-effects:
            * Calls ``model.eval()`` immediately.
            * Moves the model to the resolved device.

        Raises:
            InferenceError: If the model cannot be moved to the
                requested device.
        """
        self.config: Config = config
        self.model: BaseFusionModel = model

        # --- resolve device ------------------------------------------------
        requested = str(config.get("inference.device", "cpu"))
        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "event=device_fallback requested=cuda resolved=cpu "
                "reason=cuda_not_available"
            )
            requested = "cpu"
        self.device: torch.device = torch.device(requested)

        # --- set eval mode & move to device --------------------------------
        try:
            self.model.eval()
            self.model.to(self.device)
        except Exception as exc:
            raise InferenceError(
                f"Failed to place model on device '{self.device}'",
                details={"device": str(self.device), "error": str(exc)},
            ) from exc

        log_event(
            logger, logging.INFO, "predictor_init",
            device=str(self.device),
            model_class=type(self.model).__name__,
            num_params=sum(p.numel() for p in self.model.parameters()),
        )

    # ------------------------------------------------------------------ #
    # Core prediction
    # ------------------------------------------------------------------ #

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run a full forward pass with pre- and post-processing.

        **Preprocessing** (per modality):
            * If the value is already a ``torch.Tensor`` it is moved to
              ``self.device``.
            * If the value is a ``ModalTensor`` its ``.data`` tensor is
              moved to ``self.device`` and it is re-wrapped.
            * If the value is a list/ndarray it is converted to a
              ``torch.Tensor`` first.

        **Postprocessing** (task-type aware):
            * ``classification`` → softmax applied to ``logits`` output.
            * ``generation``     → argmax (greedy decode) applied to
              ``logits`` output, returned as ``token_ids``.
            * Other task types   → outputs returned as-is.

        All output tensors are detached and moved to CPU.

        Args:
            inputs (Dict[str, Any]): Mapping of modality name (str) to
                raw input data (tensor, list, ndarray, or ModalTensor).
                Example::

                    {"vision": image_tensor, "language": token_ids_tensor}

        Returns:
            Dict[str, Any]: Postprocessed outputs keyed by head name.
                For classification::

                    {"logits": Tensor, "probabilities": Tensor,
                     "predicted_class": Tensor}

                For generation::

                    {"logits": Tensor, "token_ids": Tensor}

        Raises:
            InferenceError: On preprocessing failure (bad type / shape)
                or forward-pass error.
            ModalityError: If the model's schema rejects the inputs.
        """
        # --- preprocess ----------------------------------------------------
        processed = self._preprocess(inputs)

        # --- forward pass --------------------------------------------------
        try:
            with torch.no_grad():
                raw_outputs: Dict[str, torch.Tensor] = self.model(processed)
        except ModalityError:
            raise  # let schema errors surface as-is
        except Exception as exc:
            log_event(
                logger, logging.ERROR, "predict_forward_error",
                error=str(exc),
                input_keys=list(inputs.keys()),
            )
            raise InferenceError(
                "Forward pass failed during predict()",
                details={
                    "input_keys": list(inputs.keys()),
                    "error": str(exc),
                },
            ) from exc

        # --- postprocess ---------------------------------------------------
        return self._postprocess(raw_outputs)

    # ------------------------------------------------------------------ #
    # Batch prediction
    # ------------------------------------------------------------------ #

    def predict_batch(
        self, inputs_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Predict on a list of inputs in a single batched forward pass.

        Individual samples are stacked along dim-0 into a single batch
        tensor per modality, the model runs once, and outputs are split
        back into per-sample dicts preserving the original order.

        Args:
            inputs_list (List[Dict[str, Any]]): Each element is a
                single-sample input dict (same schema as ``predict()``).
                All samples must share the same set of modality keys.

        Returns:
            List[Dict[str, Any]]: One postprocessed output dict per
            input sample, in the same order.

        Raises:
            InferenceError: If inputs are empty, keys are inconsistent,
                or the forward pass fails.
        """
        if not inputs_list:
            raise InferenceError(
                "predict_batch() received an empty inputs list",
                details={"length": 0},
            )

        # --- stack into a single batch per modality -----------------------
        batch_keys = list(inputs_list[0].keys())
        batched: Dict[str, Any] = {}
        for key in batch_keys:
            tensors = []
            for idx, sample in enumerate(inputs_list):
                if key not in sample:
                    raise InferenceError(
                        f"Sample {idx} is missing modality key '{key}'",
                        details={"index": idx, "expected_keys": batch_keys},
                    )
                val = sample[key]
                t = self._to_tensor(val)
                # Ensure batch dim exists (if single sample has no batch dim)
                if t.dim() == 0:
                    t = t.unsqueeze(0)
                tensors.append(t)
            batched[key] = torch.cat(tensors, dim=0)

        batch_size = len(inputs_list)

        # --- single predict call -------------------------------------------
        batch_outputs = self.predict(batched)

        # --- split back into per-sample dicts ------------------------------
        results: List[Dict[str, Any]] = []
        for i in range(batch_size):
            sample_out: Dict[str, Any] = {}
            for out_key, out_val in batch_outputs.items():
                if isinstance(out_val, torch.Tensor) and out_val.shape[0] == batch_size:
                    sample_out[out_key] = out_val[i]
                else:
                    # scalar or non-batchable output — replicate
                    sample_out[out_key] = out_val
            results.append(sample_out)

        return results

    # ------------------------------------------------------------------ #
    # Probability extraction
    # ------------------------------------------------------------------ #

    def predict_proba(self, inputs: Dict[str, Any]) -> np.ndarray:
        """Return class probabilities as a numpy array.

        Calls ``predict()`` internally.  If the postprocessor already
        attached ``probabilities`` (classification task), those are
        returned directly.  Otherwise softmax is applied to the raw
        ``logits`` key.

        Args:
            inputs (Dict[str, Any]): Same schema as ``predict()``.

        Returns:
            np.ndarray: Probability array with shape
            ``(batch_size, num_classes)``.

        Raises:
            InferenceError: If neither ``probabilities`` nor ``logits``
                are present in the model output.
        """
        outputs = self.predict(inputs)

        if "probabilities" in outputs:
            probs = outputs["probabilities"]
        elif "logits" in outputs:
            logits = outputs["logits"]
            if isinstance(logits, torch.Tensor):
                softmax_dim = int(self.config.get("inference.softmax_dim", -1))
                probs = F.softmax(logits.float(), dim=softmax_dim)
            else:
                probs = logits  # already numpy or similar
        else:
            raise InferenceError(
                "Model output contains neither 'probabilities' nor 'logits'",
                details={"available_keys": list(outputs.keys())},
            )

        if isinstance(probs, torch.Tensor):
            return probs.detach().cpu().numpy()
        return np.asarray(probs)

    # ------------------------------------------------------------------ #
    # Embedding extraction
    # ------------------------------------------------------------------ #

    def embed(self, inputs: Dict[str, Any]) -> np.ndarray:
        """Extract the fused embedding (encode + fuse, no prediction head).

        This runs only the first two stages of the model pipeline
        (``encode`` → ``fuse``) and returns the fused representation
        before it reaches any task head.  The output is suitable for
        building downstream search indexes (e.g. FAISS, ScaNN).

        Args:
            inputs (Dict[str, Any]): Same schema as ``predict()``.

        Returns:
            np.ndarray: Embedding array with shape
            ``(batch_size, fused_dim)`` where ``fused_dim`` is the
            dimensionality of the model's fused representation space.

        Raises:
            InferenceError: If encode/fuse fails.
        """
        processed = self._preprocess(inputs)

        try:
            with torch.no_grad():
                encoded = self.model.encode(processed)
                fused: torch.Tensor = self.model.fuse(encoded)
        except Exception as exc:
            log_event(
                logger, logging.ERROR, "embed_error",
                error=str(exc),
                input_keys=list(inputs.keys()),
            )
            raise InferenceError(
                "Embedding extraction failed during encode/fuse",
                details={
                    "input_keys": list(inputs.keys()),
                    "error": str(exc),
                },
            ) from exc

        return fused.detach().cpu().numpy()

    # ------------------------------------------------------------------ #
    # Factory: from_checkpoint
    # ------------------------------------------------------------------ #

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: Config,
        model_class: Optional[Type[BaseFusionModel]] = None,
    ) -> "FusionPredictor":
        """Construct a ready-to-use predictor from a saved checkpoint.

        Args:
            checkpoint_path (str): Path to a ``.pt`` / ``.pth``
                checkpoint file written by ``BaseFusionModel.save()``
                or ``save_checkpoint()``.
            config (Config): Runtime configuration.  Must contain
                enough information for the model constructor (the
                checkpoint's stored config is used for model weights).
            model_class (Optional[Type[BaseFusionModel]]): The concrete
                ``BaseFusionModel`` subclass to instantiate.  If
                ``None``, the class name stored inside the checkpoint
                is logged but the caller **must** supply the class.

        Returns:
            FusionPredictor: A fully initialised predictor instance.

        Raises:
            CheckpointError: If the checkpoint file is missing or
                corrupt.
            InferenceError: If ``model_class`` is ``None`` and cannot
                be resolved.
        """
        if not os.path.isfile(checkpoint_path):
            raise CheckpointError(
                f"Checkpoint file not found: '{checkpoint_path}'",
                details={"path": checkpoint_path},
            )

        if model_class is None:
            raise InferenceError(
                "model_class must be provided to from_checkpoint(). "
                "Automatic registry lookup is not yet implemented.",
                details={"checkpoint_path": checkpoint_path},
            )

        log_event(
            logger, logging.INFO, "loading_checkpoint",
            path=checkpoint_path,
            model_class=model_class.__name__,
        )

        # Construct the model from config, then load weights via the
        # existing load_checkpoint() utility.
        device = torch.device(
            str(config.get("inference.device", "cpu"))
        )
        model = model_class(config)
        load_checkpoint(checkpoint_path, model, device=device)

        log_event(
            logger, logging.INFO, "checkpoint_loaded",
            path=checkpoint_path,
        )

        return cls(model=model, config=config)

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _preprocess(
        self, inputs: Dict[str, Any]
    ) -> Dict[Modality, Any]:
        """Convert and move each input value to the model's device.

        Handles four value types:
            1. ``torch.Tensor`` → moved to ``self.device``.
            2. ``ModalTensor``  → inner ``.data`` moved, re-wrapped.
            3. ``np.ndarray``   → converted to tensor, then moved.
            4. ``list``         → converted via ``torch.tensor()``.

        Modality keys are resolved from string → ``Modality`` enum
        when possible; unrecognised keys are passed through as-is
        (the model's schema validation handles rejection).

        Returns:
            Dict[Modality, Any]: Device-resident tensors keyed by
            ``Modality`` enum members.

        Raises:
            InferenceError: On unsupported input types.
        """
        result: Dict[Modality, Any] = {}

        for key, value in inputs.items():
            modality = self._resolve_modality(key)

            try:
                tensor = self._to_tensor(value).to(self.device)
            except Exception as exc:
                raise InferenceError(
                    f"Failed to preprocess input for modality '{key}'",
                    details={
                        "modality": key,
                        "type": type(value).__name__,
                        "error": str(exc),
                    },
                ) from exc

            result[modality] = tensor

        return result

    def _postprocess(
        self, raw_outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Apply task-specific transformations to model outputs.

        All tensors are detached and moved to CPU.

        Returns:
            Dict[str, Any]: Postprocessed output dict.
        """
        task_type = str(self.config.get("inference.task_type", "classification"))
        softmax_dim = int(self.config.get("inference.softmax_dim", -1))
        result: Dict[str, Any] = {}

        for name, tensor in raw_outputs.items():
            if not isinstance(tensor, torch.Tensor):
                result[name] = tensor
                continue
            tensor = tensor.detach().cpu()
            result[name] = tensor

        # --- task-specific extras ------------------------------------------
        if task_type == "classification" and "logits" in result:
            logits = result["logits"]
            probs = F.softmax(logits.float(), dim=softmax_dim)
            result["probabilities"] = probs
            result["predicted_class"] = torch.argmax(probs, dim=softmax_dim)

        elif task_type == "generation" and "logits" in result:
            logits = result["logits"]
            result["token_ids"] = torch.argmax(logits, dim=softmax_dim)

        return result

    @staticmethod
    def _to_tensor(value: Any) -> torch.Tensor:
        """Coerce *value* to a ``torch.Tensor``.

        Raises:
            InferenceError: If the type is unsupported.
        """
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, ModalTensor):
            return value.data
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        if isinstance(value, (list, tuple)):
            return torch.tensor(value)
        raise InferenceError(
            f"Unsupported input type: {type(value).__name__}",
            details={"type": type(value).__name__},
        )

    @staticmethod
    def _resolve_modality(key: str) -> Modality:
        """Map a string key to a ``Modality`` enum member.

        Falls back to returning the raw string wrapped in a best-effort
        enum conversion.  If the string doesn't match any member the
        raw string is returned (the model's schema will handle
        rejection).

        Returns:
            Modality: Resolved enum member, or raises on truly
            unknown modalities.

        Raises:
            InferenceError: If the key cannot be resolved.
        """
        # Direct match on value (e.g. "vision" → Modality.VISION)
        for member in Modality:
            if member.value == key:
                return member

        # Case-insensitive fallback
        key_upper = key.upper()
        for member in Modality:
            if member.name == key_upper:
                return member

        raise InferenceError(
            f"Unknown modality key: '{key}'. "
            f"Supported: {[m.value for m in Modality]}",
            details={
                "key": key,
                "supported": [m.value for m in Modality],
            },
        )

    # ------------------------------------------------------------------ #
    # Repr
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"FusionPredictor("
            f"model={type(self.model).__name__}, "
            f"device={self.device}, "
            f"task_type={self.config.get('inference.task_type', 'classification')})"
        )


__all__ = ["FusionPredictor"]
