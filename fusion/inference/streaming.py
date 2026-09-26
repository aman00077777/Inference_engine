"""
fusion/inference/streaming.py

Phase 12 — Inference Engine: StreamingPredictor (Owner: Aman Hingawe)

Token-by-token generation wrapper around any generative BaseFusionModel.
Encodes the inputs into a model context, then repeatedly runs the model's
generation path, samples or argmaxes the next token, yields the decoded
token string, and appends the new token to the context for the next step.
Stops on EOS or max_new_tokens.

Depends on:
    Phase 1  — Config
    Phase 2  — Modality, ModalTensor
    Phase 5  — BaseFusionModel
    Phase 12 — InferenceError (fusion.exceptions)

Assumptions flagged for team review:
    * This repo ships no fusion/models/heads/ package and no tokenizer
      (Phase 10) yet, so the generation contract used here is
      model.predict(fused)["logits"] (same key FusionPredictor
      postprocesses), and decoding is injected via decode_fn / tokenizer
      with a documented placeholder fallback.
    * BaseFusionModel exposes no token-embedding or head internals, so the
      autoregressive update appends the new token id to the context_key
      input tensor (default "language") and re-runs the public
      encode -> fuse -> predict pipeline each step. Models consuming
      embeddings instead of token ids can inject context_update_fn.
    * Streaming is single-sample (batch size 1); larger batches raise
      InferenceError.
    * EOS resolution order: eos_token_id kwarg -> config
      inference.eos_token_id -> tokenizer.eos_token_id. If none exists,
      only max_new_tokens stops generation.

Build order step 2b: depends only on the base model; no other Phase 12
module imports from this file.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Generator, Optional

import numpy as np
import torch

from fusion.constants import Modality
from fusion.core.modal_tensor import ModalTensor
from fusion.exceptions import InferenceError
from fusion.models.base import BaseFusionModel
from fusion.utils.config import Config
from fusion.utils.logging import get_logger, log_event

logger = get_logger(__name__)


class StreamingPredictor:
    """Stream generated tokens one at a time from a generative model."""

    def __init__(
        self,
        model: BaseFusionModel,
        config: Optional[Config] = None,
        *,
        decode_fn: Optional[Callable[[int], str]] = None,
        tokenizer: Optional[Any] = None,
        eos_token_id: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        do_sample: bool = False,
        context_key: str = "language",
        context_update_fn: Optional[Callable[[Dict[Any, Any], int], Dict[Any, Any]]] = None,
    ) -> None:
        if not isinstance(model, BaseFusionModel):
            raise InferenceError(
                "StreamingPredictor requires a BaseFusionModel instance",
                details={"got": type(model).__name__},
            )
        self.config: Config = config if config is not None else Config({})
        self.model: BaseFusionModel = model

        requested = str(self.config.get("inference.device", "cpu"))
        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "event=device_fallback requested=cuda resolved=cpu "
                "reason=cuda_not_available"
            )
            requested = "cpu"
        self.device: torch.device = torch.device(requested)

        try:
            self.model.eval()
            self.model.to(self.device)
        except Exception as exc:
            raise InferenceError(
                f"Failed to place model on device '{self.device}'",
                details={"device": str(self.device), "error": str(exc)},
            ) from exc

        self.decode_fn = decode_fn
        self.tokenizer = tokenizer
        self.context_key = context_key
        self.context_update_fn = context_update_fn
        self.temperature = float(temperature)
        self.do_sample = bool(do_sample)

        if max_new_tokens is not None:
            self.max_new_tokens = int(max_new_tokens)
        else:
            self.max_new_tokens = int(self.config.get("inference.max_new_tokens", 64))
        if self.max_new_tokens < 1:
            raise InferenceError(
                "max_new_tokens must be >= 1",
                details={"max_new_tokens": self.max_new_tokens},
            )

        tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
        resolved_eos = (
            eos_token_id
            if eos_token_id is not None
            else self.config.get("inference.eos_token_id", tokenizer_eos)
        )
        self.eos_token_id = int(resolved_eos) if resolved_eos is not None else None

        log_event(
            logger, logging.INFO, "streaming_init",
            device=str(self.device),
            model_class=type(self.model).__name__,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.eos_token_id,
        )

    # ------------------------------------------------------------------ #
    # Streaming generation
    # ------------------------------------------------------------------ #
    def stream(self, inputs: Dict[str, Any]) -> Generator[str, None, None]:
        """Yield decoded token strings one at a time.

        Args:
            inputs: Single-sample modality inputs, e.g.
                {"language": token_ids_tensor_of_shape_(1, L)}.

        Yields:
            str: The decoded token produced at each generation step.

        Raises:
            InferenceError: On empty inputs, batch size > 1, missing
                context modality, missing 'logits' output, or any
                failure inside the generation loop.
        """
        if not inputs:
            raise InferenceError("stream() received an empty inputs dict", details={})

        processed = self._preprocess(inputs)
        for name, tensor in processed.items():
            if isinstance(tensor, torch.Tensor) and tensor.shape[0] != 1:
                raise InferenceError(
                    "StreamingPredictor.stream() supports single-sample inputs "
                    f"(batch size 1); modality '{name}' has batch size "
                    f"{tensor.shape[0]}",
                    details={"modality": str(name), "batch_size": int(tensor.shape[0])},
                )

        ctx_key = self._resolve_modality(self.context_key)
        if ctx_key not in processed:
            raise InferenceError(
                f"stream() requires a '{self.context_key}' input to append "
                f"generated tokens to; got keys: "
                f"{sorted(str(k) for k in processed)}",
                details={
                    "context_key": self.context_key,
                    "available": sorted(str(k) for k in processed),
                },
            )

        log_event(
            logger, logging.INFO, "stream_start",
            context_key=self.context_key,
            max_new_tokens=self.max_new_tokens,
        )

        generated = 0
        try:
            with torch.no_grad():
                while generated < self.max_new_tokens:
                    encoded = self.model.encode(processed)
                    fused = self.model.fuse(encoded)
                    outputs = self.model.predict(fused)
                    if "logits" not in outputs:
                        raise InferenceError(
                            "Model predict() output has no 'logits' key; a "
                            "generative model is required for streaming",
                            details={"available_keys": sorted(outputs.keys())},
                        )
                    logits = outputs["logits"]
                    if logits.dim() == 3:
                        logits = logits[:, -1, :]  # last sequence position
                    token = self._sample(logits[0])
                    if self.eos_token_id is not None and token == self.eos_token_id:
                        break
                    yield self._decode(token)
                    processed = self._append_token(processed, ctx_key, token)
                    generated += 1
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(
                "Streaming generation failed",
                details={"step": generated, "error": str(exc)},
            ) from exc

        log_event(logger, logging.INFO, "stream_end", tokens_generated=generated)

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #
    def _sample(self, step_logits: torch.Tensor) -> int:
        if self.do_sample:
            probs = torch.softmax(step_logits / max(self.temperature, 1e-8), dim=-1)
            return int(torch.multinomial(probs, num_samples=1).item())
        return int(torch.argmax(step_logits, dim=-1).item())

    def _decode(self, token: int) -> str:
        if self.decode_fn is not None:
            return str(self.decode_fn(token))
        if self.tokenizer is not None:
            decode = getattr(self.tokenizer, "decode", None)
            if callable(decode):
                try:
                    return str(decode([token]))
                except TypeError:
                    return str(decode(token))
            batch_decode = getattr(self.tokenizer, "batch_decode", None)
            if callable(batch_decode):
                return str(batch_decode([[token]])[0])
        return f"<tok_{token}>"  # placeholder until Phase 10 tokenizer lands

    def _append_token(
        self, processed: Dict[Any, Any], ctx_key: Any, token: int
    ) -> Dict[Any, Any]:
        if self.context_update_fn is not None:
            updated = self.context_update_fn(dict(processed), token)
            if not isinstance(updated, dict):
                raise InferenceError(
                    "context_update_fn must return a dict of processed inputs",
                    details={"got": type(updated).__name__},
                )
            return updated
        ctx = processed[ctx_key]
        if not isinstance(ctx, torch.Tensor) or ctx.dim() != 2:
            raise InferenceError(
                "Default context update expects a (1, L) token-id tensor for "
                f"'{self.context_key}'; inject context_update_fn for other "
                "input formats",
                details={"context_key": self.context_key,
                         "shape": list(getattr(ctx, "shape", []))},
            )
        tok = torch.tensor([[token]], dtype=ctx.dtype, device=ctx.device)
        updated = dict(processed)
        updated[ctx_key] = torch.cat([ctx, tok], dim=-1)
        return updated

    def _preprocess(self, inputs: Dict[str, Any]) -> Dict[Any, torch.Tensor]:
        result: Dict[Any, torch.Tensor] = {}
        for key, value in inputs.items():
            modality = self._resolve_modality(key)
            try:
                tensor = self._to_tensor(value).to(self.device)
            except InferenceError:
                raise
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

    @staticmethod
    def _to_tensor(value: Any) -> torch.Tensor:
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
        for member in Modality:
            if member.value == key:
                return member
        key_upper = key.upper()
        for member in Modality:
            if member.name == key_upper:
                return member
        raise InferenceError(
            f"Unknown modality key: '{key}'. "
            f"Supported: {[m.value for m in Modality]}",
            details={"key": key, "supported": [m.value for m in Modality]},
        )

    def __repr__(self) -> str:
        return (
            f"StreamingPredictor(model={type(self.model).__name__}, "
            f"device={self.device}, max_new_tokens={self.max_new_tokens}, "
            f"eos_token_id={self.eos_token_id})"
        )


__all__ = ["StreamingPredictor"]