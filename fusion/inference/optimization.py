"""fusion/inference/optimization.py

Phase 12 — Inference Engine: inference-time speed optimizations.

Owner: Shantanu Warghane
Build order: Step 3 — depends on Aman Sharma's fusion/inference/predictor.py
(FusionPredictor) being merged and stable, since warmup() and
benchmark_latency() call predictor.predict() directly.

Usage (once a FusionPredictor is constructed):

    predictor = FusionPredictor(model, config)
    predictor.model = optimize_for_inference(predictor.model)
    warmup(predictor, sample_input, n_warmup=10)
    stats = benchmark_latency(predictor, sample_input)  # run again post-optimize to compare

optimize_for_inference() operates on the raw nn.Module (predictor.model),
not on FusionPredictor itself, per the Phase 12 spec. warmup() and
benchmark_latency() operate on the FusionPredictor, since they need the
full pre/postprocessing path (predict()), not just the bare forward pass.
"""

from __future__ import annotations

import logging
import statistics
import time
from typing import TYPE_CHECKING, Any, Dict, List

import torch
import torch.nn as nn

from fusion.utils.logging import get_logger, log_event

if TYPE_CHECKING:
    from fusion.inference.predictor import FusionPredictor

logger = get_logger(__name__)


def optimize_for_inference(model: nn.Module) -> nn.Module:
    """Optimize a trained model for faster inference.

    Tries ``torch.compile(model, mode="reduce-overhead")`` first (requires
    PyTorch 2.0+). Falls back to ``torch.jit.script(model)`` if
    ``torch.compile`` is unavailable, or if it raises (e.g. unsupported ops
    or dynamic control flow in the model's forward()). If both fail, the
    original, unoptimized model is returned unchanged so callers never get
    ``None`` or a broken model back.

    Args:
        model: A trained nn.Module — typically ``predictor.model``, already
            in eval() mode (FusionPredictor.__init__ calls model.eval()
            before this would typically be invoked).

    Returns:
        nn.Module: The optimized model, or the original model if neither
        optimization path succeeds.
    """
    if hasattr(torch, "compile"):
        try:
            compiled_model = torch.compile(model, mode="reduce-overhead")
            log_event(
                logger, logging.INFO, "optimize_for_inference",
                strategy="torch.compile", mode="reduce-overhead",
            )
            return compiled_model
        except Exception as exc:  # torch.compile can fail for many model-specific reasons
            log_event(
                logger, logging.WARNING, "torch_compile_failed",
                error=str(exc), fallback="torch.jit.script",
            )
    else:
        log_event(
            logger, logging.INFO, "torch_compile_unavailable",
            torch_version=torch.__version__, fallback="torch.jit.script",
        )

    try:
        scripted_model = torch.jit.script(model)
        log_event(logger, logging.INFO, "optimize_for_inference", strategy="torch.jit.script")
        return scripted_model
    except Exception as exc:
        log_event(
            logger, logging.WARNING, "torch_jit_script_failed",
            error=str(exc), fallback="unoptimized_model",
        )
        return model


def warmup(
    predictor: "FusionPredictor",
    sample_input: Dict[str, Any],
    n_warmup: int = 10,
) -> None:
    """Warm up CUDA kernels / JIT compilation before serving real traffic.

    The first few calls after torch.compile or model.to(device) are
    typically much slower than steady state (kernel autotuning, lazy CUDA
    context init, graph capture, etc.). Calling predictor.predict() a few
    times on a throwaway input absorbs that cost at startup instead of on a
    real request.

    Args:
        predictor: A ready FusionPredictor (already constructed; call this
            after predictor.model has been swapped for the
            optimize_for_inference() result, if you're using both).
        sample_input: A representative input dict, in the same shape/format
            predictor.predict() expects at serving time.
        n_warmup: Number of warmup calls to run. Defaults to 10. A value
            of 0 or less is a no-op.
    """
    if n_warmup <= 0:
        log_event(logger, logging.DEBUG, "warmup_skipped", n_warmup=n_warmup)
        return

    for i in range(n_warmup):
        start = time.perf_counter()
        predictor.predict(sample_input)
        elapsed = time.perf_counter() - start
        log_event(logger, logging.DEBUG, "warmup_call", call=i + 1, of=n_warmup, seconds=round(elapsed, 4))

    log_event(logger, logging.INFO, "warmup_complete", n_warmup=n_warmup)


def benchmark_latency(
    predictor: "FusionPredictor",
    sample_input: Dict[str, Any],
    n_runs: int = 50,
) -> Dict[str, float]:
    """Measure predictor.predict() latency over repeated calls.

    Not in the Phase 12 deliverables checklist, but the task notes ask to
    "benchmark before/after latency on a sample input and share numbers with
    the team" — this makes that measurement reproducible instead of ad hoc.
    Call it once with the unoptimized predictor.model and once after
    swapping in optimize_for_inference()'s result, on the same
    sample_input, and compare.

    Args:
        predictor: A ready FusionPredictor.
        sample_input: Representative input dict for predictor.predict().
        n_runs: Number of timed calls (after a few untimed warmup calls so
            the numbers reflect steady state, not cold start).

    Returns:
        Dict[str, float]: {"mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms"}.
    """
    # A few untimed calls so this measures steady-state latency, not warmup cost.
    for _ in range(min(3, n_runs)):
        predictor.predict(sample_input)

    timings_ms: List[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        predictor.predict(sample_input)
        timings_ms.append((time.perf_counter() - start) * 1000)

    timings_ms.sort()
    p95_index = min(len(timings_ms) - 1, int(round(0.95 * (len(timings_ms) - 1))))

    stats = {
        "mean_ms": statistics.mean(timings_ms),
        "median_ms": statistics.median(timings_ms),
        "p95_ms": timings_ms[p95_index],
        "min_ms": timings_ms[0],
        "max_ms": timings_ms[-1],
    }
    log_event(logger, logging.INFO, "benchmark_latency", n_runs=n_runs, **{k: round(v, 4) for k, v in stats.items()})
    return stats