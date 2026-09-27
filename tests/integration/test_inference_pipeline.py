"""
tests/integration/test_inference_pipeline.py

Phase 12 — Inference Engine: full-pipeline integration tests (Owner: Atharav)

Unlike tests/test_predictor.py (unit tests for FusionPredictor in
isolation), this suite exercises the Phase 12 modules *together*, the way
a real caller would:

    checkpoint -> FusionPredictor -> predict / predict_batch / embed
                                   -> BatchInference (file output)
                                   -> StreamingPredictor (generation)
                                   -> EmbeddingCache (persistence)
                                   -> optimize_for_inference / warmup

One of the five Phase 12 files (optimization.py) is still being built by
Shantanu at the time this file was written. Rather than hard-failing the
whole suite on `import`, the tests that need it use
`pytest.importorskip`, so:

    * Right now: predictor, streaming, cache, and batch-inference
      integration tests run and must pass. Optimization tests report as
      SKIPPED (not failed, not silently absent).
    * Once optimization.py lands: its test class activates automatically,
      no edits needed here.

(predictor.py, streaming.py, cache.py, and batch_inference.py are all
merged as of this revision — batch_inference.py ships its own
`FusionDataset`, used directly below.)

Before marking Phase 12 complete, re-run this file and confirm there are
zero skips left.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

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
from fusion.inference.streaming import StreamingPredictor
from fusion.inference.cache import EmbeddingCache


# ====================================================================
# Shared dummy models
# ====================================================================

class TinyClassifier(BaseFusionModel):
    """Two-modality classifier: encode=identity, fuse=sum, predict=linear.

    Mirrors the DummyFusionModel used in tests/test_predictor.py so both
    suites agree on what a minimal valid BaseFusionModel looks like.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        hidden_dim = int(config.get("model.hidden_dim", 32))
        num_classes = int(config.get("model.num_classes", 5))
        self.head = nn.Linear(hidden_dim, num_classes)

    def encode(self, inputs: Dict[Modality, Any]) -> Dict[Modality, ModalTensor]:
        return {
            mod: (ModalTensor(data=t, modality=mod) if isinstance(t, torch.Tensor) else t)
            for mod, t in inputs.items()
        }

    def fuse(self, encoded: Dict[Modality, ModalTensor]) -> torch.Tensor:
        return sum(mt.data for mt in encoded.values())

    def predict(self, fused: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"logits": self.head(fused)}


class TinyGenerator(BaseFusionModel):
    """Single-modality (language) token predictor used for streaming tests.

    encode: identity wrap.
    fuse:   embed token ids, mean-pool over sequence -> (1, hidden).
    predict: linear head over vocab -> (1, vocab).

    The head's weight/bias can be hand-set by tests to make next-token
    selection deterministic (see `force_argmax_token`), since streaming
    behaviour (EOS stop vs. max_new_tokens stop) needs to be verified
    without depending on random init.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        vocab_size = int(config.get("model.vocab_size", 20))
        hidden_dim = int(config.get("model.hidden_dim", 16))
        self.vocab_size = vocab_size
        self.embed_layer = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def encode(self, inputs: Dict[Modality, Any]) -> Dict[Modality, ModalTensor]:
        return {
            mod: (ModalTensor(data=t, modality=mod) if isinstance(t, torch.Tensor) else t)
            for mod, t in inputs.items()
        }

    def fuse(self, encoded: Dict[Modality, ModalTensor]) -> torch.Tensor:
        lang = encoded[Modality.LANGUAGE].data  # (1, L) long
        return self.embed_layer(lang).mean(dim=1)  # (1, hidden)

    def predict(self, fused: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"logits": self.head(fused)}

    def force_argmax_token(self, token_id: int) -> None:
        """Zero the head so every forward pass argmaxes to `token_id`,
        regardless of context. Makes stream() deterministic for testing.
        """
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()
            self.head.bias[token_id] = 100.0


# ====================================================================
# Fixtures
# ====================================================================

@pytest.fixture
def classifier_config() -> Config:
    return Config({
        "model": {"hidden_dim": 32, "num_classes": 5},
        "inference": {"device": "cpu", "task_type": "classification"},
    })


@pytest.fixture
def classifier_model(classifier_config: Config) -> TinyClassifier:
    return TinyClassifier(classifier_config)


@pytest.fixture
def predictor(classifier_model: TinyClassifier, classifier_config: Config) -> FusionPredictor:
    return FusionPredictor(model=classifier_model, config=classifier_config)


@pytest.fixture
def sample_input() -> Dict[str, torch.Tensor]:
    return {"vision": torch.randn(1, 32), "language": torch.randn(1, 32)}


@pytest.fixture
def sample_inputs_batch() -> List[Dict[str, torch.Tensor]]:
    return [
        {"vision": torch.randn(1, 32), "language": torch.randn(1, 32)}
        for _ in range(4)
    ]


# ====================================================================
# 1. Predictor pipeline (checkpoint -> predict / predict_batch / embed)
# ====================================================================

class TestPredictorPipeline:

    def test_from_checkpoint_predict_batch_embed_round_trip(
        self, classifier_config: Config
    ) -> None:
        model = TinyClassifier(classifier_config)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = os.path.join(tmp, "model.pt")
            save_checkpoint(ckpt_path, model, epoch=0)

            loaded = FusionPredictor.from_checkpoint(
                checkpoint_path=ckpt_path,
                config=classifier_config,
                model_class=TinyClassifier,
            )

            single = loaded.predict({"vision": torch.randn(1, 32), "language": torch.randn(1, 32)})
            assert "logits" in single and "probabilities" in single

            batch = loaded.predict_batch([
                {"vision": torch.randn(1, 32), "language": torch.randn(1, 32)}
                for _ in range(3)
            ])
            assert len(batch) == 3
            assert all("predicted_class" in r for r in batch)

            emb = loaded.embed({"vision": torch.randn(1, 32), "language": torch.randn(1, 32)})
            assert isinstance(emb, np.ndarray)
            assert emb.shape == (1, 32)

    def test_missing_checkpoint_raises_checkpoint_error(self, classifier_config: Config) -> None:
        with pytest.raises(CheckpointError):
            FusionPredictor.from_checkpoint(
                checkpoint_path="/nonexistent/checkpoint.pt",
                config=classifier_config,
                model_class=TinyClassifier,
            )

    def test_bad_modality_raises_inference_error(self, predictor: FusionPredictor) -> None:
        with pytest.raises(InferenceError):
            predictor.predict({"smell": torch.randn(1, 32)})


# ====================================================================
# 2a. Batch inference (fusion/inference/batch_inference.py, owner: Gauri)
# ====================================================================
#
# batch_inference.py landed in the repo with its own FusionDataset class
# (a thin wrapper around a list of sample dicts). Gauri's tests
# (tests/test_batch_inference.py) already cover BatchInference against a
# *mocked* predictor. What's missing — and what belongs at the
# integration level — is proving the same code path works against a
# real FusionPredictor wired to a real model, end to end.

class TestBatchInference:

    def _make_dataset(self, n: int, hidden_dim: int = 32) -> "FusionDataset":
        from fusion.inference.batch_inference import FusionDataset

        samples = [
            {"vision": torch.randn(1, hidden_dim), "language": torch.randn(1, hidden_dim)}
            for _ in range(n)
        ]
        return FusionDataset(samples)

    def test_run_writes_one_json_line_per_sample(self, predictor: FusionPredictor) -> None:
        from fusion.inference.batch_inference import BatchInference

        dataset = self._make_dataset(n=6)
        with tempfile.TemporaryDirectory() as tmp:
            output_path = os.path.join(tmp, "predictions.jsonl")
            BatchInference(predictor, batch_size=2, num_workers=0).run(dataset, output_path)

            assert os.path.isfile(output_path)
            with open(output_path, "r", encoding="utf-8") as fh:
                lines = [line for line in fh if line.strip()]

            assert len(lines) == len(dataset)
            for line in lines:
                record = json.loads(line)  # must be valid JSON
                assert "predicted_class" in record
                assert "probabilities" in record
                assert isinstance(record["probabilities"], list)

    def test_run_matches_predict_batch_directly(
        self, predictor: FusionPredictor, classifier_config: Config
    ) -> None:
        """The JSONL output should agree with calling predict_batch directly
        on the same inputs — i.e. BatchInference must not silently reorder,
        drop, or reshape samples on the way through the DataLoader."""
        from fusion.inference.batch_inference import BatchInference, FusionDataset

        torch.manual_seed(0)
        samples = [
            {"vision": torch.randn(1, 32), "language": torch.randn(1, 32)}
            for _ in range(5)
        ]
        direct = predictor.predict_batch(samples)

        with tempfile.TemporaryDirectory() as tmp:
            output_path = os.path.join(tmp, "predictions.jsonl")
            BatchInference(predictor, batch_size=2, num_workers=0).run(
                FusionDataset(samples), output_path
            )
            with open(output_path, "r", encoding="utf-8") as fh:
                via_batch = [json.loads(line) for line in fh if line.strip()]

        assert len(via_batch) == len(direct)
        for direct_r, batch_r in zip(direct, via_batch):
            assert direct_r["predicted_class"].item() == batch_r["predicted_class"]

    def test_run_rejects_plain_list_not_fusion_dataset(self, predictor: FusionPredictor) -> None:
        from fusion.exceptions import DataError
        from fusion.inference.batch_inference import BatchInference

        with tempfile.TemporaryDirectory() as tmp:
            output_path = os.path.join(tmp, "out.jsonl")
            with pytest.raises(DataError):
                BatchInference(predictor, batch_size=2, num_workers=0).run(
                    [{"vision": torch.randn(1, 32)}], output_path  # type: ignore[arg-type]
                )


# ====================================================================
# 2b. Streaming generation
# ====================================================================

class TestStreamingPredictor:

    def _make_generator(self, vocab_size: int = 20) -> TinyGenerator:
        config = Config({"model": {"vocab_size": vocab_size, "hidden_dim": 16}})
        return TinyGenerator(config)

    def test_stops_immediately_on_eos(self) -> None:
        model = self._make_generator()
        model.force_argmax_token(7)  # every step argmaxes to token 7

        sp = StreamingPredictor(
            model=model,
            eos_token_id=7,
            max_new_tokens=10,
            decode_fn=lambda tok: f"tok{tok}",
        )
        tokens = list(sp.stream({"language": torch.randint(0, 20, (1, 5), dtype=torch.long)}))
        assert tokens == []  # EOS hit on first step, nothing yielded

    def test_stops_at_max_new_tokens_when_no_eos(self) -> None:
        model = self._make_generator()
        model.force_argmax_token(3)  # every step argmaxes to token 3, never EOS

        sp = StreamingPredictor(
            model=model,
            eos_token_id=99,  # unreachable
            max_new_tokens=4,
            decode_fn=lambda tok: f"tok{tok}",
        )
        tokens = list(sp.stream({"language": torch.randint(0, 20, (1, 5), dtype=torch.long)}))
        assert tokens == ["tok3", "tok3", "tok3", "tok3"]

    def test_batch_size_greater_than_one_raises(self) -> None:
        model = self._make_generator()
        sp = StreamingPredictor(model=model, max_new_tokens=2)
        with pytest.raises(InferenceError):
            list(sp.stream({"language": torch.randint(0, 20, (2, 5), dtype=torch.long)}))

    def test_missing_context_key_raises(self) -> None:
        model = self._make_generator()
        sp = StreamingPredictor(model=model, max_new_tokens=2, context_key="language")
        with pytest.raises(InferenceError):
            list(sp.stream({"vision": torch.randn(1, 8)}))


# ====================================================================
# 2c. Embedding cache
# ====================================================================

class TestEmbeddingCache:

    def test_put_then_get_returns_same_array(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(cache_dir=str(tmp_path), max_size_gb=1.0)
        emb = np.random.randn(32).astype(np.float32)
        cache.put("sample-key", emb)

        retrieved = cache.get("sample-key")
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, emb)

    def test_get_missing_key_returns_none(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(cache_dir=str(tmp_path), max_size_gb=1.0)
        assert cache.get("does-not-exist") is None

    def test_disk_persists_across_new_instance(self, tmp_path: Path) -> None:
        emb = np.random.randn(16).astype(np.float32)
        cache1 = EmbeddingCache(cache_dir=str(tmp_path), max_size_gb=1.0)
        cache1.put("persisted", emb)

        # Fresh instance, empty in-memory LRU, same cache_dir on disk.
        cache2 = EmbeddingCache(cache_dir=str(tmp_path), max_size_gb=1.0)
        retrieved = cache2.get("persisted")
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, emb)

    def test_clear_removes_disk_files_and_memory(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(cache_dir=str(tmp_path), max_size_gb=1.0)
        cache.put("a", np.zeros(8, dtype=np.float32))
        cache.put("b", np.ones(8, dtype=np.float32))
        assert list(tmp_path.glob("*.npy"))

        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None
        assert list(tmp_path.glob("*.npy")) == []

    def test_evicts_oldest_when_over_max_size(self, tmp_path: Path) -> None:
        # Write one embedding to measure its on-disk footprint, then size
        # the cache to hold ~1.5 files so each new put forces one eviction.
        probe = EmbeddingCache(cache_dir=str(tmp_path), max_size_gb=1.0)
        probe.put("probe", np.random.randn(64).astype(np.float32))
        one_file_bytes = next(tmp_path.glob("*.npy")).stat().st_size
        probe.clear()

        max_size_gb = (one_file_bytes * 1.5) / (1024 ** 3)
        cache = EmbeddingCache(cache_dir=str(tmp_path), max_size_gb=max_size_gb)

        # Age each file well into the past *before* writing the next one,
        # so eviction order is deterministic instead of racing on
        # filesystem mtime resolution when writes land in the same tick.
        old_time = time.time() - 1000
        cache.put("k1", np.random.randn(64).astype(np.float32))
        os.utime(cache._cache_path("k1"), (old_time, old_time))

        cache.put("k2", np.random.randn(64).astype(np.float32))
        # k1+k2 > budget -> k1 (aged) evicted, k2 (fresh) survives.
        remaining = {p.stem for p in tmp_path.glob("*.npy")}
        assert cache._hash_key("k1") not in remaining
        assert cache._hash_key("k2") in remaining

        os.utime(cache._cache_path("k2"), (old_time, old_time))
        cache.put("k3", np.random.randn(64).astype(np.float32))
        remaining = {p.stem for p in tmp_path.glob("*.npy")}
        assert cache._hash_key("k2") not in remaining
        assert cache._hash_key("k3") in remaining

    def test_predictor_embed_round_trips_through_cache(
        self, predictor: FusionPredictor, sample_input: Dict[str, torch.Tensor], tmp_path: Path
    ) -> None:
        """End-to-end: embed() output can be cached and retrieved byte-for-byte,
        the way a search-index build would use it."""
        cache = EmbeddingCache(cache_dir=str(tmp_path), max_size_gb=1.0)
        emb = predictor.embed(sample_input)
        cache.put("doc-1", emb)
        retrieved = cache.get("doc-1")
        np.testing.assert_array_equal(retrieved, emb)


# ====================================================================
# 3. Inference optimization — SKIPS until optimization.py exists
# ====================================================================

class TestOptimization:

    @pytest.fixture(autouse=True)
    def _require_module(self):
        pytest.importorskip(
            "fusion.inference.optimization",
            reason="optimization.py not yet in the repo (owner: Shantanu)",
        )

    def test_optimize_for_inference_returns_nn_module(self, classifier_model: TinyClassifier) -> None:
        from fusion.inference.optimization import optimize_for_inference

        optimized = optimize_for_inference(classifier_model)
        assert isinstance(optimized, nn.Module)

    def test_warmup_calls_predict_n_times(
        self, predictor: FusionPredictor, sample_input: Dict[str, torch.Tensor]
    ) -> None:
        from fusion.inference.optimization import warmup

        call_count = {"n": 0}
        original_predict = predictor.predict

        def counting_predict(inputs):
            call_count["n"] += 1
            return original_predict(inputs)

        predictor.predict = counting_predict  # type: ignore[method-assign]
        warmup(predictor, sample_input, n_warmup=5)
        assert call_count["n"] == 5


# ====================================================================
# 4. Full pipeline, end to end
# ====================================================================

class TestFullPipelineEndToEnd:
    """Deliverables-checklist smoke test: every finished Phase 12 piece
    working together against one checkpoint. Modules not yet merged are
    skipped inline rather than failing the whole scenario.
    """

    def test_checkpoint_to_predictions_to_cache(
        self, classifier_config: Config, tmp_path: Path
    ) -> None:
        model = TinyClassifier(classifier_config)
        ckpt_path = str(tmp_path / "model.pt")
        save_checkpoint(ckpt_path, model, epoch=0)

        predictor = FusionPredictor.from_checkpoint(
            checkpoint_path=ckpt_path, config=classifier_config, model_class=TinyClassifier,
        )

        inputs = {"vision": torch.randn(1, 32), "language": torch.randn(1, 32)}
        result = predictor.predict(inputs)
        assert "predicted_class" in result

        emb = predictor.embed(inputs)
        cache = EmbeddingCache(cache_dir=str(tmp_path / "cache"), max_size_gb=1.0)
        cache.put("item-1", emb)
        assert cache.get("item-1") is not None

        from fusion.inference.batch_inference import BatchInference, FusionDataset

        out_path = str(tmp_path / "batch_out.jsonl")
        dataset = FusionDataset([
            {"vision": torch.randn(1, 32), "language": torch.randn(1, 32)}
            for _ in range(4)
        ])
        BatchInference(predictor, batch_size=2, num_workers=0).run(dataset, out_path)
        with open(out_path) as fh:
            assert sum(1 for line in fh if line.strip()) == 4

        try:
            from fusion.inference.optimization import optimize_for_inference, warmup
        except ImportError:
            pytest.skip("optimization.py not yet in the repo (owner: Shantanu)")
        else:
            optimize_for_inference(model)
            warmup(predictor, inputs, n_warmup=2)
