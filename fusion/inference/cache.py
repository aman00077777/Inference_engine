from collections import OrderedDict
from pathlib import Path
from typing import Optional
import hashlib

import numpy as np


class EmbeddingCache:
    """Cache fused embeddings in memory and on disk."""

    def __init__(self, cache_dir: str, max_size_gb: float):
        """
        Initialize the embedding cache.

        Args:
            cache_dir: Directory used for disk caching.
            max_size_gb: Maximum disk cache size in GB.
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_gb = max_size_gb

        # Create cache directory if it does not exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory LRU cache
        self._memory_cache = OrderedDict()

    def _hash_key(self, key: str) -> str:
        """Generate a stable hash for a cache key."""
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        """Return the disk path associated with a cache key."""
        return self.cache_dir / f"{self._hash_key(key)}.npy"

    def get(self, key: str) -> Optional[np.ndarray]:
        """
        Retrieve an embedding from the cache.

        Checks the in-memory LRU cache first.
        If not found, checks the disk cache.

        Returns:
            The cached embedding, or None if it does not exist.
        """

        # Check in-memory LRU first
        if key in self._memory_cache:
            embedding = self._memory_cache.pop(key)

            # Move recently used item to the end
            self._memory_cache[key] = embedding

            return embedding

        # Check disk cache
        cache_path = self._cache_path(key)

        if cache_path.exists():
            embedding = np.load(cache_path)

            # Add disk-loaded embedding to memory cache
            self._memory_cache[key] = embedding

            return embedding

        return None

    def put(self, key: str, embedding: np.ndarray) -> None:
        """
        Store an embedding in memory and on disk.
        """

        # Update memory cache
        if key in self._memory_cache:
            self._memory_cache.pop(key)

        self._memory_cache[key] = embedding

        # Save to disk
        cache_path = self._cache_path(key)
        np.save(cache_path, embedding)

        # Enforce disk size limit
        self._evict_if_needed()

    def _disk_usage(self) -> int:
        """Return total size of cached .npy files in bytes."""
        return sum(
            path.stat().st_size
            for path in self.cache_dir.glob("*.npy")
            if path.is_file()
        )

    def _evict_if_needed(self) -> None:
        """Remove oldest cache files if the disk limit is exceeded."""

        max_size_bytes = self.max_size_gb * (1024 ** 3)

        while self._disk_usage() > max_size_bytes:
            cache_files = [
                path
                for path in self.cache_dir.glob("*.npy")
                if path.is_file()
            ]

            if not cache_files:
                break

            # Oldest file first
            oldest = min(
                cache_files,
                key=lambda path: path.stat().st_mtime
            )

            oldest.unlink()

            # Remove corresponding entry from memory cache
            for key in list(self._memory_cache.keys()):
                if self._cache_path(key) == oldest:
                    del self._memory_cache[key]
                    break

    def clear(self) -> None:
        """Remove all cached embeddings from memory and disk."""

        # Remove disk cache files
        for path in self.cache_dir.glob("*.npy"):
            if path.is_file():
                path.unlink()

        # Clear memory cache
        self._memory_cache.clear()


__all__ = ["EmbeddingCache"]