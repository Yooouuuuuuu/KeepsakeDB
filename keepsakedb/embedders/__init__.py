"""Embedders: anything with ``name``, ``dims`` and ``embed(texts) -> (n, dims) float32``,
L2-normalized so cosine == dot product.

- OpenAIEmbedder: any OpenAI-compatible ``/v1/embeddings`` endpoint (llama-server,
  Ollama, LM Studio, vLLM, OpenAI). Zero extra dependencies (urllib).
- LlamaCppEmbedder: in-process GGUF via llama-cpp-python (``keepsakedb[llama]``).
- CachedEmbedder: wraps any embedder with an on-disk SQLite cache keyed by text+model.

The memory language and the embedder travel together: a store written in zh-TW needs
an embedder that covers Chinese (bge-m3 covers 100+ languages). Changing the embedder
means re-embedding the store (``Keepsake.reembed``).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import urllib.request
from typing import List, Optional, Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    name: str
    dims: Optional[int]

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def _normalize(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return a / n


class OpenAIEmbedder:
    def __init__(self, base_url: str = "http://127.0.0.1:55602/v1", model: str = "bge-m3",
                 api_key: str = "", batch_size: int = 32, timeout: float = 120.0,
                 dims: Optional[int] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.batch_size = batch_size
        self.timeout = timeout
        self.dims = dims
        self.name = f"openai:{model}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = list(texts[i:i + self.batch_size])
            body = json.dumps({"model": self.model, "input": chunk}).encode()
            req = urllib.request.Request(self.base_url + "/embeddings", data=body,
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": f"Bearer {self.api_key or 'none'}"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode())
            rows = sorted(data["data"], key=lambda d: d.get("index", 0))
            out.append(np.asarray([d["embedding"] for d in rows], dtype=np.float32))
        if not out:
            return np.zeros((0, self.dims or 0), dtype=np.float32)
        arr = _normalize(np.concatenate(out))
        self.dims = arr.shape[1]
        return arr


class LlamaCppEmbedder:
    def __init__(self, model_path: str, n_ctx: int = 8192, n_threads: Optional[int] = None,
                 pooling: str = "cls", verbose: bool = False) -> None:
        from llama_cpp import Llama, LLAMA_POOLING_TYPE_CLS, LLAMA_POOLING_TYPE_MEAN

        pool = {"cls": LLAMA_POOLING_TYPE_CLS, "mean": LLAMA_POOLING_TYPE_MEAN}[pooling]
        self._llm = Llama(model_path=model_path, embedding=True, n_ctx=n_ctx,
                          n_threads=n_threads, pooling_type=pool, verbose=verbose)
        self.name = f"llama:{model_path.rsplit('/', 1)[-1].rsplit(chr(92), 1)[-1]}"
        self.dims: Optional[int] = None

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._llm.embed(list(texts))
        arr = _normalize(np.asarray(vecs, dtype=np.float32))
        self.dims = arr.shape[1]
        return arr


class CachedEmbedder:
    """On-disk cache so repeated texts (benchmarks, retries, dedup checks) embed once."""

    def __init__(self, inner: Embedder, path: str) -> None:
        self.inner = inner
        self.name = inner.name
        self.dims = inner.dims
        self.conn = sqlite3.connect(path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, vec BLOB)")
        self.hits = 0
        self.misses = 0

    def _key(self, t: str) -> str:
        return hashlib.sha1((self.name + "\x00" + t).encode()).hexdigest()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        found = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i:i + 500]
            q = f"SELECT key, vec FROM cache WHERE key IN ({','.join('?' * len(chunk))})"
            found.update({k: np.frombuffer(v, dtype=np.float32) for k, v in self.conn.execute(q, chunk)})
        missing = [i for i, k in enumerate(keys) if k not in found]
        self.hits += len(keys) - len(missing)
        self.misses += len(missing)
        if missing:
            vecs = self.inner.embed([texts[i] for i in missing])
            self.dims = self.inner.dims
            with self.conn:
                for i, v in zip(missing, vecs):
                    found[keys[i]] = v
                    self.conn.execute("INSERT OR REPLACE INTO cache(key, vec) VALUES (?, ?)", (keys[i], v.tobytes()))
        arr = np.stack([found[k] for k in keys]) if keys else np.zeros((0, self.dims or 0), dtype=np.float32)
        if self.dims is None and arr.size:
            self.dims = arr.shape[1]
        return arr


class HashEmbedder:
    """Deterministic pseudo-embedder for tests (no semantics, only plumbing)."""

    def __init__(self, dims: int = 32) -> None:
        self.dims = dims
        self.name = f"hash:{dims}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha1(t.encode()).digest()[:8], "little")
            rows.append(np.random.default_rng(seed).standard_normal(self.dims))
        return _normalize(np.asarray(rows, dtype=np.float32)) if rows else np.zeros((0, self.dims), dtype=np.float32)
