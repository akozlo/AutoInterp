"""Pluggable text embeddings with disk cache. Title-only when abstract not in graph."""

import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional

DEFAULT_EMBED_CACHE_DIR = "output/embeddings"
DETERMINISTIC_SEED = 42


class Embedder:
    """Abstract embedder: given list of texts, return list of vectors (same order)."""

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    """Local embeddings via sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        model = self._get_model()
        import numpy as np
        arr = model.encode(texts, show_progress_bar=False)
        return arr.tolist()


class OpenAIEmbedder(Embedder):
    """OpenAI embeddings when API key is available."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY not set; cannot use OpenAI embedder")
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self._api_key)
            out = client.embeddings.create(input=texts, model=self.model)
            return [e.embedding for e in out.data]
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding failed: {e}") from e


def _cache_key(embedder_id: str, texts: List[str]) -> str:
    raw = embedder_id + "\n" + "\n".join(texts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_embedder(
    backend: str = "sentence-transformers",
    model: Optional[str] = None,
) -> Embedder:
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name=model or "all-MiniLM-L6-v2")
    if backend == "openai":
        return OpenAIEmbedder(model=model or "text-embedding-3-small")
    raise ValueError(f"Unknown embedder backend: {backend}")


def embed_papers_cached(
    node_ids: List[str],
    texts: List[str],
    embedder: Embedder,
    cache_dir: str | Path,
) -> List[List[float]]:
    """
    Embed texts for papers; use disk cache keyed by (embedder_id + sorted node_id + text).
    Returns list of vectors in same order as node_ids/texts.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    embedder_id = getattr(embedder, "model_name", None) or getattr(embedder, "model", "embedder")
    if isinstance(embedder_id, str) and "/" in embedder_id:
        embedder_id = embedder_id.replace("/", "_")
    index_path = cache_dir / "index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
    else:
        index = {}

    to_compute: List[int] = []
    results: List[Optional[List[float]]] = [None] * len(node_ids)
    for i, (nid, text) in enumerate(zip(node_ids, texts)):
        key = _cache_key(embedder_id, [nid, text])
        if key in index:
            path = cache_dir / (index[key] + ".json")
            if path.exists():
                with open(path) as f:
                    results[i] = json.load(f)
                continue
        to_compute.append(i)

    if to_compute:
        batch_texts = [texts[i] for i in to_compute]
        batch_vectors = embedder.embed(batch_texts)
        for idx, vec in zip(to_compute, batch_vectors):
            results[idx] = vec
            nid, text = node_ids[idx], texts[idx]
            key = _cache_key(embedder_id, [nid, text])
            fname = key[:16] + ".json"
            index[key] = key[:16]
            with open(cache_dir / (key[:16] + ".json"), "w") as f:
                json.dump(vec, f)
        with open(index_path, "w") as f:
            json.dump(index, f, indent=0)

    return results
