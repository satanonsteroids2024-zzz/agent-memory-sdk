"""Optional embedding support for the SQLite backend.

Installed via the ``semantic`` extra::

    pip install agent-memory-sdk[semantic]

which pulls in ``sqlite-vec`` (vector index inside SQLite) and ``fastembed``
(ONNX MiniLM embeddings, no torch). Everything here degrades gracefully:
if the extras are missing, the SQLite backend stays lexical-only.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

# An embedder maps a batch of texts to a batch of float vectors.
Embedder = Callable[[list[str]], list[list[float]]]


@lru_cache(maxsize=1)
def get_default_embedder() -> Embedder | None:
    """Return an embedding callable from whichever optional backend is installed.

    Tries fastembed first (lightweight ONNX), then sentence-transformers.
    Returns None when neither is available. Cached so the model loads once.
    """
    try:
        from fastembed import TextEmbedding

        model = TextEmbedding("BAAI/bge-small-en-v1.5")

        def _fastembed(texts: list[str]) -> list[list[float]]:
            return [vec.tolist() for vec in model.embed(texts)]

        return _fastembed
    except ImportError:
        pass

    try:
        from sentence_transformers import SentenceTransformer

        st_model = SentenceTransformer("all-MiniLM-L6-v2")

        def _sentence_transformers(texts: list[str]) -> list[list[float]]:
            vectors: list[list[float]] = st_model.encode(
                texts, normalize_embeddings=True
            ).tolist()
            return vectors

        return _sentence_transformers
    except ImportError:
        return None


def embedding_dimension(embedder: Embedder) -> int:
    """Probe the embedder once to learn its output dimension."""
    return len(embedder(["dimension probe"])[0])
