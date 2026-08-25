"""Styla · Representation & Retrieval (Rol C).

FashionCLIP embedding-ləri, vector search (numpy / pgvector) və
reference outfit matching.

İctimai API:
    from ml.retrieval import FashionCLIPEmbedder, Searcher, Matcher, get_store
    from ml.retrieval.metrics import recall_at_k, evaluate_ranking

Qeyd: `config` birbaşa import olunur, qalanları lazy — `import ml.retrieval`
torch/transformers-i yükləməsin deyə.
"""
from ml.retrieval import config

__all__ = [
    "config",
    "FashionCLIPEmbedder",
    "Searcher",
    "Matcher",
    "get_store",
    "SearchResult",
    "VectorStore",
]


def __getattr__(name: str):
    """Lazy import — ağır asılılıqlar yalnız istifadə olunanda yüklənir."""
    if name == "FashionCLIPEmbedder":
        from ml.retrieval.embedder import FashionCLIPEmbedder

        return FashionCLIPEmbedder
    if name == "Searcher":
        from ml.retrieval.search import Searcher

        return Searcher
    if name == "Matcher":
        from ml.retrieval.matcher import Matcher

        return Matcher
    if name in ("get_store", "SearchResult", "VectorStore"):
        from ml.retrieval import store  # noqa: F401
        from ml.retrieval.store import base

        return getattr(base, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
