from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# Prefer new package, fall back to community for compatibility with older envs
try:
    from langchain_qdrant import Qdrant as _Qdrant
except Exception:  # pragma: no cover
    from langchain_community.vectorstores import Qdrant as _Qdrant

# Local fallback vector store (Chroma)
from langchain_community.vectorstores import Chroma

from .config import AppConfig


def create_qdrant_client(config: AppConfig) -> QdrantClient:
    # Only pass api_key if it is truthy to avoid insecure-connection warnings on localhost
    # Increase timeout for Qdrant Cloud to reduce write timeouts on large upserts
    timeout = max(float(config.http_timeout_seconds), 30.0)
    if config.qdrant_api_key:
        return QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key, timeout=timeout)
    return QdrantClient(url=config.qdrant_url, timeout=timeout)


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    existing = client.get_collections()
    names = {c.name for c in existing.collections}
    if collection_name in names:
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
    )


def recreate_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    """Delete and recreate the collection so it contains only the next ingest."""
    existing = client.get_collections()
    names = {c.name for c in existing.collections}
    if collection_name in names:
        client.delete_collection(collection_name=collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
    )


def get_vectorstore(client: QdrantClient, collection_name: str, embeddings) -> _Qdrant:
    return _Qdrant(client=client, collection_name=collection_name, embeddings=embeddings)


def get_chroma_vectorstore(collection_name: str, embeddings, persist_directory: str = ".chroma") -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

