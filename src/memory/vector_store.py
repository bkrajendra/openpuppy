"""ChromaDB vector store for semantic long-term memory."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

_vector_client: Any = None
_collection_name = "agent_memory"


def _get_client(persist_path: str | Path = "data/vector_store") -> Any:
    global _vector_client
    if _vector_client is None:
        import chromadb
        from chromadb.config import Settings
        path = Path(persist_path)
        path.mkdir(parents=True, exist_ok=True)
        _vector_client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
    return _vector_client


class VectorStore:
    """
    Semantic memory store using ChromaDB.
    Use store_memory/retrieve_memory tools or direct add/query.
    """

    def __init__(self, persist_path: str | Path = "data/vector_store", collection_name: str = "agent_memory") -> None:
        self.persist_path = Path(persist_path)
        self.collection_name = collection_name
        self._coll: Any = None

    def _collection(self):
        if self._coll is None:
            client = _get_client(self.persist_path)
            self._coll = client.get_or_create_collection(name=self.collection_name, metadata={"description": "Agent long-term semantic memory"})
        return self._coll

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        """Add a memory (text) and return its id."""
        doc_id = str(uuid.uuid4())
        meta = metadata or {}
        self._collection().add(documents=[text], metadatas=[meta], ids=[doc_id])
        logger.info("vector_store_add", id=doc_id, text_len=len(text))
        return doc_id

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search by semantic similarity; returns list of {id, document, metadata, distance}."""
        results = self._collection().query(query_texts=[query], n_results=top_k, include=["documents", "metadatas", "distances"])
        out = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for i, doc in enumerate(docs):
            out.append({
                "id": ids[i] if i < len(ids) else "",
                "document": doc,
                "metadata": metas[i] if i < len(metas) else {},
                "distance": dists[i] if i < len(dists) else None,
            })
        return out
