"""Vector store for financial knowledge retrieval."""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    """Document with vector embedding."""
    id: str
    content: str
    metadata: dict[str, Any]
    embedding: Optional[np.ndarray] = None


@dataclass
class SearchResult:
    """Search result with score."""
    document: VectorDocument
    score: float


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add(self, documents: list[VectorDocument]) -> None:
        """Add documents to the store."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> list[SearchResult]:
        """Search for similar documents."""
        pass

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete documents by ID."""
        pass

    @abstractmethod
    def get(self, id: str) -> Optional[VectorDocument]:
        """Get document by ID."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return number of documents."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all documents."""
        pass


class InMemoryVectorStore(VectorStore):
    """In-memory vector store using numpy for similarity search."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._documents: dict[str, VectorDocument] = {}
        self._embeddings: Optional[np.ndarray] = None
        self._ids: list[str] = []

    def add(self, documents: list[VectorDocument]) -> None:
        new_embeddings = []
        new_ids = []

        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"Document {doc.id} has no embedding")
            if doc.embedding.shape[0] != self.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.dimension}, got {doc.embedding.shape[0]}"
                )

            self._documents[doc.id] = doc
            new_embeddings.append(doc.embedding)
            new_ids.append(doc.id)

        if new_embeddings:
            new_embeddings_array = np.stack(new_embeddings)
            if self._embeddings is None:
                self._embeddings = new_embeddings_array
                self._ids = new_ids
            else:
                self._embeddings = np.vstack([self._embeddings, new_embeddings_array])
                self._ids.extend(new_ids)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> list[SearchResult]:
        if self._embeddings is None or len(self._ids) == 0:
            return []

        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        # Normalize stored embeddings
        doc_norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-8
        normalized_embeddings = self._embeddings / doc_norms

        # Cosine similarity
        similarities = np.dot(normalized_embeddings, query_norm)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get more for filtering

        results = []
        for idx in top_indices:
            doc_id = self._ids[idx]
            doc = self._documents[doc_id]

            # Apply metadata filter if provided
            if filter_metadata:
                match = all(
                    doc.metadata.get(k) == v for k, v in filter_metadata.items()
                )
                if not match:
                    continue

            results.append(SearchResult(document=doc, score=float(similarities[idx])))

            if len(results) >= top_k:
                break

        return results

    def delete(self, ids: list[str]) -> None:
        # Rebuild without deleted IDs
        keep_indices = [i for i, doc_id in enumerate(self._ids) if doc_id not in ids]

        if len(keep_indices) == len(self._ids):
            return  # Nothing to delete

        for doc_id in ids:
            self._documents.pop(doc_id, None)

        if keep_indices:
            self._embeddings = self._embeddings[keep_indices]
            self._ids = [self._ids[i] for i in keep_indices]
        else:
            self._embeddings = None
            self._ids = []

    def get(self, id: str) -> Optional[VectorDocument]:
        return self._documents.get(id)

    def count(self) -> int:
        return len(self._ids)

    def clear(self) -> None:
        self._documents.clear()
        self._embeddings = None
        self._ids = []

    def save(self, path: str) -> None:
        """Save vector store to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "dimension": self.dimension,
            "documents": {},
            "ids": self._ids,
        }

        for doc_id, doc in self._documents.items():
            data["documents"][doc_id] = {
                "content": doc.content,
                "metadata": doc.metadata,
                "embedding": doc.embedding.tolist() if doc.embedding is not None else None,
            }

        if self._embeddings is not None:
            data["embeddings"] = self._embeddings.tolist()

        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "InMemoryVectorStore":
        """Load vector store from disk."""
        with open(path, "r") as f:
            data = json.load(f)

        store = cls(dimension=data["dimension"])
        store._ids = data["ids"]

        if "embeddings" in data and data["embeddings"]:
            store._embeddings = np.array(data["embeddings"])
        else:
            store._embeddings = None

        for doc_id, doc_data in data["documents"].items():
            embedding = np.array(doc_data["embedding"]) if doc_data["embedding"] else None
            store._documents[doc_id] = VectorDocument(
                id=doc_id,
                content=doc_data["content"],
                metadata=doc_data["metadata"],
                embedding=embedding,
            )

        return store


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store for production use."""

    def __init__(self, dimension: int = 384, index_type: str = "flat"):
        self.dimension = dimension
        self.index_type = index_type
        self._index = None
        self._documents: dict[str, VectorDocument] = {}
        self._id_to_index: dict[str, int] = {}
        self._next_index = 0
        self._init_index()

    def _init_index(self) -> None:
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS not installed. Install with: pip install faiss-cpu")

        if self.index_type == "flat":
            self._index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine with normalized vectors)
        elif self.index_type == "ivf":
            nlist = 100
            quantizer = faiss.IndexFlatIP(self.dimension)
            self._index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        elif self.index_type == "hnsw":
            self._index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

    def add(self, documents: list[VectorDocument]) -> None:
        if self._index is None:
            self._init_index()

        embeddings = []
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"Document {doc.id} has no embedding")

            # Normalize for cosine similarity
            embedding = doc.embedding / (np.linalg.norm(doc.embedding) + 1e-8)
            embeddings.append(embedding)

            self._documents[doc.id] = doc
            self._id_to_index[doc.id] = self._next_index
            self._next_index += 1

        if embeddings:
            embeddings_array = np.stack(embeddings).astype(np.float32)

            # Train IVF index if needed
            if self.index_type == "ivf" and not self._index.is_trained:
                self._index.train(embeddings_array)

            self._index.add(embeddings_array)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> list[SearchResult]:
        if self._index is None or self._index.ntotal == 0:
            return []

        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        query_norm = query_norm.astype(np.float32).reshape(1, -1)

        # Search
        scores, indices = self._index.search(query_norm, min(top_k * 3, self._index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            # Find document ID by index
            doc_id = None
            for did, i in self._id_to_index.items():
                if i == idx:
                    doc_id = did
                    break

            if doc_id is None:
                continue

            doc = self._documents[doc_id]

            # Apply metadata filter
            if filter_metadata:
                match = all(
                    doc.metadata.get(k) == v for k, v in filter_metadata.items()
                )
                if not match:
                    continue

            results.append(SearchResult(document=doc, score=float(score)))

            if len(results) >= top_k:
                break

        return results

    def delete(self, ids: list[str]) -> None:
        # FAISS doesn't support efficient deletion, rebuild index
        for doc_id in ids:
            if doc_id in self._documents:
                del self._documents[doc_id]
                del self._id_to_index[doc_id]

        # Rebuild index
        self._init_index()
        if self._documents:
            self.add(list(self._documents.values()))

    def get(self, id: str) -> Optional[VectorDocument]:
        return self._documents.get(id)

    def count(self) -> int:
        return len(self._documents)

    def clear(self) -> None:
        self._documents.clear()
        self._id_to_index.clear()
        self._next_index = 0
        self._init_index()

    def save(self, path: str) -> None:
        """Save FAISS index and documents."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        if self._index is not None:
            import faiss
            faiss.write_index(self._index, f"{path}.faiss")

        # Save documents separately
        import json
        doc_data = {}
        for doc_id, doc in self._documents.items():
            doc_data[doc_id] = {
                "content": doc.content,
                "metadata": doc.metadata,
            }

        with open(f"{path}.json", "w") as f:
            json.dump({
                "documents": doc_data,
                "id_to_index": self._id_to_index,
                "next_index": self._next_index,
                "dimension": self.dimension,
                "index_type": self.index_type,
            }, f)

    @classmethod
    def load(cls, path: str) -> "FAISSVectorStore":
        """Load FAISS index and documents."""
        import faiss
        import json

        with open(f"{path}.json", "r") as f:
            data = json.load(f)

        store = cls(dimension=data["dimension"], index_type=data["index_type"])
        store._id_to_index = data["id_to_index"]
        store._next_index = data["next_index"]

        store._index = faiss.read_index(f"{path}.faiss")

        for doc_id, doc_data in data["documents"].items():
            store._documents[doc_id] = VectorDocument(
                id=doc_id,
                content=doc_data["content"],
                metadata=doc_data["metadata"],
                embedding=None,  # Embeddings stored in FAISS index
            )

        return store


def create_vector_store(
    store_type: str = "memory",
    dimension: int = 384,
    **kwargs,
) -> VectorStore:
    """Factory function to create vector store."""
    if store_type == "memory":
        return InMemoryVectorStore(dimension=dimension)
    elif store_type == "faiss":
        return FAISSVectorStore(dimension=dimension, **kwargs)
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")