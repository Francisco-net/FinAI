"""RAG (Retrieval-Augmented Generation) module for FinAI."""

from finai.rag.vector_store import (
    VectorStore,
    VectorDocument,
    SearchResult,
    InMemoryVectorStore,
    FAISSVectorStore,
    create_vector_store,
)
from finai.rag.embeddings import (
    EmbeddingModel,
    EmbeddingConfig,
    SentenceTransformerEmbedding,
    OpenAIEmbedding,
    CohereEmbedding,
    MockEmbedding,
    create_embedding_model,
    FinancialTextProcessor,
)
from finai.rag.pipeline import (
    RAGConfig,
    RAGResult,
    FinancialRAG,
    HybridRetriever,
    create_financial_rag,
)

__all__ = [
    # Vector Store
    "VectorStore",
    "VectorDocument",
    "SearchResult",
    "InMemoryVectorStore",
    "FAISSVectorStore",
    "create_vector_store",
    # Embeddings
    "EmbeddingModel",
    "EmbeddingConfig",
    "SentenceTransformerEmbedding",
    "OpenAIEmbedding",
    "CohereEmbedding",
    "MockEmbedding",
    "create_embedding_model",
    "FinancialTextProcessor",
    # Pipeline
    "RAGConfig",
    "RAGResult",
    "FinancialRAG",
    "HybridRetriever",
    "create_financial_rag",
]