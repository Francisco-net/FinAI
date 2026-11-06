"""RAG pipeline for financial knowledge retrieval and generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from finai.rag.vector_store import VectorStore, VectorDocument, SearchResult, create_vector_store
from finai.rag.embeddings import EmbeddingModel, EmbeddingConfig, create_embedding_model, FinancialTextProcessor

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    vector_store_type: str = "memory"
    embedding_model_type: str = "sentence-transformer"
    embedding_config: Optional[EmbeddingConfig] = None
    top_k: int = 5
    similarity_threshold: float = 0.3
    max_context_length: int = 4000
    include_metadata: bool = True


@dataclass
class RAGResult:
    """Result from RAG query."""
    answer: str
    sources: list[SearchResult]
    query: str
    context_used: str


class FinancialRAG:
    """RAG system for financial knowledge."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.vector_store = create_vector_store(
            store_type=config.vector_store_type,
            dimension=config.embedding_config.dimension if config.embedding_config else 384,
        )
        self.embedding_model = create_embedding_model(
            model_type=config.embedding_model_type,
            **(config.embedding_config.__dict__ if config.embedding_config else {}),
        )
        self.text_processor = FinancialTextProcessor()

    def add_documents(
        self,
        texts: list[str],
        metadata: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        """Add documents to the knowledge base."""
        # Process texts
        documents = self.text_processor.create_documents(texts, metadata)

        # Generate embeddings
        contents = [doc["content"] for doc in documents]
        embeddings = self.embedding_model.encode(contents, show_progress=True)

        # Create vector documents
        vector_docs = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = ids[i] if ids and i < len(ids) else f"doc_{len(self.vector_store._documents) + i}"
            vector_docs.append(VectorDocument(
                id=doc_id,
                content=doc["content"],
                metadata=doc["metadata"],
                embedding=embedding,
            ))

        # Add to vector store
        self.vector_store.add(vector_docs)
        logger.info(f"Added {len(vector_docs)} documents to knowledge base")

        return [doc.id for doc in vector_docs]

    def add_financial_knowledge(self, knowledge_items: list[dict]) -> list[str]:
        """Add structured financial knowledge items."""
        texts = []
        metadata = []
        ids = []

        for item in knowledge_items:
            # Create rich text representation
            text_parts = [
                f"Termo: {item.get('term', '')}",
                f"Definição: {item.get('definition', '')}",
                f"Categoria: {item.get('category', '')}",
            ]
            if item.get("synonyms"):
                text_parts.append(f"Sinônimos: {', '.join(item['synonyms'])}")
            if item.get("related_terms"):
                text_parts.append(f"Termos relacionados: {', '.join(item['related_terms'])}")

            texts.append("\n".join(text_parts))
            metadata.append({
                "term": item.get("term", ""),
                "category": item.get("category", ""),
                "type": "financial_knowledge",
            })
            ids.append(f"knowledge_{item.get('term', '').lower().replace(' ', '_')}")

        return self.add_documents(texts, metadata, ids)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[dict] = None,
    ) -> list[SearchResult]:
        """Search for relevant documents."""
        top_k = top_k or self.config.top_k

        # Expand query terms
        expanded_query = self.text_processor.expand_terms(query)

        # Generate query embedding
        query_embedding = self.embedding_model.encode_single(expanded_query)

        # Search vector store
        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

        # Filter by similarity threshold
        filtered = [r for r in results if r.score >= self.config.similarity_threshold]

        return filtered

    def get_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[dict] = None,
    ) -> tuple[str, list[SearchResult]]:
        """Get formatted context for LLM."""
        results = self.search(query, top_k, filter_metadata)

        if not results:
            return "", []

        context_parts = []
        total_length = 0

        for i, result in enumerate(results):
            doc = result.document
            metadata_str = ""
            if self.config.include_metadata and doc.metadata:
                meta_parts = [f"{k}: {v}" for k, v in doc.metadata.items() if k != "source_index"]
                if meta_parts:
                    metadata_str = f" [{'; '.join(meta_parts)}]"

            chunk = f"[Fonte {i+1} (score: {result.score:.3f}){metadata_str}]\n{doc.content}\n"
            chunk_length = len(chunk)

            if total_length + chunk_length > self.config.max_context_length:
                break

            context_parts.append(chunk)
            total_length += chunk_length

        context = "\n".join(context_parts)
        return context, results

    def query(
        self,
        query: str,
        llm_client: Any,  # Anthropic or other LLM client
        system_prompt: Optional[str] = None,
        top_k: Optional[int] = None,
        filter_metadata: Optional[dict] = None,
    ) -> RAGResult:
        """Query the RAG system with an LLM."""
        context, sources = self.get_context(query, top_k, filter_metadata)

        if not context:
            return RAGResult(
                answer="Não encontrei informações relevantes na base de conhecimento para responder sua pergunta.",
                sources=[],
                query=query,
                context_used="",
            )

        # Default system prompt for financial assistant
        if system_prompt is None:
            system_prompt = """Você é um assistente financeiro especializado no mercado brasileiro.
Use APENAS as informações fornecidas no contexto para responder.
Se a informação não estiver no contexto, diga que não sabe.
Cite as fontes usando [Fonte X] no final de cada afirmação.
Responda em português, de forma clara e educativa."""

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Contexto:\n{context}\n\nPergunta: {query}"},
        ]

        # Call LLM
        try:
            # Assuming Anthropic-style client
            response = llm_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                messages=messages,
            )
            answer = response.content[0].text
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            # Fallback: return context as answer
            answer = f"Baseado no contexto recuperado:\n\n{context}"

        return RAGResult(
            answer=answer,
            sources=sources,
            query=query,
            context_used=context,
        )


class HybridRetriever:
    """Hybrid retriever combining vector search with keyword/BM25 search."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        keyword_weight: float = 0.3,
        vector_weight: float = 0.7,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight
        self._bm25_index = None
        self._documents: list[VectorDocument] = []

    def add_documents(self, documents: list[VectorDocument]) -> None:
        """Add documents to both vector and keyword indices."""
        self.vector_store.add(documents)
        self._documents.extend(documents)
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        """Build BM25 index for keyword search."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed, keyword search disabled")
            return

        corpus = [doc.content.split() for doc in self._documents]
        self._bm25_index = BM25Okapi(corpus)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> list[SearchResult]:
        """Hybrid search combining vector and keyword scores."""
        # Vector search
        query_embedding = self.embedding_model.encode_single(query)
        vector_results = self.vector_store.search(query_embedding, top_k * 2, filter_metadata)

        # Keyword search
        keyword_scores = {}
        if self._bm25_index is not None:
            tokenized_query = query.split()
            scores = self._bm25_index.get_scores(tokenized_query)
            for i, score in enumerate(scores):
                if i < len(self._documents):
                    doc = self._documents[i]
                    if filter_metadata is None or all(
                        doc.metadata.get(k) == v for k, v in filter_metadata.items()
                    ):
                        keyword_scores[doc.id] = score

        # Combine scores
        combined_results = []
        for v_result in vector_results:
            doc_id = v_result.document.id
            vector_score = v_result.score
            keyword_score = keyword_scores.get(doc_id, 0)

            # Normalize keyword score (simple min-max)
            max_kw = max(keyword_scores.values()) if keyword_scores else 1
            norm_keyword = keyword_score / max_kw if max_kw > 0 else 0

            combined_score = (
                self.vector_weight * vector_score +
                self.keyword_weight * norm_keyword
            )

            combined_results.append(SearchResult(
                document=v_result.document,
                score=combined_score,
            ))

        # Sort by combined score
        combined_results.sort(key=lambda x: x.score, reverse=True)

        return combined_results[:top_k]


def create_financial_rag(
    vector_store_type: str = "memory",
    embedding_model_type: str = "sentence-transformer",
    top_k: int = 5,
    **embedding_kwargs,
) -> FinancialRAG:
    """Factory function to create FinancialRAG with default config."""
    embedding_config = EmbeddingConfig(**embedding_kwargs)
    config = RAGConfig(
        vector_store_type=vector_store_type,
        embedding_model_type=embedding_model_type,
        embedding_config=embedding_config,
        top_k=top_k,
    )
    return FinancialRAG(config)