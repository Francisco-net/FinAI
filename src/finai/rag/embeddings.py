"""Embedding generation for financial text."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    dimension: int = 384
    max_length: int = 512
    batch_size: int = 32
    device: str = "auto"  # auto, cpu, cuda
    normalize: bool = True


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def encode(
        self,
        texts: list[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts into embeddings."""
        pass

    @abstractmethod
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass


class SentenceTransformerEmbedding(EmbeddingModel):
    """Sentence Transformers embedding model."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model = None
        self._device = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.config.model_name)
            self._model.to(self.device)
        return self._model

    @property
    def device(self) -> str:
        if self._device is None:
            if self.config.device == "auto":
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.config.device
        return self._device

    def encode(
        self,
        texts: list[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        batch_size = batch_size or self.config.batch_size
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
            device=self.device,
        )
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    @property
    def dimension(self) -> int:
        return self.config.dimension


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI embedding model (text-embedding-3-small/large)."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimension: int = 1536,
    ):
        self.model_name = model_name
        self._dimension = dimension
        self._client = None

        if api_key:
            import os
            os.environ["OPENAI_API_KEY"] = api_key

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def encode(
        self,
        texts: list[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        # OpenAI API handles batching internally
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        embeddings = np.array([d.embedding for d in response.data])
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimension


class CohereEmbedding(EmbeddingModel):
    """Cohere embedding model."""

    def __init__(
        self,
        model_name: str = "embed-multilingual-v3.0",
        api_key: Optional[str] = None,
        dimension: int = 1024,
    ):
        self.model_name = model_name
        self._dimension = dimension
        self._client = None

        if api_key:
            import os
            os.environ["COHERE_API_KEY"] = api_key

    @property
    def client(self):
        if self._client is None:
            import cohere
            self._client = cohere.Client()
        return self._client

    def encode(
        self,
        texts: list[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        response = self.client.embed(
            model=self.model_name,
            texts=texts,
            input_type="search_document",
        )
        embeddings = np.array(response.embeddings)
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimension


class MockEmbedding(EmbeddingModel):
    """Mock embedding for testing without external dependencies."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self._cache: dict[str, np.ndarray] = {}

    def encode(
        self,
        texts: list[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        embeddings = []
        for text in texts:
            if text in self._cache:
                embeddings.append(self._cache[text])
            else:
                # Deterministic pseudo-random embedding based on text hash
                np.random.seed(hash(text) % 2**32)
                emb = np.random.randn(self._dimension).astype(np.float32)
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                self._cache[text] = emb
                embeddings.append(emb)
        return np.stack(embeddings)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimension


def create_embedding_model(
    model_type: str = "sentence-transformer",
    **kwargs,
) -> EmbeddingModel:
    """Factory function to create embedding model."""
    if model_type == "sentence-transformer":
        config = EmbeddingConfig(**kwargs)
        return SentenceTransformerEmbedding(config)
    elif model_type == "openai":
        return OpenAIEmbedding(**kwargs)
    elif model_type == "cohere":
        return CohereEmbedding(**kwargs)
    elif model_type == "mock":
        return MockEmbedding(**kwargs)
    else:
        raise ValueError(f"Unknown embedding model type: {model_type}")


class FinancialTextProcessor:
    """Process financial text for better embedding quality."""

    # Financial term expansions for better semantic matching
    TERM_EXPANSIONS = {
        "cdi": "Certificado de Depósito Interbancário taxa de juros",
        "selic": "Sistema Especial de Liquidação e Custódia taxa básica juros",
        "ipca": "Índice Nacional de Preços ao Consumidor Amplo inflação",
        "igpm": "Índice Geral de Preços do Mercado inflação aluguel",
        "tesouro direto": "Tesouro Direto títulos públicos governo federal",
        "tesouro selic": "Tesouro Selic LFT título pós-fixado taxa Selic",
        "tesouro ipca": "Tesouro IPCA+ NTN-B título inflação taxa real",
        "tesouro prefixado": "Tesouro Prefixado LTN título taxa fixa",
        "cdb": "Certificado de Depósito Bancário renda fixa banco",
        "lci": "Letra de Crédito Imobiliário isento IR pessoa física",
        "lca": "Letra de Crédito do Agronegócio isento IR pessoa física",
        "debênture": "debênture título dívida privada empresa",
        "debênture incentivada": "debênture incentivada isenta IR infraestrutura",
        "cri": "Certificado de Recebíveis Imobiliários lastro imobiliário",
        "cra": "Certificado de Recebíveis do Agronegócio lastro agro",
        "fii": "Fundo de Investimento Imobiliário imóveis aluguel dividendos",
        "etf": "Exchange Traded Fund fundo índice bolsa",
        "acoes": "ações ações ordinárias preferenciais bolsa B3",
        "dividendos": "dividendos proventos lucro acionistas isento IR",
        "jcp": "Juros sobre Capital Próprio JCP proventos IR 15%",
        "fundo multimercado": "fundo multimercado estratégia livre vários ativos",
        "fundo renda fixa": "fundo renda fixa títulos dívida conservador",
        "fundo ações": "fundo ações bolsa brasileiro exterior",
        "previdência privada": "previdência privada PGBL VGBL aposentadoria",
        "reserva emergência": "reserva emergência liquidez diária Tesouro Selic CDB",
        "perfil risco": "perfil risco conservador moderado arrojado suitability",
        "alocação ativos": "alocação ativos diversificação carteira otimização",
        "rebalanceamento": "rebalanceamento carteira pesos alvo desvio",
        "marcação mercado": "marcação a mercado MTM preço atual título",
        "duration": "duration duração sensibilidade juros título",
        "convexidade": "convexidade segunda derivada preço juros",
        "sharpe": "Sharpe ratio retorno risco livre volatilidade",
        "sortino": "Sortino ratio retorno risco downside volatilidade negativa",
        "var": "Value at Risk VaR perda máxima confiança",
        "cvar": "Conditional Value at Risk CVaR expected shortfall",
        "drawdown": "drawdown queda pico vale recuperação",
        "volatilidade": "volatilidade desvio padrão retornos risco",
        "correlação": "correlação co-movimento ativos diversificação",
        "beta": "beta risco sistemático mercado CAPM",
        "alpha": "alpha retorno excesso benchmark gestão ativa",
    }

    def __init__(self):
        pass

    def expand_terms(self, text: str) -> str:
        """Expand financial abbreviations and terms."""
        expanded = text.lower()
        for term, expansion in self.TERM_EXPANSIONS.items():
            # Replace whole words only
            import re
            pattern = r'\b' + re.escape(term) + r'\b'
            expanded = re.sub(pattern, expansion, expanded, flags=re.IGNORECASE)
        return expanded

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> list[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def create_documents(
        self,
        texts: list[str],
        metadata: Optional[list[dict]] = None,
        expand_terms: bool = True,
    ) -> list[dict]:
        """Create documents with processed text."""
        documents = []

        for i, text in enumerate(texts):
            processed = self.expand_terms(text) if expand_terms else text
            doc_metadata = metadata[i] if metadata and i < len(metadata) else {}
            doc_metadata["source_index"] = i

            documents.append({
                "content": processed,
                "original": text,
                "metadata": doc_metadata,
            })

        return documents