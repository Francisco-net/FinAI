"""Loop do agente: conversa com o Claude via streaming, executa ferramentas e emite eventos.

O motor é independente de interface: CLI, API (SSE) e Streamlit consomem o mesmo
gerador de ``AgentEvent``.

Detalhes importantes da API (Claude Opus 5.5):
- thinking é sempre adaptativo; o "quanto pensar" é controlado por ``output_config.effort``;
- o histórico é APENAS ACRESCENTADO (append-only), devolvendo ``response.content`` intacto,
  inclusive blocos de thinking — editar turnos anteriores invalida o raciocínio preservado;
- ``tool_choice`` forçado não é suportado: o modelo decide chamar ferramentas (auto);
- ``fallbacks="default"`` refaz no servidor uma requisição recusada pelos classificadores
  de segurança em outro modelo recomendado.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import anthropic

from finai.agent.prompts import SYSTEM_PROMPT
from finai.agent.tools import execute_tool, tool_definitions
from finai.config import Settings, get_settings
from finai.rag.pipeline import FinancialRAG, RAGConfig, create_financial_rag

log = logging.getLogger(__name__)

EventKind = Literal[
    "text", "thinking", "tool_start", "tool_result", "notice", "refusal", "error", "done", "rag_context"
]


@dataclass
class AgentEvent:
    kind: EventKind
    text: str = ""
    tool_name: str | None = None
    data: Any = None
    is_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "text": self.text,
            "tool_name": self.tool_name,
            "data": self.data,
            "is_error": self.is_error,
        }


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    requests: int = 0

    def add(self, usage: Any) -> None:
        self.requests += 1
        self.input_tokens += usage.input_tokens or 0
        self.output_tokens += usage.output_tokens or 0
        self.cache_read_input_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cache_creation_input_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0


@dataclass
class FinAIAgent:
    client: anthropic.Anthropic | None = None
    settings: Settings = field(default_factory=get_settings)
    messages: list[dict[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    last_portfolio: dict[str, Any] | None = None
    last_simulation: dict[str, Any] | None = None
    rag: FinancialRAG | None = None
    use_rag: bool = True

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = anthropic.Anthropic()
        self._tools = tool_definitions()
        self._system = [
            {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
        ]

        # Initialize RAG if enabled
        if self.use_rag:
            self._init_rag()

    def _init_rag(self) -> None:
        """Initialize RAG system with financial knowledge."""
        try:
            self.rag = create_financial_rag(
                vector_store_type="memory",
                embedding_model_type="mock",  # Use mock for testing, change to sentence-transformer for production
                top_k=5,
            )

            # Load financial knowledge base
            self._load_financial_knowledge()
        except Exception as e:
            log.warning(f"Failed to initialize RAG: {e}")
            self.rag = None

    def _load_financial_knowledge(self) -> None:
        """Load financial knowledge into RAG system."""
        if not self.rag:
            return

        from finai.knowledge.glossary import consultar_glossario
        from finai.knowledge.assets import asset_class_info

        # Load glossary terms
        glossary_terms = []
        for term in consultar_glossario("").get("termos_disponiveis", []):
            result = consultar_glossario(term)
            if result.get("encontrado"):
                glossary_terms.append({
                    "term": result["termo"],
                    "definition": result["definicao"],
                    "category": "glossario",
                })

        # Load asset class info
        asset_classes = []
        for cls_id, info in asset_class_info().items():
            asset_classes.append({
                "term": info.nome,
                "definition": f"{info.descricao}. Veículos típicos: {', '.join(info.veiculos)}. "
                             f"Retorno real esperado: {info.retorno_real_esperado*100:.1f}% a.a. "
                             f"Volatilidade: {info.volatilidade*100:.1f}% a.a. "
                             f"Liquidez: {info.liquidez}. Risco: {info.risco}.",
                "category": "classe_ativo",
            })

        all_knowledge = glossary_terms + asset_classes
        if all_knowledge:
            self.rag.add_financial_knowledge(all_knowledge)
            log.info(f"Loaded {len(all_knowledge)} financial knowledge items into RAG")

    def reset(self) -> None:
        self.messages.clear()
        self.last_portfolio = None
        self.last_simulation = None

    # ------------------------------------------------------------------ request
    def _request_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": self._system,
            "tools": self._tools,
            "messages": self.messages,
            "thinking": {"type": "adaptive", "display": self.settings.thinking_display},
            "output_config": {"effort": self.settings.effort},
        }
        if self.settings.refusal_fallback:
            params["betas"] = ["server-side-fallback-2026-07-01"]
            params["fallbacks"] = "default"
        return params

    def _enhance_with_rag(self, user_text: str) -> str:
        """Enhance user query with RAG context."""
        if not self.rag or not self.use_rag:
            return user_text

        try:
            context, sources = self.rag.get_context(user_text)
            if context:
                enhanced = f"{user_text}\n\n[Contexto do conhecimento financeiro:]\n{context}"
                return enhanced
        except Exception as e:
            log.warning(f"RAG enhancement failed: {e}")

        return user_text

    # ------------------------------------------------------------------ loop
    def chat(self, user_text: str) -> Iterator[AgentEvent]:
        """Envia uma mensagem do usuário e gera eventos até o assistente concluir a resposta."""
        assert self.client is not None

        # Enhance with RAG context
        enhanced_text = self._enhance_with_rag(user_text)
        if enhanced_text != user_text and self.rag:
            # Get sources for event emission
            context, sources = self.rag.get_context(user_text)
            if sources:
                yield AgentEvent(
                    "rag_context",
                    text=f"Recuperados {len(sources)} documentos relevantes da base de conhecimento.",
                    data=[{
                        "content": s.document.content[:200] + "...",
                        "score": s.score,
                        "metadata": s.document.metadata,
                    } for s in sources],
                )

        self.messages.append({"role": "user", "content": enhanced_text})
        json_retries = 0

        for _ in range(self.settings.max_tool_iterations):
            try:
                with self.client.beta.messages.stream(**self._request_params()) as stream:
                    for event in stream:
                        if (
                            event.type == "content_block_start"
                            and event.content_block.type == "tool_use"
                        ):
                            yield AgentEvent("tool_start", tool_name=event.content_block.name)
                        elif event.type == "content_block_delta":
                            if event.delta.type == "text_delta":
                                yield AgentEvent("text", text=event.delta.text)
                            elif event.delta.type == "thinking_delta":
                                yield AgentEvent("thinking", text=event.delta.thinking)
                    response = stream.get_final_message()
                json_retries = 0
            except ValueError:
                # JSON de ferramenta impossível de interpretar: refaz o turno (limitado)
                json_retries += 1
                if json_retries > 2:
                    yield AgentEvent(
                        "error", text="Falha repetida ao interpretar a chamada de ferramenta."
                    )
                    return
                yield AgentEvent("notice", text="Refazendo a última etapa...")
                continue
            except anthropic.APIError as exc:
                yield AgentEvent("error", text=_friendly_api_error(exc))
                return
            except TypeError as exc:
                # O SDK só descobre a falta de credenciais ao montar a requisição
                if "authentication" not in str(exc).lower():
                    raise
                yield AgentEvent("error", text=MISSING_CREDENTIALS)
                return

            self.usage.add(response.usage)

            if response.stop_reason == "refusal":
                # Não acrescenta a resposta parcial ao histórico; a interface deve descartá-la.
                yield AgentEvent(
                    "refusal",
                    text="Não consigo ajudar com esse pedido. Posso ajudar com seu planejamento de investimentos?",
                )
                return

            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "pause_turn":
                continue

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                if response.stop_reason == "max_tokens":
                    yield AgentEvent(
                        "notice", text="A resposta foi interrompida pelo limite de tamanho."
                    )
                yield AgentEvent("done")
                return

            results: list[dict[str, Any]] = []
            for block in tool_uses:
                if response.stop_reason == "max_tokens":
                    # Entrada possivelmente truncada: não executa, pede para o modelo refazer.
                    results.append(
                        _tool_result(
                            block.id,
                            "Entrada truncada pelo limite de tokens; refaça a chamada.",
                            True,
                        )
                    )
                    continue
                outcome = execute_tool(block.name, block.input)
                if not outcome.is_error:
                    if block.name == "montar_carteira":
                        self.last_portfolio = outcome.data
                    elif block.name == "simular_patrimonio":
                        self.last_simulation = outcome.data
                yield AgentEvent(
                    "tool_result",
                    tool_name=block.name,
                    data=outcome.data if not outcome.is_error else outcome.content,
                    is_error=outcome.is_error,
                )
                results.append(_tool_result(block.id, outcome.content, outcome.is_error))
            # Todos os resultados em UMA mensagem (mantém as chamadas paralelas funcionando bem)
            self.messages.append({"role": "user", "content": results})

        yield AgentEvent(
            "error", text="Limite de etapas atingido nesta resposta. Tente reformular o pedido."
        )


def _tool_result(tool_use_id: str, content: str, is_error: bool) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        result["is_error"] = True
    return result


MISSING_CREDENTIALS = (
    "Nenhuma credencial da Anthropic encontrada. Defina ANTHROPIC_API_KEY no arquivo .env "
    "(crie uma chave em https://platform.claude.com) e reinicie a aplicação."
)


def _friendly_api_error(exc: anthropic.APIError) -> str:
    if isinstance(exc, anthropic.AuthenticationError):
        return "Chave da API inválida. Confira ANTHROPIC_API_KEY no arquivo .env."
    if isinstance(exc, anthropic.PermissionDeniedError):
        return "A chave da API não tem permissão para este modelo/recurso."
    if isinstance(exc, anthropic.NotFoundError):
        return "Modelo ou endpoint não encontrado. Verifique FINAI_MODEL."
    if isinstance(exc, anthropic.RateLimitError):
        return "Limite de requisições atingido. Aguarde alguns segundos e tente novamente."
    if isinstance(exc, anthropic.BadRequestError):
        log.error("Requisição inválida: %s", exc)
        return f"Requisição inválida para a API: {exc.message}"
    if isinstance(exc, anthropic.APIStatusError):
        return f"Erro no serviço de IA (HTTP {exc.status_code}). Tente novamente em instantes."
    if isinstance(exc, anthropic.APIConnectionError):
        return "Sem conexão com o serviço de IA. Verifique sua internet."
    return f"Erro inesperado na API: {exc}"